"""Physical world-model BPTT using the existing VAIC teacher interfaces."""

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from hydra.core.config_store import ConfigStore
from tensordict import TensorDict

from ..ppo.ppo_vel import PPOConfig, PPOVEL
from .core import ObservationCodec
from .replay import CompactSequenceReplay, ReplayPrefetcher
from .actor_step import ActorStepController, FixedRateActorStep
from .execution import CUDAGraphBackward
from .schedule import FrameTrainingSchedule


@dataclass
class WMConfig(PPOConfig):
    _target_: str = "active_adaptation.learning.wm.WM"
    name: str = "wm"
    phase: str = "train"
    vecnorm: str = "train"
    in_keys: List[str] = ("command", "policy", "object_", "priv", "object_geo_", "height", "vel_command")
    raw_world_model_observations: bool = True
    # Real collection steps per iteration (PPO keeps 32).
    train_every: int = 16
    # WM-training-only task settings. scripts/train.py merges them into `task`
    # before the environment is built; an explicit `task.*` command-line
    # override wins, and play.py/PPO keep the task file's values. Strict body
    # error termination keeps imagined failures informative, the lost-contact
    # rule is position-only (frc_thres=0), and starts use return-based RSI.
    task_defaults: Dict[str, Any] = field(default_factory=lambda: {
        "termination": {
            "cum_body_pos_error": {"min_steps": 2},
            "cum_body_ori_error": {"min_steps": 2},
            "cum_body_pos_error_local": {"min_steps": 2},
            "cum_body_ori_error_local": {"min_steps": 2},
            "cum_lost_contact_steps": {"frc_thres": 0.0, "min_steps": 8},
        },
        "command": {"reset_sampling": "return"},
    })
    wm_hidden_dims: List[int] = (1024, 1024, 1024)
    # Teacher actor MLP and critic MLP hidden widths (defaults: the PPO teacher's).
    # SuperMimic uses actor (2048, 1024, 512, 256) and critic (256, 128).
    actor_hidden_dims: List[int] = (512, 256, 256)
    critic_hidden_dims: List[int] = (512, 256, 128)
    wm_lr: float = 2e-4
    wm_updates: int = 48  # Fixed-mode model updates per iteration (epoch mode: 3x16).
    wm_batch_size: int = 1024
    wm_sequence_length: int = 8
    # Optimization units only; task rewards and reported physical errors stay unchanged.
    # Defaults are SuperMimic-style Bryson nominal errors (root pose in cm/deg,
    # joints in 0.1 rad), so multi-step position/orientation drift dominates the
    # fit instead of noisy velocities. Hydra merges dict overrides key by key;
    # the legacy units are root_pos=.5, root_rotation=.5, joint_pos=.25,
    # joint_velocity=2.5 (root_velocity=2.5 is unchanged by default).
    # object_root_pos/rotation/velocity override the root units of the object
    # only: object_root_velocity=1.0 gives the board twist 6.25x the robot's weight.
    wm_loss_scales: dict[str, float] = field(default_factory=lambda: {
        "root_pos": .02, "root_rotation": .05, "joint_pos": .1, "joint_velocity": 3.,
        "object_root_velocity": 1.})
    # Root mode models the board pose/twist, without wheel-state prediction.
    wm_object_dynamics: str = "root"
    # Convert compatible articulated checkpoints to the default root model;
    # only this architecture conversion resets the WM optimizer.
    wm_root_warmstart: bool = True
    # SuperMimic-style dynamics: False predicts the next state from generalized
    # state, actuator state and domain parameters only, without contact sensor
    # inputs, contact presence/force heads or their losses. Requires the
    # kinematic imagination reward and position-only imagined lost contact.
    # Checkpoints trained with contact prediction need algo.wm_contact_model=true.
    wm_contact_model: bool = False
    # "heading" (SuperMimic/NeRD-style): the WM network sees the robot root as
    # height, body-frame gravity and twist, and the object relative to the robot
    # heading frame; root increments are predicted in heading frames. This makes
    # the dynamics invariant to horizontal translation and yaw. "world" keeps
    # raw world-frame root inputs/outputs (checkpoints before 2026-11-18 night).
    wm_frame: str = "heading"
    # Add every robot body link position in each object's root frame to the WM
    # input (exact FK of the input state, never predicted). Object-agnostic:
    # the same features apply to a board, box or cart. Checkpoints trained
    # without it need algo.wm_object_body_features=false.
    wm_object_body_features: bool = True
    # Limit compounding object/robot dynamics error while keeping H8 WM fitting.
    imagination_horizon: int = 16
    # SuperMimic-style default: each actor step imagines from a fresh replay
    # batch, alternating with WM fitting (8 model updates per actor step here).
    # "epochs" reuses one start pool for ppo_epochs passes, a PPO structure
    # that pathwise gradients do not need and that lets the actor exploit
    # model error on the same states.
    schedule_mode: str = "fixed"
    wm_epochs: int = 3
    wm_num_minibatches: int = 16
    # ppo_epochs/num_minibatches retain the common config names, but control
    # BPTT actor/critic minibatches here, never a PPO surrogate objective.
    # The following sample ratios apply only to legacy frame-mode runs.
    wm_predictions_per_frame: float = 1.0
    actor_starts_per_frame: float = 0.03125
    # Fixed-mode counts per iteration: SuperMimic's 4096 fresh starts per actor
    # step; wm_updates are spread evenly between the actor steps.
    actor_updates: int = 6
    actor_batch_size: int = 4096
    actor_microbatch_size: int = 1024
    # Bound CPU tensor workers for collection/replay; all model fitting stays on CUDA.
    cpu_threads: int = 4
    # Epoch mode pairs each actor minibatch with exactly one critic step.
    critic_epochs: int = 3
    critic_num_minibatches: int = 8
    critic_microbatch_size: int = 8192
    critic_cache_device: str = "auto"
    critic_cache_reserve_gib: float = 4.0
    # Initial rates, adapted using measured action-distribution changes.
    # Keep critic's inherited cfg.lr independent from the BPTT actor rate.
    actor_lr: float = 1.5e-5
    encoder_lr: float = 5e-6
    actor_lr_max: float = 1e-4
    actor_target_kl: float = 0.01
    actor_max_kl: float = 0.02
    actor_drift_batch_size: int = 256
    # Actor update rule. "adam" follows SuperMimic's BPTT learner: a fixed rate,
    # Adam beta2=.95 and per-element gradient clipping after the norm clip.
    # "kl" restores the legacy measured-KL trust region (actor_lr, encoder_lr,
    # actor_lr_max, actor_target_kl, actor_max_kl), whose step size scales with
    # the decaying exploration std. Non-actor optimizer groups keep the
    # configured encoder_lr/actor_lr ratio in both modes.
    actor_optimizer: str = "adam"
    actor_adam_lr: float = 2e-4
    actor_adam_beta2: float = .95
    actor_grad_clip_value: float = .1
    # Optional start filter: keep a start when every pose/object consecutive-
    # violation counter is at most this many steps, dropping the long
    # pre-termination fall that large min_steps log as ordinary replay.
    # Lost-contact counters are never filtered: a foot off the board is
    # recoverable, and those starts are the only ones where imagined contact
    # termination can fire. Negative (default) disables; short body min_steps
    # already end falls in the real env.
    actor_start_violation_tolerance: int = -1
    # Imagined (actor BPTT) lost-contact termination: False judges contact by
    # FK foot-to-target distance only; True also requires the learned contact
    # force, matching the real env. The real env and replay counters are unchanged.
    imagination_contact_force_termination: bool = False
    # Imagined actor/critic rewards use only functions of the predicted
    # kinematics (SuperMimic): contact-timer, impact, slip and torque terms are
    # dropped and the contact kernel keeps only its FK position factor. WM
    # fitting/validation and the real env keep the exact task reward.
    imagination_kinematic_reward: bool = True
    # Clamp critic bootstrap values and lambda-return targets at the analytic
    # ceiling max_step_reward / (1 - gamma) per reward group (SuperMimic v_max).
    critic_value_ceiling: bool = True
    cuda_graph_wm: bool = True
    cuda_graph_actor: bool = True
    replay_prefetch: bool = True
    compact_model_transport: bool = True
    critic_updates: int = 4
    warmup_updates: int = 2
    warmup_model_updates: int = 10
    # SuperMimic-style alternating model/policy updates after a short prefill.
    # Accuracy metrics remain diagnostics unless explicitly requested as a gate.
    require_model_validation: bool = False
    replay_capacity: int = 524288
    validation_capacity: int = 131072
    validation_env_stride: int = 10
    validation_batch_size: int = 256
    # Longer diagnostics do not expand the optional H2/H4/H8 admission gate.
    validation_horizon: int = 32
    validation_good_checks: int = 3
    validation_max_reward_rmse: float = 0.04
    validation_max_error_ratio: float = 0.95
    validation_max_auxiliary_nmse: float = 0.75
    validation_max_contact_event_error: float = 0.1
    validation_max_contact_event_miss_rate: float = 0.5
    validation_min_contact_events: int = 4
    replay_device: str = "cpu"
    profile_timing: bool = False
    # Preserve recent progress when a long WM run is interrupted between sparse checkpoints.
    checkpoint_interval_seconds: float = 600.0
    wm_reward_coef: float = 0.1
    wm_auxiliary_coef: float = 1.0
    entropy_coef_start: float = 0.0
    entropy_coef_end: float = 0.0
    init_noise_scale: float = 0.05
    exploration_noise_end: float = 0.05
    exploration_decay_frames: int = 20000000
    # Unlike PPO warm-starts, a WM resume must preserve learned exploration.
    load_noise_scale: float | None = None


ConfigStore.instance().store(name="wm_train", node=WMConfig(), group="algo")


def _trunk_widths(world_model_state):
    """Hidden widths of the physical dynamics trunk's linear layers, in order."""
    widths = []
    for key, value in world_model_state.items():
        parts = key.split(".")
        if (len(parts) == 4 and parts[:2] == ["physics", "trunk"] and parts[3] == "weight"
                and parts[2].isdigit() and getattr(value, "ndim", 0) == 2):
            widths.append((int(parts[2]), int(value.shape[0])))
    return tuple(width for _, width in sorted(widths))


def _hidden_widths(module_state):
    """Output widths of a module's linear layers, excluding its output head."""
    widths = [int(value.shape[0]) for key, value in module_state.items()
              if key.endswith(".weight") and getattr(value, "ndim", 0) == 2]
    return tuple(widths[:-1])


def validation_reward_groups(world_model, count):
    """Per reward group: (modeled, contact_free) flags for validation metrics.

    ``modeled`` is False for groups a world model without contact prediction
    cannot reproduce (the factual reward-loss mask). ``contact_free`` marks the
    groups whose exact terms never read contact; it is reported in every mode so
    runs with and without contact prediction share one comparable aggregate.
    """
    mask = getattr(world_model, "reward_loss_group_mask", None)
    modeled = [True] * count if mask is None else [bool(value) for value in mask.tolist()]
    groups = getattr(getattr(world_model, "reward_model", None), "contact_free_groups", None)
    contact_free = list(groups()) if callable(groups) else None
    if len(modeled) != count or (contact_free is not None and len(contact_free) != count):
        raise RuntimeError("Reward-group validation flags do not match the reward width")
    return modeled, contact_free


@contextmanager
def frozen_parameters(*modules):
    """Freeze weights while retaining derivatives with respect to inputs."""
    parameters = list(dict.fromkeys(p for m in modules for p in m.parameters()))
    flags = [p.requires_grad for p in parameters]
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


def sequence_mask(batch):
    """Score terminal transitions, never a reset or post-terminal transition."""
    done = batch["next", "done"].squeeze(-1)
    valid = ~batch["is_init"].squeeze(-1).bool()
    if done.shape[1] > 1:
        valid[:, 1:] &= (~done[:, :-1]).cumprod(1).bool()
    return valid


def lambda_returns(reward, next_value, continuation, boundary, gamma, lmbda):
    """Detached targets; boundary cuts recursion but retains its bootstrap."""
    result = torch.empty_like(reward)
    carry = next_value[:, -1]
    for t in reversed(range(reward.shape[1])):
        tail = torch.where(boundary[:, t], next_value[:, t], carry)
        carry = reward[:, t] + gamma * continuation[:, t] * (
            (1 - lmbda) * next_value[:, t] + lmbda * tail
        )
        result[:, t] = carry
    return result


class WM(PPOVEL):
    requires_next_observations = True
    # train.py logs completed real episodes after every (long) WM iteration.
    log_every_iteration = True
    # Retain these modules only for compatibility with existing physical-WM
    # checkpoint schemas. They do not participate in collection or learning.
    student_module_names = (
        "adapt_module", "adapt_ema", "actor_adapt", "object_adapt", "object_adapt_ema",
        "object_pred_transform", "depth_cnn", "temporal_depth_gru", "temporal_depth_gru_ema",
        "dr_estimator",
    )

    def __init__(self, cfg, observation_spec, action_spec, reward_spec, device, env):
        if cfg.phase != "train":
            raise ValueError("wm_train trains the teacher; use its teacher checkpoint for evaluation.")
        for key in ("wm_updates", "wm_batch_size", "wm_sequence_length", "imagination_horizon",
                    "wm_epochs", "wm_num_minibatches", "ppo_epochs", "num_minibatches",
                    "actor_updates", "actor_batch_size", "actor_microbatch_size", "critic_epochs",
                    "critic_num_minibatches", "critic_microbatch_size", "replay_capacity",
                    "validation_capacity", "validation_batch_size", "validation_horizon", "validation_good_checks",
                    "validation_min_contact_events"):
            if getattr(cfg, key) < 1:
                raise ValueError(f"{key} must be positive")
        if cfg.schedule_mode not in ("epochs", "frames", "fixed"):
            raise ValueError("schedule_mode must be 'fixed' (default), 'epochs', or legacy 'frames'")
        if cfg.schedule_mode == "epochs":
            if cfg.actor_batch_size % cfg.num_minibatches:
                raise ValueError("Actor start pool must divide evenly into actor minibatches")
            if (cfg.critic_epochs, cfg.critic_num_minibatches) != (cfg.ppo_epochs, cfg.num_minibatches):
                raise ValueError("Epoch mode pairs actor and critic epochs/minibatches one-to-one")
            if cfg.wm_epochs != cfg.ppo_epochs:
                raise ValueError("Epoch mode requires matching model and policy epoch counts")
        if cfg.critic_cache_device not in ("cpu", "auto"):
            raise ValueError("critic_cache_device must be 'cpu' or bounded 'auto'")
        if not math.isfinite(cfg.critic_cache_reserve_gib) or cfg.critic_cache_reserve_gib < 0:
            raise ValueError("critic_cache_reserve_gib must be finite and nonnegative")
        if cfg.warmup_model_updates < 0 or cfg.warmup_updates < 0:
            raise ValueError("Warmup update counts must be nonnegative")
        if cfg.wm_sequence_length > cfg.train_every:
            raise ValueError("WM training sequence must not exceed train_every")
        if cfg.imagination_horizon < 2:
            raise ValueError("Imagination horizon must be at least two steps")
        if cfg.actor_drift_batch_size < 0:
            raise ValueError("actor_drift_batch_size must be nonnegative (zero disables diagnostics)")
        if cfg.replay_capacity < cfg.wm_sequence_length:
            raise ValueError("replay_capacity must hold at least one training sequence")
        if cfg.value_norm:
            raise ValueError("wm_train currently requires value_norm=false for target-critic consistency")
        if "wm_observation" not in observation_spec:
            raise ValueError("wm_train requires RawWorldModelObservation before VecNorm")
        if cfg.entropy_coef_start != 0 or cfg.entropy_coef_end != 0:
            raise ValueError("WM uses deterministic BPTT; configure collection noise instead of entropy")
        if cfg.validation_env_stride < 2 or env.num_envs < 2:
            raise ValueError("WM needs at least two environments for disjoint validation")
        if not 0 <= cfg.validation_max_contact_event_miss_rate <= 1:
            raise ValueError("validation_max_contact_event_miss_rate must be between zero and one")
        if cfg.train_dr_estimator:
            raise ValueError("wm_train is teacher-only; train_dr_estimator must be false")
        for key in ("wm_hidden_dims", "actor_hidden_dims", "critic_hidden_dims"):
            widths = getattr(cfg, key)
            if not len(widths) or any(isinstance(width, bool) or int(width) != width or width < 1
                                      for width in widths):
                raise ValueError(f"{key} must be a non-empty list of positive integers")
        if cfg.actor_optimizer not in ("adam", "kl"):
            raise ValueError("actor_optimizer must be 'adam' (SuperMimic fixed rate) or legacy 'kl'")
        if cfg.wm_frame not in ("world", "heading"):
            raise ValueError("wm_frame must be 'heading' (default) or 'world'")
        if not cfg.wm_contact_model and (not cfg.imagination_kinematic_reward
                                         or cfg.imagination_contact_force_termination):
            raise ValueError("algo.wm_contact_model=false predicts no contact force or timers: it requires "
                             "algo.imagination_kinematic_reward=true and "
                             "algo.imagination_contact_force_termination=false")
        super().__init__(cfg, observation_spec, action_spec, reward_spec, device, env)
        from .teacher import configure_raw_teacher
        configure_raw_teacher(self, observation_spec)
        for name in self.student_module_names:
            module = getattr(self, name, None)
            if module is not None:
                module.requires_grad_(False)
        for name in ("opt_adapt", "opt_adapt_actor", "opt_dr_estimator"):
            if hasattr(self, name):
                delattr(self, name)
        raw_spec = observation_spec["wm_observation"]
        self.codec = ObservationCodec(raw_spec, list(raw_spec.keys()))
        self.replay = CompactSequenceReplay(cfg.replay_capacity, device=cfg.replay_device)
        self.validation_replay = CompactSequenceReplay(cfg.validation_capacity, device=cfg.replay_device)
        object.__setattr__(self, "_observation_normalizer", None)
        self._normalizer_mode = None
        self._norm_snapshot = {}
        self._model_updates_since_resume = 0
        self._optimizer_model_updates_since_resume = 0
        self._context_schema = ()
        self._validated_horizon = 0
        self._good_checks = {h: 0 for h in (2, 4, 8) if h <= cfg.imagination_horizon}
        self._oracle_checked = False
        self._collected_frames = 0
        # Noise is an independent collection schedule, not a BPTT objective.
        self.actor.module[0][2].module.actor_std.requires_grad_(False)
        if cfg.actor_optimizer == "adam":
            ratios = (1.,) + (cfg.encoder_lr / cfg.actor_lr,) * (len(self.opt_policy.param_groups) - 1)
            self._actor_step = FixedRateActorStep(
                self.opt_policy, lr=cfg.actor_adam_lr, beta2=cfg.actor_adam_beta2,
                clip_value=cfg.actor_grad_clip_value, group_lr_ratios=ratios,
            )
        else:
            self._actor_step = ActorStepController(
                self.opt_policy, actor_lr=cfg.actor_lr, encoder_lr=cfg.encoder_lr,
                max_actor_lr=cfg.actor_lr_max,
                target_kl=cfg.actor_target_kl, max_kl=cfg.actor_max_kl,
            )
        self.lr_policy = self.opt_policy.param_groups[0]["lr"]
        self._wm_backward = None
        self._actor_backward = None
        self._actor_backward_horizon = None
        self._critic_payload_cache = None
        self._critic_payload_signature = None
        self._critic_cache_reclaim_signatures = set()
        self._frame_schedule = FrameTrainingSchedule(cfg.wm_predictions_per_frame, cfg.actor_starts_per_frame)

    def get_rollout_policy(self, mode="train"):
        from .teacher import raw_teacher_rollout_policy
        return raw_teacher_rollout_policy(self)

    def make_tensordict_primer(self):
        # The teacher has no recurrent student hidden state to initialize.
        return None

    def set_world_model_context(self, transform):
        from .physics import PhysicalWorldModel
        from .adapter import VAICWorldModelAdapter
        from .reward import VAICRewardModel
        from .grounded import GroundedWorldModel
        from .imagination import VAICImaginationReference
        self._context_schema = transform.context_schema
        env = transform._base_env
        self.world_model = GroundedWorldModel(
            PhysicalWorldModel(self.codec, transform, env, self.cfg.wm_hidden_dims, self.device,
                               training_loss_scales=self.cfg.wm_loss_scales,
                               object_dynamics=self.cfg.wm_object_dynamics,
                               contact_model=bool(self.cfg.wm_contact_model),
                               frame=str(self.cfg.wm_frame),
                               object_body_features=bool(self.cfg.wm_object_body_features)),
            VAICWorldModelAdapter(env, transform, self.codec),
            VAICRewardModel(env, transform, self.codec).to(self.device),
        )
        if self.cfg.wm_object_dynamics == "root":
            from .object_mode import validate_root_only_task
            validate_root_only_task(self.world_model.adapter, self.world_model.reward_model)
        self.world_model.set_imagination_reference(
            VAICImaginationReference(env, self.world_model.adapter).to(self.device))
        self.world_model.set_soft_contact_force(bool(self.cfg.imagination_contact_force_termination))
        self.world_model.set_kinematic_imagination_reward(bool(self.cfg.imagination_kinematic_reward))
        ceilings = self.world_model.reward_model.max_step_rewards(
            kinematic=bool(self.cfg.imagination_kinematic_reward)) / (1 - self.cfg.gamma)
        if not self.cfg.critic_value_ceiling:
            ceilings = torch.full_like(ceilings, math.inf)
        self._value_ceiling = ceilings.to(self.device)
        self.opt_world_model = torch.optim.Adam(self.world_model.parameters(), lr=self.cfg.wm_lr)

    def set_observation_normalizer(self, normalizer, mode):
        # A live transform must not become a policy submodule/checkpoint child.
        object.__setattr__(self, "_observation_normalizer", normalizer)
        self._normalizer_mode = mode

    def _snapshot_normalizer(self):
        norm = self._observation_normalizer
        values = {}
        if norm is not None and self._normalizer_mode is not None:
            loc, scale = norm.loc, norm.scale
            values = {
                key: (loc[key].detach().clone(), scale[key].detach().clone().clamp_min(norm.eps))
                for key in norm.in_keys if key in self.codec.keys
            }
        compatible = values.keys() == self._norm_snapshot.keys() and all(
            old.shape == new.shape and old.device == new.device and old.dtype == new.dtype
            for key in values for old, new in zip(self._norm_snapshot[key], values[key]))
        if compatible:
            # Captured actor graphs retain these tensor addresses. Statistics
            # still refresh once per real collection, in place.
            for key in values:
                for old, new in zip(self._norm_snapshot[key], values[key]):
                    old.copy_(new)
        else:
            self._norm_snapshot = values
            if self._actor_backward is not None:
                self._actor_backward.clear()
                self._actor_backward = None

    def _policy_observation(self, state):
        td = self.codec.unpack(state)
        for key, (loc, scale) in self._norm_snapshot.items():
            td[key] = (td[key] - loc) / scale
        return td

    def _teacher_dist(self, state):
        from .teacher import teacher_dist
        return teacher_dist(self, self._policy_observation(state))

    def _value(self, state):
        return self.critic(self._policy_observation(state))["state_value"]

    def _add_replay(self, rollout):
        if not hasattr(self, "world_model"):
            raise RuntimeError("The physical WM adapter must be initialized before training")
        # Fixed environment split: no overlapping windows/episodes can leak
        # between model training and validation, even across replay insertions.
        ids = torch.arange(rollout.shape[0], device=rollout.device)
        heldout = ids % self.cfg.validation_env_stride == 0
        for selected, replay in ((~heldout, self.replay), (heldout, self.validation_replay)):
            selected_ids = selected.nonzero(as_tuple=True)[0]
            batch_size = (selected_ids.numel(), *rollout.batch_size[1:])
            def pack_partition(observations):
                # Materialize each selected group before copying into the packed
                # feature slice. Direct index_select(out=...) into a strided
                # destination has unstable CPU thread scaling at task scale.
                first = observations[self.codec.keys[0]]
                packed = first.new_empty(*batch_size, self.codec.state_dim)
                for key, section in self.codec.slices.items():
                    packed[..., section].copy_(
                        observations[key][selected].reshape(*batch_size, -1))
                return packed
            replay_batch = TensorDict({
                "state": pack_partition(rollout["wm_observation"]),
                "action": rollout["action"].index_select(0, selected_ids),
                "is_init": rollout["is_init"].index_select(0, selected_ids),
                "next": TensorDict({
                    "state": pack_partition(rollout["next", "wm_observation"]),
                    **{key: rollout["next", key].index_select(0, selected_ids)
                       for key in ("reward", "terminated", "done", "discount")},
                }, batch_size),
            }, batch_size)
            valid = ~replay_batch["is_init"].squeeze(-1)
            if replay is self.replay and valid.any():
                self.world_model.update_stats(
                    replay_batch["state"], replay_batch["action"],
                    replay_batch["next", "state"], replay_batch["next", "reward"], valid=valid)
            replay.add(replay_batch.detach())
            del replay_batch
        self._collected_frames += rollout.numel()

    def _sample(self, length, batch_size=None):
        # Replay guarantees owned, ordinary tensors even after inference-mode
        # collection. A second full clone only duplicates hundreds of MB.
        return self.replay.sample_compact(batch_size or self.cfg.wm_batch_size, length, device=self.device)

    def _sample_actor_starts(self, batch_size=None):
        return self.replay.sample_starts(batch_size or self.cfg.actor_batch_size, device=self.device)

    def _diagnostic_rng_devices(self):
        device = torch.device(self.device)
        return ([device.index if device.index is not None else torch.cuda.current_device()]
                if device.type == "cuda" else [])

    @torch.no_grad()
    def _capture_collection_actor_anchor(self):
        """Snapshot a fixed replay anchor without consuming training randomness."""
        count = self.cfg.actor_drift_batch_size
        if not count or not len(self.replay):
            return None
        with torch.random.fork_rng(devices=self._diagnostic_rng_devices()):
            try:
                starts = self._sample_actor_starts(count)
            except ValueError as error:
                if str(error) == "Replay has no non-reset initial states":
                    return None
                raise
            distribution = self._teacher_dist(starts["state"])
            return starts, distribution.mean.detach().clone(), distribution.stddev.detach().clone()

    @torch.no_grad()
    def _collection_actor_drift(self, anchor):
        """Measure cumulative mean drift using the same normalization and std.

        This is a diagnostic, not an update clamp. Holding the anchor's
        collection std fixed separates policy-mean changes from the independent
        exploration schedule. The ordinary actor update KL remains unchanged.
        """
        if anchor is None:
            return {}
        starts, initial_mean, initial_std = anchor
        with torch.random.fork_rng(devices=self._diagnostic_rng_devices()):
            difference = self._teacher_dist(starts["state"]).mean - initial_mean
            valid = ~starts["is_init"].squeeze(-1).bool()
            difference, initial_std = difference[valid], initial_std[valid]
            if not difference.numel():
                return {}
            return {"actor/collection_mean_kl": (.5 * (difference / initial_std).square().sum(-1)).mean().item(),
                    "actor/collection_action_delta_rms": difference.square().mean().sqrt().item(),
                    "actor/collection_anchor_samples": int(valid.sum().item())}

    def _model_state_columns(self):
        # Grounded WM fitting reads physical context and recorded reward only.
        # Actor, oracle, validation, and critic inputs ALWAYS retain all fields.
        if (self.cfg.compact_model_transport and hasattr(self.world_model, "physics")
                and hasattr(self.world_model, "adapter")):
            return self.codec.slices["wm_context"]
        return None

    def _sample_model(self, length=None):
        return self.replay.sample_compact(self.cfg.wm_batch_size, length or self.cfg.wm_sequence_length,
                                         device=self.device, state_columns=self._model_state_columns())

    def _world_model_loss(self, batch):
        """Pure tensor objective, shared by eager and captured execution."""
        valid = sequence_mask(batch)
        state = batch["state"][:, 0]
        sums = {}
        denom = valid.sum().clamp_min(1)
        for t in range(batch.shape[1]):
            target = batch["next", "state"][:, t]
            losses, prediction = self.world_model.transition_loss(
                state, batch["action"][:, t], target,
                batch["next", "reward"][:, t], batch["next", "terminated"][:, t],
                batch["next", "discount"][:, t],
                return_prediction=True,
            )
            for key, value in losses.items():
                sums[key] = sums.get(key, 0) + (value * valid[:, t]).sum() / denom
            state = torch.where(valid[:, t, None], prediction, target.detach())
        loss = (sums["dynamics"] + self.cfg.wm_reward_coef * sums["reward"]
                + self.cfg.wm_auxiliary_coef * sums.get("auxiliary", 0))
        metrics = {f"wm/{key}_loss": value.detach() for key, value in sums.items()}
        metrics["wm/valid_fraction"] = valid.float().mean()
        return loss, metrics

    def _fit_world_model(self, batch):
        self.opt_world_model.zero_grad(set_to_none=True)
        captured = self.cfg.cuda_graph_wm and torch.device(self.device).type == "cuda"
        if captured:
            if self._wm_backward is None:
                self._wm_backward = CUDAGraphBackward(
                    self._world_model_loss_tensors, self.world_model.parameters())
            loss, metrics = self._wm_backward.run(tuple(batch[key] for key in self._model_batch_keys))
            metrics = dict(metrics)
        else:
            loss, metrics = self._world_model_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite physical world-model training loss")
        if not captured:
            loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.cfg.max_grad_norm, error_if_nonfinite=True)
        self.opt_world_model.step()
        metrics["wm/grad_norm"] = grad.detach()
        values = torch.stack(list(metrics.values())).tolist()
        return {**dict(zip(metrics, values)), "performance/wm_cuda_graph": float(captured)}

    _model_batch_keys = ("state", "action", "is_init", ("next", "state"),
                         ("next", "reward"), ("next", "terminated"),
                         ("next", "done"), ("next", "discount"))

    def _world_model_loss_tensors(self, *inputs):
        return self._world_model_loss(TensorDict(
            dict(zip(self._model_batch_keys, inputs)), inputs[0].shape[:2]))

    def _critic_observation(self, state):
        # Store only the original critic's normalized inputs, not thousands of
        # simulator-context fields that the critic never reads.
        return self._policy_observation(state).select(*self.critic.in_keys)

    def _imagine_microbatch(self, starts, horizon, denominator):
        """One pure H-step rollout; caller accumulates its scaled gradient.

        No replay future or factual done enters this path. The reference
        provider advances motion independently from the sampled real state.
        The critic is frozen here but keeps its state derivatives for the
        actor's final bootstrap. All critic training data/targets are detached.
        """
        state = starts["state"]
        valid = ~starts["is_init"].squeeze(-1).bool()
        weight = valid[:, None].to(state.dtype)
        survival = weight.clone()
        immediate = torch.zeros_like(weight)
        bootstrap_weight = torch.zeros_like(weight)
        bootstrap_state = state
        seen_boundary = torch.zeros_like(weight, dtype=torch.bool)
        observations, rewards, continuations, boundaries, alive_steps = [], [], [], [], []
        alive = valid.clone()
        effective_steps = state.new_zeros(())
        soft_model = None
        if hasattr(self.world_model, "begin_soft_imagination"):
            soft_model, termination_state = self.world_model.begin_soft_imagination(state, horizon)
        hazard_sum = state.new_zeros(())
        hazard_samples = state.new_zeros(())
        for t in range(horizon):
            alive_steps.append(alive.clone())
            observations.append(self._critic_observation(state).detach())
            dist = self._teacher_dist(state)
            action = dist.mean
            if t == 0:
                initial_action = action.detach().clone()
                initial_std = getattr(dist, "stddev", torch.ones_like(action)).detach().clone()
            if soft_model is None:
                prediction, reward, term_prob, discount, truncated = self.world_model.imagine(state, action)
            else:
                prediction, reward, term_prob, discount, truncated, termination_state = self.world_model.imagine_soft(
                    state, action, soft_model, termination_state)
            continuation = discount * (1 - term_prob)
            hazard_sum = hazard_sum + (survival.detach() * term_prob.detach()).sum()
            hazard_samples = hazard_samples + survival.detach().sum()
            effective_steps = effective_steps + survival.detach().sum()
            immediate = immediate + weight * (reward * self.reward_scales).sum(-1, keepdim=True)
            weight = weight * self.cfg.gamma * continuation
            boundary = truncated.bool() | (t == horizon - 1)
            first_boundary = boundary & ~seen_boundary
            bootstrap_state = torch.where(first_boundary, prediction, bootstrap_state)
            bootstrap_weight = bootstrap_weight + weight * first_boundary
            seen_boundary = seen_boundary | boundary
            weight = weight * (~boundary)
            survival = survival * continuation * (~boundary)
            rewards.append(reward.detach())
            continuations.append(continuation.detach())
            boundaries.append(truncated.detach().bool())
            # Failure is a probability, not a boolean episode boundary. Keep
            # predicting through the horizon, weighting returns by survival.
            # Only true motion/time truncation stops rows; converting a positive
            # hazard to bool would incorrectly stop every row after one step.
            state = torch.where(alive[:, None], prediction, state)
            alive = alive & ~truncated.bool().squeeze(-1)
        bootstrap_value = self._value(bootstrap_state)
        ceiling = getattr(self, "_value_ceiling", None)
        if ceiling is not None:
            raw_value = bootstrap_value
            bootstrap_value = torch.minimum(bootstrap_value, ceiling)
            # Return removed by the ceiling, in objective units (0 when it never binds).
            clamped = bootstrap_weight * ((raw_value - bootstrap_value) * self.reward_scales).sum(-1, keepdim=True)
        else:
            clamped = torch.zeros_like(bootstrap_weight)
        bootstrap = bootstrap_weight * (bootstrap_value * self.reward_scales).sum(-1, keepdim=True)
        objective = immediate + bootstrap
        loss = -objective.sum() / denominator
        observations = torch.stack(observations, 1)
        with torch.no_grad():
            # Same critic, same generated rollout, before any actor/critic
            # optimizer step. No target critic or real-replay value targets.
            # The original critic only adds TensorDict keys (CatTensors keeps
            # its inputs); isolate that structure without duplicating every
            # H-step observation tensor in the CUDA graph's memory pool.
            values = self.critic(observations.clone(False))["state_value"]
            last_value = self._value(state.detach())
            next_values = torch.cat((values[:, 1:], last_value[:, None]), 1)
            targets = lambda_returns(torch.stack(rewards, 1), next_values,
                                     torch.stack(continuations, 1), torch.stack(boundaries, 1),
                                     self.cfg.gamma, self.cfg.lmbda)
            if ceiling is not None:
                targets = torch.minimum(targets, ceiling)
        payload = TensorDict({"observation": observations.detach(), "target": targets.detach(),
                              "valid": torch.stack(alive_steps, 1)}, [len(starts), horizon])
        metrics = {"model_return": objective.detach().sum() / denominator,
                   "bootstrap_return": bootstrap.detach().sum() / denominator,
                   "bootstrap_clamped_value": clamped.detach().sum() / denominator,
                   "immediate_return": immediate.detach().sum() / denominator,
                   "effective_horizon": effective_steps / denominator,
                   "termination_probability_sum": hazard_sum,
                   "termination_probability_samples": hazard_samples,
                   "soft_termination": valid.sum().to(state.dtype) / denominator
                                       if soft_model is not None else state.new_zeros(())}
        return loss, payload, metrics, initial_action, initial_std

    def _actor_loss_tensors(self, state, is_init, denominator):
        starts = TensorDict({"state": state, "is_init": is_init}, [state.shape[0]])
        loss, payload, metrics, action, std = self._imagine_microbatch(
            starts, self._actor_backward_horizon, denominator)
        outputs = {"metric/" + key: value for key, value in metrics.items()}
        for key, value in payload.items(True, True):
            path = (key,) if isinstance(key, str) else key
            outputs["payload/" + "/".join(path)] = value
        outputs.update(action=action, std=std)
        return loss, outputs

    def _actor_microbatch_backward(self, starts, horizon, denominator):
        captured = getattr(self.cfg, "cuda_graph_actor", False) and starts["state"].is_cuda
        if not captured:
            loss, payload, metrics, action, std = self._imagine_microbatch(starts, horizon, denominator)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite imagined actor objective")
            loss.backward()
            return payload, metrics, action, std, False
        if self._actor_backward is None or self._actor_backward_horizon != horizon:
            if self._actor_backward is not None:
                self._actor_backward.clear()
            self._actor_backward_horizon = horizon
            parameters = [p for group in self.opt_policy.param_groups for p in group["params"]]
            self._actor_backward = CUDAGraphBackward(self._actor_loss_tensors, parameters)
        loss, outputs = self._actor_backward.run(
            (starts["state"], starts["is_init"], denominator), accumulate=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite captured imagined actor objective")
        payload = TensorDict({tuple(key[8:].split("/")): value for key, value in outputs.items()
                              if key.startswith("payload/")}, [len(starts), horizon])
        metrics = {key[7:]: value for key, value in outputs.items() if key.startswith("metric/")}
        return payload, metrics, outputs["action"], outputs["std"], True

    def _fit_imagined_critic(self, payload, *, cache_metrics=None, epochs=None, num_minibatches=None):
        """Fit detached imagined targets with exact gradient accumulation.

        The cache holds only fixed normalized critic inputs and GAE targets.
        It moves to CUDA once if free memory allows a separate safety reserve;
        otherwise the exact same minibatches stream from CPU.
        Each logical minibatch accumulates its complete mean loss before one
        optimizer step; microbatching changes memory use, not sample weighting.
        """
        # Keep the original CPU permutation RNG even when imagination writes
        # observations/targets directly into the persistent CUDA cache.
        epochs = self.cfg.critic_epochs if epochs is None else epochs
        num_minibatches = self.cfg.critic_num_minibatches if num_minibatches is None else num_minibatches
        indices = payload["valid"].reshape(-1).to("cpu").nonzero(as_tuple=True)[0]
        if not indices.numel():
            return {"critic/value_loss": 0., "critic/grad_norm": 0.,
                    "critic/updates": 0, "critic/sample_passes": 0}
        if indices.numel() < num_minibatches:
            raise ValueError("Imagined critic rollout has fewer samples than configured minibatches")
        # Draw the same permutation before optional transport. Cache placement
        # must not change which RNG/device selects the training minibatches.
        permutation = indices[torch.randperm(indices.numel(), device=indices.device)]
        if cache_metrics is None:
            data, cache_metrics = self._cache_imagined_critic(payload)
        else:
            data = payload
        data = data.reshape(-1)
        permutation = permutation.to(data["valid"].device)
        minibatches = permutation.tensor_split(num_minibatches)
        losses, gradients, sample_passes = [], [], 0
        for _ in range(epochs):
            for selected in minibatches:
                self.opt_critic.zero_grad(set_to_none=True)
                total_loss = torch.zeros((), device=self.device)
                for micro in selected.split(self.cfg.critic_microbatch_size):
                    batch = data[micro].to(self.device)
                    estimate = self.critic(batch["observation"])["state_value"]
                    loss = (estimate - batch["target"]).square().mean(-1).sum() / selected.numel()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite imagined critic loss")
                    loss.backward()
                    total_loss = total_loss + loss.detach()
                gradient = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm,
                                                          error_if_nonfinite=True)
                self.opt_critic.step()
                losses.append(total_loss)
                gradients.append(gradient.detach())
                sample_passes += selected.numel()
        return {"critic/value_loss": torch.stack(losses).mean().item(),
                "critic/grad_norm": torch.stack(gradients).mean().item(),
                "critic/updates": len(losses), "critic/sample_passes": sample_passes,
                **cache_metrics}

    def _cache_imagined_critic(self, payload, *, batch_size=None):
        """Copy a complete payload or acquire owned storage for microbatch writes.

        With ``batch_size`` the payload is only a shape/dtype template. No
        observations are transferred: return an empty full-batch CUDA cache,
        or ``None`` when CPU staging is required. The caller fills every slice
        before fitting the critic and must copy before a graph output is reused.
        """
        fields = tuple(payload.items(True, True))
        acquire_only = batch_size is not None
        batch_size = torch.Size(payload.batch_size if batch_size is None else batch_size)
        if len(batch_size) != payload.ndim or batch_size[1:] != payload.batch_size[1:]:
            raise ValueError("Critic cache template may resize only its first batch dimension")
        shapes = {key: (*batch_size, *value.shape[payload.ndim:]) for key, value in fields}
        size = sum(math.prod(shapes[key]) * value.element_size() for key, value in fields)
        cache_on_gpu, reused = False, False
        auto_cuda = self.cfg.critic_cache_device == "auto" and torch.device(self.device).type == "cuda"
        free = 0
        reclaim_attempted, reclaimed_free = False, 0
        if auto_cuda:
            signature = (str(torch.device(self.device)), tuple(batch_size), tuple(sorted(
                (((key,) if isinstance(key, str) else key), shapes[key], value.dtype)
                for key, value in fields)))
            cache = getattr(self, "_critic_payload_cache", None)
            compatible = cache is not None and getattr(self, "_critic_payload_signature", None) == signature
            if not compatible:
                # Release incompatible owned storage. CUDA's reserved graph
                # pools are deliberately NOT counted as available memory.
                self._critic_payload_cache = self._critic_payload_signature = None
                cache = None
            free, _ = torch.cuda.mem_get_info(self.device)
            if compatible:
                # This allocation already passed the memory-reserve check.
                # Requiring another allocation's worth of global free memory
                # would incorrectly fall back after the allocator reserves it.
                if not acquire_only:
                    for key, value in fields:
                        cache[key].copy_(value)
                cache_on_gpu = reused = True
            else:
                reserve = self.cfg.critic_cache_reserve_gib * 1024 ** 3
                attempts = getattr(self, "_critic_cache_reclaim_signatures", None)
                if attempts is None:
                    attempts = self._critic_cache_reclaim_signatures = set()
                if free < size + reserve and signature not in attempts:
                    # Driver free memory excludes reusable ordinary allocator
                    # blocks left by previous WM/actor work. Release only unused
                    # blocks once for this shape before choosing CPU staging.
                    # Live tensors and captured graph pools stay owned; never
                    # count reserved bytes as free or relax the safety reserve.
                    attempts.add(signature)
                    reclaim_attempted = True
                    before_release = free
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info(self.device)
                    reclaimed_free = max(free - before_release, 0)
                if free >= size + reserve:
                    if acquire_only:
                        cache = TensorDict({key: torch.empty(shapes[key], dtype=value.dtype, device=self.device)
                                            for key, value in fields}, batch_size=batch_size)
                    else:
                        cache = payload.to(self.device)
                        # Also guarantee ownership for an already-CUDA caller.
                        if any(cache[key].data_ptr() == value.data_ptr() for key, value in fields):
                            cache = cache.clone()
                    self._critic_payload_cache = cache
                    self._critic_payload_signature = signature
                    cache_on_gpu = True
            if cache_on_gpu:
                payload = self._critic_payload_cache
        else:
            self._critic_payload_cache = self._critic_payload_signature = None
        if acquire_only and not cache_on_gpu:
            payload = None
        return payload, {"performance/critic_gpu_cache": float(cache_on_gpu),
                         "performance/critic_cache_reused": float(reused),
                         "performance/critic_cache_bytes": size,
                         "performance/critic_cache_free_bytes": free,
                         "performance/critic_cache_reclaim_attempted": float(reclaim_attempted),
                         "performance/critic_cache_reclaimed_free_bytes": reclaimed_free,
                         "performance/critic_cache_fallback": float(auto_cuda and not cache_on_gpu)}

    def _filter_failing_starts(self, starts):
        """Mark replay starts that are already failing as invalid imagination starts.

        Consecutive-violation terminations (min_steps>1) log every step between
        the first violation and the termination as ordinary replay. BPTT from
        there spends the actor batch on falls the model predicts worst and whose
        saturated rewards carry little useful gradient. ``is_init`` already
        removes a row from the objective, critic targets and KL measurement.

        Lost-contact counters are skipped: those states are recoverable, and
        they are the only starts from which imagined contact termination can
        fire within the horizon.

        Returns ``(starts, kept_fraction)``; the fraction is ``None`` when the
        filter is disabled or the task has no pose/object counters. If no start
        survives, the minibatch is returned unchanged (and the fraction reports 0).
        """
        tolerance = self.cfg.actor_start_violation_tolerance
        reward = getattr(getattr(self, "world_model", None), "reward_model", None)
        terms = tuple(term for term in getattr(reward, "termination_terms", ())
                      if getattr(term, "kind", None) != "cum_lost_contact_steps")
        if tolerance < 0 or not terms:
            return starts, None
        state, is_init = starts["state"], starts["is_init"]
        batch = state.shape[:-1]
        counts = torch.stack([
            reward._get(state, f"termination.{term.name}._cum_error_mixin__cum_steps").reshape(batch)
            for term in terms], -1)
        valid = ~is_init.reshape(batch).bool()
        keep = valid & (counts <= tolerance).all(-1)
        valid_count, kept = int(valid.sum().item()), int(keep.sum().item())
        if kept == 0:
            return starts, 0.
        filtered = TensorDict({"state": state, "is_init": (~keep).reshape(is_init.shape).to(is_init.dtype)},
                              starts.batch_size)
        return filtered, kept / max(valid_count, 1)

    def _imagine(self, starts, horizon=None, *, train_critic=True):
        # Accept old callers' sequence containers by selecting only their first
        # state. Their factual future/done fields are deliberately never read.
        def synchronize():
            if getattr(self.cfg, "profile_timing", False) and torch.device(self.device).type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
        synchronize()
        actor_started = time.perf_counter()
        if starts.ndim == 2:
            starts = starts[:, 0]
        starts = starts.select("state", "is_init")
        filter_starts = getattr(self, "_filter_failing_starts", None)
        starts, healthy_fraction = filter_starts(starts) if filter_starts is not None else (starts, None)
        horizon = self.cfg.imagination_horizon if horizon is None else horizon
        valid = ~starts["is_init"].squeeze(-1).bool()
        denominator = valid.sum().clamp_min(1)
        metrics, payloads, initial_actions, initial_stds = {}, [], [], []
        self.entropy_coef = 0.
        self.opt_policy.zero_grad(set_to_none=True)
        captured_any = False
        critic_payload, critic_cache_metrics, payload_offset = None, None, 0
        with frozen_parameters(self.world_model, self.critic):
            for microbatch_index, micro in enumerate(starts.split(self.cfg.actor_microbatch_size)):
                payload, info, action, std, captured = self._actor_microbatch_backward(micro, horizon, denominator)
                captured_any |= captured
                # Graph outputs reuse storage on the following microbatch.
                if train_critic:
                    if microbatch_index == 0:
                        critic_payload, critic_cache_metrics = self._cache_imagined_critic(
                            payload, batch_size=(len(starts), horizon))
                    if critic_payload is None:
                        payloads.append(payload.to("cpu"))
                    else:
                        for key, value in payload.items(True, True):
                            critic_payload[key][payload_offset:payload_offset + len(micro)].copy_(value)
                    payload_offset += len(micro)
                initial_actions.append(action.clone())
                initial_stds.append(std.clone())
                for name, value in info.items():
                    metrics[name] = metrics.get(name, 0) + value
                del payload
            params = [p for group in self.opt_policy.param_groups for p in group["params"]]
            grad = torch.nn.utils.clip_grad_norm_(params, self.cfg.max_grad_norm, error_if_nonfinite=True)
            def evaluate_mean():
                return torch.cat([self._teacher_dist(micro["state"]).mean
                                  for micro in starts.split(self.cfg.actor_microbatch_size)])
            step_info = self._actor_step.step(evaluate_mean, torch.cat(initial_actions),
                                               torch.cat(initial_stds), valid)
            self.lr_policy = self.opt_policy.param_groups[0]["lr"]
        info = {"actor/" + key: value.item() for key, value in metrics.items()}
        info["actor/termination_probability"] = (
            info.pop("actor/termination_probability_sum") /
            max(1., info.pop("actor/termination_probability_samples")))
        info.update({"actor/policy_loss": -info["actor/model_return"], "actor/grad_norm": grad.item(),
                     "actor/encoder_priv_grad_norm": 0.,
                     "actor/entropy_coef": 0., "actor/std_grad_norm": 0.,
                     "actor/planned_starts": len(starts), "actor/valid_starts": valid.sum().item(),
                     "actor/imagined_steps": len(starts) * horizon,
                     "actor/accepted_starts": valid.sum().item() * step_info["actor/step_accepted"],
                     "performance/actor_cuda_graph": float(captured_any),
                     **step_info})
        if healthy_fraction is not None:
            info["actor/healthy_start_fraction"] = healthy_fraction
        synchronize()
        info["performance/wm_imagination_seconds"] = time.perf_counter() - actor_started
        if train_critic:
            critic_started = time.perf_counter()
            if critic_payload is None:
                critic_payload = torch.cat(payloads, 0)
                # The selected CPU fallback stays CPU for this logical batch.
            fit_options = {"epochs": 1, "num_minibatches": 1} if self.cfg.schedule_mode == "epochs" else {}
            info.update(self._fit_imagined_critic(critic_payload, cache_metrics=critic_cache_metrics, **fit_options))
            synchronize()
            info["performance/wm_critic_seconds"] = time.perf_counter() - critic_started
        return info

    @torch.no_grad()
    def verify_oracle(self, batch, *, raise_on_error=True):
        """Check equations against REAL transitions before trusting gradients."""
        valid = sequence_mask(batch)
        state = batch["state"][valid][:256]
        target = batch["next", "state"][valid][:256]
        action = batch["action"][valid][:256]
        if state.shape[0] == 0:
            raise RuntimeError("No valid real transitions available for WM oracle checks")
        reconstructed = self.world_model.adapter.rebuild(target, target, action, previous_state=state)
        errors = {}
        for key in self.codec.keys:
            if key != "wm_context":
                errors[f"oracle/observation/{key}"] = (reconstructed[..., self.codec.slices[key]]
                                                        - target[..., self.codec.slices[key]]).abs().max().item()
        reward, terminated, discount = self.world_model.reward_model(state, target, action, target)
        errors["oracle/reward"] = (reward - batch["next", "reward"][valid][:256]).abs().max().item()
        errors["oracle/termination"] = (terminated.bool() != batch["next", "terminated"][valid][:256]).float().mean().item()
        errors["oracle/discount"] = (discount - batch["next", "discount"][valid][:256]).abs().max().item()
        failed = {key: value for key, value in errors.items() if not torch.isfinite(torch.tensor(value)) or value > 2e-4}
        physics = getattr(self.world_model, "physics", None)
        if physics is not None and hasattr(physics, "oracle_contact_transition"):
            # The sensor clock and timer recurrence is known, not learned. Even
            # a tiny rounding difference can flip the original first-contact
            # threshold, so require exact equality for this part of the oracle.
            for name, prediction in physics.oracle_contact_transition(state, target).items():
                actual = physics.context(target, name)
                key = f"oracle/{name}"
                errors[key] = (prediction - actual).abs().max().item()
                if not torch.equal(prediction, actual):
                    failed[key] = errors[key]
        if failed and raise_on_error:
            raise RuntimeError(f"WM equations do not match the existing VAIC environment: {failed}")
        self._oracle_checked = not failed
        return errors

    @torch.no_grad()
    def _validate_world_model(self):
        horizon = min(self.cfg.validation_horizon, self.cfg.train_every, self.cfg.imagination_horizon)
        batch = self.validation_replay.sample(self.cfg.validation_batch_size, horizon, self.device)
        info = self.verify_oracle(batch) if not self._oracle_checked else {}
        valid = sequence_mask(batch)
        # Starts must be actual observations, never the environment's zero reset.
        valid &= valid[:, :1]
        state, baseline = batch["state"][:, 0], batch["state"][:, 0]
        accepted = []
        prefix_good = True
        contact_prefix_good = True
        reward_groups = None
        limits = {"joint_pos": .15, "robot_root_pos": .03,
                  "robot_root_rotation": .15, "robot_root_velocity": 1.0,
                  "object_root_pos": .03, "object_root_rotation": .15,
                  "object_root_velocity": 1.0}
        for t in range(horizon):
            target = batch["next", "state"][:, t]
            action = batch["action"][:, t]
            state, reward, terminated, _ = self.world_model(state, action, target)
            baseline = self.world_model.baseline_prediction(baseline, action, target)
            if t + 1 not in (1, 2, 4, 8, 16, 32, horizon):
                continue
            mask = valid[:, t]
            denom = mask.sum().clamp_min(1)
            valid_count = mask.sum().item()
            reduce = lambda value: (value * mask).sum().item() / max(valid_count, 1)
            def aggregate(prediction):
                measurements = self.world_model.validation_errors(prediction, target)
                # Transfer all reductions together instead of synchronizing the
                # device once for every scalar metric at every horizon.
                keys = list(measurements)
                sums = (torch.stack([measurements[key] for key in keys], -1)
                        * mask[:, None]).sum(0).tolist()
                totals = dict(zip(keys, sums))
                errors = {key: value / max(valid_count, 1) for key, value in totals.items()}
                for prefix in ("", "all_sensor_"):
                    true_key, miss_key = (f"{prefix}contact_event_{kind}_count" for kind in ("true", "miss"))
                    count_keys = {true_key, miss_key}
                    if count_keys & measurements.keys():
                        if not count_keys <= measurements.keys():
                            raise RuntimeError("Contact-event validation requires both true and missed event counts")
                        # Event-free transitions must not hide missed contacts.
                        for key in count_keys:
                            errors[key] = totals[key]
                        errors[f"{prefix}contact_event_miss_rate"] = errors[miss_key] / max(errors[true_key], 1)
                return errors
            errors = aggregate(state)
            base_errors = aggregate(baseline)
            # These fields form the documented readiness contract. Missing
            # measurements must fail closed rather than silently opening BPTT.
            missing = (set(limits) | {"normalized_dynamics"}) - errors.keys()
            if missing:
                raise RuntimeError(f"Physical WM validation is missing metrics: {sorted(missing)}")
            reward_error = reward - batch["next", "reward"][:, t]
            group_mse = (reward_error.square() * mask[:, None]).sum(0) / denom
            if reward_groups is None:
                reward_groups = validation_reward_groups(self.world_model, reward.shape[-1])
            modeled, contact_free = reward_groups
            # A pooled RMSE retains rare large errors; averaging per-row square
            # roots would underweight them. Groups the model cannot reproduce
            # (contact-reading groups without contact prediction) are reported
            # separately and excluded from the headline and gate metrics.
            group_rmse = group_mse.sqrt().tolist()
            modeled_ids = [group for group, keep in enumerate(modeled) if keep]
            reward_rmse = group_mse[modeled_ids].mean().sqrt().item() if modeled_ids else 0.
            reward_weights = getattr(self, "reward_scales", 1. / reward.shape[-1])
            if not all(modeled):
                reward_weights = reward_weights * torch.tensor(modeled, dtype=reward.dtype, device=reward.device)
            weighted_rmse = reduce((reward_error * reward_weights).sum(-1).square()) ** .5
            if contact_free is not None and any(contact_free):
                free_ids = [group for group, keep in enumerate(contact_free) if keep]
                info[f"validation/h{t+1}/reward_contact_free_rmse"] = group_mse[free_ids].mean().sqrt().item()
            term_error = reduce((terminated.bool() != batch["next", "terminated"][:, t]).float().squeeze(-1))
            ratio = errors["normalized_dynamics"] / max(base_errors["normalized_dynamics"], 1e-8)
            for key, value in errors.items():
                info[f"validation/h{t+1}/{key}"] = value
                info[f"validation/h{t+1}/baseline_{key}"] = base_errors[key]
            info.update({f"validation/h{t+1}/error_ratio": ratio,
                         f"validation/h{t+1}/reward_rmse": reward_rmse,
                         f"validation/h{t+1}/reward_max_group_rmse": max((group_rmse[group] for group in modeled_ids),
                                                                         default=0.),
                         f"validation/h{t+1}/reward_weighted_rmse": weighted_rmse,
                         f"validation/h{t+1}/termination_error": term_error,
                         f"validation/h{t+1}/valid_samples": valid_count})
            for group, error in enumerate(group_rmse):
                name = "reward_group" if modeled[group] else "reward_contact_group"
                info[f"validation/h{t+1}/{name}_{group}_rmse"] = error
            absolute_ok = all(errors[key] <= limit for key, limit in limits.items())
            # Near-static states can have a practically perfect constant-velocity
            # baseline; accept only very small absolute errors in that case.
            tiny = all(errors[key] <= limit * .1 for key, limit in limits.items())
            auxiliary_ok = True
            if "auxiliary" in errors:
                auxiliary_ok = errors["auxiliary"] <= self.cfg.validation_max_auxiliary_nmse and (
                    errors["auxiliary"] <= .05 or errors["auxiliary"] <= self.cfg.validation_max_error_ratio * base_errors["auxiliary"])
            contact_ok = True
            if "contact_event_error" in errors:
                contact_ok &= errors["contact_event_error"] <= self.cfg.validation_max_contact_event_error
            if "contact_event_true_count" in errors:
                event_support = errors["contact_event_true_count"] >= self.cfg.validation_min_contact_events
                info[f"validation/h{t+1}/contact_event_support"] = float(event_support)
                contact_ok &= (event_support and errors["contact_event_miss_rate"]
                               <= self.cfg.validation_max_contact_event_miss_rate)
            finite = all(math.isfinite(v) for v in (*errors.values(), *base_errors.values(), reward_rmse, term_error, ratio))
            good = (finite and valid_count >= max(8, self.cfg.validation_batch_size // 4)
                    and absolute_ok and auxiliary_ok and (ratio <= self.cfg.validation_max_error_ratio or tiny)
                    and reward_rmse <= self.cfg.validation_max_reward_rmse and term_error <= .02)
            info[f"validation/h{t+1}/continuous_ok"] = float(good)
            info[f"validation/h{t+1}/contact_ok"] = float(contact_ok)
            # H16/H32 are diagnostic-only. Keep every existing gate counter,
            # contact readiness flag, and actor admission limited to H<=8.
            if t + 1 <= 8:
                prefix_good = prefix_good and good
                contact_prefix_good = contact_prefix_good and contact_ok
            if t + 1 <= 8 and t + 1 in self._good_checks:
                # Hard contact events have zero pathwise derivative. Validate
                # short, delayed-action BPTT with physical/force/reward/termination
                # errors; require event fidelity before expanding beyond H=2.
                # This is limited policy admission, not full contact accuracy.
                admitted = prefix_good and (t + 1 == 2 or contact_prefix_good)
                self._good_checks[t+1] = self._good_checks[t+1] + 1 if admitted else 0
                if self._good_checks[t+1] >= self.cfg.validation_good_checks:
                    accepted.append(t+1)
        # Accuracy regressions close the gate again; a one-time warmup is not
        # sufficient as the actor changes the training distribution.
        self._validated_horizon = max(accepted, default=0)
        info["validation/approved_horizon"] = self._validated_horizon
        info["validation/short_pathwise_ready"] = float(self._good_checks[2] >= self.cfg.validation_good_checks)
        info["validation/contact_ready"] = float(contact_prefix_good)
        # Preserve the legacy <=H8 real-transition Bellman diagnostic even when
        # physical rollouts are inspected over a longer imagination horizon.
        critic_batch, critic_valid = batch[:, :8], valid[:, :8]
        next_value = self._value(critic_batch["next", "state"])
        targets = lambda_returns(critic_batch["next", "reward"], next_value,
                                 critic_batch["next", "discount"] * (~critic_batch["next", "terminated"]).float(),
                                 critic_batch["next", "done"], self.cfg.gamma, self.cfg.lmbda)
        critic_error = (self._value(critic_batch["state"]) - targets).square().mean(-1)
        info["validation/critic_rmse"] = ((critic_error * critic_valid).sum() / critic_valid.sum().clamp_min(1)).sqrt().item()
        return info

    def train_op(self, tensordict):
        timings = {}
        def timed(name, operation, *args):
            synchronize = self.cfg.profile_timing and torch.device(self.device).type == "cuda"
            if synchronize:
                torch.cuda.current_stream(self.device).synchronize()
            started = time.perf_counter()
            result = operation(*args)
            if synchronize:
                torch.cuda.current_stream(self.device).synchronize()
            key = f"performance/wm_{name}_seconds"
            timings[key] = timings.get(key, 0.) + time.perf_counter() - started
            return result
        self._snapshot_normalizer()
        timed("replay", self._add_replay, tensordict)
        collection_anchor = self._capture_collection_actor_anchor()
        frames = tensordict.numel()
        frame_mode = self.cfg.schedule_mode == "frames"
        epoch_mode = self.cfg.schedule_mode == "epochs"
        model_cost = self.cfg.wm_batch_size * self.cfg.wm_sequence_length
        actor_minibatch_size = (self.cfg.actor_batch_size // self.cfg.num_minibatches
                                if epoch_mode else self.cfg.actor_batch_size)
        if epoch_mode:
            planned_models = self.cfg.wm_epochs * self.cfg.wm_num_minibatches
            planned_actors = self.cfg.ppo_epochs * self.cfg.num_minibatches
        elif frame_mode:
            self._frame_schedule.accrue(frames)
            planned_models = self._frame_schedule.available("wm", model_cost)
            planned_actors = self._frame_schedule.available("actor", self.cfg.actor_batch_size)
        else:
            planned_models, planned_actors = self.cfg.wm_updates, self.cfg.actor_updates
        infos = []
        model_updates, actor_attempts = 0, 0
        transferred_bytes, materialized_bytes = 0, 0
        prefetch = (self.cfg.replay_prefetch and self.replay.device.type == "cpu"
                    and torch.device(self.device).type == "cuda")

        def model_step(batch):
            nonlocal model_updates
            result = timed("model", self._fit_world_model, batch)
            infos.append(result)
            model_updates += 1
            self._optimizer_model_updates_since_resume += 1
            if frame_mode:
                self._frame_schedule.consume("wm", model_cost)

        def fit_models(count):
            nonlocal transferred_bytes, materialized_bytes
            if not count:
                return
            if (self.cfg.cuda_graph_wm and torch.device(self.device).type == "cuda"
                    and (self._wm_backward is None or not self._wm_backward.has_cache)):
                # Capture without a concurrently active H2D worker.
                batch = timed("sample", self._sample_model, self.cfg.wm_sequence_length)
                model_step(batch)
                del batch
                count -= 1
            if prefetch and count:
                with ReplayPrefetcher(self.replay, self.cfg.wm_batch_size,
                                      self.cfg.wm_sequence_length, count, self.device,
                                      compact=True, state_columns=self._model_state_columns()) as batches:
                    for _ in range(count):
                        model_step(timed("sample", next, batches))
                    transferred_bytes += batches.transferred_bytes
                    materialized_bytes += batches.materialized_bytes
            else:
                for _ in range(count):
                    model_step(timed("sample", self._sample_model, self.cfg.wm_sequence_length))

        ready, actor_horizon = False, self.cfg.imagination_horizon
        if epoch_mode:
            from .epoch import EpochReplayPlan
            # Keep only sample indices between epochs. Replay remains on CPU;
            # each fixed-size minibatch is materialized for GPU training on demand.
            model_plan = EpochReplayPlan(
                self.replay, self.cfg.wm_num_minibatches * self.cfg.wm_batch_size,
                self.cfg.wm_batch_size, self.cfg.wm_epochs,
                sequence_length=self.cfg.wm_sequence_length)
            actor_plan = EpochReplayPlan(
                self.replay, self.cfg.actor_batch_size, actor_minibatch_size,
                self.cfg.ppo_epochs, starts_only=True)
            for epoch in range(self.cfg.wm_epochs):
                first_batch = 0
                if (self.cfg.cuda_graph_wm and torch.device(self.device).type == "cuda"
                        and (self._wm_backward is None or not self._wm_backward.has_cache)):
                    # Graph capture cannot overlap the prefetch transfer worker.
                    model_step(timed("sample", lambda: model_plan.sample(
                        epoch, 0, self.device, state_columns=self._model_state_columns())))
                    first_batch = 1
                if prefetch and first_batch < self.cfg.wm_num_minibatches:
                    with model_plan.prefetch(epoch, self.device,
                                             state_columns=self._model_state_columns(),
                                             start_batch=first_batch) as batches:
                        for _ in range(first_batch, self.cfg.wm_num_minibatches):
                            model_step(timed("sample", next, batches))
                        transferred_bytes += batches.transferred_bytes
                        materialized_bytes += batches.materialized_bytes
                else:
                    for minibatch in range(first_batch, self.cfg.wm_num_minibatches):
                        model_step(timed("sample", lambda: model_plan.sample(
                            epoch, minibatch, self.device, state_columns=self._model_state_columns())))
                if epoch == 0:
                    validation = timed("validation", self._validate_world_model)
                    validation["validation/model_updates_at_check"] = model_updates
                    infos.append(validation)
                actor_horizon = (self._validated_horizon if self.cfg.require_model_validation
                                 else self.cfg.imagination_horizon)
                ready = (self._optimizer_model_updates_since_resume >= self.cfg.warmup_model_updates
                         and actor_horizon >= 2 and self._oracle_checked)
                if ready:
                    for minibatch in range(self.cfg.num_minibatches):
                        starts = timed("sample", lambda: actor_plan.sample(epoch, minibatch, self.device))
                        infos.append(timed("actor", self._imagine, starts, actor_horizon))
                        actor_attempts += 1
        # Spread model fitting between successive policy/critic updates instead
        # of fitting the entire replay budget against one stale policy first in
        # legacy schedules. Epoch mode above alternates whole model/policy epochs.
        cycles = 0 if epoch_mode else max(planned_actors, 1)
        for cycle in range(cycles):
            target_models = math.ceil((cycle + 1) * planned_models / cycles)
            fit_models(target_models - model_updates)
            if cycle == 0:
                infos.append(timed("validation", self._validate_world_model))
            actor_horizon = (self._validated_horizon if self.cfg.require_model_validation
                             else self.cfg.imagination_horizon)
            warmed = (self._optimizer_model_updates_since_resume >= self.cfg.warmup_model_updates
                      if frame_mode else self._model_updates_since_resume >= self.cfg.warmup_updates)
            ready = warmed and actor_horizon >= 2 and self._oracle_checked
            if ready and cycle < planned_actors:
                starts = timed("sample", self._sample_actor_starts)
                infos.append(timed("actor", self._imagine, starts, actor_horizon))
                actor_attempts += 1
                if frame_mode:
                    self._frame_schedule.consume("actor", self.cfg.actor_batch_size)
        self.num_updates += 1
        self._model_updates_since_resume += 1
        collection_drift = self._collection_actor_drift(collection_anchor)
        with torch.no_grad():
            progress = min(1., self._collected_frames / max(1, self.cfg.exploration_decay_frames))
            self.actor.module[0][2].module.actor_std.fill_(
                self.cfg.init_noise_scale + progress * (self.cfg.exploration_noise_end - self.cfg.init_noise_scale))
        keys = set().union(*(item.keys() for item in infos))
        info = {key: sum(item[key] for item in infos if key in item) / sum(key in item for item in infos)
                for key in keys}
        total = lambda key: sum(item.get(key, 0) for item in infos)
        if "actor/update_mean_kl" in info:
            info["actor/update_max_kl"] = max(item["actor/update_mean_kl"] for item in infos
                                             if "actor/update_mean_kl" in item)
        scored = total("wm/valid_fraction") * model_cost
        info.update({"wm/replay_size": len(self.replay), "wm/policy_ready": float(ready),
                     "wm/object_root_only": float(getattr(self.cfg, "wm_object_dynamics", "articulated") == "root"),
                     "wm/contact_model": float(getattr(self.cfg, "wm_contact_model", True)),
                     "wm/heading_frame": float(getattr(self.cfg, "wm_frame", "world") == "heading"),
                     "wm/object_body_features": float(getattr(self.cfg, "wm_object_body_features", False)),
                     "wm/model_updates": model_updates,
                     "wm/planned_model_updates": planned_models,
                     "wm/model_prediction_samples": model_updates * model_cost,
                     "wm/scored_predictions": scored,
                     "wm/actor_updates": total("actor/step_accepted"),
                     "wm/actor_update_attempts": actor_attempts,
                     "wm/planned_actor_updates": planned_actors,
                     "wm/planned_actor_starts": planned_actors * actor_minibatch_size,
                     "wm/actor_starts": total("actor/valid_starts"),
                     "wm/accepted_actor_starts": total("actor/accepted_starts"),
                     "wm/actor_imagined_steps": total("actor/imagined_steps"),
                     "wm/actor_horizon": actor_horizon if actor_attempts else 0,
                     "wm/accuracy_gate_enabled": float(self.cfg.require_model_validation),
                     "wm/validation_size": len(self.validation_replay),
                     "wm/replay_gb": (self.replay.bytes + self.validation_replay.bytes) / 1e9,
                     "wm/scored_steps_per_frame": scored / frames,
                     "wm/imagined_steps_per_frame": total("actor/imagined_steps") / frames,
                     "wm/frame_schedule": float(frame_mode),
                     "wm/epoch_schedule": float(epoch_mode),
                     "wm/raw_teacher_input": 1.,
                     "wm/actor_minibatch_size": actor_minibatch_size,
                     "wm/prediction_credit": float(self._frame_schedule.wm_credit),
                     "wm/actor_start_credit": float(self._frame_schedule.actor_credit),
                     "critic/updates": total("critic/updates"),
                     "critic/sample_passes": total("critic/sample_passes"),
                     "performance/wm_imagination_seconds": total("performance/wm_imagination_seconds"),
                     "performance/wm_critic_seconds": total("performance/wm_critic_seconds"),
                     "performance/wm_replay_prefetch": float(prefetch),
                     "performance/wm_transfer_bytes": transferred_bytes,
                     "performance/wm_materialized_bytes": materialized_bytes,
                     "actor_std/mean": self.actor.module[0][2].module.actor_std.detach().mean().item()})
        info.update(timings)
        info.update(collection_drift)
        return info

    def state_dict(self):
        # PPOVEL.state_dict copies teacher noise into the student and refreshes
        # student EMAs. A teacher-only WM save must not mutate retained students.
        state = {name: module.state_dict() for name, module in self.named_children()}
        state.update(last_phase=self.cfg.phase, last_iter=self.env.current_iter, lr_policy=self.lr_policy)
        state["wm_version"] = 2
        state["wm_teacher_input_schema"] = self.teacher_input_schema
        state["wm_schema"] = tuple((key, tuple(self.codec.shapes[key])) for key in self.codec.keys)
        state["wm_context_schema"] = self._context_schema
        state["wm_physics_schema"] = getattr(getattr(self.world_model, "physics", None), "schema", None)
        state["wm_initialization"] = getattr(self, "_wm_initialization", None)
        state["num_updates"] = self.num_updates
        state["collected_frames"] = self._collected_frames
        state["optimizers"] = {name: getattr(self, name).state_dict() for name in self._optimizer_names()}
        state["wm_actor_step"] = self._actor_step.state_dict()
        state["wm_frame_schedule"] = self._frame_schedule.state_dict()
        return state

    def _optimizer_names(self):
        return [name for name in ("opt_world_model", "opt_policy", "opt_critic")
                if hasattr(self, name)]

    def load_state_dict(self, state_dict, strict=True):
        if state_dict.get("wm_version") != 2:
            raise ValueError("wm_train expects a physical WM v2 checkpoint; PPO and old observation-WM checkpoints are incompatible")
        if state_dict.get("wm_teacher_input_schema") != self.teacher_input_schema:
            raise ValueError("WM raw-input teacher architecture does not match this checkpoint. "
                             "Checkpoints with encoder_priv cannot resume this architecture; start a new training run.")
        schema = tuple((key, tuple(self.codec.shapes[key])) for key in self.codec.keys)
        if state_dict.get("wm_schema") != schema:
            raise ValueError("WM checkpoint observation/context schema does not match this environment")
        if state_dict.get("wm_context_schema") != self._context_schema:
            raise ValueError("WM checkpoint simulator-context fields do not match this environment")
        saved_widths = _trunk_widths(state_dict.get("world_model", {}))
        target_widths = _trunk_widths(self.world_model.state_dict())
        if saved_widths and target_widths and saved_widths != target_widths:
            widths = ",".join(str(width) for width in saved_widths)
            raise ValueError(f"This WM checkpoint has dynamics hidden widths {list(saved_widths)}, but this run "
                             f"uses {list(target_widths)}; pass 'algo.wm_hidden_dims=[{widths}]' to load it")
        for name, flag in (("actor", "actor_hidden_dims"), ("critic", "critic_hidden_dims")):
            saved = _hidden_widths(state_dict.get(name, {}))
            current = _hidden_widths(getattr(self, name).state_dict())
            if saved and current and saved != current:
                widths = ",".join(str(width) for width in saved)
                raise ValueError(f"This checkpoint has {name} hidden widths {list(saved)}, but this run uses "
                                 f"{list(current)}; pass 'algo.{flag}=[{widths}]' to load it")
        target_physics = getattr(self.world_model, "physics", None)
        migrated_world_model = None
        migration_info = None
        if state_dict.get("wm_physics_schema") != getattr(target_physics, "schema", None):
            from .physics import schema_contact_model, schema_frame, schema_object_body_features
            saved_schema = state_dict.get("wm_physics_schema")
            target_schema = getattr(target_physics, "schema", None)
            mismatched = []
            boolean = lambda value: str(value).lower()
            for flag, read, show in (("wm_contact_model", schema_contact_model, boolean),
                                     ("wm_frame", schema_frame, str),
                                     ("wm_object_body_features", schema_object_body_features, boolean)):
                saved, target = read(saved_schema), read(target_schema)
                if None not in (saved, target) and saved != target:
                    mismatched.append((flag, show(saved), show(target)))
            if mismatched:
                saved_flags = " ".join(f"algo.{flag}={saved}" for flag, saved, _ in mismatched)
                target_flags = " ".join(f"algo.{flag}={target}" for flag, _, target in mismatched)
                raise ValueError(f"This WM checkpoint was trained with {saved_flags}, but this run uses "
                                 f"{target_flags}; pass {saved_flags} to load it")
            if not getattr(self.cfg, "wm_root_warmstart", False):
                raise ValueError("WM checkpoint robot/object kinematic structure does not match this environment; "
                                 "an articulated-to-root warm-start requires algo.wm_root_warmstart=true")
            from .object_migration import migrate_object_root_physics
            source_world_model = state_dict["world_model"]
            source_physics = {name.removeprefix("physics."): value
                              for name, value in source_world_model.items() if name.startswith("physics.")}
            converted, migration_info = migrate_object_root_physics(
                source_physics, state_dict.get("wm_physics_schema"), target_physics)
            migrated_world_model = {name: value for name, value in source_world_model.items()
                                    if not name.startswith("physics.")}
            migrated_world_model.update({f"physics.{name}": value for name, value in converted.items()})
        if strict:
            missing = set(self._optimizer_names()) - state_dict.get("optimizers", {}).keys()
            if missing:
                raise ValueError(f"WM checkpoint is missing optimizer states: {sorted(missing)}")
        # Fail loudly rather than silently continuing with random dynamics.
        for name, module in self.named_children():
            module.load_state_dict(migrated_world_model if name == "world_model" and migrated_world_model is not None
                                   else state_dict[name], strict=strict)
        for name in self._optimizer_names():
            if name == "opt_world_model" and migrated_world_model is not None:
                # Input/output projections change parameter shapes. Keep the
                # teacher and critic histories, but start fresh WM Adam moments.
                self.opt_world_model = torch.optim.Adam(self.world_model.parameters(), lr=self.cfg.wm_lr)
                continue
            if name in state_dict.get("optimizers", {}):
                getattr(self, name).load_state_dict(state_dict["optimizers"][name])
        self._wm_initialization = (migration_info if migration_info is not None
                                   else state_dict.get("wm_initialization"))
        self._actor_step.load_state_dict(state_dict.get("wm_actor_step"))
        self._frame_schedule.load_state_dict(state_dict.get("wm_frame_schedule"))
        self._critic_payload_cache = self._critic_payload_signature = None
        self._critic_cache_reclaim_signatures = set()
        if self._wm_backward is not None:
            self._wm_backward.clear()
            self._wm_backward = None
        if self._actor_backward is not None:
            self._actor_backward.clear()
            self._actor_backward = None
            self._actor_backward_horizon = None
        self.num_updates = state_dict.get("num_updates", 0)
        self._collected_frames = state_dict.get("collected_frames", 0)
        self.env.set_progress(state_dict.get("last_iter", 0))
        self.lr_policy = self.opt_policy.param_groups[0]["lr"]
        # Replay is deliberately not serialized; rebuild it before actor BPTT.
        self._model_updates_since_resume = 0
        self._optimizer_model_updates_since_resume = 0
        self.replay = CompactSequenceReplay(self.cfg.replay_capacity, device=self.cfg.replay_device)
        self.validation_replay = CompactSequenceReplay(self.cfg.validation_capacity, device=self.cfg.replay_device)
        self._validated_horizon = 0
        self._good_checks = {h: 0 for h in (2, 4, 8) if h <= self.cfg.imagination_horizon}
        self._oracle_checked = False
        self._norm_snapshot = {}
        return []
