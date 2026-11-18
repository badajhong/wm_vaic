"""Regression checks for replacing PPO without changing the skateboard task.

These tests compose the real Hydra config and construct the policies on CPU;
they do not import Isaac Sim or require motion/asset files.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torchrl.data import Composite, Unbounded

import active_adaptation.learning  # noqa: F401: registers algorithm configs
from active_adaptation.learning.ppo.ppo_vel import PPOConfig, PPOVEL
from active_adaptation.learning.wm import WM, WMConfig


CONFIG_DIR = Path(__file__).resolve().parents[1] / "cfg"
TASK = "G1/vaic/skateboard_tea"


def _compose_pair(overrides=()):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        return tuple(
            compose(
                config_name="train",
                overrides=[f"algo={algorithm}", f"task={TASK}", *overrides],
            )
            for algorithm in ("ppo_vel_train", "wm_train")
        )


def _plain(config):
    return OmegaConf.to_container(config, resolve=True)


def _effective_observations(config):
    # This is the public input-selection contract used by make_env_policy:
    # actor/critic keys plus auxiliary groups with a trailing underscore.
    return {
        key: _plain(group)
        for key, group in config.task.observation.items()
        if key in config.algo.in_keys or key.endswith("_")
    }


@pytest.mark.parametrize(
    "overrides",
    [
        (),
        (
            "seed=19",
            "task.num_envs=8",
            "total_frames=16384",
            "save_interval=7",
            "wandb.mode=disabled",
            "task.randomization.object_body_randomization.mass_range=[1.3,2.1]",
        ),
    ],
)
def test_wm_preserves_task_and_shared_experiment_config(overrides):
    ppo, wm = _compose_pair(overrides)

    assert _plain(wm.task) == _plain(ppo.task)
    # The algorithm and run names intentionally distinguish the two runs.
    ppo_run, wm_run = _plain(ppo), _plain(wm)
    for run in (ppo_run, wm_run):
        run.pop("algo")
        run.pop("exp_name")
        run["wandb"].pop("name")
    assert wm_run == ppo_run
    assert wm.algo._target_ == "active_adaptation.learning.wm.WM"
    assert wm.algo.name != ppo.algo.name


def test_wm_preserves_observations_rewards_and_teacher_settings():
    ppo, wm = _compose_pair()

    assert issubclass(WMConfig, PPOConfig)
    assert tuple(wm.algo.in_keys) == tuple(ppo.algo.in_keys)
    assert _effective_observations(wm) == _effective_observations(ppo)
    assert "object_" in _effective_observations(wm)
    assert "object_geo_" in _effective_observations(wm)
    assert "ref_joint_pos_" in _effective_observations(wm)
    assert "depth" not in _effective_observations(wm)

    for key in ("reward", "termination", "randomization", "command", "action"):
        assert _plain(wm.task[key]) == _plain(ppo.task[key])
    assert list(wm.task.reward) == list(ppo.task.reward)
    assert "object_tracking" in wm.task.reward
    assert wm.task.reward.feet.feet_air_time_skateboard.enabled
    assert len(wm.task.command.object_joint_name) == 4

    for key in (
        "phase",
        "vecnorm",
        "gamma",
        "lmbda",
        "latent_dim",
        "adapt_module",
        "adapt_module_input_cmd",
        "use_object_adapt",
        "use_depth",
        "enable_residual_distillation",
        "distill_with_priv_pred",
        "train_dr_estimator",
        "value_norm",
        "normalize_before_sum",
        "clip_neg_reward",
    ):
        assert wm.algo[key] == ppo.algo[key], key
    assert wm.algo.entropy_coef_start == wm.algo.entropy_coef_end == 0
    # SuperMimic-style low collection noise and a shorter collection window.
    assert wm.algo.init_noise_scale == wm.algo.exploration_noise_end == .05
    assert wm.algo.train_every == 16 and ppo.algo.train_every == 32


def test_wm_physical_loss_and_object_mode_overrides_compose_without_task_changes():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        base = compose(config_name="train", overrides=["algo=wm_train", f"task={TASK}"])
        changed = compose(config_name="train", overrides=[
            "algo=wm_train", f"task={TASK}", "algo.wm_object_dynamics=root",
            "algo.wm_root_warmstart=true",
            "+algo.wm_loss_scales={root_pos:0.05,root_rotation:0.2,object_joint_pos:1.0,object_joint_velocity:12.0}",
        ])
    assert _plain(base.task) == _plain(changed.task)
    # Hydra merges the override into the Bryson-unit defaults key by key.
    assert _plain(base.algo.wm_loss_scales) == {
        "root_pos": .02, "root_rotation": .05, "joint_pos": .1, "joint_velocity": 3.,
        "object_root_velocity": 1.}
    assert _plain(changed.algo.wm_loss_scales) == {
        "root_pos": .05, "root_rotation": .2, "joint_pos": .1, "joint_velocity": 3.,
        "object_root_velocity": 1., "object_joint_pos": 1., "object_joint_velocity": 12.}
    assert changed.algo.wm_object_dynamics == "root"
    assert changed.algo.wm_root_warmstart
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        board = compose(config_name="train", overrides=[
            "algo=wm_train", f"task={TASK}", "++algo.wm_loss_scales.object_root_velocity=2.5"])
    # One board key is replaced without dropping the Bryson defaults.
    assert _plain(board.algo.wm_loss_scales) == {
        "root_pos": .02, "root_rotation": .05, "joint_pos": .1, "joint_velocity": 3.,
        "object_root_velocity": 2.5}
    assert base.algo.actor_microbatch_size == 1024
    assert base.algo.replay_capacity == 524288
    assert base.algo.cpu_threads == 4


def test_wm_default_uses_root_object_and_fresh_data_budget_without_cli_overrides():
    ppo, wm = _compose_pair()
    assert wm.algo.wm_object_dynamics == "root"
    assert wm.algo.wm_root_warmstart is True
    # SuperMimic-style fresh-start schedule: 6 actor steps of 4096 new starts,
    # 48 WM updates spread between them (the epoch-mode model budget).
    assert wm.algo.schedule_mode == "fixed"
    assert wm.algo.actor_updates == 6
    assert wm.algo.wm_updates == 48
    assert wm.algo.imagination_kinematic_reward is True
    assert wm.algo.critic_value_ceiling is True
    # SuperMimic-style dynamics without contact sensor inputs/heads.
    assert wm.algo.wm_contact_model is False
    assert wm.algo.wm_frame == "heading"
    assert wm.algo.wm_object_body_features is True
    assert wm.algo.imagination_contact_force_termination is False
    assert wm.algo.wm_epochs == 3
    assert wm.algo.wm_num_minibatches == 16
    assert wm.algo.ppo_epochs == 3
    assert wm.algo.num_minibatches == 8
    assert wm.algo.wm_predictions_per_frame == 1.
    assert wm.algo.actor_starts_per_frame == .03125
    assert wm.algo.critic_epochs == 3
    assert wm.algo.critic_num_minibatches == 8
    # Preserve model/teacher hidden widths and the total actor start pool.
    assert tuple(wm.algo.wm_hidden_dims) == (1024, 1024, 1024)
    assert wm.algo.wm_batch_size == 1024
    assert wm.algo.wm_sequence_length == 8
    assert wm.algo.actor_batch_size == 4096
    assert wm.algo.imagination_horizon == 16
    assert wm.task.num_envs == 4096 and wm.algo.train_every == 16
    assert _plain(wm.task) == _plain(ppo.task)
    assert _effective_observations(wm) == _effective_observations(ppo)


def test_articulated_mode_and_disabled_warmstart_remain_explicit_overrides():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        default = compose(config_name="train", overrides=["algo=wm_train", f"task={TASK}"])
        legacy = compose(config_name="train", overrides=[
            "algo=wm_train", f"task={TASK}", "algo.wm_object_dynamics=articulated",
            "algo.wm_root_warmstart=false",
        ])
    assert legacy.algo.wm_object_dynamics == "articulated"
    assert legacy.algo.wm_root_warmstart is False
    assert _plain(default.task) == _plain(legacy.task)


def _module_contract(module):
    return (
        tuple(module.in_keys),
        tuple(module.out_keys),
        tuple((name, type(child).__qualname__) for name, child in module.named_modules()),
        {key: (tuple(value.shape), value.dtype) for key, value in module.state_dict().items()},
    )


@pytest.fixture(scope="module")
def policies():
    # Small batches with the same observation-group types used by the real
    # teacher. The actual task dimensions are supplied by Isaac at runtime.
    ppo_cfg, wm_cfg = _compose_pair()
    batch, actions = 2, 29
    shapes = {
        "command": (18,),
        "policy": (32,),
        "priv": (48,),
        "object_": (24,),
        "object_geo_": (128 * 3,),
        "vel_command": (9,),
        "ref_joint_pos_": (actions,),
    }
    observation_spec = Composite(
        {key: Unbounded((batch, *shape)) for key, shape in shapes.items()},
        shape=(batch,),
    )
    wm_observation_spec = observation_spec.clone()
    raw_spec = observation_spec.clone()
    raw_spec["wm_context"] = Unbounded((batch, 17))
    wm_observation_spec["wm_observation"] = raw_spec
    action_spec = Unbounded((batch, actions))
    reward_spec = Composite(
        reward=Unbounded((batch, len(ppo_cfg.task.reward))), shape=(batch,)
    )
    env = SimpleNamespace(
        cfg=ppo_cfg.task,
        action_manager=SimpleNamespace(joint_names=[f"joint_{i}" for i in range(actions)]),
        current_iter=0,
        set_progress=lambda _: None,
        num_envs=batch,
        device="cpu",
    )

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        ppo = PPOVEL(ppo_cfg.algo, observation_spec, action_spec, reward_spec, "cpu", env)
        wm = WM(wm_cfg.algo, wm_observation_spec, action_spec, reward_spec, "cpu", env)
        yield ppo, wm, observation_spec
    finally:
        torch.set_num_threads(old_threads)


def test_wm_keeps_critic_object_and_student_modules_but_uses_raw_teacher_inputs(policies):
    ppo, wm, _ = policies

    assert isinstance(wm, PPOVEL)
    assert WM.train_op is not PPOVEL.train_op
    assert wm.reward_groups == ppo.reward_groups
    torch.testing.assert_close(wm.reward_scales, ppo.reward_scales)
    for name in (
        "critic",
        "object_transform",
        "actor_adapt",
        "adapt_module",
        "object_adapt",
    ):
        assert _module_contract(getattr(wm, name)) == _module_contract(getattr(ppo, name)), name
    assert hasattr(ppo, "encoder_priv") and not hasattr(wm, "encoder_priv")
    assert tuple(wm.actor.module[0][0].in_keys) == (
        "command", "policy", "priv", "object_")
    for actor in (ppo.actor, wm.actor):
        layers = actor.module[0][1].module
        assert [layer.out_features for layer in layers if isinstance(layer, torch.nn.Linear)] == [512, 256, 256]
    assert wm.actor.module[0][1].module[0].in_features == sum(width for _, width in wm.teacher_input_schema[1])
    assert wm.teacher_input_schema[0] == "raw_teacher_input_v1"
    optimized = {id(parameter) for group in wm.opt_policy.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in wm.actor.parameters()}


def test_teacher_critic_outputs_match_ppo_with_identical_weights(policies):
    ppo, wm, observation_spec = policies
    for name in ("critic", "object_transform"):
        getattr(wm, name).load_state_dict(getattr(ppo, name).state_dict())

    observations = observation_spec.rand()
    outputs = []
    with torch.no_grad():
        for policy in (ppo, wm):
            inputs = observations.clone()
            policy.object_transform(inputs)
            values = policy.critic(inputs)["state_value"]
            outputs.append(values)
    torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)


def _train_defaults(overrides=(), algo="wm_train"):
    import sys
    sys.path.insert(0, str(CONFIG_DIR.parent / "scripts"))
    from helpers import apply_algo_task_defaults
    cli = [f"algo={algo}", f"task={TASK}", *overrides]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="train", overrides=cli)
    OmegaConf.set_struct(cfg, False)
    return cfg, apply_algo_task_defaults(cfg, cli)


def test_wm_training_task_defaults_apply_in_train_only_and_explicit_overrides_win():
    base_task = None
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        base_task = _plain(compose(config_name="train", overrides=["algo=wm_train", f"task={TASK}"]).task)
    cfg, (applied, skipped) = _train_defaults()
    assert skipped == []
    for name in ("cum_body_pos_error", "cum_body_ori_error", "cum_body_pos_error_local", "cum_body_ori_error_local"):
        assert cfg.task.termination[name].min_steps == 2
        # Only the listed keys change; thresholds and bodies stay from the task.
        assert cfg.task.termination[name].threshold == base_task["termination"][name]["threshold"]
    assert cfg.task.termination.cum_lost_contact_steps.frc_thres == 0.
    assert cfg.task.termination.cum_lost_contact_steps.min_steps == 8
    assert cfg.task.command.reset_sampling == "return"
    assert len(applied) == 7
    # A task.* command-line override (or a parent) beats the WM default.
    cfg, _ = _train_defaults(["task.termination.cum_body_pos_error.min_steps=8",
                              "task.command.reset_sampling=uniform",
                              "task.termination.cum_lost_contact_steps={frc_thres:1.0,min_steps:3,pos_thres:0.2}"])
    assert cfg.task.termination.cum_body_pos_error.min_steps == 8
    assert cfg.task.termination.cum_body_ori_error.min_steps == 2
    assert cfg.task.command.reset_sampling == "uniform"
    assert cfg.task.termination.cum_lost_contact_steps.frc_thres == 1.
    assert cfg.task.termination.cum_lost_contact_steps.min_steps == 3


def test_ppo_has_no_training_task_defaults_and_missing_terms_are_never_added():
    cfg, result = _train_defaults(algo="ppo_vel_train")
    assert result == ([], [])
    assert cfg.task.command.reset_sampling == "uniform"
    cfg, (applied, skipped) = _train_defaults(["++algo.task_defaults.termination.not_a_term.min_steps=1"])
    assert skipped == ["termination.not_a_term.min_steps"]
    assert "not_a_term" not in cfg.task.termination
