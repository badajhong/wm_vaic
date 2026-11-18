"""Exercise the full WM learner on CPU without simulator assets or Isaac Sim."""

import copy
import io
import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs.transforms import VecNorm

from active_adaptation.learning.wm.algorithm import WM, WMConfig, lambda_returns, sequence_mask
from active_adaptation.learning.wm.core import ObservationCodec
from wm_test_models import DifferentiableWorldModel
from active_adaptation.learning.wm.replay import CompactSequenceReplay


@pytest.fixture(autouse=True)
def single_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(123)
    yield
    torch.set_num_threads(original)


def make_policy(*, height=False, **overrides):
    cfg = WMConfig(
        train_every=4, num_minibatches=2, latent_dim=32,
        wm_hidden_dims=(32, 32), wm_updates=1, wm_batch_size=4,
        wm_sequence_length=2, imagination_horizon=2,
        actor_updates=1, critic_updates=1, warmup_updates=0,
        schedule_mode="fixed", critic_epochs=1, critic_num_minibatches=1,
        actor_microbatch_size=4, critic_microbatch_size=16,
        require_model_validation=True,
        replay_capacity=64, validation_capacity=32, actor_batch_size=4,
        entropy_coef_start=0., entropy_coef_end=0.,
    )
    for name, value in overrides.items():
        setattr(cfg, name, value)
    envs, actions = 4, 6
    shapes = {
        "command": (18,), "policy": (32,), "priv": (48,),
        "object_": (24,), "object_geo_": (128 * 3,),
        "vel_command": (9,), "ref_joint_pos_": (actions,),
    }
    if height:
        shapes["height"] = (1, 16, 16)
    observations = Composite(
        {key: Unbounded((envs, *shape)) for key, shape in shapes.items()}, shape=(envs,)
    )
    raw = observations.clone()
    raw["wm_context"] = Unbounded((envs, 17))
    observations["wm_observation"] = raw
    env = SimpleNamespace(
        cfg=SimpleNamespace(reward={"tracking": {}, "object_tracking": {}}),
        action_manager=SimpleNamespace(joint_names=[f"joint_{i}" for i in range(actions)]),
        current_iter=3, num_envs=envs, device="cpu",
    )
    env.set_progress = lambda value: setattr(env, "current_iter", value)
    policy = WM(
        cfg, observations, Unbounded((envs, actions)),
        Composite(reward=Unbounded((envs, 2)), shape=(envs,)), "cpu", env,
    )
    # This fixture isolates learner mechanics with small synthetic transitions.
    # Production construction requires the physical adapter, tested separately.
    policy.world_model = DifferentiableWorldModel(policy.codec, actions, 2, (32, 32))
    def imagine(state, action):
        # A standalone reference advance, deliberately independent of replay
        # future/done. The production provider is tested against real VAIC.
        reference = state.detach().clone()
        reference[..., policy.codec.slices["command"]] += .01
        prediction, reward, termination, discount = policy.world_model(state, action, reference)
        return prediction, reward, (termination > .5).float(), discount, torch.zeros_like(termination, dtype=torch.bool)
    policy.world_model.imagine = imagine
    policy.opt_world_model = torch.optim.Adam(policy.world_model.parameters(), lr=cfg.wm_lr)
    def approve_synthetic_model():
        policy._validated_horizon = 2
        policy._oracle_checked = True
        return {"validation/approved_horizon": 2}
    policy._validate_world_model = approve_synthetic_model
    return policy


def make_rollout(policy):
    """Create action-dependent object trajectories using the real rollout actor."""
    with torch.inference_mode():
        raw = policy.observation_spec["wm_observation"].rand()
        actor = policy.get_rollout_policy()
        steps = []
        for t in range(policy.cfg.train_every):
            step = raw.exclude("wm_context").clone()
            step["wm_observation"] = raw.clone()
            step["is_init"] = torch.zeros(policy.env.num_envs, 1, dtype=torch.bool)
            step["step_count"] = torch.full((policy.env.num_envs, 1), t + 2)
            actor(step)
            action = step["action"]
            next_raw = raw.clone()
            next_raw["policy"][..., :policy.action_dim] += 0.1 * action
            next_raw["object_"][..., :policy.action_dim] += 0.2 * action
            next_raw["priv"][..., :policy.action_dim] += 0.05 * action
            next_raw["wm_context"][..., :policy.action_dim] += 0.3 * action
            next_raw["command"] += 0.01
            next_raw["ref_joint_pos_"] += 0.01
            step["next", "wm_observation"] = next_raw.clone()
            step["next", "reward"] = torch.stack((
                action.mean(-1), -next_raw["object_"][..., :3].square().mean(-1),
            ), -1)
            step["next", "done"] = torch.zeros(policy.env.num_envs, 1, dtype=torch.bool)
            step["next", "terminated"] = torch.zeros(policy.env.num_envs, 1, dtype=torch.bool)
            step["next", "discount"] = torch.ones(policy.env.num_envs, 1)
            raw = next_raw
            steps.append(step)
        return torch.stack(steps, dim=1)


def parameters(module):
    return {name: value.detach().clone() for name, value in module.named_parameters()}


def changed(before, module):
    return any(not torch.equal(before[name], value) for name, value in module.named_parameters())


@pytest.mark.parametrize("time_major", [False, True])
@pytest.mark.parametrize("matrix_dtype", [torch.float32, torch.float64])
def test_replay_partition_pack_preserves_layout_events_dtype_split_and_source_independence(time_major, matrix_dtype):
    environments, steps = 5, 4
    shapes = {"vector": (3,), "matrix": (2, 3), "scalar": ()}
    spec = Composite({key: Unbounded((environments, *shape)) for key, shape in shapes.items()},
                     shape=(environments,))
    codec = ObservationCodec(spec, tuple(shapes))
    frames = torch.arange(environments * (steps + 1) * codec.state_dim, dtype=torch.float32).reshape(
        environments, steps + 1, codec.state_dim)
    raw, following = codec.unpack(frames[:, :-1]).clone(), codec.unpack(frames[:, 1:]).clone()
    raw["matrix"] = raw["matrix"].to(matrix_dtype)
    following["matrix"] = following["matrix"].to(matrix_dtype)
    following["matrix"][1, 1] += .125  # Preserve a non-adjacent terminal observation.
    rollout = TensorDict({"wm_observation": raw,
        "action": torch.arange(environments * steps * 2).reshape(environments, steps, 2).double(),
        "is_init": torch.zeros(environments, steps, 1, dtype=torch.bool),
        "next": {"wm_observation": following,
                 "reward": torch.arange(environments * steps).reshape(environments, steps, 1).double(),
                 "terminated": torch.zeros(environments, steps, 1, dtype=torch.bool),
                 "done": torch.zeros(environments, steps, 1, dtype=torch.bool),
                 "discount": torch.ones(environments, steps, 1)}}, [environments, steps])
    rollout["is_init"][1, 2] = True
    rollout["next", "terminated"][1, 1] = rollout["next", "done"][1, 1] = True
    if time_major:
        rollout = rollout.apply(lambda value: value.transpose(0, 1).contiguous().transpose(0, 1))
        assert not rollout["wm_observation", "matrix"].is_contiguous()
        assert rollout["wm_observation", "matrix"][:, 0].is_contiguous()
    original = rollout.clone()
    stats_inputs = []
    def update_stats(*args, valid):
        stats_inputs.append((*[value.clone() for value in args], valid.clone()))
    harness = SimpleNamespace(codec=codec, cfg=SimpleNamespace(validation_env_stride=2),
        world_model=SimpleNamespace(update_stats=update_stats), _collected_frames=0,
        replay=CompactSequenceReplay(64), validation_replay=CompactSequenceReplay(64))
    WM._add_replay(harness, rollout)
    rollout.zero_()  # A later collection must never overwrite replay storage.
    expected_partitions = []
    for ids, replay in ((torch.tensor([1, 3]), harness.replay),
                        (torch.tensor([0, 2, 4]), harness.validation_replay)):
        def expected_pack(group):
            # Match the prior assignment contract, including first-group dtype.
            return torch.cat([group[key][ids].reshape(len(ids), steps, -1).float()
                              for key in shapes], -1)
        expected = TensorDict({"state": expected_pack(original["wm_observation"]),
            "action": original["action"][ids], "is_init": original["is_init"][ids],
            "next": {"state": expected_pack(original["next", "wm_observation"]),
                     **{key: original["next", key][ids]
                        for key in ("reward", "terminated", "done", "discount")}}}, [len(ids), steps])
        actual = replay.sample(len(ids), steps, draws=torch.arange(len(ids)))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual["state"].dtype == torch.float32
        assert actual["action"].dtype == actual["next", "reward"].dtype == torch.float64
        assert actual["is_init"].dtype == actual["next", "done"].dtype == torch.bool
        expected_partitions.append(expected)
    assert harness._collected_frames == environments * steps
    assert len(stats_inputs) == 1
    train = expected_partitions[0]
    for actual, expected in zip(stats_inputs[0], (train["state"], train["action"], train["next", "state"],
                                                train["next", "reward"], ~train["is_init"].squeeze(-1))):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("height", [False, True])
def test_train_op_updates_teacher_dynamics_critic_without_ppo_or_students(monkeypatch, height):
    policy = make_policy(height=height)
    rollout = make_rollout(policy)
    assert rollout["action"].is_inference()
    learned = ["actor", "world_model", "critic"] + (["height_cnn"] if height else [])
    before = {name: parameters(getattr(policy, name)) for name in learned}
    assert not hasattr(policy, "encoder_priv")

    def forbidden(*args, **kwargs):
        raise AssertionError("The PPO policy/advantage pipeline must never participate in WM training")

    monkeypatch.setattr(policy, "_update_ppo", forbidden)
    monkeypatch.setattr(policy, "_compute_advantage", forbidden)
    monkeypatch.setattr(policy, "train_policy", forbidden)
    monkeypatch.setattr(policy, "_train_adapt_no_depth", forbidden)
    info = policy.train_op(rollout)
    assert all(math.isfinite(value) for value in info.values())
    assert info["wm/policy_ready"] == 1
    assert info["wm/actor_updates"] == policy.cfg.actor_updates
    assert info["wm/replay_size"] + info["wm/validation_size"] == rollout.numel()
    assert info["actor/grad_norm"] > 0
    assert info["actor/encoder_priv_grad_norm"] == 0
    assert info["wm/grad_norm"] > 0
    assert not any(key.startswith("adapt/") for key in info)
    for name, snapshot in before.items():
        assert changed(snapshot, getattr(policy, name)), name
    assert policy.num_updates == 1
    assert policy.world_model.state_stats.count == rollout[1:].numel() * 2
    assert all(p.requires_grad for p in policy.world_model.parameters())


def test_closed_gate_reports_zero_actor_steps_and_preserves_teacher_weights():
    policy = make_policy()
    def reject_model():
        policy._validated_horizon = 0
        return {"validation/approved_horizon": 0}
    policy._validate_world_model = reject_model
    rollout = make_rollout(policy)
    before = {name: {key: parameter.detach().clone() for key, parameter in
                    getattr(policy, name).named_parameters() if parameter.requires_grad}
              for name in ("actor",)}
    info = policy.train_op(rollout)
    assert info["wm/policy_ready"] == info["wm/actor_updates"] == 0
    for name, values in before.items():
        for key, parameter in getattr(policy, name).named_parameters():
            if key in values:
                torch.testing.assert_close(parameter, values[key], rtol=0, atol=0)
    assert not any(key.startswith("actor/") and not key.startswith("actor/collection_") for key in info)
    assert info["actor/collection_mean_kl"] == info["actor/collection_action_delta_rms"] == 0


def test_default_supermimic_style_admission_starts_after_warmup_despite_accuracy_diagnostics():
    assert WMConfig().require_model_validation is False
    policy = make_policy(require_model_validation=False, warmup_updates=2)
    def inaccurate_model():
        policy._validated_horizon = 0
        policy._oracle_checked = True
        return {"validation/approved_horizon": 0, "validation/contact_ready": 0.}
    policy._validate_world_model = inaccurate_model
    for iteration in range(3):
        info = policy.train_op(make_rollout(policy))
        assert info["validation/approved_horizon"] == 0
        assert info["wm/accuracy_gate_enabled"] == 0
        assert info["wm/actor_updates"] == (policy.cfg.actor_updates if iteration == 2 else 0)
        assert info["wm/actor_horizon"] == (policy.cfg.imagination_horizon if iteration == 2 else 0)
    assert info["actor/grad_norm"] > 0
    assert not hasattr(policy, "encoder_priv")


def test_warmup_admission_still_requires_equation_oracle():
    policy = make_policy(require_model_validation=False)
    policy._validate_world_model = lambda: {"validation/approved_horizon": 0}
    assert not policy._oracle_checked
    info = policy.train_op(make_rollout(policy))
    assert info["wm/actor_updates"] == info["wm/policy_ready"] == 0


def test_actor_bptt_does_not_update_or_accumulate_gradients_on_world_model():
    policy = make_policy()
    policy._add_replay(make_rollout(policy))
    policy._fit_world_model(policy._sample(2))
    policy.opt_world_model.zero_grad(set_to_none=True)
    model_before = parameters(policy.world_model)
    actor_before = parameters(policy.actor)
    info = policy._imagine(policy._sample(2))
    assert info["actor/grad_norm"] > 0
    assert changed(actor_before, policy.actor)
    assert not changed(model_before, policy.world_model)
    assert all(p.grad is None for p in policy.world_model.parameters())
    assert all(p.requires_grad for p in policy.world_model.parameters())


def test_raw_teacher_is_owned_by_actor_optimizer_and_updates_only_during_bptt():
    policy = make_policy()
    optimizer_ids = {id(parameter) for group in policy.opt_policy.param_groups for parameter in group["params"]}
    assert not hasattr(policy, "encoder_priv")
    assert optimizer_ids == {id(parameter) for parameter in policy.actor.parameters()}
    policy._add_replay(make_rollout(policy))
    snapshot = parameters(policy.actor)
    policy._fit_world_model(policy._sample(2))
    assert not changed(snapshot, policy.actor)
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    info = policy._imagine(policy._sample(2))
    assert changed(snapshot, policy.actor)
    assert info["actor/grad_norm"] > 0
    assert "encoder_priv" not in policy.state_dict()
    assert policy.state_dict()["wm_teacher_input_schema"] == policy.teacher_input_schema


def test_students_remain_frozen_unchanged_and_without_gradients_during_train_and_save():
    policy = make_policy()
    students = {name: getattr(policy, name) for name in policy.student_module_names if hasattr(policy, name)}
    # Keep an EMA deliberately different to detect parent checkpoint-copy side effects.
    with torch.no_grad():
        next(policy.adapt_module.parameters()).add_(.1)
    snapshots = {name: copy.deepcopy(module.state_dict()) for name, module in students.items()}
    assert all(not parameter.requires_grad for module in students.values() for parameter in module.parameters())
    for name in ("opt_adapt", "opt_adapt_actor", "opt_dr_estimator"):
        assert not hasattr(policy, name)
    policy.train_op(make_rollout(policy))
    with torch.no_grad():
        policy.actor.module[0][2].module.actor_std.fill_(.37)
    checkpoint = policy.state_dict()
    assert set(checkpoint["optimizers"]) == {"opt_world_model", "opt_policy", "opt_critic"}
    for name, module in students.items():
        assert_tree_equal(module.state_dict(), snapshots[name])
        assert_tree_equal(checkpoint[name], snapshots[name])
        assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.parametrize("mode", ["train", "eval", "deploy"])
def test_all_wm_rollout_modes_use_teacher_without_student_execution_or_hidden_state(monkeypatch, mode):
    policy = make_policy()
    def forbidden(*args, **kwargs):
        raise AssertionError("A teacher-only WM rollout must not execute student modules")
    for name in policy.student_module_names:
        if hasattr(policy, name):
            monkeypatch.setattr(getattr(policy, name), "forward", forbidden)
    raw = policy.observation_spec["wm_observation"].rand()
    with torch.no_grad():
        expected = policy._teacher_dist(policy.codec.pack(raw)).mean
        output = policy.get_rollout_policy(mode)(raw.exclude("wm_context").clone())
    torch.testing.assert_close(output["loc"], expected)
    assert "priv_pred" not in output.keys()
    assert ("next", "adapt_hx") not in output.keys(True, True)
    assert policy.make_tensordict_primer() is None


def test_teacher_reference_action_residual_is_preserved_without_student_training():
    policy = make_policy()
    assert policy.cfg.enable_residual_distillation
    raw = policy.observation_spec["wm_observation"].rand()
    shifted = raw.clone()
    shifted["ref_joint_pos_"] += .25
    with torch.no_grad():
        original = policy._teacher_dist(policy.codec.pack(raw)).mean
        modified = policy._teacher_dist(policy.codec.pack(shifted)).mean
    torch.testing.assert_close(modified, original + .25)


def test_sequence_mask_scores_terminal_but_not_reset_or_post_boundary_transitions():
    batch = TensorDict({
        "is_init": torch.tensor([
            [False, False, True, False],
            [False, False, False, False],
            [True, False, False, False],
        ]).unsqueeze(-1),
        "next": TensorDict({"done": torch.tensor([
            [False, True, False, False],
            [False, False, False, False],
            [False, False, False, False],
        ]).unsqueeze(-1)}, batch_size=[3, 4]),
    }, batch_size=[3, 4])
    expected = torch.tensor([
        [True, True, False, False],
        [True, True, True, True],
        [False, True, True, True],
    ])
    torch.testing.assert_close(sequence_mask(batch), expected)


def test_world_model_update_is_unchanged_by_data_after_episode_boundary():
    policy = make_policy()
    policy._add_replay(make_rollout(policy))
    batch = policy._sample(4)
    batch["next", "done"][:, 1] = True
    batch["is_init"][:, 2] = True
    altered = batch.clone()
    # Finite, adversarially different data from the next episode must neither
    # enter the reported objective nor change any learned dynamics parameter.
    altered["state"][:, 2:] += 100
    altered["action"][:, 2:] *= -50
    altered["next", "state"][:, 2:] -= 100
    altered["next", "reward"][:, 2:] += 500
    altered["next", "terminated"][:, 2:] = True
    other = make_policy()
    other.world_model.load_state_dict(policy.world_model.state_dict())
    expected_info = policy._fit_world_model(batch)
    actual_info = other._fit_world_model(altered)
    for name in expected_info:
        assert actual_info[name] == pytest.approx(expected_info[name], rel=1e-6)
    for name, value in policy.world_model.state_dict().items():
        torch.testing.assert_close(other.world_model.state_dict()[name], value)


def test_lambda_returns_bootstrap_truncation_stop_termination_and_cut_episode_leakage():
    # Environments share a boundary at t=1. The first terminates, while the
    # second times out and must retain the terminal state's bootstrap value.
    rewards = torch.tensor([[[1.], [2.], [1000.]], [[1.], [2.], [1000.]]])
    values = torch.tensor([[[10.], [20.], [30.]], [[10.], [20.], [30.]]])
    continuation = torch.ones(2, 3, 1)
    continuation[0, 1] = 0
    done = torch.tensor([[[False], [True], [False]], [[False], [True], [False]]])
    returns = lambda_returns(rewards, values, continuation, done, gamma=0.9, lmbda=1.)
    torch.testing.assert_close(returns[0, :2, 0], torch.tensor([2.8, 2.]))
    torch.testing.assert_close(returns[1, :2, 0], torch.tensor([19., 20.]))
    assert returns[:, 2, 0].tolist() == [1027., 1027.]


def test_replay_retains_raw_values_while_actor_uses_a_frozen_normalizer_snapshot():
    policy = make_policy()
    rollout = make_rollout(policy)
    raw_policy = rollout["wm_observation", "policy"].clone()
    normalizer = VecNorm(in_keys=["policy", "priv", "object_"], decay=0.)
    calibration = rollout[:, 0].clone()
    calibration["policy"] += 5
    normalizer(calibration)
    policy.set_observation_normalizer(normalizer, "train")
    policy._snapshot_normalizer()
    packed = policy.codec.pack(rollout["wm_observation"]).clone()
    actor_input = policy._policy_observation(packed)
    expected = (raw_policy - normalizer.loc["policy"]) / normalizer.scale["policy"].clamp_min(normalizer.eps)
    torch.testing.assert_close(actor_input["policy"], expected)
    policy._add_replay(rollout)
    stored = torch.cat([chunk[0]["state"] for chunk in policy.replay._chunks], 0)
    # Sampling environments permutes their order but does not normalize values.
    sorted_input = packed[1:, 0, policy.codec.slices["policy"]][:, 0].sort().values
    sorted_stored = stored[:, 0, policy.codec.slices["policy"]][:, 0].sort().values
    torch.testing.assert_close(sorted_stored, sorted_input)
    previous = actor_input["policy"].clone()
    calibration["policy"] += 50
    normalizer(calibration)
    torch.testing.assert_close(policy._policy_observation(packed)["policy"], previous)
    policy._snapshot_normalizer()
    assert not torch.equal(policy._policy_observation(packed)["policy"], previous)
    assert "_observation_normalizer" not in dict(policy.named_children())


def assert_tree_equal(actual, expected):
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_tree_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_tree_equal(a, b)
    else:
        assert actual == expected


def test_checkpoint_roundtrip_restores_optimizers_noise_moments_and_resets_replay():
    policy = make_policy()
    policy.train_op(make_rollout(policy))
    with torch.no_grad():
        policy.actor.module[0][2].module.actor_std.fill_(0.37)
    stream = io.BytesIO()
    torch.save(policy.state_dict(), stream)
    stream.seek(0)
    checkpoint = torch.load(stream, weights_only=False)
    restored = make_policy()
    restored.load_state_dict(checkpoint, strict=True)
    for name, module in restored.named_children():
        assert_tree_equal(module.state_dict(), checkpoint[name])
    for name in restored._optimizer_names():
        assert_tree_equal(getattr(restored, name).state_dict(), checkpoint["optimizers"][name])
    torch.testing.assert_close(
        restored.actor.module[0][2].module.actor_std,
        torch.full((restored.action_dim,), 0.37), rtol=0, atol=0,
    )
    assert restored.world_model.state_stats.count == policy.world_model.state_stats.count
    assert restored.num_updates == policy.num_updates
    assert restored.env.current_iter == policy.env.current_iter
    assert restored._model_updates_since_resume == 0
    assert len(restored.replay) == 0
    # Optimizer state and ordinary tensors must permit a real update after load.
    info = restored.train_op(make_rollout(restored))
    assert all(math.isfinite(value) for value in info.values())


def test_checkpoint_with_removed_target_critic_entry_still_loads():
    # Checkpoints saved before the unused target critic was removed carry its
    # weights; only modules this learner owns are restored.
    policy = make_policy()
    assert not hasattr(policy, "target_critic")
    checkpoint = policy.state_dict()
    checkpoint["target_critic"] = copy.deepcopy(checkpoint["critic"])
    restored = make_policy()
    restored.load_state_dict(checkpoint, strict=True)
    assert_tree_equal(restored.critic.state_dict(), checkpoint["critic"])
    assert not hasattr(restored, "target_critic")


@pytest.mark.parametrize("old_architecture", ["missing_schema", "latent", "geometry_present"])
def test_incompatible_teacher_checkpoint_is_rejected_before_any_module_or_optimizer_mutation(old_architecture):
    policy = make_policy()
    policy.train_op(make_rollout(policy))
    checkpoint = copy.deepcopy(policy.state_dict())
    if old_architecture == "missing_schema":
        checkpoint.pop("wm_teacher_input_schema")
    elif old_architecture == "geometry_present":
        version, fields, *rest = checkpoint["wm_teacher_input_schema"]
        assert all(field[0] != "object_trans" for field in fields)
        checkpoint["wm_teacher_input_schema"] = (version, (*fields, ("object_trans", 384)), *rest)
    else:
        checkpoint["wm_teacher_input_schema"] = ("latent_teacher_v0",)
    checkpoint["encoder_priv"] = {"legacy_weight": torch.ones(3, 4)}
    before = copy.deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match="raw-input teacher.*|encoder_priv"):
        policy.load_state_dict(checkpoint, strict=True)
    assert_tree_equal(policy.state_dict(), before)


@pytest.mark.parametrize("optimizer", ["kl", "adam"])
@pytest.mark.parametrize("height", [False, True])
def test_raw_actor_resume_without_controller_metadata_restores_rates_and_moments(height, optimizer):
    policy = make_policy(height=height, actor_optimizer=optimizer)
    policy.train_op(make_rollout(policy))
    checkpoint = copy.deepcopy(policy.state_dict())
    checkpoint.pop("wm_actor_step")
    checkpoint["lr_policy"] = 3e-4
    for group in checkpoint["optimizers"]["opt_policy"]["param_groups"]:
        group["lr"] = 3e-4
        group["betas"] = (.9, .999)
    restored = make_policy(height=height, actor_optimizer=optimizer)
    original_groups = [[id(p) for p in group["params"]] for group in restored.opt_policy.param_groups]
    restored.load_state_dict(checkpoint, strict=True)
    assert [[id(p) for p in group["params"]] for group in restored.opt_policy.param_groups] == original_groups
    cfg = restored.cfg
    actor_rate = cfg.actor_lr if optimizer == "kl" else cfg.actor_adam_lr
    expected_rates = [actor_rate]
    if height:
        expected_rates.append(cfg.encoder_lr if optimizer == "kl"
                              else cfg.actor_adam_lr * (cfg.encoder_lr / cfg.actor_lr))
    assert [group["lr"] for group in restored.opt_policy.param_groups] == expected_rates
    expected_beta2 = .999 if optimizer == "kl" else cfg.actor_adam_beta2
    assert all(group["betas"][1] == expected_beta2 for group in restored.opt_policy.param_groups)
    assert restored.lr_policy == actor_rate
    assert restored._actor_step.scale == 1
    # Every Adam first/second moment and step counter survives the migration;
    # only policy learning rates change, leaving critic/model training intact.
    assert_tree_equal(restored.opt_policy.state_dict()["state"],
                      checkpoint["optimizers"]["opt_policy"]["state"])
    for name in ("opt_critic", "opt_world_model"):
        assert_tree_equal(getattr(restored, name).state_dict(), checkpoint["optimizers"][name])
    for name in ("actor",):
        assert_tree_equal(getattr(restored, name).state_dict(), checkpoint[name])


def test_actor_controller_resume_reproduces_next_bptt_step_and_measured_kl_bound():
    policy = make_policy(actor_optimizer="kl")
    policy.train_op(make_rollout(policy))
    batch = policy._sample(2)
    checkpoint = copy.deepcopy(policy.state_dict())
    restored = make_policy(actor_optimizer="kl")
    restored.load_state_dict(checkpoint, strict=True)
    assert_tree_equal(restored._actor_step.state_dict(), checkpoint["wm_actor_step"])
    assert restored.lr_policy == policy.lr_policy
    # Reproduce the critic permutation as well as weights/optimizer state.
    # Different summation orders can otherwise differ by a float32 ulp.
    rng = torch.random.get_rng_state()
    expected = policy._imagine(batch)
    torch.random.set_rng_state(rng)
    actual = restored._imagine(batch)
    assert {key: value for key, value in actual.items() if not key.endswith("_seconds")} == {
        key: value for key, value in expected.items() if not key.endswith("_seconds")}
    assert actual["performance/wm_imagination_seconds"] > 0
    assert actual["performance/wm_critic_seconds"] > 0
    assert actual["actor/step_accepted"] == 1
    assert 0 < actual["actor/update_mean_kl"] <= restored.cfg.actor_max_kl
    for name in ("actor",):
        assert_tree_equal(getattr(restored, name).state_dict(), getattr(policy, name).state_dict())
    assert_tree_equal(restored.opt_policy.state_dict(), policy.opt_policy.state_dict())
    assert_tree_equal(restored._actor_step.state_dict(), policy._actor_step.state_dict())


def test_train_reports_accepted_steps_separately_from_rejected_attempts(monkeypatch):
    policy = make_policy(actor_updates=2)
    original = policy._actor_step.step
    calls = 0

    def reject_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            # Model an explicitly rejected step; never replace it with a hidden
            # update or report the configured attempt count as accepted work.
            return {"actor/step_accepted": 0., "actor/update_mean_kl": 0.,
                    "actor/action_delta_rms": 0.}
        return original(*args, **kwargs)

    monkeypatch.setattr(policy._actor_step, "step", reject_second)
    info = policy.train_op(make_rollout(policy))
    assert calls == 2
    assert info["wm/actor_update_attempts"] == 2
    assert info["wm/actor_updates"] == 1
    assert info["actor/step_accepted"] == .5


def test_checkpoint_rejects_ppo_missing_model_weights_and_schema_changes():
    policy = make_policy()
    checkpoint = copy.deepcopy(policy.state_dict())
    with pytest.raises(ValueError, match="WM checkpoint"):
        policy.load_state_dict({"last_phase": "train"})
    changed_schema = copy.deepcopy(checkpoint)
    changed_schema["wm_schema"] = tuple(
        (key, (999,) if key == "object_" else shape)
        for key, shape in changed_schema["wm_schema"]
    )
    with pytest.raises(ValueError, match="schema"):
        policy.load_state_dict(changed_schema)
    reordered_schema = copy.deepcopy(checkpoint)
    reordered_schema["wm_schema"] = tuple(reversed(checkpoint["wm_schema"]))
    with pytest.raises(ValueError, match="schema"):
        policy.load_state_dict(reordered_schema)
    changed_context = copy.deepcopy(checkpoint)
    changed_context["wm_context_schema"] = (("different_field", (17,)),)
    with pytest.raises(ValueError, match="context fields"):
        policy.load_state_dict(changed_context)
    missing_optimizer = copy.deepcopy(checkpoint)
    missing_optimizer["optimizers"].pop("opt_world_model")
    with pytest.raises(ValueError, match="optimizer states"):
        policy.load_state_dict(missing_optimizer, strict=True)
    missing_weight = copy.deepcopy(checkpoint)
    missing_weight["world_model"].pop("dynamics_head.weight")
    with pytest.raises(RuntimeError, match="Missing key"):
        policy.load_state_dict(missing_weight, strict=True)


def test_teacher_only_load_accepts_legacy_student_optimizer_entries_without_enabling_students():
    policy = make_policy()
    policy.train_op(make_rollout(policy))
    checkpoint = copy.deepcopy(policy.state_dict())
    # Earlier physical-WM v2 saves included these optimizer states and possibly
    # trained student weights. They remain loadable only with an identical
    # observation/context/physics schema; the extra optimizers stay unused.
    checkpoint["optimizers"]["opt_adapt"] = {"state": {}, "param_groups": []}
    checkpoint["optimizers"]["opt_adapt_actor"] = {"state": {}, "param_groups": []}
    student_key = next(key for key, value in checkpoint["adapt_module"].items()
                       if value.is_floating_point())
    checkpoint["adapt_module"][student_key].add_(0.75)
    restored = make_policy()
    restored.load_state_dict(checkpoint, strict=True)
    assert_tree_equal(restored.adapt_module.state_dict(), checkpoint["adapt_module"])
    for name in restored.student_module_names:
        if hasattr(restored, name):
            assert all(not parameter.requires_grad for parameter in getattr(restored, name).parameters())
    assert not hasattr(restored, "opt_adapt")
    assert not hasattr(restored, "opt_adapt_actor")
    assert restored._validated_horizon == 0
    assert not restored._oracle_checked
    assert restored._model_updates_since_resume == 0
    assert len(restored.replay) == len(restored.validation_replay) == 0


def _attach_checkpoint_physics(policy, schema, output_width):
    """Populate real WM Adam moments around a small schema-controlled module."""
    physics = torch.nn.Linear(3, output_width)
    physics.schema = schema
    policy.world_model.physics = physics
    policy.opt_world_model = torch.optim.Adam(policy.world_model.parameters(), lr=policy.cfg.wm_lr)
    for parameter in policy.world_model.parameters():
        parameter.grad = torch.full_like(parameter, .125)
    policy.opt_world_model.step()
    policy.opt_world_model.zero_grad(set_to_none=True)
    assert policy.opt_world_model.state
    return physics


def test_root_warmstart_load_respects_disabled_flag_and_resets_only_wm_optimizer(monkeypatch):
    from active_adaptation.learning.wm import object_migration as wm_object_migration

    source = make_policy()
    source.train_op(make_rollout(source))
    _attach_checkpoint_physics(source, ("articulated-test",), output_width=4)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint_before = copy.deepcopy(checkpoint)
    for name in ("opt_world_model", "opt_policy", "opt_critic"):
        assert checkpoint["optimizers"][name]["state"]

    restored = make_policy(wm_root_warmstart=False)
    target = _attach_checkpoint_physics(restored, ("root-test",), output_width=2)
    target_before = copy.deepcopy(restored.state_dict())
    migration_calls = []
    metadata = {"kind": "articulated_to_root", "removed_object_joint_channels": 8}

    def convert(physics_state, schema, target_physics):
        migration_calls.append((schema, target_physics))
        assert schema == checkpoint["wm_physics_schema"]
        assert target_physics is target
        assert set(physics_state) == {"weight", "bias"}
        assert_tree_equal(physics_state["weight"], checkpoint["world_model"]["physics.weight"])
        return {"weight": physics_state["weight"][:2].clone(),
                "bias": physics_state["bias"][:2].clone()}, metadata.copy()

    monkeypatch.setattr(wm_object_migration, "migrate_object_root_physics", convert)
    with pytest.raises(ValueError, match="wm_root_warmstart=true"):
        restored.load_state_dict(checkpoint, strict=True)
    assert not migration_calls
    assert_tree_equal(restored.state_dict(), target_before)

    restored.cfg.wm_root_warmstart = True
    previous_wm_optimizer = restored.opt_world_model
    teacher_optimizer, critic_optimizer = restored.opt_policy, restored.opt_critic
    restored.load_state_dict(checkpoint, strict=True)
    assert len(migration_calls) == 1
    assert restored.opt_world_model is not previous_wm_optimizer
    assert not restored.opt_world_model.state
    assert restored.opt_world_model.param_groups[0]["lr"] == restored.cfg.wm_lr
    assert restored.opt_policy is teacher_optimizer
    assert restored.opt_critic is critic_optimizer
    for name in ("opt_policy", "opt_critic"):
        assert_tree_equal(getattr(restored, name).state_dict(), checkpoint["optimizers"][name])
    for name, module in restored.named_children():
        if name != "world_model":
            assert_tree_equal(module.state_dict(), checkpoint[name])
    for name, value in restored.world_model.state_dict().items():
        expected = checkpoint["world_model"][name]
        if name.startswith("physics."):
            expected = expected[:2]
        assert_tree_equal(value, expected)
    saved = restored.state_dict()
    assert saved["wm_physics_schema"] == ("root-test",)
    assert saved["wm_initialization"] == metadata
    assert restored.num_updates == checkpoint["num_updates"]
    assert restored._collected_frames == checkpoint["collected_frames"]
    assert not restored._oracle_checked
    assert restored._validated_horizon == 0
    assert len(restored.replay) == len(restored.validation_replay) == 0
    assert_tree_equal(checkpoint, checkpoint_before)


def test_root_same_schema_resume_keeps_wm_moments_even_with_warmstart_flag(monkeypatch):
    from active_adaptation.learning.wm import object_migration as wm_object_migration

    source = make_policy(wm_root_warmstart=True)
    source.train_op(make_rollout(source))
    _attach_checkpoint_physics(source, ("root-test",), output_width=2)
    source._wm_initialization = {"kind": "previous_articulated_to_root"}
    checkpoint = copy.deepcopy(source.state_dict())
    restored = make_policy(wm_root_warmstart=True)
    _attach_checkpoint_physics(restored, ("root-test",), output_width=2)
    optimizer_before = restored.opt_world_model

    def forbidden(*args, **kwargs):
        pytest.fail("Same-schema root resume must not invoke architecture migration")

    monkeypatch.setattr(wm_object_migration, "migrate_object_root_physics", forbidden)
    restored.load_state_dict(checkpoint, strict=True)
    assert restored.opt_world_model is optimizer_before
    assert restored.opt_world_model.state
    for name in restored._optimizer_names():
        assert_tree_equal(getattr(restored, name).state_dict(), checkpoint["optimizers"][name])
    for name, module in restored.named_children():
        assert_tree_equal(module.state_dict(), checkpoint[name])
    assert restored.state_dict()["wm_initialization"] == checkpoint["wm_initialization"]


@pytest.mark.parametrize("schema_key", ["wm_schema", "wm_context_schema"])
def test_root_warmstart_rejects_wrong_observation_or_context_before_migration(monkeypatch, schema_key):
    from active_adaptation.learning.wm import object_migration as wm_object_migration

    source = make_policy()
    _attach_checkpoint_physics(source, ("articulated-test",), output_width=4)
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint[schema_key] = (("incompatible_field", (17,)),)
    restored = make_policy(wm_root_warmstart=True)
    _attach_checkpoint_physics(restored, ("root-test",), output_width=2)
    before = copy.deepcopy(restored.state_dict())

    def forbidden(*args, **kwargs):
        pytest.fail("Architecture migration must not bypass observation/context schema checks")

    monkeypatch.setattr(wm_object_migration, "migrate_object_root_physics", forbidden)
    with pytest.raises(ValueError, match="schema|context fields"):
        restored.load_state_dict(checkpoint, strict=True)
    assert_tree_equal(restored.state_dict(), before)


def test_legacy_frame_budget_train_op_executes_16_model_one_actor_and_one_critic_update(monkeypatch):
    # Legacy frame budgets were defined for the 32-step PPO collection window.
    defaults = WMConfig(schedule_mode="frames", critic_epochs=1, critic_num_minibatches=1, train_every=32)
    # Scale environment count and both logical batches by the same factor of 1024.
    # Keep production collection/model/imagination lengths and replay ratios,
    # so real CPU optimizer calls exercise the exact full-task update counts.
    policy = make_policy(
        train_every=defaults.train_every,
        schedule_mode=defaults.schedule_mode,
        wm_predictions_per_frame=defaults.wm_predictions_per_frame,
        actor_starts_per_frame=defaults.actor_starts_per_frame,
        wm_batch_size=defaults.wm_batch_size // 1024,
        wm_sequence_length=defaults.wm_sequence_length,
        actor_batch_size=defaults.actor_batch_size // 1024,
        imagination_horizon=defaults.imagination_horizon,
        critic_epochs=defaults.critic_epochs,
        critic_num_minibatches=defaults.critic_num_minibatches,
        warmup_model_updates=defaults.warmup_model_updates,
        require_model_validation=defaults.require_model_validation,
        replay_capacity=128,
    )
    calls = {"model": 0, "actor": 0, "critic": 0}
    def count(kind):
        def hook(*_args, **_kwargs):
            calls[kind] += 1
        return hook
    model_hook = policy.opt_world_model.register_step_post_hook(count("model"))
    actor_hook = policy.opt_policy.register_step_post_hook(count("actor"))
    critic_hook = policy.opt_critic.register_step_post_hook(count("critic"))
    valid_critic_samples = []
    fit_critic = policy._fit_imagined_critic
    def record_critic_payload(payload, **kwargs):
        valid_critic_samples.append(int(payload["valid"].sum()))
        return fit_critic(payload, **kwargs)
    monkeypatch.setattr(policy, "_fit_imagined_critic", record_critic_payload)
    try:
        for iteration in range(2):
            before = calls.copy()
            metrics = policy.train_op(make_rollout(policy))
            assert calls["model"] - before["model"] == 16
            assert calls["actor"] - before["actor"] == 1
            assert calls["critic"] - before["critic"] == 1
            assert metrics["wm/model_updates"] == metrics["wm/planned_model_updates"] == 16
            assert metrics["wm/actor_updates"] == metrics["wm/actor_update_attempts"] == 1
            assert metrics["wm/planned_actor_updates"] == 1
            assert metrics["critic/updates"] == 1
            assert len(valid_critic_samples) == iteration + 1
            assert metrics["critic/sample_passes"] == valid_critic_samples[-1]
            # One logical critic batch still accumulates several microbatches
            # before its single optimizer step; no valid sample is dropped.
            assert valid_critic_samples[-1] > policy.cfg.critic_microbatch_size
            assert metrics["wm/model_prediction_samples"] == 4 * 32
            assert metrics["wm/actor_starts"] == 4
            assert metrics["wm/actor_horizon"] == 16
            assert metrics["wm/prediction_credit"] == metrics["wm/actor_start_credit"] == 0
            assert policy._collected_frames == (iteration + 1) * 4 * 32
    finally:
        model_hook.remove()
        actor_hook.remove()
        critic_hook.remove()


@pytest.mark.parametrize("saved,target", [
    ("physical_causal_contact_substeps_root_v1", "physical_causal_nocontact_root_v1"),
    ("physical_causal_nocontact_root_v1", "physical_causal_contact_substeps_root_v1"),
])
def test_contact_model_mismatch_names_the_flag_and_never_migrates(monkeypatch, saved, target):
    from active_adaptation.learning.wm import object_migration as wm_object_migration

    source = make_policy()
    source.train_op(make_rollout(source))
    _attach_checkpoint_physics(source, (saved,), output_width=2)
    checkpoint = copy.deepcopy(source.state_dict())
    restored = make_policy()
    _attach_checkpoint_physics(restored, (target,), output_width=2)
    before = copy.deepcopy(restored.state_dict())
    monkeypatch.setattr(wm_object_migration, "migrate_object_root_physics",
                        lambda *args, **kwargs: pytest.fail("contact mismatch must not migrate"))
    expected = "true" if "nocontact" not in saved else "false"
    with pytest.raises(ValueError, match=f"pass algo.wm_contact_model={expected}"):
        restored.load_state_dict(checkpoint, strict=True)
    assert_tree_equal(restored.state_dict(), before)


@pytest.mark.parametrize("overrides", [dict(imagination_kinematic_reward=False),
                                       dict(imagination_contact_force_termination=True)])
def test_contact_free_world_model_rejects_contact_reading_imagination(overrides):
    with pytest.raises(ValueError, match="wm_contact_model=false"):
        make_policy(wm_contact_model=False, **overrides)
    make_policy(wm_contact_model=True, **overrides)


def test_frame_mismatch_names_the_flag_and_never_migrates(monkeypatch):
    from active_adaptation.learning.wm import object_migration as wm_object_migration

    source = make_policy()
    source.train_op(make_rollout(source))
    _attach_checkpoint_physics(source, ("physical_causal_nocontact_root_v1",), output_width=2)
    checkpoint = copy.deepcopy(source.state_dict())
    restored = make_policy()
    _attach_checkpoint_physics(restored, ("physical_causal_nocontact_root_v1_heading",), output_width=2)
    monkeypatch.setattr(wm_object_migration, "migrate_object_root_physics",
                        lambda *args, **kwargs: pytest.fail("frame mismatch must not migrate"))
    with pytest.raises(ValueError, match="pass algo.wm_frame=world to load it"):
        restored.load_state_dict(checkpoint, strict=True)


def test_world_model_frame_option_is_validated():
    with pytest.raises(ValueError, match="wm_frame"):
        make_policy(wm_frame="body")


def test_object_body_feature_mismatch_names_the_flag(monkeypatch):
    from active_adaptation.learning.wm import object_migration as wm_object_migration

    source = make_policy()
    source.train_op(make_rollout(source))
    _attach_checkpoint_physics(source, ("physical_causal_nocontact_root_v1_heading",), output_width=2)
    checkpoint = copy.deepcopy(source.state_dict())
    restored = make_policy()
    _attach_checkpoint_physics(restored, ("physical_causal_nocontact_root_v1_heading_objectbodies",), output_width=2)
    monkeypatch.setattr(wm_object_migration, "migrate_object_root_physics",
                        lambda *args, **kwargs: pytest.fail("feature mismatch must not migrate"))
    with pytest.raises(ValueError, match="pass algo.wm_object_body_features=false to load it"):
        restored.load_state_dict(checkpoint, strict=True)


def test_dynamics_width_mismatch_names_the_hidden_dims_flag():
    from active_adaptation.learning.wm.algorithm import _trunk_widths

    state = {"physics.trunk.0.weight": torch.zeros(1024, 7), "physics.trunk.1.weight": torch.zeros(1024),
             "physics.trunk.3.weight": torch.zeros(1024, 1024), "physics.trunk.6.weight": torch.zeros(512, 1024),
             "physics.velocity_head.weight": torch.zeros(3, 512)}
    assert _trunk_widths(state) == (1024, 1024, 512)
    source = make_policy()
    source.train_op(make_rollout(source))
    checkpoint = copy.deepcopy(source.state_dict())
    checkpoint["world_model"] = {**checkpoint["world_model"], **state}
    restored = make_policy()
    restored.world_model.physics = torch.nn.Module()
    restored.world_model.physics.trunk = torch.nn.Sequential(torch.nn.Linear(7, 2048))
    with pytest.raises(ValueError, match=r"pass 'algo.wm_hidden_dims=\[1024,1024,512\]' to load it"):
        restored.load_state_dict(checkpoint, strict=True)


def test_actor_and_critic_hidden_widths_are_configurable_and_checked_on_load():
    from active_adaptation.learning.wm.algorithm import _hidden_widths
    default = make_policy()
    # Defaults keep the PPO teacher widths.
    assert _hidden_widths(default.actor.state_dict()) == (512, 256, 256)
    assert _hidden_widths(default.critic.state_dict()) == (512, 256, 128)
    small = make_policy(actor_hidden_dims=(64, 32), critic_hidden_dims=(16,))
    assert _hidden_widths(small.actor.state_dict()) == (64, 32)
    assert _hidden_widths(small.critic.state_dict()) == (16,)
    # Optimizers own the rebuilt modules' parameters.
    assert {id(p) for group in small.opt_policy.param_groups for p in group["params"]} >= {
        id(p) for p in small.actor.parameters()}
    assert {id(p) for group in small.opt_critic.param_groups for p in group["params"]} == {
        id(p) for p in small.critic.parameters()}
    info = small.train_op(make_rollout(small))
    assert all(math.isfinite(value) for value in info.values())
    with pytest.raises(ValueError, match=r"'algo\.actor_hidden_dims=\[512,256,256\]'"):
        small.load_state_dict(default.state_dict())
    critic_only = make_policy(critic_hidden_dims=(16,))
    with pytest.raises(ValueError, match=r"'algo\.critic_hidden_dims=\[512,256,128\]'"):
        critic_only.load_state_dict(default.state_dict())
    for bad in ((), (0,), (1.5,)):
        with pytest.raises(ValueError, match="actor_hidden_dims"):
            make_policy(actor_hidden_dims=bad)
