"""The WM teacher consumes raw privileged features without changing PPO."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torchrl.data import Composite, Unbounded
from torchrl.envs.utils import ExplorationType, set_exploration_type

from active_adaptation.learning.ppo.ppo_vel import PPOConfig, PPOVEL
from active_adaptation.learning.wm.teacher import (
    configure_raw_teacher, raw_teacher_rollout_policy, teacher_dist,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(73)
    yield
    torch.set_num_threads(previous)


def make_teacher(height=False, residual=True, object_width=24):
    count, action_dim = 2, 6
    shapes = {"command": (18,), "policy": (32,), "priv": (48,),
              "object_": (object_width,), "object_geo_": (384,), "vel_command": (9,),
              "ref_joint_pos_": (action_dim,)}
    if height:
        shapes["height"] = (1, 16, 16)
    spec = Composite({key: Unbounded((count, *shape)) for key, shape in shapes.items()}, shape=(count,))
    cfg = PPOConfig(latent_dim=32, init_noise_scale=.3, load_noise_scale=None,
                    enable_residual_distillation=residual)
    env = SimpleNamespace(cfg=SimpleNamespace(reward={"tracking": {}, "object_tracking": {}}),
                          action_manager=SimpleNamespace(joint_names=list(range(action_dim))), current_iter=0,
                          num_envs=count, device="cpu", set_progress=lambda _: None)
    policy = PPOVEL(cfg, spec, Unbounded((count, action_dim)),
                    Composite(reward=Unbounded((count, 2)), shape=(count,)), "cpu", env)
    return policy, spec


@pytest.mark.parametrize("height", [False, True])
def test_raw_teacher_removes_encoder_and_rebuilds_only_actor_and_its_optimizer(height):
    policy, spec = make_teacher(height)
    encoder_ids = {id(parameter) for parameter in policy.encoder_priv.parameters()}
    old_actor_ids = {id(parameter) for parameter in policy.actor.parameters()}
    preserved = {name: (module, {key: value.clone() for key, value in module.state_dict().items()})
                 for name, module in policy.named_children() if name not in ("actor", "encoder_priv")}
    original_raw_keys = tuple(key for key in policy.encoder_priv[0].in_keys
                              if key not in ("object_geo_", "object_trans"))
    schema = configure_raw_teacher(policy, spec)
    assert schema == policy.teacher_input_schema
    assert schema[0] == "raw_teacher_input_v1"
    assert not hasattr(policy, "encoder_priv")
    assert "encoder_priv" not in dict(policy.named_children())
    assert not any(name.startswith("encoder_priv.") for name, _ in policy.named_parameters())
    actor_mlp = policy.actor.module[0][1].module
    assert [module.out_features for module in actor_mlp if isinstance(module, nn.Linear)] == [512, 256, 256]
    keys = tuple(policy.actor.module[0][0].in_keys)
    assert keys == ("command", "policy", *original_raw_keys)
    assert not {"priv_feature", "object_geo_", "object_trans"} & set(keys)
    assert actor_mlp[0].in_features == sum(width for _, width in schema[1])
    assert [group["name"] for group in policy.opt_policy.param_groups] == (["actor", "height"] if height else ["actor"])
    optimized = {id(parameter) for group in policy.opt_policy.param_groups for parameter in group["params"]}
    expected = {id(parameter) for parameter in policy.actor.parameters()}
    if height:
        expected |= {id(parameter) for parameter in policy.height_cnn.parameters()}
    assert optimized == expected
    assert not optimized & (encoder_ids | old_actor_ids)
    assert not policy.actor.module[0][2].module.actor_std.requires_grad
    for name, (original, state) in preserved.items():
        assert getattr(policy, name) is original
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="once"):
        configure_raw_teacher(policy, spec)


@pytest.mark.parametrize("height", [False, True])
def test_raw_inputs_reach_actor_unchanged_and_receive_gradients_without_latent_or_students(height):
    policy, spec = make_teacher(height)
    configure_raw_teacher(policy, spec)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Student modules must not execute for raw teacher inference")
    for name in ("adapt_module", "actor_adapt", "object_adapt", "object_pred_transform", "object_transform"):
        getattr(policy, name).forward = forbidden
    observations = spec.rand().apply(lambda value: value.detach().requires_grad_())
    # A stale latent from an external TensorDict must have no influence.
    observations["priv_feature"] = torch.full((2, policy.cfg.latent_dim), float("nan"))
    distribution = teacher_dist(policy, observations)
    expected = torch.cat([observations[key] for key, _ in policy.teacher_input_schema[1]], -1)
    torch.testing.assert_close(observations["_actor_inp"], expected, rtol=0, atol=0)
    assert torch.isfinite(distribution.mean).all()
    differentiable_keys = ["command", "policy", "priv", "object_"]
    if height:
        differentiable_keys.append("height")
    gradient = torch.autograd.grad(distribution.mean.square().sum(),
                                   [observations[key] for key in differentiable_keys])
    for key, value in zip(differentiable_keys, gradient):
        assert torch.isfinite(value).all() and value.abs().sum() > 0, key
    before = torch.random.get_rng_state().clone()
    distribution = teacher_dist(policy, observations.clone())
    assert torch.equal(before, torch.random.get_rng_state())
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        collected = raw_teacher_rollout_policy(policy)(observations.clone())
    torch.testing.assert_close(collected["action"], distribution.mean, rtol=0, atol=0)
    assert not any("adapt" in str(key) for key in collected.keys(True, True) if isinstance(key, tuple))


def test_raw_teacher_never_reads_or_transforms_point_clouds_even_when_omitted():
    policy, spec = make_teacher()
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Raw teacher initialization/inference must not transform object points")
    policy.object_transform.forward = forbidden
    configure_raw_teacher(policy, spec)
    observations = spec.rand()
    expected = teacher_dist(policy, observations.clone())
    changed = observations.clone()
    changed["object_geo_"] = (torch.randn_like(changed["object_geo_"]) * 100).requires_grad_()
    changed["object_trans"] = torch.full_like(changed["object_geo_"], float("nan")).requires_grad_()
    actual = teacher_dist(policy, changed)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(actual.stddev, expected.stddev, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.mean.sum(),
                                    [changed["object_geo_"], changed["object_trans"]], allow_unused=True)
    assert gradients == (None, None)
    omitted = observations.exclude("object_geo_", "object_trans")
    no_geometry = teacher_dist(policy, omitted.clone())
    torch.testing.assert_close(no_geometry.mean, expected.mean, rtol=0, atol=0)
    rollout = raw_teacher_rollout_policy(policy)
    assert "object_geo_" not in rollout.in_keys and "object_trans" not in rollout.in_keys
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        collected = rollout(omitted)
    torch.testing.assert_close(collected["action"], expected.mean, rtol=0, atol=0)


@pytest.mark.parametrize("residual", [False, True])
def test_raw_teacher_preserves_reference_residual_and_gaussian_noise(residual):
    policy, spec = make_teacher(residual=residual)
    configure_raw_teacher(policy, spec)
    head = policy.actor.module[0][2].module
    with torch.no_grad():
        head.actor_mean.weight.zero_()
        head.actor_mean.bias.zero_()
    observation = spec.rand()
    distribution = teacher_dist(policy, observation)
    expected = observation["ref_joint_pos_"] if residual else torch.zeros_like(observation["ref_joint_pos_"])
    torch.testing.assert_close(distribution.mean, expected, rtol=0, atol=0)
    torch.testing.assert_close(distribution.stddev, torch.full_like(expected, .3), rtol=0, atol=0)
    assert policy.teacher_input_schema[-1] == residual


def test_raw_teacher_schema_is_independent_of_environment_batch_size():
    policy, spec = make_teacher()
    first = configure_raw_teacher(policy, spec)
    other, same_spec = make_teacher()
    # Only initialization samples change; event widths define the contract.
    larger = Composite({key: Unbounded((5, *leaf.shape[1:])) for key, leaf in same_spec.items()}, shape=(5,))
    second = configure_raw_teacher(other, larger)
    assert first == second


def declare_duplicate_object_pose(policy):
    policy.env.cfg.observation = {
        "object_": {"object_xy_b": {"noise_std": .01},
                    "object_heading_b": {"noise_std": .01},
                    "ref_contact_pos_b": {"noise_std": .01, "episodic_noise_std": .02, "yaw_only": True},
                    "object_pos_b": {}, "object_ori_b": {}},
        "priv": {"object_pos_b": {}, "object_ori_b": {}},
    }
    policy.env.cfg.command = {"contact_eef_body_name": ["right_ankle_roll_link", "left_ankle_roll_link"]}


def test_duplicate_object_pose_projection_is_actor_only_and_preserves_raw_privileged_pose():
    policy, spec = make_teacher(object_width=22)
    declare_duplicate_object_pose(policy)
    original = spec.rand()
    # Privileged normalization is outside this helper; retaining an affine copy
    # preserves the measurement without requiring the two input vectors equal.
    original["priv"][..., :12] = (original["object_"][..., -12:] - .7) / 2
    critic_before = policy.critic(original.clone())["state_value"].detach().clone()
    critic_state = {key: value.clone() for key, value in policy.critic.state_dict().items()}
    schema = configure_raw_teacher(policy, spec)
    assert schema[1] == (("command", 18), ("policy", 32), ("priv", 48), ("_actor_object", 10))
    assert policy.raw_teacher_projection.module.stop == 10
    observations = original.clone().apply(lambda value: value.detach().requires_grad_())
    source = observations["object_"]
    expected = teacher_dist(policy, observations)
    assert observations["object_"] is source
    assert observations["_actor_object"].untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    torch.testing.assert_close(source, original["object_"], rtol=0, atol=0)
    direct_object_grad = torch.autograd.grad(observations["_actor_object"].sum(), source, retain_graph=True)[0]
    assert torch.equal(direct_object_grad[..., :10], torch.ones_like(direct_object_grad[..., :10]))
    assert torch.count_nonzero(direct_object_grad[..., 10:]) == 0
    object_grad, priv_grad = torch.autograd.grad(expected.mean.sum(), (source, observations["priv"]))
    assert object_grad[..., :10].abs().sum() > 0
    assert torch.count_nonzero(object_grad[..., 10:]) == 0
    assert priv_grad[..., :12].abs().sum() > 0
    changed = original.clone()
    changed["object_"][..., 10:] = torch.randn_like(changed["object_"][..., 10:]) * 100
    actual = teacher_dist(policy, changed)
    torch.testing.assert_close(changed["_actor_object"], observations["_actor_object"], rtol=0, atol=0)
    torch.testing.assert_close(actual.mean, expected.mean, rtol=0, atol=0)
    changed = original.clone()
    changed["priv"][..., :12] += 100
    assert not torch.equal(teacher_dist(policy, changed).mean, expected.mean)
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        rollout = raw_teacher_rollout_policy(policy)(original.clone())
    torch.testing.assert_close(rollout["action"], expected.mean, rtol=0, atol=0)
    torch.testing.assert_close(policy.critic(original.clone())["state_value"], critic_before, rtol=0, atol=0)
    for name, value in policy.critic.state_dict().items():
        torch.testing.assert_close(value, critic_state[name], rtol=0, atol=0)


@pytest.mark.parametrize("case", ["missing_config", "wrong_order", "missing_privileged_pose", "options",
                                   "matching_noise_options", "unknown_prefix_options", "wrong_width", "unknown_contacts"])
def test_duplicate_pose_projection_requires_proven_noiseless_layout(case):
    policy, spec = make_teacher(object_width=24 if case == "wrong_width" else 22)
    declare_duplicate_object_pose(policy)
    if case == "missing_config":
        del policy.env.cfg.observation
    elif case == "wrong_order":
        group = policy.env.cfg.observation["object_"]
        group["object_pos_b"] = group.pop("object_pos_b")
    elif case == "missing_privileged_pose":
        del policy.env.cfg.observation["priv"]["object_ori_b"]
    elif case == "options":
        policy.env.cfg.observation["object_"]["object_pos_b"] = {"noise_std": .1}
    elif case == "matching_noise_options":
        for group in ("object_", "priv"):
            policy.env.cfg.observation[group]["object_pos_b"] = {"noise_std": .1}
    elif case == "unknown_prefix_options":
        policy.env.cfg.observation["object_"]["ref_contact_pos_b"]["history"] = 2
    elif case == "unknown_contacts":
        policy.env.cfg.command["contact_eef_body_name"] = ".*ankle_roll_link"
    schema = configure_raw_teacher(policy, spec)
    assert not hasattr(policy, "raw_teacher_projection")
    assert dict(schema[1])["object_"] == spec["object_"].shape[-1]
    assert "object_trans" not in dict(schema[1])


def test_real_skateboard_config_selects_the_ten_element_object_prefix():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "cfg"), version_base=None):
        config = compose(config_name="train", overrides=["algo=ppo_vel_train", "task=G1/vaic/skateboard_tea"])
    policy, spec = make_teacher(object_width=22)
    policy.env.cfg = config.task
    schema = configure_raw_teacher(policy, spec)
    assert dict(schema[1])["_actor_object"] == 10
    assert "object_trans" not in dict(schema[1])
