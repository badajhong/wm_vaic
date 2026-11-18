"""Reward-port parity against the original compute methods, without Isaac Sim."""

import ast
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from active_adaptation.learning.wm import reward as wm


ROOT = Path(__file__).parents[1]
HDMI = ROOT / "active_adaptation/envs/mdp/commands/hdmi/rewards.py"
COMMON = ROOT / "active_adaptation/envs/mdp/rewards/common.py"
FEET = ROOT / "active_adaptation/envs/mdp/rewards/feet.py"


def source_method(path, class_name, method="compute"):
    """Execute the repository's original method, without simulator imports."""
    tree = ast.parse(path.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == class_name)
    function = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == method)
    function.decorator_list = []
    namespace = {"torch": torch, **{name: getattr(wm, name) for name in
        ("quat_apply_inverse", "quat_mul", "quat_conjugate", "axis_angle_from_quat", "yaw_quat")}}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method]


def term(kind, **kwargs):
    obj = type(kind, (), {})()
    obj.__dict__.update(weight=0.7, enabled=True, sigma=0.5, tolerance=0.03, num_bodies=2,
                        body_indices_asset=[2, 0], body_indices_motion=[0, 2],
                        joint_indices_asset=[2, 0], joint_indices_motion=[0, 2],
                        joint_ids=[0, 2], body_ids=[0, 2], articulation_body_ids=[2, 0],
                        pos_sigma=0.3, pos_tolerance=0.01, frc_sigma=40.0,
                        frc_thres=10.0, gain=5.0, thres=0.2, soft_discount=0.8,
                        default_mass_total=torch.tensor(30., dtype=torch.double))
    obj.__dict__.update(kwargs)
    return obj


def fixture(kind, *, multiplicative=False, mult_dt=True):
    torch.manual_seed(42)
    batch, bodies, joints = 3, 4, 3
    rand = lambda *shape: torch.randn(batch, *shape, dtype=torch.double)
    quaternion = lambda *shape: torch.nn.functional.normalize(rand(*shape, 4), dim=-1)
    state = {
        "robot.root_link_pos_w": rand(3), "robot.root_link_quat_w": quaternion(),
        "object.root_link_pos_w": rand(3), "object.root_link_quat_w": quaternion(),
        "command.robot_root_pos_w": rand(3), "command.robot_root_quat_w": quaternion(),
        "command.object_pos_w": rand(3), "command.object_quat_w": quaternion(),
        "command.contact_target_pos_w": rand(2, 3), "command.contact_eef_pos_w": rand(2, 3),
        "command.eef_contact_forces_b": rand(2, 3),
        "command.is_standing_env": torch.tensor([[False], [True], [False]]),
        "action.action_buf": rand(2, 3),
    }
    refs = {
        "ref_body_pos_w": rand(bodies, 3), "ref_body_quat_w": quaternion(bodies),
        "ref_body_lin_vel_w": rand(bodies, 3), "ref_body_ang_vel_w": rand(bodies, 3),
        "ref_root_pos_w": rand(3), "ref_root_quat_w": quaternion(),
        "ref_joint_pos": rand(joints), "ref_joint_vel": rand(joints),
        "ref_object_pos_w": rand(3), "ref_object_quat_w": quaternion(),
        "ref_body_contact": torch.tensor([[True, True], [True, False], [False, True]]),
    }
    state.update({"command." + key: value for key, value in refs.items()})
    nxt = {
        "robot.root_link_pos_w": rand(3), "robot.root_link_quat_w": quaternion(),
        "robot.body_link_pos_w": rand(bodies, 3), "robot.body_link_quat_w": quaternion(bodies),
        "robot.body_com_vel_w": rand(bodies, 6), "robot.joint_pos": rand(joints),
        "robot.joint_vel": rand(joints), "robot.applied_torque": rand(joints) * 20,
        "object.root_link_pos_w": rand(3), "object.root_link_quat_w": quaternion(),
        "contact.net_forces_w_history": rand(4, bodies, 3) * 50,
        "contact.current_contact_time": torch.tensor([[0., .005, .025, .5],
            [.01, 0., .005, .5], [.015, .4, 0., .02]], dtype=torch.double),
        "contact.last_air_time": rand(bodies).abs() / 10,
        "reward.main.test.joint_vel": rand(2, 2),
        "reward.main.test.soft_limits": torch.tensor([[[-.5, .5], [-.8, .8]]], dtype=torch.double).expand(batch, -1, -1),
    }
    nxt.update({"pre." + key: value for key, value in refs.items()})
    if kind == "joint_torque_limits":
        nxt["reward.main.test.soft_limits"] = torch.full((batch, 2), 10., dtype=torch.double)
    reward = term(kind)
    env = SimpleNamespace(command_manager=SimpleNamespace(object2=None), step_dt=.02,
        mult_dt=mult_dt, termination_funcs={}, reward_groups=OrderedDict([
            ("empty", SimpleNamespace(funcs={}, multiplicative=False)),
            ("main", SimpleNamespace(funcs={"test": reward}, multiplicative=multiplicative))]))
    schema = {key: tuple(value.shape[1:]) for key, value in {**state, **nxt}.items()}
    transform = SimpleNamespace(context_schema=tuple(schema.items()), context_slices={})
    model = wm.VAICRewardModel(env, transform, None).double()
    return model, state, nxt, rand(2), reward, env


def original_reward(reward, state, nxt, action):
    """Feed exactly the same mixed-time tensors to the production method."""
    data = SimpleNamespace(**{name.split(".", 1)[1]: value for name, value in nxt.items() if name.startswith("robot.")})
    data.body_quat_w = data.body_link_quat_w
    data.body_com_lin_vel_w = data.body_com_vel_w[..., :3]
    data.body_com_ang_vel_w = data.body_com_vel_w[..., 3:]
    data.body_lin_vel_w = data.body_com_lin_vel_w
    command = SimpleNamespace(**{name.split(".", 1)[1]: value for name, value in state.items() if name.startswith("command.")})
    command.asset = SimpleNamespace(data=data)
    command.object2 = None
    sensor_data = SimpleNamespace(**{name.split(".", 1)[1]: value for name, value in nxt.items() if name.startswith("contact.")})
    sensor = SimpleNamespace(data=sensor_data,
        compute_first_contact=lambda dt: (sensor_data.current_contact_time > 0) & (sensor_data.current_contact_time < dt + 1e-8))
    reward.asset = command.asset
    reward.command_manager = command
    reward.contact_sensor = sensor
    history = state["action.action_buf"]
    reward.action_manager = SimpleNamespace(action_buf=torch.cat((action.unsqueeze(-1), history[..., :-1]), -1))
    reward.joint_vel = nxt["reward.main.test.joint_vel"]
    reward.soft_limits = nxt["reward.main.test.soft_limits"]
    reward.env = SimpleNamespace(step_dt=.02, command_manager=command, discount=torch.ones(3, 1, dtype=torch.double))
    reward.num_envs = 3
    reward.device = "cpu"
    kind = type(reward).__name__
    if kind == "eef_contact_exp":
        reward.eef_pos_error = torch.zeros(3, 2, dtype=torch.double)
        reward.eef_frc = torch.zeros(3, 2, 3, dtype=torch.double)
        source_method(HDMI, kind, "update")(reward)
    path = COMMON if kind in {"action_rate_l2", "action_rate2_l2", "joint_vel_l2", "joint_pos_limits", "joint_torque_limits", "survival"} else FEET if kind in {"impact_force_l2", "feet_air_time_skateboard"} else HDMI
    return source_method(path, kind)(reward), reward.env.discount


@pytest.mark.parametrize("kind", sorted(wm.VAICRewardModel.reward_kinds))
def test_every_supported_reward_matches_original_compute_method(kind):
    model, state, nxt, action, reward, _ = fixture(kind)
    expected, discount = original_reward(reward, state, nxt, action)
    expected = expected.to(action)
    actual, terminated, actual_discount = model(state, nxt, action)
    torch.testing.assert_close(actual[:, 1:], expected * reward.weight * .02, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual[:, :1], torch.zeros_like(expected))
    torch.testing.assert_close(discount, actual_discount)
    assert not terminated.any()


def test_reward_timing_uses_old_object_cache_and_current_body_with_old_local_anchor():
    model, state, nxt, action, _, _ = fixture("object_pos_tracking")
    before = model(state, nxt, action)[0]
    nxt["object.root_link_pos_w"] = nxt["object.root_link_pos_w"] + 100
    nxt["pre.object_pos_w"] = state["command.object_pos_w"] + 500
    torch.testing.assert_close(model(state, nxt, action)[0], before)
    state["command.object_pos_w"] = state["command.object_pos_w"] + 100
    assert not torch.allclose(model(state, nxt, action)[0], before)
    model, state, nxt, action, _, _ = fixture("keypoint_pos_tracking_local_product")
    before = model(state, nxt, action)[0]
    nxt["robot.root_link_pos_w"] = nxt["robot.root_link_pos_w"] + 100
    torch.testing.assert_close(model(state, nxt, action)[0], before)
    nxt["robot.body_link_pos_w"] = nxt["robot.body_link_pos_w"] + .1
    assert not torch.allclose(model(state, nxt, action)[0], before)


def test_multiplicative_reward_multiplies_weights_before_single_dt_factor():
    model, state, nxt, action, reward, env = fixture("survival", multiplicative=True)
    env.reward_groups["main"].funcs["other"] = term("survival", weight=2.)
    model = wm.VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None)
    actual = model(state, nxt, action)[0]
    torch.testing.assert_close(actual[:, 1], torch.full((3,), .7 * 2 * .02, dtype=torch.double))
    env.mult_dt = False
    model = wm.VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None)
    torch.testing.assert_close(model(state, nxt, action)[0][:, 1], torch.full((3,), 1.4, dtype=torch.double))


def test_analytic_tracking_and_action_rate_have_correct_action_directional_derivatives():
    model, state, nxt, action, _, env = fixture("joint_pos_tracking_product")
    env.reward_groups["main"].funcs["rate"] = term("action_rate_l2", weight=.1)
    model = wm.VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None).double()
    base = nxt["robot.joint_pos"].clone()
    action.requires_grad_()

    def objective(value):
        predicted = dict(nxt)
        predicted["robot.joint_pos"] = base + torch.cat((value, value[..., :1]), -1) * .1
        return model(state, predicted, value)[0]

    assert torch.autograd.gradcheck(objective, (action,), atol=1e-6, rtol=1e-4)
    objective(action).sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0


def test_quaternion_identity_has_finite_backward_and_nonzero_rotations_pass_gradcheck():
    identity = torch.tensor([[1., 0., 0., 0.]], dtype=torch.double, requires_grad=True)
    wm.orientation_error(identity.detach(), identity).sum().backward()
    assert torch.isfinite(identity.grad).all()
    q = torch.tensor([[.9, .1, -.2, .3]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(wm.axis_angle_from_quat, (q,))


def test_termination_is_consecutive_and_uses_new_object_pose():
    model, state, nxt, action, _, env = fixture("survival")
    name = "cum_object_pos_error"
    env.termination_funcs[name] = term(name, threshold=.25, min_steps=3)
    counter_key = "termination." + name + "._cum_error_mixin__cum_steps"
    state[counter_key] = torch.tensor([2., 2., 0.], dtype=torch.double)
    target = nxt["pre.ref_object_pos_w"]
    nxt["object.root_link_pos_w"] = target + torch.tensor([[1., 0, 0], [.1, 0, 0], [.25, 0, 0]], dtype=torch.double)
    model = wm.VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None)
    details = model.termination_details(state, nxt)[name]
    torch.testing.assert_close(details["count"], torch.tensor([3., 0., 1.], dtype=torch.double))
    assert details["terminated"].tolist() == [[True], [False], [False]]
    updated, terminated = model.update_termination_state(state, nxt)
    torch.testing.assert_close(updated[counter_key], details["count"])
    # Wrapper calls forward after writing counters: it must not count twice.
    torch.testing.assert_close(model(state, updated, action)[1], terminated)
    assert counter_key not in nxt


def test_unsupported_active_rewards_and_disabled_discount_side_effects_fail_closed():
    _, _, _, _, _, env = fixture("survival")
    env.reward_groups["main"].funcs["test"] = term("custom_reward")
    transform = SimpleNamespace(context_schema=(), context_slices={})
    with pytest.raises(ValueError, match="Unsupported enabled"):
        wm.VAICRewardModel(env, transform, None)
    env.reward_groups["main"].funcs["test"] = term("custom_reward", enabled=False, soft_discount=.5)
    with pytest.raises(ValueError, match="changes discount"):
        wm.VAICRewardModel(env, transform, None)


def test_resume_keeps_fresh_environment_reward_constants_and_rejects_old_persisted_metadata():
    source, state, nxt, action, _, _ = fixture("impact_force_l2")
    target, _, _, _, _, _ = fixture("impact_force_l2")
    target.reward_terms[0].default_mass_total.fill_(80.)
    expected = target(state, nxt, action)[0]
    checkpoint = source.state_dict()
    assert not any("default_mass_total" in key for key in checkpoint)
    target.load_state_dict(checkpoint, strict=True)
    assert target.reward_terms[0].default_mass_total.item() == 80.
    torch.testing.assert_close(target(state, nxt, action)[0], expected, rtol=0, atol=0)
    assert not torch.equal(expected, source(state, nxt, action)[0])
    # Constants remain buffers so moving/casting the module still moves them.
    assert "reward_terms.0.default_mass_total" in dict(target.named_buffers())
    assert target.float().reward_terms[0].default_mass_total.dtype == torch.float32
    with pytest.raises(RuntimeError, match="Unexpected key"):
        target.load_state_dict({"reward_terms.0.default_mass_total": torch.tensor(30.)}, strict=True)


@pytest.mark.parametrize("kind", sorted(wm.VAICRewardModel.termination_kinds))
def test_every_termination_error_and_counter_matches_original_update_method(kind):
    _, state, nxt, action, reward, env = fixture("survival")
    original_reward(reward, state, nxt, action)  # Build the original-method tensor proxy.
    command = reward.command_manager
    command.object = SimpleNamespace(data=SimpleNamespace(
        root_link_pos_w=nxt["object.root_link_pos_w"], root_link_quat_w=nxt["object.root_link_quat_w"]))
    termination = term(kind, threshold=.5, min_steps=3, pos_thres=.2, frc_thres=1.)
    env.termination_funcs[kind] = termination
    key = "termination." + kind + "._cum_error_mixin__cum_steps"
    state[key] = torch.tensor([0., 2., 7.], dtype=torch.double)
    model = wm.VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None)

    path = ROOT / "active_adaptation/envs/mdp/commands/hdmi/terminations.py"
    tree = ast.parse(path.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name in ("_cum_error_mixin", kind)]
    for cls in classes:
        cls.body = [node for node in cls.body if not isinstance(node, ast.FunctionDef) or node.name != "__init__"]
    namespace = {"torch": torch, "RobotTrackTermination": object, "RobotObjectTrackTermination": object,
        **{name: getattr(wm, name) for name in
           ("quat_apply_inverse", "quat_mul", "quat_conjugate", "axis_angle_from_quat", "yaw_quat")}}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), namespace)
    original = namespace[kind]()
    original.__dict__.update(termination.__dict__)
    original.command_manager = command
    original.error = torch.zeros(3, dtype=torch.double)
    original._cum_error_mixin__cum_steps = state[key].clone()
    original._cum_error_mixin__exceeded = torch.zeros(3, dtype=torch.bool)
    original.update()
    actual = model.termination_details(state, nxt)[kind]
    torch.testing.assert_close(actual["error"], original.error, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual["count"], original._cum_error_mixin__cum_steps)
    torch.testing.assert_close(actual["terminated"], original())


def test_packed_state_arbitrary_batch_and_reference_template_preserve_gradients():
    _, state, nxt, action, _, env = fixture("keypoint_pos_tracking_local_product")
    kind = "cum_object_pos_error"
    env.termination_funcs[kind] = term(kind, threshold=.5, min_steps=3)
    state["termination." + kind + "._cum_error_mixin__cum_steps"] = torch.tensor([0., 2., 4.], dtype=torch.double)
    state["termination." + kind + ".error"] = torch.zeros(3, dtype=torch.double)
    state["termination." + kind + "._cum_error_mixin__exceeded"] = torch.zeros(3, dtype=torch.double)
    shapes = {name: tuple(value.shape[1:]) for name, value in {**state, **nxt}.items()}
    slices, width = {}, 0
    for name, shape in shapes.items():
        size = torch.Size(shape).numel()
        slices[name] = slice(width, width + size)
        width += size
    transform = SimpleNamespace(context_schema=tuple(shapes.items()), context_slices=slices)
    codec = SimpleNamespace(slices={"wm_context": slice(5, width + 5)})
    model = wm.VAICRewardModel(env, transform, codec).double()
    reference_model = wm.VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None).double()

    def pack(fields):
        output = torch.zeros(3, width + 5, dtype=torch.double)
        for name, value in fields.items():
            region = slices[name]
            output[:, region.start + 5:region.stop + 5] = value.reshape(3, -1).to(output)
        return output

    packed = pack(state).unsqueeze(0).expand(2, -1, -1).clone().requires_grad_()
    next_packed = pack(nxt).unsqueeze(0).expand(2, -1, -1).clone().requires_grad_()
    actions = action.unsqueeze(0).expand(2, -1, -1)
    expected = reference_model(state, nxt, action)
    actual = model(packed, next_packed, actions, next_packed.detach())
    for output, target in zip(actual, expected):
        torch.testing.assert_close(output, target.unsqueeze(0).expand_as(output))
    corrected, terminated = model.update_termination_state(packed, next_packed)
    torch.testing.assert_close(terminated, actual[1])
    # Compare the functional replacement against the previous dense-write
    # implementation, including gradients through the analytic error/counter.
    legacy = next_packed.clone()
    values = model.termination_details(packed, next_packed)[kind]
    for suffix, value in (("error", values["error"]),
                          ("_cum_error_mixin__exceeded", values["exceeded"]),
                          ("_cum_error_mixin__cum_steps", values["count"])):
        region = slices["termination." + kind + "." + suffix]
        legacy[..., region.start + 5:region.stop + 5] = value.unsqueeze(-1).to(legacy.dtype)
    torch.testing.assert_close(corrected, legacy, rtol=0, atol=0)
    weights = torch.randn_like(corrected)
    actual_grads = torch.autograd.grad((corrected * weights).sum(), (packed, next_packed), retain_graph=True)
    expected_grads = torch.autograd.grad((legacy * weights).sum(), (packed, next_packed), retain_graph=True)
    for gradient, expected_gradient in zip(actual_grads, expected_grads):
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    # Disable the call-local decoder to exercise the previous independent
    # packed-slice read path. Values and both input gradients must be unchanged.
    decode = model._decode_context
    model._decode_context = lambda value: value
    old_read_reward = model(packed, next_packed, actions, next_packed.detach())[0]
    model._decode_context = decode
    reward_weights = torch.randn_like(actual[0])
    actual_reward_grads = torch.autograd.grad((actual[0] * reward_weights).sum(),
                                             (packed, next_packed), retain_graph=True)
    old_reward_grads = torch.autograd.grad((old_read_reward * reward_weights).sum(),
                                          (packed, next_packed), retain_graph=True)
    torch.testing.assert_close(actual[0], old_read_reward, rtol=0, atol=0)
    for gradient, expected_gradient in zip(actual_reward_grads, old_reward_grads):
        torch.testing.assert_close(gradient, expected_gradient, rtol=0, atol=0)
    actual[0].sum().backward()
    assert next_packed.grad.isfinite().all() and next_packed.grad.abs().sum() > 0
    assert corrected.shape == next_packed.shape
