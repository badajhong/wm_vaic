"""Physical-model invariants and gradients, independent of Isaac runtime."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.wm.core import ObservationCodec
from active_adaptation.learning.wm.physics import (
    DifferentiableKinematics, PhysicalWorldModel, quat_exp, quat_multiply,
    quat_rotate, quat_normalize, quat_conjugate, quaternion_error, fixed_action_contract,
    contact_sensor_transition, contact_force_support, is_contact_field, schema_contact_model,
    schema_frame, quat_axes, heading_basis, world_to_heading, heading_to_world,
    schema_object_body_features,
)


def make_tree(prefix):
    return DifferentiableKinematics(
        [f"{prefix}_root", f"{prefix}_child"], [f"{prefix}_joint"],
        [dict(name=f"{prefix}_joint", parent=f"{prefix}_root", child=f"{prefix}_child",
              type="revolute", local_pos0=(1., 0., 0.), local_pos1=(-.3, 0., 0.), axis=(0., 0., 1.))],
    )


def make_model(batch=4, hidden=(32, 32), action_scale=1., training_loss_scales=None,
               prefixes=("robot", "object"), object_dynamics="articulated",
               object_torques=False, object_body_id=0, contact_model=True, frame="world",
               object_body_features=False):
    shapes = {}
    for prefix in prefixes:
        for name, shape in {
            "root_link_pos_w": (3,), "root_link_quat_w": (4,), "root_com_vel_w": (6,),
            "body_link_pos_w": (2, 3), "body_link_quat_w": (2, 4), "body_com_vel_w": (2, 6),
            "joint_pos": (1,), "joint_vel": (1,), "physics.com": (2, 7),
            "kinematic_scale": (3,), "joint_pos_substeps": (2, 1), "joint_vel_substeps": (2, 1),
        }.items():
            shapes[f"{prefix}.{name}"] = shape
    shapes.update({
        "action.action_buf": (1, 3), "action.applied_action": (1,),
        "action.alpha": (1,), "action.delay": (1,), "action.offset": (1,),
        "episode_length": (1,), "command.t": (1,),
        "contact.net_forces_w_history": (3, 2, 3), "contact.current_air_time": (2,),
        "contact.current_contact_time": (2,),
        "contact.last_air_time": (2,), "contact.last_contact_time": (2,),
        "contact.net_forces_w_substeps": (4, 2, 3),
        "contact.timestamp": (), "contact.timestamp_last_update": (),
        "robot.applied_torque": (1,), "reward.loco.joint_vel_l2.joint_vel": (2, 1),
        "pre.ref_joint_pos": (1,),
    })
    if object_torques:
        shapes.update({"object.applied_torque": (1,), "object.custom_torques": ()})
    sections, offset = {}, 0
    for name, shape in shapes.items():
        width = int(torch.tensor(shape).prod())
        sections[name] = slice(offset, offset + width)
        offset += width
    transform = SimpleNamespace(context_schema=tuple(shapes.items()), context_slices=sections,
                                exogenous_context_slices={"pre.ref_joint_pos": sections["pre.ref_joint_pos"]})
    spec = Composite(policy=Unbounded((4,)), object_=Unbounded((12,)), command=Unbounded((2,)),
                     wm_context=Unbounded((offset,)))
    codec = ObservationCodec(spec, list(spec.keys()))
    env = SimpleNamespace(physics_dt=.005, decimation=4,
                          action_manager=SimpleNamespace(action_dim=1, joint_ids=[0], joint_names=["robot_joint"],
                                                         action_scaling=torch.tensor([action_scale]),
                                                         default_joint_pos=torch.zeros(batch, 1)),
                          command_manager=SimpleNamespace(asset=SimpleNamespace(cfg=SimpleNamespace(actuators={
                              "motor": SimpleNamespace(joint_names_expr=["robot_joint"], effort_limit_sim=10.)})),
                              object=object(), object2=object(), object_body_id_asset=object_body_id),
                          reward_groups={"loco": SimpleNamespace(funcs={
                              "joint_vel_l2": SimpleNamespace(joint_ids=[0])})})
    trees = {prefix: make_tree(prefix) for prefix in prefixes}
    model = PhysicalWorldModel(codec, transform, env, hidden, kinematics=trees,
                               training_loss_scales=training_loss_scales, object_dynamics=object_dynamics,
                               contact_model=contact_model, frame=frame,
                               object_body_features=object_body_features)
    state = torch.zeros(batch, codec.state_dim)
    updates = {"action.alpha": torch.full((batch, 1), .8),
               "action.delay": torch.tensor([[0], [2], [4], [6]])[:batch].float()}
    for prefix in prefixes:
        updates[f"{prefix}.root_link_pos_w"] = torch.tensor([0., 0., 1.]).expand(batch, 3).clone()
        updates[f"{prefix}.root_link_quat_w"] = torch.tensor([1., 0., 0., 0.]).expand(batch, 4).clone()
        updates[f"{prefix}.kinematic_scale"] = torch.ones(batch, 3)
    state = model.set_context(state, updates)
    state = model.baseline_prediction(state, torch.zeros(batch, 1))
    return model, state


def test_fixed_action_schema_detects_changed_scale_but_ignores_num_envs():
    original, _ = make_model(batch=4, action_scale=.5)
    different_batch, _ = make_model(batch=2, action_scale=.5)
    changed_scale, _ = make_model(batch=4, action_scale=.6)
    assert original.schema == different_batch.schema
    assert original.schema != changed_scale.schema


def test_root_object_mode_reduces_only_object_channels_and_has_explicit_schema():
    legacy, _ = make_model(object_torques=True)
    root, _ = make_model(object_dynamics="root", object_torques=True)
    assert legacy.object_dynamics == "articulated"
    assert legacy.schema[0] == "physical_causal_contact_substeps_v2"
    assert root.schema[0] == "physical_causal_contact_substeps_root_v1"
    assert root.schema[2:] == legacy.schema[2:]
    assert root.codec.state_dim == legacy.codec.state_dim
    assert root.velocity_slices["robot"] == legacy.velocity_slices["robot"]
    assert root.velocity_slices["object"].stop - root.velocity_slices["object"].start == 6
    assert root.velocity_head.out_features == legacy.velocity_head.out_features - 1
    assert root.trunk[0].in_features == legacy.trunk[0].in_features - 2
    assert root.trunk[0].out_features == legacy.trunk[0].out_features
    for actual, expected in zip(root.trunk[1:].parameters(), legacy.trunk[1:].parameters()):
        assert actual.shape == expected.shape
    assert set(legacy.input_keys) - set(root.input_keys) == {"object.joint_pos", "object.joint_vel"}
    assert set(legacy.auxiliary_keys) - set(root.auxiliary_keys) == {
        "object.applied_torque", "object.custom_torques"}
    assert root.auxiliary_head.out_features == legacy.auxiliary_head.out_features - 2
    assert root.kinematics["object"].schema == legacy.kinematics["object"].schema
    with pytest.raises(ValueError, match="object_dynamics"):
        make_model(object_dynamics="rigid_typo")
    with pytest.raises(ValueError, match="tracked contact body"):
        make_model(object_dynamics="root", object_body_id=1)


def test_root_object_prediction_and_gradients_cannot_read_wheel_placeholders():
    torch.manual_seed(128)
    model, initial = make_model(object_dynamics="root", object_torques=True)
    wheel_names = tuple(f"object.{name}" for name in (
        "joint_pos", "joint_vel", "joint_pos_substeps", "joint_vel_substeps",
        "applied_torque", "custom_torques", "body_link_pos_w", "body_link_quat_w", "body_com_vel_w"))
    altered = model.set_context(initial, {name: torch.randn_like(model.context(initial, name)) * 30
                                        for name in wheel_names})
    first = initial.detach().requires_grad_()
    other = altered.detach().requires_grad_()
    action = torch.randn(4, 1, requires_grad=True)
    prediction, counterpart = first, other
    for _ in range(3):
        prediction = model(prediction, action)
        counterpart = model(counterpart, action)
        torch.testing.assert_close(prediction, counterpart, rtol=0, atol=0)
        for name in wheel_names[:6]:
            assert torch.count_nonzero(model.context(prediction, name)) == 0
        torch.testing.assert_close(model.context(prediction, "object.body_link_pos_w")[:, 0],
                                   model.context(prediction, "object.root_link_pos_w"), rtol=0, atol=0)
        torch.testing.assert_close(model.context(prediction, "object.body_link_quat_w")[:, 0],
                                   model.context(prediction, "object.root_link_quat_w"), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(model.context(prediction, "object.body_com_vel_w")[:, 0],
                                   model.context(prediction, "object.root_com_vel_w"), rtol=0, atol=0)
    objective = model.context(prediction, "object.root_link_pos_w").square().sum()
    objective = objective + model.context(prediction, "robot.joint_pos").square().sum()
    state_grad, action_grad = torch.autograd.grad(objective, (first, action))
    for name in wheel_names:
        assert torch.count_nonzero(model.context(state_grad, name)) == 0
    assert torch.isfinite(state_grad).all() and action_grad.abs().sum() > 0


def test_root_object_losses_statistics_and_metrics_ignore_wheel_targets():
    torch.manual_seed(198)
    model, state = make_model(object_dynamics="root", object_torques=True)
    action = torch.randn(4, 1)
    target = model(state, action).detach()
    irrelevant = ("object.joint_pos", "object.joint_vel", "object.applied_torque", "object.custom_torques")
    changed = model.set_context(target, {key: torch.randn_like(model.context(target, key)) * 100
                                        for key in irrelevant})
    before = model.transition_loss(state, action, target)
    after = model.transition_loss(state, action, changed)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    metrics = model.validation_errors(target, changed)
    assert "object_joint_pos" not in metrics and "object_joint_velocity" not in metrics
    assert "joint_pos" in metrics and "robot_joint_velocity" in metrics
    torch.testing.assert_close(metrics["normalized_dynamics"],
                               model.validation_errors(target, target)["normalized_dynamics"], rtol=0, atol=0)
    other, _ = make_model(object_dynamics="root", object_torques=True)
    other.load_state_dict(model.state_dict())
    model.update_stats(state, action, target)
    other.update_stats(state, action, changed)
    for name in ("state_stats", "auxiliary_stats"):
        for key, value in getattr(model, name).state_dict().items():
            torch.testing.assert_close(value, getattr(other, name).state_dict()[key], rtol=0, atol=0)


def test_root_object_mode_preserves_robot_integration_and_root_com_kinematics():
    articulated, state = make_model()
    root, _ = make_model(object_dynamics="root")
    torch.manual_seed(46)
    state = articulated.set_context(state, {"object.physics.com": torch.randn(4, 2, 7) * .03,
                                           "object.root_com_vel_w": torch.randn(4, 6)})
    action = torch.randn(4, 1)
    increments = torch.randn(4, 4, articulated.velocity_dim) * .02
    shorter = torch.cat((increments[..., articulated.velocity_slices["robot"]],
                         increments[..., articulated.velocity_slices["object"]][..., :6]), -1)
    expected = articulated._integrate(state, action, increments)
    actual = root._integrate(state, action, shorter)
    for key in root.context_slices:
        if key.startswith("robot.") or key.startswith("object.root_"):
            torch.testing.assert_close(root.context(actual, key), root.context(expected, key), rtol=0, atol=0)
    torch.testing.assert_close(root.context(actual, "object.body_com_vel_w")[:, 0],
                               root.context(actual, "object.root_com_vel_w"), rtol=1e-6, atol=1e-7)


def test_fixed_action_contract_rejects_variable_defaults_and_tracks_configured_limits():
    manager = SimpleNamespace(joint_ids=[0], joint_names=["joint"], action_scaling=torch.tensor([.5]),
                              default_joint_pos=torch.zeros(4, 1))
    config = SimpleNamespace(joint_names_expr=["joint"], effort_limit_sim=10.)
    asset = SimpleNamespace(cfg=SimpleNamespace(actuators={"motor": config}))
    original = fixed_action_contract(manager, asset)
    config.effort_limit_sim = 11.
    assert fixed_action_contract(manager, asset) != original
    manager.default_joint_pos[1, 0] = .1
    with pytest.raises(ValueError, match="Per-environment default_joint_pos"):
        fixed_action_contract(manager, asset)


def test_quaternion_exponential_is_unit_smooth_at_zero_and_matches_rotation():
    vector = torch.zeros(5, 3, requires_grad=True)
    q = quat_exp(vector)
    torch.testing.assert_close(q[:, 0], torch.ones(5))
    q[:, 1:].sum().backward()
    torch.testing.assert_close(vector.grad, torch.full_like(vector, .5))
    rotation = quat_exp(torch.tensor([[0., 0., torch.pi / 2]]))
    point = quat_rotate(rotation, torch.tensor([[1., 0., 0.]]))
    torch.testing.assert_close(point, torch.tensor([[0., 1., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(quaternion_error(rotation, -rotation), torch.zeros(1), atol=1e-7, rtol=0)


def test_fk_joint_frames_and_spatial_velocities_agree_with_finite_differences():
    tree = make_tree("robot").double()
    p = torch.tensor([[.2, -.1, .9]], dtype=torch.float64)
    q = quat_exp(torch.tensor([[.1, -.2, .3]], dtype=torch.float64))
    joints = torch.tensor([[.5]], dtype=torch.float64, requires_grad=True)
    velocity = torch.tensor([[.5, -.2, .1, .2, -.1, .4]], dtype=torch.float64)
    qdot = torch.tensor([[1.2]], dtype=torch.float64)
    scale = torch.full((1, 3), 1.1, dtype=torch.float64)
    com = torch.tensor([[[.1, 0., 0.], [.15, -.03, 0.]]], dtype=torch.float64)
    pos, quat, body_velocity = tree(p, q, joints, velocity, qdot, scale, com)
    eps = 1e-6
    points = []
    for sign in (-1, 1):
        root_q = quat_multiply(quat_exp(sign * eps * velocity[:, 3:]), q)
        bp, bq, _ = tree(p + sign * eps * velocity[:, :3], root_q,
                        joints + sign * eps * qdot, scale=scale)
        points.append(bp + quat_rotate(bq, com))
    numerical = (points[1] - points[0]) / (2 * eps)
    torch.testing.assert_close(body_velocity[..., :3], numerical, atol=1e-8, rtol=1e-6)
    pos[..., 1, 1].sum().backward()
    assert torch.isfinite(joints.grad).all()
    assert joints.grad.abs().sum() > .01
    torch.testing.assert_close(quat.norm(dim=-1), torch.ones_like(quat[..., 0]))


def _serial_fk_reference(tree, root_pos, root_quat, joint_pos, root_velocity,
                         joint_vel, scale, com_offsets):
    """Frozen per-link equations from before sibling batching, for parity."""
    positions, rotations = [root_pos], [quat_normalize(root_quat)]
    linear, angular = [root_velocity[..., :3]], [root_velocity[..., 3:]]
    for index, (parent, joint, kind) in enumerate(zip(tree.parents, tree.joint_indices, tree.joint_types)):
        p, q, v, w = positions[parent], rotations[parent], linear[parent], angular[parent]
        r0 = quat_rotate(q, tree.local_pos0[index] * scale)
        origin = p + r0
        joint_rotation = quat_multiply(q, tree.local_quat0[index].expand_as(q))
        axis = quat_rotate(joint_rotation, tree.axis[index].expand_as(p))
        v_joint = v + torch.cross(w, r0, dim=-1)
        if kind == "revolute":
            relative = quat_exp(tree.axis[index] * joint_pos[..., joint, None])
            child_quat = quat_multiply(quat_multiply(joint_rotation, relative),
                                       quat_conjugate(tree.local_quat1[index]).expand_as(q))
            child_angular = w + axis * joint_vel[..., joint, None]
        else:
            child_quat = quat_multiply(joint_rotation, quat_conjugate(tree.local_quat1[index]).expand_as(q))
            child_angular = w
            if kind == "prismatic":
                offset = axis * joint_pos[..., joint, None]
                origin = origin + offset
                v_joint = v_joint + torch.cross(w, offset, dim=-1) + axis * joint_vel[..., joint, None]
        child_quat = quat_normalize(child_quat)
        r1 = quat_rotate(child_quat, tree.local_pos1[index] * scale)
        positions.append(origin - r1)
        rotations.append(child_quat)
        linear.append(v_joint - torch.cross(child_angular, r1, dim=-1))
        angular.append(child_angular)
    pos, quat, lin, ang = (torch.stack([values[i] for i in tree.output_indices], -2)
                           for values in (positions, rotations, linear, angular))
    lin = lin + torch.cross(ang, quat_rotate(quat, com_offsets), dim=-1)
    return pos, quat, torch.cat((lin, ang), -1)


@pytest.mark.parametrize("batch", [(5,), (2, 3)])
def test_batched_fk_matches_serial_branches_fixed_prismatic_and_all_input_gradients(batch):
    torch.manual_seed(31)
    topology = [("a", "root", "revolute"), ("b", "root", "prismatic"),
                ("c", "root", "fixed"), ("d", "a", "fixed"),
                ("e", "b", "revolute"), ("f", "c", "prismatic"),
                ("g", "e", "revolute")]
    joints = [dict(name=child, parent=parent, child=child, type=kind,
                   local_pos0=torch.randn(3).tolist(), local_pos1=torch.randn(3).tolist(),
                   local_quat0=quat_exp(torch.randn(3)).tolist(),
                   local_quat1=quat_exp(torch.randn(3)).tolist(), axis=(0., 0., 1.))
              for child, parent, kind in topology]
    moving = [name for name, _, kind in topology if kind != "fixed"]
    tree = DifferentiableKinematics(["root", "g", "c", "a", "f", "b", "e", "d"], moving, joints).double()
    inputs = [torch.randn(*batch, width, dtype=torch.double, requires_grad=True)
              for width in (3, 4, len(moving), 6, len(moving), 3)]
    inputs.append(torch.randn(*batch, 8, 3, dtype=torch.double, requires_grad=True))
    actual, expected = tree(*inputs), _serial_fk_reference(tree, *inputs)
    weights = [torch.randn_like(value) for value in actual]
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(actual, weights)), inputs)
    expected_grad = torch.autograd.grad(sum((value * weight).sum() for value, weight in zip(expected, weights)), inputs)
    for value, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, reference, rtol=1e-11, atol=1e-11)
    # Derived execution indices must not alter saved physical checkpoint keys.
    assert set(tree.state_dict()) == {"local_pos0", "local_pos1", "local_quat0", "local_quat1", "axis"}


def test_training_metrics_skip_reporting_without_changing_objective_or_gradient():
    model, state = make_model()
    prediction = model(state, torch.randn(4, 1)).detach().requires_grad_()
    complete = model.validation_errors(prediction, state)
    compact = model.validation_errors(prediction, state, reporting=False)
    assert set(compact) == {"normalized_dynamics", "auxiliary"}
    for key in compact:
        torch.testing.assert_close(compact[key], complete[key], rtol=0, atol=0)
    full_grad = torch.autograd.grad(sum(complete[key].sum() for key in compact), prediction)[0]
    compact_grad = torch.autograd.grad(sum(value.sum() for value in compact.values()), prediction)[0]
    torch.testing.assert_close(compact_grad, full_grad, rtol=0, atol=0)


def _legacy_physical_loss(model, state, action, target):
    """Frozen objective before configurable optimization units were introduced."""
    prediction, logits, forces = model(state, action, target, return_auxiliary=True)
    terms = []
    for prefix in model.prefixes:
        for suffix, scale in (("root_link_pos_w", .5), ("root_com_vel_w", 2.5),
                              ("joint_pos", .25), ("joint_vel", 2.5)):
            key = f"{prefix}.{suffix}"
            if key in model.context_slices:
                terms.append((model.context(prediction, key) - model.context(target, key)).square().mean(-1) / scale ** 2)
        terms.append(quaternion_error(model.context(prediction, f"{prefix}.root_link_quat_w"),
                                      model.context(target, f"{prefix}.root_link_quat_w")).square() / .5 ** 2)
    dynamics = torch.stack(terms, -1).mean(-1)
    auxiliary = ((model._auxiliaries(prediction) - model._auxiliaries(target)) /
                 model.auxiliary_stats.std).square().mean(-1)
    target_forces = model.context(target, "contact.net_forces_w_substeps")
    active = (target_forces.norm(dim=-1) > model.contact_force_threshold).to(logits.dtype)
    auxiliary = auxiliary + torch.nn.functional.binary_cross_entropy_with_logits(
        logits, active, reduction="none").mean((-1, -2))
    auxiliary = auxiliary + ((forces.flatten(-2) - target_forces.flatten(-2)) /
                             model.contact_force_stats.std).square().mean((-1, -2))
    dynamics = dynamics + (model.context(prediction, "robot.joint_pos_substeps") -
                           model.context(target, "robot.joint_pos_substeps")).square().mean((-1, -2)) / .25 ** 2
    return dynamics, auxiliary


@pytest.mark.parametrize("scales", [None, {}, {"root_pos": .5, "root_rotation": .5,
                                               "object_joint_pos": .25, "object_joint_velocity": 2.5}])
def test_default_training_loss_and_all_parameter_gradients_match_frozen_legacy(scales):
    torch.manual_seed(43)
    model, state = make_model(training_loss_scales=scales)
    action = torch.randn(4, 1)
    target = state + .01 * torch.randn_like(state)
    actual = model.transition_loss(state, action, target)
    dynamics, auxiliary = _legacy_physical_loss(model, state, action, target)
    torch.testing.assert_close(actual["dynamics"], dynamics, rtol=0, atol=0)
    torch.testing.assert_close(actual["auxiliary"], auxiliary, rtol=0, atol=0)
    parameters = tuple(model.parameters())
    actual_gradients = torch.autograd.grad((actual["dynamics"] + actual["auxiliary"]).sum(), parameters,
                                           allow_unused=True)
    expected_gradients = torch.autograd.grad((dynamics + auxiliary).sum(), parameters, allow_unused=True)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("scales", [
    {"object_joint_pos": 1., "object_joint_velocity": 12.},
    {"root_pos": .05, "root_rotation": .2, "object_joint_pos": 1., "object_joint_velocity": 12.},
    # SuperMimic-style Bryson units: generic keys reach the robot joints/twist
    # and the robot substep positions; object_* keys still override object joints.
    {"root_pos": .02, "root_rotation": .05, "root_velocity": 1., "joint_pos": .1,
     "joint_velocity": 3., "object_joint_pos": 1.},
    # Object root units override the generic root units for that asset only.
    {"object_root_velocity": 1.},
    {"root_pos": .02, "root_rotation": .05, "root_velocity": 2.5, "object_root_pos": .01,
     "object_root_rotation": .02, "object_root_velocity": 1., "object_joint_velocity": 12.},
])
def test_training_scales_change_only_selected_terms_and_leave_validation_and_checkpoint_unchanged(scales):
    torch.manual_seed(79)
    legacy, state = make_model(prefixes=("robot", "object", "object2"))
    configured, _ = make_model(training_loss_scales=scales, prefixes=legacy.prefixes)
    assert set(configured.state_dict()) == set(legacy.state_dict())
    configured.load_state_dict(legacy.state_dict(), strict=True)
    assert configured.schema == legacy.schema
    action = torch.randn(4, 1)
    target = state + .1 * torch.randn_like(state)
    baseline, baseline_prediction = legacy.transition_loss(state, action, target, return_prediction=True)
    actual, prediction = configured.transition_loss(state, action, target, return_prediction=True)
    torch.testing.assert_close(prediction, baseline_prediction, rtol=0, atol=0)
    # Check independent term weights, including the untouched robot/object2
    # joints. The optimization group count and substep coefficient stay fixed.
    terms = []
    for prefix in configured.prefixes:
        joint_pos_scale, joint_vel_scale = scales.get("joint_pos", .25), scales.get("joint_velocity", 2.5)
        root_pos_scale, root_vel_scale = scales.get("root_pos", .5), scales.get("root_velocity", 2.5)
        rotation_scale = scales.get("root_rotation", .5)
        if prefix == "object":
            joint_pos_scale = scales.get("object_joint_pos", joint_pos_scale)
            joint_vel_scale = scales.get("object_joint_velocity", joint_vel_scale)
            root_pos_scale = scales.get("object_root_pos", root_pos_scale)
            root_vel_scale = scales.get("object_root_velocity", root_vel_scale)
            rotation_scale = scales.get("object_root_rotation", rotation_scale)
        for suffix, scale in (("root_link_pos_w", root_pos_scale),
                              ("root_com_vel_w", root_vel_scale),
                              ("joint_pos", joint_pos_scale),
                              ("joint_vel", joint_vel_scale)):
            key = f"{prefix}.{suffix}"
            terms.append((configured.context(prediction, key) - configured.context(target, key)).square().mean(-1) / scale ** 2)
        angle = quaternion_error(configured.context(prediction, f"{prefix}.root_link_quat_w"),
                                 configured.context(target, f"{prefix}.root_link_quat_w"))
        terms.append(angle.square() / rotation_scale ** 2)
    substep = (configured.context(prediction, "robot.joint_pos_substeps") -
               configured.context(target, "robot.joint_pos_substeps")).square().mean((-1, -2)) / scales.get("joint_pos", .25) ** 2
    torch.testing.assert_close(actual["substep_joint_pos"], substep, rtol=0, atol=0)
    expected = torch.stack(terms, -1).mean(-1) + substep
    torch.testing.assert_close(actual["dynamics"], expected, rtol=0, atol=0)
    assert not torch.allclose(actual["dynamics"], baseline["dynamics"])
    for key in baseline.keys() - {"dynamics", "substep_joint_pos"}:
        torch.testing.assert_close(actual[key], baseline[key], rtol=0, atol=0)
    for reporting in (False, True):
        configured_errors = configured.validation_errors(prediction, target, reporting=reporting)
        legacy_errors = legacy.validation_errors(baseline_prediction, target, reporting=reporting)
        assert configured_errors.keys() == legacy_errors.keys()
        for key in legacy_errors:
            torch.testing.assert_close(configured_errors[key], legacy_errors[key], rtol=0, atol=0)


def test_object_root_scales_reweight_only_the_object_root_terms():
    torch.manual_seed(83)
    legacy, state = make_model(object_dynamics="root")
    action = torch.randn(4, 1)
    target = state + .1 * torch.randn_like(state)
    base = legacy.transition_loss(state, action, target)["dynamics"]
    # Dynamics is the mean of one rotation term per asset plus its present
    # position/twist/joint terms; root mode drops the object joints.
    def present(prefix, suffix):
        return f"{prefix}.{suffix}" in legacy.context_slices and not (
            prefix == "object" and suffix in ("joint_pos", "joint_vel"))
    term_count = sum(1 + sum(present(prefix, suffix) for suffix in
                             ("root_link_pos_w", "root_com_vel_w", "joint_pos", "joint_vel"))
                     for prefix in legacy.prefixes)
    assert term_count == 8
    for key, suffix in (("object_root_pos", "root_link_pos_w"), ("object_root_velocity", "root_com_vel_w"),
                        ("object_root_rotation", None)):
        default = {"object_root_pos": .5, "object_root_velocity": 2.5, "object_root_rotation": .5}[key]
        configured, _ = make_model(object_dynamics="root", training_loss_scales={key: default / 2})
        configured.load_state_dict(legacy.state_dict(), strict=True)
        prediction = configured(state, action, target)
        if suffix is None:
            error = quaternion_error(configured.context(prediction, "object.root_link_quat_w"),
                                     configured.context(target, "object.root_link_quat_w")).square()
        else:
            error = (configured.context(prediction, f"object.{suffix}") -
                     configured.context(target, f"object.{suffix}")).square().mean(-1)
        # Halving the unit multiplies that one term by four: +3x its default share.
        expected = base + 3 * error / default ** 2 / term_count
        actual = configured.transition_loss(state, action, target)["dynamics"]
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("scales", [
    [], {"robot_joint_pos": .3}, {"object2_joint_pos": .3}, {"root_position": .1},
    {"robot_root_velocity": 1.}, {"object2_root_velocity": 1.},
    {"root_pos": 0}, {"root_pos": -1}, {"root_pos": float("nan")},
    {"root_rotation": float("inf")}, {"object_joint_velocity": None},
    {"object_joint_pos": True}, {"root_pos": "invalid"},
])
def test_training_scales_reject_unknown_nonfinite_or_nonpositive_values(scales):
    with pytest.raises(ValueError, match="training_loss_scales"):
        make_model(training_loss_scales=scales)


def test_fk_rejects_mismatched_joints_and_disconnected_bodies():
    with pytest.raises(ValueError, match="moving joints differ"):
        DifferentiableKinematics(["root"], ["missing"], [])
    with pytest.raises(ValueError, match="missing from USD tree"):
        DifferentiableKinematics(["root", "orphan"], [], [])


def test_fk_startup_oracle_reads_synchronized_physics_not_stale_data_caches():
    model, state = make_model()
    tree = model.kinematics["robot"]
    root_pos = model.context(state, "robot.root_link_pos_w")
    root_quat = model.context(state, "robot.root_link_quat_w")
    joint_pos = torch.full((4, 1), .3)
    body_pos, body_quat, _ = tree(root_pos, root_quat, joint_pos)
    refreshed = []

    def read(value):
        assert refreshed, "PhysX kinematics must be synchronized before reading"
        return value

    # Deliberately omit asset.data: its timestamped fields can disagree during
    # initial scene construction and must not be used by this startup oracle.
    asset = SimpleNamespace(
        _physics_sim_view=SimpleNamespace(update_articulations_kinematic=lambda: refreshed.append(True)),
        root_physx_view=SimpleNamespace(
            get_root_transforms=lambda: read(torch.cat((root_pos, root_quat[..., [1, 2, 3, 0]]), -1)),
            get_dof_positions=lambda: read(joint_pos),
            get_link_transforms=lambda: read(torch.cat((body_pos, body_quat[..., [1, 2, 3, 0]]), -1)),
        ),
        cfg=SimpleNamespace(spawn=SimpleNamespace(scale=None)),
    )
    model._verify_kinematics("robot", asset, tree)
    assert refreshed == [True]


def test_contact_validation_checks_first_contact_and_separates_force_and_torque():
    model, target = make_model()
    target = model.set_context(target, {
        "contact.current_contact_time": torch.tensor([[.005, 0.], [.02, .04], [0., 0.], [.04, .04]]),
    })
    prediction = model.set_context(target, {
        "contact.current_contact_time": torch.tensor([[0., 0.], [.02, .005], [0., 0.], [.04, .04]]),
        "robot.applied_torque": torch.ones(4, 1),
    })
    errors = model.validation_errors(prediction, target)
    torch.testing.assert_close(errors["contact_event_error"], torch.tensor([.5, .5, 0., 0.]))
    torch.testing.assert_close(errors["contact_event_miss_rate"], torch.tensor([1., 0., 0., 0.]))
    torch.testing.assert_close(errors["contact_event_true_count"], torch.tensor([1., 1., 0., 0.]))
    torch.testing.assert_close(errors["contact_event_miss_count"], torch.tensor([1., 0., 0., 0.]))
    assert (errors["torque"] > 0).all()
    torch.testing.assert_close(errors["contact_force"], torch.zeros(4))


def test_contact_presence_reconstructs_timers_history_and_force_support():
    model, state = make_model()
    action = torch.zeros(4, 1)
    prediction = model(state, action)
    torch.testing.assert_close(model.context(prediction, "contact.current_contact_time"), torch.zeros(4, 2))
    assert (model.context(prediction, "contact.current_air_time") > 0).all()
    assert not hasattr(model, "contact_mode_head")
    assert all(not key.startswith("contact.") for key in model.learned_auxiliary_keys)
    with torch.no_grad():
        model.contact_presence_head.bias.fill_(10.)
    prediction = model(state, action)
    forces = model.context(prediction, "contact.net_forces_w_substeps")
    assert (forces.norm(dim=-1) > model.contact_force_threshold).all()
    torch.testing.assert_close(model.context(prediction, "contact.current_air_time"), torch.zeros(4, 2))
    for key, value in model.oracle_contact_transition(state, prediction).items():
        assert torch.equal(model.context(prediction, key), value)
    torch.testing.assert_close(model.context(prediction, "contact.net_forces_w_history"), forces.flip(-3)[:, :3])


def test_contact_counts_are_training_only_and_count_every_physics_substep():
    model, state = make_model()
    before = model.contact_presence_counts.clone()
    model(state, torch.zeros(4, 1))
    model.validation_errors(state, state)
    torch.testing.assert_close(model.contact_presence_counts, before)
    model.update_stats(state, torch.zeros(4, 1), state)
    torch.testing.assert_close(model.contact_presence_counts, before + torch.tensor([32., 0.]))


def test_contact_presence_and_raw_force_learn_when_initial_support_is_incorrect():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        model, state = make_model()
        state = model.set_context(state, {"action.delay": torch.zeros(4, 1)})
        action = torch.tensor([[-1.5], [-.5], [.5], [1.5]])
        forces = torch.zeros(4, 4, 2, 3)
        forces[2:, :, :, 2] = 10.
        target = model.set_context(state, model._contact_transition(state, forces))
        model.update_stats(state, action, target)
        losses = model.transition_loss(state, action, target)
        losses["auxiliary"].mean().backward()
        assert model.contact_presence_head.weight.grad.abs().sum() > 0
        assert model.contact_force_head.weight.grad.abs().sum() > 0
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        for _ in range(100):
            optimizer.zero_grad()
            model.transition_loss(state, action, target)["auxiliary"].mean().backward()
            optimizer.step()
        prediction = model(state, action)
        # Air/contact support matches; first-contact timing is checked below.
        for key in model.contact_timer_keys:
            assert torch.equal(model.context(prediction, key) > 0, model.context(target, key) > 0)
        torch.testing.assert_close(model.validation_errors(prediction, target)["contact_event_error"], torch.zeros(4))
    finally:
        torch.set_num_threads(old_threads)


def test_substep_timer_recurrence_distinguishes_detachment_and_recontact():
    forces = torch.zeros(2, 4, 1, 3)
    forces[..., 2] = 2.
    forces[1, 0] = 0.
    result = contact_sensor_transition(forces, torch.zeros(2, 1), torch.full((2, 1), .1),
                                       torch.full((2, 1), .2), torch.full((2, 1), .3),
                                       torch.zeros(2), torch.zeros(2), .005)
    torch.testing.assert_close(result["current_contact_time"], torch.tensor([[.12], [.015]]))
    assert torch.equal(result["net_forces_w_history"][0], result["net_forces_w_history"][1])
    torch.testing.assert_close(result["last_contact_time"], torch.tensor([[.3], [.105]]))
    torch.testing.assert_close(result["last_air_time"], torch.tensor([[.2], [.01]]))


def test_contact_clock_preserves_float32_boundary_and_short_history_shift():
    timestamp = torch.zeros(1)
    for _ in range(16):
        timestamp += .005
    forces = torch.zeros(1, 4, 1, 3)
    forces[..., 2] = 2.
    result = contact_sensor_transition(forces, timestamp[:, None], torch.zeros(1, 1),
                                       torch.zeros(1, 1), torch.zeros(1, 1), timestamp,
                                       timestamp, .005)
    duration = result["current_contact_time"]
    assert (duration > .02).all()
    assert not (duration < .02 + 1e-8).any()
    previous = torch.arange(9.).reshape(1, 3, 1, 3)
    result = contact_sensor_transition(forces[:, :1], torch.zeros(1, 1), torch.zeros(1, 1),
                                       torch.zeros(1, 1), torch.zeros(1, 1), torch.zeros(1),
                                       torch.zeros(1), .005, previous_force_history=previous)
    torch.testing.assert_close(result["net_forces_w_history"], torch.cat((forces[:, :1], previous[:, :2]), 1))


def test_contact_force_projection_has_exact_hard_support_and_finite_continuous_gradients():
    torch.manual_seed(7)
    raw = (torch.randn(3, 4, 2, 3) * 5).requires_grad_()
    active = torch.rand(3, 4, 2) > .5
    projected = contact_force_support(raw, active)
    assert torch.equal(projected.norm(dim=-1) > 1., active)
    projected.square().sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    zero = torch.zeros(2, 3, requires_grad=True)
    zero_projection = contact_force_support(zero, torch.tensor([False, True]))
    assert torch.equal(zero_projection.norm(dim=-1) > 1., torch.tensor([False, True]))
    zero_projection.sum().backward()
    assert torch.isfinite(zero.grad).all()
    # Direction-normalization roundoff previously produced false contacts for
    # about 0.5% of vectors clipped exactly onto the threshold.
    many_vectors = torch.randn(100000, 3) * 10
    inactive = contact_force_support(many_vectors, torch.zeros(100000, dtype=torch.bool))
    assert not (inactive.norm(dim=-1) > 1.).any()
    active_forces = contact_force_support(many_vectors, torch.ones(100000, dtype=torch.bool))
    assert (active_forces.norm(dim=-1) > 1.).all()


def test_contact_substep_heads_obey_queued_action_causality_across_two_steps():
    torch.manual_seed(9)
    model, state = make_model()
    with torch.no_grad():
        model.contact_presence_head.weight.normal_(0., .1)
    previous_action = torch.randn(4, 1, requires_grad=True)
    current_action = torch.randn(4, 1, requires_grad=True)
    first_state = model(state, previous_action)
    _, logits, forces = model(first_state, current_action, return_auxiliary=True)
    delay = model.context(state, "action.delay").squeeze(-1)
    for index in range(model.decimation):
        for output in (logits[:, index], forces[:, index]):
            previous, current = torch.autograd.grad(output.sum(), (previous_action, current_action), retain_graph=True)
            previous_blocked = delay > model.decimation + index
            current_blocked = delay > index
            torch.testing.assert_close(previous[previous_blocked], torch.zeros_like(previous[previous_blocked]))
            torch.testing.assert_close(current[current_blocked], torch.zeros_like(current[current_blocked]))
            assert (previous[~previous_blocked].abs() > 0).all()
            assert (current[~current_blocked].abs() > 0).all()


def test_functional_context_replacement_preserves_values_and_gradients():
    model, state = make_model()
    state = state.detach().requires_grad_()
    replacement = torch.randn(4, 3, requires_grad=True)
    actual = model.set_context(state, {"robot.root_link_pos_w": replacement})
    section = model.context_slices["robot.root_link_pos_w"]
    expected = state.clone()
    expected[..., section] = replacement
    torch.testing.assert_close(actual, expected)
    weights = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * weights).sum(), (state, replacement))
    expected_grad = torch.autograd.grad((expected * weights).sum(), (state, replacement))
    for observed, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(observed, reference)
    assert type(actual.grad_fn).__name__ == "CatBackward0"


def test_history_and_delay_filter_have_exact_action_derivatives():
    model, state = make_model()
    action = torch.tensor([[1.], [2.], [3.], [4.]], requires_grad=True)
    next_state = model.baseline_prediction(state, action)
    history = model.context(next_state, "action.action_buf")
    torch.testing.assert_close(history[..., 0], action)
    derivative = torch.autograd.grad(history[..., 0].sum(), action, retain_graph=True)[0]
    torch.testing.assert_close(derivative, torch.ones_like(action))
    applied = model.context(next_state, "action.applied_action")
    # Delay 0 sees the new action four times; delay 2 twice; >=4 not yet.
    factors = torch.tensor([[1 - .2 ** 4], [1 - .2 ** 2], [0.], [0.]])
    torch.testing.assert_close(applied, action * factors)
    derivative = torch.autograd.grad(applied.sum(), action)[0]
    torch.testing.assert_close(derivative, factors)


def test_new_action_has_no_physical_or_contact_effect_until_actuator_delay():
    torch.manual_seed(3)
    model, state = make_model()
    action = torch.randn(4, 1, requires_grad=True)
    increments, _ = model._causal_network(state, action)
    delay = model.context(state, "action.delay").squeeze(-1).long()
    for substep in range(model.decimation):
        derivative = torch.autograd.grad(increments[:, substep].sum(), action, retain_graph=True)[0].squeeze(-1)
        torch.testing.assert_close(derivative[delay > substep], torch.zeros_like(derivative[delay > substep]))
        assert (derivative[delay <= substep].abs() > 0).all()
    prediction = model(state, action)
    alternate = model(state, action + 3.)
    for key in ("robot.root_link_pos_w", "robot.root_link_quat_w", "robot.root_com_vel_w",
                "robot.joint_pos", "robot.body_link_pos_w", "object.root_link_pos_w",
                "contact.net_forces_w_history", "robot.applied_torque"):
        value = model.context(prediction, key)
        torch.testing.assert_close(value[delay >= model.decimation], model.context(alternate, key)[delay >= model.decimation])
        derivative = torch.autograd.grad(value.sum(), action, retain_graph=True)[0]
        torch.testing.assert_close(derivative[delay >= model.decimation], torch.zeros_like(derivative[delay >= model.decimation]))
    # History still records the new action immediately, despite physical delay.
    history = model.context(prediction, "action.action_buf")[..., 0]
    torch.testing.assert_close(torch.autograd.grad(history.sum(), action)[0], torch.ones_like(action))


def test_delay_three_keeps_penultimate_joint_sample_independent_of_new_action():
    model, state = make_model()
    state = model.set_context(state, {"action.delay": torch.full((4, 1), 3.)})
    action = torch.randn(4, 1, requires_grad=True)
    prediction = model(state, action)
    substeps = model.context(prediction, "robot.joint_pos_substeps")
    before = torch.autograd.grad(substeps[:, 0].sum(), action, retain_graph=True)[0]
    after = torch.autograd.grad(substeps[:, 1].sum(), action)[0]
    torch.testing.assert_close(before, torch.zeros_like(action))
    assert (after.abs() > 0).all()


def test_delayed_action_can_influence_following_step_through_recorded_history():
    torch.manual_seed(8)
    model, state = make_model()
    state = model.set_context(state, {"action.delay": torch.full((4, 1), 4.)})
    first_action = torch.randn(4, 1, requires_grad=True)
    second_action = torch.randn(4, 1, requires_grad=True)
    first_state = model(state, first_action)
    second_state = model(first_state, second_action)
    joint_pos = model.context(second_state, "robot.joint_pos")
    earlier_gradient, current_gradient = torch.autograd.grad(joint_pos.sum(), (first_action, second_action))
    assert (earlier_gradient.abs() > 0).all()
    torch.testing.assert_close(current_gradient, torch.zeros_like(current_gradient))


def test_filtered_trajectory_matches_two_step_closed_form_for_delays_zero_two_four_six():
    model, state = make_model()
    first_action = torch.randn(4, 1, requires_grad=True)
    second_action = torch.randn(4, 1, requires_grad=True)
    _, first_trajectory = model._actuator_trajectory(state, first_action)
    first_state = model.baseline_prediction(state, first_action)
    _, second_trajectory = model._actuator_trajectory(first_state, second_action)
    trajectory = torch.cat((first_trajectory, second_trajectory), -2)
    delays = model.context(state, "action.delay")
    for time in range(8):
        elapsed = time - delays
        first_factor = torch.where(elapsed < 0, 0., torch.where(
            elapsed < 4, 1 - .2 ** (elapsed + 1), (1 - .2 ** 4) * .2 ** (elapsed - 3)))
        second_elapsed = elapsed - 4
        second_factor = torch.where(second_elapsed < 0, 0., 1 - .2 ** (second_elapsed + 1))
        expected = first_action * first_factor + second_action * second_factor
        torch.testing.assert_close(trajectory[:, time], expected)
        gradients = torch.autograd.grad(trajectory[:, time].sum(), (first_action, second_action), retain_graph=True)
        torch.testing.assert_close(gradients[0], first_factor)
        torch.testing.assert_close(gradients[1], second_factor)


def test_physics_substeps_cannot_see_future_queued_actions_across_two_control_steps():
    torch.manual_seed(21)
    model, state = make_model()
    assert "action.action_buf" not in model.input_keys
    assert model.schema[0] == "physical_causal_contact_substeps_v2"
    first_action = torch.randn(4, 1, requires_grad=True)
    second_action = torch.randn(4, 1, requires_grad=True)
    first_state = model(state, first_action)
    second_increments, _ = model._causal_network(first_state, second_action)
    delay = model.context(state, "action.delay").squeeze(-1).long()
    for substep in range(model.decimation):
        prior, current = torch.autograd.grad(second_increments[:, substep].sum(),
                                             (first_action, second_action), retain_graph=True)
        prior_blocked = delay > model.decimation + substep
        current_blocked = delay > substep
        torch.testing.assert_close(prior[prior_blocked], torch.zeros_like(prior[prior_blocked]))
        torch.testing.assert_close(current[current_blocked], torch.zeros_like(current[current_blocked]))
        assert (prior[~prior_blocked].abs() > 0).all()
        assert (current[~current_blocked].abs() > 0).all()


def test_zero_filter_alpha_blocks_both_new_and_queued_action_influence():
    model, state = make_model()
    state = model.set_context(state, {"action.alpha": torch.zeros(4, 1)})
    queued = torch.randn(4, 1, 3, requires_grad=True)
    state = model.set_context(state, {"action.action_buf": queued})
    action = torch.randn(4, 1, requires_grad=True)
    prediction = model(state, action)
    output = model.context(prediction, "robot.joint_pos").sum() + model.context(prediction, "contact.net_forces_w_history").sum()
    gradients = torch.autograd.grad(output, (action, queued))
    for value in gradients:
        torch.testing.assert_close(value, torch.zeros_like(value))


def test_statistics_mask_compact_inputs_and_normalize_applied_trajectory():
    model, state = make_model()
    reference, _ = make_model()
    action = torch.tensor([[1.], [2.], [3.], [4.]])
    valid = torch.tensor([True, False, True, False])
    model.update_stats(state, action, state, valid=valid)
    reference.update_stats(state[valid], action[valid], state[valid])
    for name in ("state_stats", "action_stats", "auxiliary_stats", "contact_force_stats"):
        for field in ("mean", "var", "count"):
            torch.testing.assert_close(getattr(getattr(model, name), field), getattr(getattr(reference, name), field))
    _, applied = model._actuator_trajectory(state, action)
    torch.testing.assert_close(model.action_stats.mean, applied[valid].mean((0, 1)))
    assert model.action_stats.count.item() == int(valid.sum()) * model.decimation
    torch.testing.assert_close(model.contact_presence_counts, reference.contact_presence_counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for CPU replay/device regression")
def test_cpu_replay_statistics_can_update_cuda_physics_model():
    model, state = make_model()
    reference, _ = make_model()
    model.cuda()
    action = torch.randn(4, 1)
    valid = torch.tensor([True, False, True, True])
    model.update_stats(state, action, state, valid=valid)
    reference.update_stats(state, action, state, valid=valid)
    for name in ("state_stats", "action_stats", "auxiliary_stats", "contact_force_stats"):
        for field in ("mean", "var", "count"):
            torch.testing.assert_close(getattr(getattr(model, name), field).cpu(), getattr(getattr(reference, name), field))
    torch.testing.assert_close(model.contact_presence_counts.cpu(), reference.contact_presence_counts)


def test_constant_velocity_integrates_com_root_orientation_joints_and_body_fk():
    model, state = make_model()
    updates = {}
    for prefix in model.prefixes:
        com = torch.zeros(4, 2, 7)
        com[:, 0, 0] = .2
        updates[f"{prefix}.physics.com"] = com
        updates[f"{prefix}.root_com_vel_w"] = torch.tensor([.1, .2, .3, 0., 0., 1.]).expand(4, 6)
        updates[f"{prefix}.joint_vel"] = torch.ones(4, 1) * 2
    state = model.set_context(state, updates)
    predicted = model.baseline_prediction(state, torch.zeros(4, 1))
    for prefix in model.prefixes:
        pos = model.context(predicted, f"{prefix}.root_link_pos_w")
        quat = model.context(predicted, f"{prefix}.root_link_quat_w")
        original_com = model.context(state, f"{prefix}.root_link_pos_w") + torch.tensor([.2, 0., 0.])
        final_com = pos + quat_rotate(quat, torch.tensor([.2, 0., 0.]))
        torch.testing.assert_close(final_com, original_com + torch.tensor([.002, .004, .006]))
        torch.testing.assert_close(quat, quat_exp(torch.tensor([0., 0., .02])).expand(4, 4))
        torch.testing.assert_close(model.context(predicted, f"{prefix}.joint_pos"), torch.full((4, 1), .04))
        torch.testing.assert_close(model.context(predicted, f"{prefix}.joint_pos_substeps"),
                                   torch.tensor([[[.03], [.04]]]).expand(4, 2, 1))
        bp, bq, _ = model.kinematics[prefix](pos, quat, model.context(predicted, f"{prefix}.joint_pos"))
        torch.testing.assert_close(bp, model.context(predicted, f"{prefix}.body_link_pos_w"))
        torch.testing.assert_close(bq, model.context(predicted, f"{prefix}.body_link_quat_w"))
    torch.testing.assert_close(model.context(predicted, "reward.loco.joint_vel_l2.joint_vel"), torch.full((4, 2, 1), 2.))


def test_model_has_no_observation_reward_or_termination_heads_and_preserves_reference():
    model, state = make_model()
    assert not hasattr(model, "reward_head")
    assert not hasattr(model, "termination_head")
    assert not any(name.startswith("obs_history") or name.startswith("pre.ref") for name in model.input_keys)
    action = torch.randn(4, 1, requires_grad=True)
    template = state.clone()
    template[:, model.codec.slices["command"]] = 7
    template = model.set_context(template, {"pre.ref_joint_pos": torch.ones(4, 1) * 4})
    prediction = model(state, action, template)
    torch.testing.assert_close(prediction[:, model.codec.slices["command"]], torch.full((4, 2), 7.))
    torch.testing.assert_close(model.context(prediction, "pre.ref_joint_pos"), torch.full((4, 1), 4.))
    torch.testing.assert_close(prediction[:, model.codec.slices["policy"]], state[:, model.codec.slices["policy"]])
    model.context(prediction, "robot.joint_pos").sum().backward()
    assert torch.isfinite(action.grad).all() and action.grad.abs().sum() > 0
    losses, predicted = model.transition_loss(state, action, template, return_prediction=True)
    assert losses["dynamics"].shape == (4,)
    for value in model.validation_errors(predicted, template).values():
        assert value.shape == (4,)
        assert torch.isfinite(value).all()


def test_physical_dynamics_fits_action_acceleration_without_independent_pose_outputs():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(7)
        model, state = make_model()
        state = model.set_context(state, {"action.delay": torch.zeros(4, 1)})
        state = state.repeat(32, 1)
        action = torch.randn(state.shape[0], 1)
        increments = torch.zeros(state.shape[0], model.decimation, model.velocity_dim)
        for prefix, section in model.velocity_slices.items():
            increments[:, :, section.start + 6] = action * .2
        target = model._integrate(state, action, increments)
        model.update_stats(state, action, target)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        before = model.transition_loss(state, action, target)["dynamics"].mean().item()
        for _ in range(100):
            loss = model.transition_loss(state, action, target)["dynamics"].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        after = model.transition_loss(state, action, target)["dynamics"].mean().item()
        assert after < before * .1
        probe = torch.tensor([[.4]], requires_grad=True)
        pred = model(state[:1], probe)
        derivative = torch.autograd.grad(model.context(pred, "robot.joint_vel").sum(), probe)[0]
        assert derivative.item() > .4  # true derivative is four increments * .2
    finally:
        torch.set_num_threads(old_threads)


def _contact_columns(model):
    return [index for name, section in model.context_slices.items() if is_contact_field(name)
            for index in range(section.start, section.stop)]


def test_contact_free_model_drops_contact_inputs_heads_and_losses():
    full, _ = make_model()
    free, state = make_model(contact_model=False)
    contact_inputs = {name for name in full.input_keys if is_contact_field(name)}
    assert contact_inputs
    assert set(full.input_keys) - set(free.input_keys) == contact_inputs
    assert not any(is_contact_field(name) for name in free.input_keys + free.auxiliary_keys)
    # Applied torque stays a learned output; it is part of teacher observations.
    assert free.learned_auxiliary_keys == full.learned_auxiliary_keys == ("robot.applied_torque",)
    assert free.contact_presence_head is None and free.contact_force_head is None
    assert free.contact_body_count == 0
    removed = sum(int(torch.tensor(full.context_shapes[name]).prod()) for name in contact_inputs)
    assert free.trunk[0].in_features == full.trunk[0].in_features - removed
    assert free.schema[0] == "physical_causal_nocontact_v1"
    assert make_model(contact_model=False, object_dynamics="root")[0].schema[0] == "physical_causal_nocontact_root_v1"
    assert schema_contact_model(free.schema) is False
    assert schema_contact_model(full.schema) is True
    assert schema_contact_model(None) is None
    losses = free.transition_loss(state, torch.zeros(4, 1), state)
    assert not {"contact_presence", "contact_force_raw"} & set(losses)
    assert all(torch.isfinite(value).all() for value in losses.values())
    errors = free.validation_errors(state, state)
    assert not any(key.startswith("contact_event") for key in errors)
    assert "contact_force" not in errors and "torque" in errors
    with pytest.raises(ValueError, match="contact_model"):
        make_model(contact_model=1)


def test_contact_free_prediction_cannot_read_contact_and_carries_it_unchanged():
    torch.manual_seed(0)
    model, state = make_model(contact_model=False)
    with torch.no_grad():
        model.velocity_head.weight.normal_()
        model.auxiliary_head.weight.normal_()
    noisy = model.set_context(state, {
        "contact.net_forces_w_history": 50 * torch.randn(4, 3, 2, 3),
        "contact.net_forces_w_substeps": 50 * torch.randn(4, 4, 2, 3),
        "contact.current_air_time": torch.rand(4, 2), "contact.current_contact_time": torch.rand(4, 2),
        "contact.last_air_time": torch.rand(4, 2), "contact.last_contact_time": torch.rand(4, 2),
    })
    contact = torch.tensor(_contact_columns(model))
    keep = torch.ones(state.shape[-1], dtype=torch.bool)
    keep[contact] = False
    action = torch.ones(4, 1)
    clean_prediction, noisy_prediction = model(state, action), model(noisy, action)
    assert not torch.equal(clean_prediction, state)
    torch.testing.assert_close(noisy_prediction[:, keep], clean_prediction[:, keep], rtol=0, atol=0)
    torch.testing.assert_close(noisy_prediction[:, contact], noisy[:, contact], rtol=0, atol=0)
    source = noisy.clone().requires_grad_(True)
    model(source, action)[:, keep].square().sum().backward()
    assert torch.count_nonzero(source.grad[:, contact]) == 0
    assert torch.count_nonzero(source.grad[:, keep]) > 0


def _yaw_quat(angle):
    angle = torch.as_tensor(angle, dtype=torch.float32)
    return torch.stack((torch.cos(angle / 2), torch.zeros_like(angle), torch.zeros_like(angle), torch.sin(angle / 2)), -1)


def test_quaternion_axes_and_heading_frames_match_rotations():
    torch.manual_seed(0)
    q = quat_normalize(torch.randn(64, 4))
    x_axis, y_axis, gravity = quat_axes(q)
    eye = torch.eye(3).expand(64, 3, 3)
    torch.testing.assert_close(x_axis, quat_rotate(q, eye[:, 0]))
    torch.testing.assert_close(y_axis, quat_rotate(q, eye[:, 1]))
    torch.testing.assert_close(gravity, quat_rotate(quat_conjugate(q), -eye[:, 2]))
    yaw = torch.rand(64) * 6.28 - 3.14
    basis = heading_basis(_yaw_quat(yaw))
    torch.testing.assert_close(basis, torch.stack((torch.cos(yaw), torch.sin(yaw)), -1))
    v = torch.randn(64, 3)
    torch.testing.assert_close(heading_to_world(basis, v), quat_rotate(_yaw_quat(yaw), v))
    torch.testing.assert_close(world_to_heading(basis, heading_to_world(basis, v)), v)


def test_heading_frame_replaces_world_root_inputs_with_invariant_features():
    world, _ = make_model()
    heading, state = make_model(frame="heading")
    roots = {f"{prefix}.{field}" for prefix in ("robot", "object")
             for field in ("root_link_pos_w", "root_link_quat_w", "root_com_vel_w")}
    assert set(world.input_keys) - set(heading.input_keys) == roots
    assert heading.canonical_dim == 10 + 15
    assert heading.state_stats.mean.numel() == world.state_stats.mean.numel() - 2 * 13 + 25
    assert heading._features(state).shape[-1] == heading.state_stats.mean.numel()
    assert heading.schema[0] == "physical_causal_contact_substeps_v2_heading"
    assert make_model(frame="heading", contact_model=False, object_dynamics="root")[0].schema[0] == \
        "physical_causal_nocontact_root_v1_heading"
    assert schema_frame(heading.schema) == "heading" and schema_frame(world.schema) == "world"
    assert schema_frame(None) is None
    with pytest.raises(ValueError, match="frame"):
        make_model(frame="body")


def _random_root_state(model, state, generator):
    updates = {}
    for prefix in model.prefixes:
        updates[f"{prefix}.root_link_pos_w"] = torch.randn(4, 3, generator=generator) + torch.tensor([0., 0., 1.])
        updates[f"{prefix}.root_link_quat_w"] = quat_normalize(torch.randn(4, 4, generator=generator))
        updates[f"{prefix}.root_com_vel_w"] = torch.randn(4, 6, generator=generator)
        updates[f"{prefix}.joint_pos"] = torch.randn(4, 1, generator=generator)
        updates[f"{prefix}.joint_vel"] = torch.randn(4, 1, generator=generator)
    return model.set_context(state, updates)


def _move_world(model, state, yaw, shift):
    rotation = _yaw_quat(torch.full((4,), yaw))
    updates = {}
    for prefix in model.prefixes:
        get = lambda name: model.context(state, f"{prefix}.{name}")
        updates[f"{prefix}.root_link_pos_w"] = quat_rotate(rotation, get("root_link_pos_w")) + shift
        updates[f"{prefix}.root_link_quat_w"] = quat_multiply(rotation, get("root_link_quat_w"))
        velocity = get("root_com_vel_w")
        updates[f"{prefix}.root_com_vel_w"] = torch.cat((quat_rotate(rotation, velocity[..., :3]),
                                                         quat_rotate(rotation, velocity[..., 3:])), -1)
        body_rotation = rotation[:, None]
        updates[f"{prefix}.body_link_pos_w"] = quat_rotate(body_rotation, get("body_link_pos_w")) + shift
        updates[f"{prefix}.body_link_quat_w"] = quat_multiply(body_rotation.expand_as(get("body_link_quat_w")),
                                                              get("body_link_quat_w"))
        body_velocity = get("body_com_vel_w")
        updates[f"{prefix}.body_com_vel_w"] = torch.cat((quat_rotate(body_rotation, body_velocity[..., :3]),
                                                         quat_rotate(body_rotation, body_velocity[..., 3:])), -1)
    return model.set_context(state, updates)


@pytest.mark.parametrize("frame", ["heading", "world"])
def test_heading_frame_dynamics_are_equivariant_to_yaw_and_horizontal_translation(frame):
    generator = torch.Generator().manual_seed(3)
    model, state = make_model(frame=frame)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=.2, generator=generator)
    state = _random_root_state(model, state, generator)
    action = torch.randn(4, 1, generator=generator)
    yaw, shift = 1.3, torch.tensor([2.5, -1.7, 0.])
    moved = _move_world(model, state, yaw, shift)
    expected = _move_world(model, model(state, action), yaw, shift)
    actual = model(moved, action)
    keys = [f"{prefix}.{field}" for prefix in model.prefixes
            for field in ("root_link_pos_w", "root_com_vel_w", "body_link_pos_w", "body_com_vel_w", "joint_pos",
                          "joint_vel")]
    close = all(torch.allclose(model.context(actual, key), model.context(expected, key), atol=2e-4, rtol=1e-4)
                for key in keys)
    rotation_close = all(quaternion_error(model.context(actual, f"{prefix}.root_link_quat_w"),
                                          model.context(expected, f"{prefix}.root_link_quat_w")).max() < 1e-4
                         for prefix in model.prefixes)
    if frame == "heading":
        torch.testing.assert_close(model._features(moved), model._features(state), atol=1e-5, rtol=1e-5)
        assert close and rotation_close
    else:
        # The raw world-frame network has no such symmetry: the test is meaningful.
        assert not close


def test_heading_frame_has_finite_gradients_when_the_robot_axis_is_vertical():
    model, state = make_model(frame="heading")
    pitched = torch.tensor([.5 ** .5, 0., .5 ** .5, 0.]).expand(4, 4)  # body x axis points straight down
    state = model.set_context(state, {"robot.root_link_quat_w": pitched, "object.root_link_quat_w": pitched})
    source = state.clone().requires_grad_(True)
    prediction = model(source, torch.ones(4, 1))
    prediction[..., :].square().sum().backward()
    assert torch.isfinite(prediction).all() and torch.isfinite(source.grad).all()


def test_object_body_features_are_robot_body_positions_in_each_object_frame():
    generator = torch.Generator().manual_seed(5)
    plain, _ = make_model(frame="heading")
    model, state = make_model(frame="heading", object_body_features=True)
    state = _random_root_state(model, state, generator)
    state = model.set_context(state, {"robot.body_link_pos_w": torch.randn(4, 2, 3, generator=generator)})
    assert model.object_body_dim == 2 * 3
    assert model.state_stats.mean.numel() == plain.state_stats.mean.numel() + 6
    assert model.schema[0] == "physical_causal_contact_substeps_v2_heading_objectbodies"
    assert schema_object_body_features(model.schema) is True
    assert schema_object_body_features(plain.schema) is False
    assert schema_frame(model.schema) == "heading"
    bodies = model.context(state, "robot.body_link_pos_w")
    position = model.context(state, "object.root_link_pos_w")
    quat = quat_normalize(model.context(state, "object.root_link_quat_w"))
    expected = quat_rotate(quat_conjugate(quat)[:, None], bodies - position[:, None]).flatten(-2)
    torch.testing.assert_close(model._object_body_positions(state), expected)
    features = model._features(state)
    torch.testing.assert_close(features[:, model.canonical_dim:model.canonical_dim + 6], expected)
    # A body placed on the object's origin reads zero, whatever the object pose.
    on_object = model.set_context(state, {"robot.body_link_pos_w": position[:, None].expand(4, 2, 3)})
    torch.testing.assert_close(model._object_body_positions(on_object), torch.zeros(4, 6), atol=1e-6, rtol=0)
    with pytest.raises(ValueError, match="object_body_features"):
        make_model(object_body_features=1)


def test_object_body_features_reach_the_network_and_keep_frame_equivariance():
    generator = torch.Generator().manual_seed(7)
    model, state = make_model(frame="heading", object_body_features=True)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=.2, generator=generator)
    state = _random_root_state(model, state, generator)
    state = model.set_context(state, {"robot.body_link_pos_w": torch.randn(4, 2, 3, generator=generator)})
    action = torch.randn(4, 1, generator=generator)
    shifted_bodies = model.set_context(state, {"robot.body_link_pos_w": model.context(state, "robot.body_link_pos_w") + .3})
    assert not torch.allclose(model(shifted_bodies, action), model(state, action))
    yaw, shift = -2.1, torch.tensor([-3., 4., 0.])
    moved = _move_world(model, state, yaw, shift)
    torch.testing.assert_close(model._features(moved), model._features(state), atol=2e-5, rtol=1e-5)
    expected = _move_world(model, model(state, action), yaw, shift)
    actual = model(moved, action)
    for prefix in model.prefixes:
        for field in ("root_link_pos_w", "root_com_vel_w", "body_link_pos_w", "joint_pos"):
            torch.testing.assert_close(model.context(actual, f"{prefix}.{field}"),
                                       model.context(expected, f"{prefix}.{field}"), atol=2e-4, rtol=1e-4)
