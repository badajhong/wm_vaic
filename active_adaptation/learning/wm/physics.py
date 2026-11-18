"""Compact robot/object dynamics with differentiable articulated kinematics.

This module predicts generalized velocities, not independent policy observations.
Root poses and joint positions are integrated at the simulator's physics rate;
body poses and velocities are reconstructed from the actual USD joint frames.
Contact forces/timing remain supervised auxiliaries: predicting those quantities
does not make their derivatives an exact model of discontinuous contact physics.
No reward or termination predictor is implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite, prod
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .core import RunningMoments


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_multiply(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.cross(av, bv, dim=-1)), -1)


def quat_rotate(q, v):
    qv, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = 2 * torch.cross(qv, v, dim=-1)
    return v + q[..., :1] * uv + torch.cross(qv, uv, dim=-1)


def quat_exp(rotation_vector):
    """Quaternion exponential for an axis-angle vector, smooth at zero."""
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    # torch.sinc includes the removable singularity at zero, unlike sin(x)/x.
    xyz = rotation_vector * (0.5 * torch.sinc(angle / (2 * torch.pi)))
    return torch.cat((torch.cos(angle * 0.5), xyz), -1)


def quat_normalize(q):
    return F.normalize(q, dim=-1, eps=1e-12)


def quat_axes(q):
    """World-frame body x and y axes and the body-frame gravity of unit ``q``.

    Closed forms of the rotation matrix columns/row avoid constant tensors,
    which a captured CUDA graph could not copy from the host.
    """
    w, x, y, z = q.unbind(-1)
    x_axis = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)), -1)
    y_axis = torch.stack((2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x)), -1)
    # R^T @ (0, 0, -1): the negated third row of R.
    gravity = -torch.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)
    return x_axis, y_axis, gravity


def heading_basis(q, eps=1e-6):
    """(cos, sin) of the yaw of unit ``q``: its x axis projected on the ground.

    Finite, with finite derivatives, even when that axis is vertical.
    """
    planar = quat_axes(q)[0][..., :2]
    return planar / torch.linalg.vector_norm(planar, dim=-1, keepdim=True).clamp_min(eps)


def world_to_heading(basis, v):
    c, s = basis[..., :1], basis[..., 1:]
    return torch.cat((c * v[..., :1] + s * v[..., 1:2], c * v[..., 1:2] - s * v[..., :1], v[..., 2:]), -1)


def heading_to_world(basis, v):
    c, s = basis[..., :1], basis[..., 1:]
    return torch.cat((c * v[..., :1] - s * v[..., 1:2], s * v[..., :1] + c * v[..., 1:2], v[..., 2:]), -1)


def quaternion_error(a, b):
    relative = quat_multiply(quat_normalize(a), quat_conjugate(quat_normalize(b)))
    return 2 * torch.atan2(torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., 0].abs())


def contact_sensor_transition(forces, current_air_time, current_contact_time,
                              last_air_time, last_contact_time, timestamp,
                              timestamp_last_update, physics_dt, force_threshold=1.,
                              history_length=3, previous_force_history=None):
    """Reproduce IsaacLab's contact sensor after chronological physics samples.

    ``forces`` is [*batch, substeps, bodies, 3]; timers are [*batch, bodies]
    and clocks are [*batch]. Float32 clock additions/subtractions intentionally
    match SensorBase and ContactSensor, including first-contact boundary roundoff.
    Contact switches use the actual hard force threshold, with no surrogate
    derivative. Forces themselves retain their ordinary continuous gradients.
    """
    if forces.ndim < 3 or forces.shape[-1] != 3 or forces.shape[-3] < 1:
        raise ValueError("Contact forces must have shape [*batch, substeps, bodies, 3]")
    if timestamp.shape != forces.shape[:-3] or timestamp_last_update.shape != timestamp.shape:
        raise ValueError("Contact sensor clocks must match the force batch shape")
    if history_length < 1:
        raise ValueError("Contact sensor transition requires a positive history length")
    for index in range(forces.shape[-3]):
        timestamp = timestamp + physics_dt
        elapsed = (timestamp - timestamp_last_update).unsqueeze(-1)
        contact = forces[..., index, :, :].norm(dim=-1) > force_threshold
        first_contact = (current_air_time > 0) & contact
        first_detached = (current_contact_time > 0) & ~contact
        last_air_time = torch.where(first_contact, current_air_time + elapsed, last_air_time)
        current_air_time = torch.where(~contact, current_air_time + elapsed, 0.)
        last_contact_time = torch.where(first_detached, current_contact_time + elapsed, last_contact_time)
        current_contact_time = torch.where(contact, current_contact_time + elapsed, 0.)
        timestamp_last_update = timestamp
    history = forces.flip(-3)
    if history.shape[-3] < history_length:
        if previous_force_history is None or previous_force_history.shape[-3] < history_length:
            raise ValueError("Previous force history is required when fewer substeps than history slots")
        history = torch.cat((history, previous_force_history), dim=-3)
    return {"net_forces_w_substeps": forces, "net_forces_w_history": history[..., :history_length, :, :],
            "current_air_time": current_air_time, "current_contact_time": current_contact_time,
            "last_air_time": last_air_time, "last_contact_time": last_contact_time,
            "timestamp": timestamp, "timestamp_last_update": timestamp_last_update}


CONTACT_FORCE_FIELDS = ("command.eef_contact_forces_w", "command.eef2_contact_forces_w")


def is_contact_field(name):
    """Contact sensor bookkeeping and end-effector contact forces."""
    return name.startswith("contact.") or name in CONTACT_FORCE_FIELDS


def schema_contact_model(schema):
    """Whether a saved physical-model schema predicts contacts (None if unknown)."""
    if not schema or not isinstance(schema[0], str):
        return None
    return "nocontact" not in schema[0]


def schema_frame(schema):
    """Network frame of a saved physical-model schema (None if unknown)."""
    if not schema or not isinstance(schema[0], str):
        return None
    return "heading" if "_heading" in schema[0] else "world"


def schema_object_body_features(schema):
    """Whether a saved physical-model schema uses object-frame body positions."""
    if not schema or not isinstance(schema[0], str):
        return None
    return "_objectbodies" in schema[0]


HEADING_ROOT_FIELDS = ("root_link_pos_w", "root_link_quat_w", "root_com_vel_w")


def contact_force_support(raw_forces, active, force_threshold=1.):
    """Project force magnitude onto the selected hard sensor contact support.

    The raw vector is supervised separately so a wrong discrete decision cannot
    suppress force-regression learning. No straight-through contact estimator is
    used; this projection is differentiable within each fixed support region.
    """
    magnitude = raw_forces.norm(dim=-1, keepdim=True)
    direction = raw_forces / magnitude.clamp_min(1e-8)
    vertical = torch.zeros_like(raw_forces)
    vertical[..., 2] = 1.
    direction = torch.where(magnitude > 1e-8, direction, vertical)
    # A representable margin makes norm(force)>threshold reliable in float32.
    supported_magnitude = torch.where(active.unsqueeze(-1),
                                      magnitude.clamp_min(force_threshold + 1e-5),
                                      magnitude.clamp_max(max(0., force_threshold - 1e-5)))
    return direction * supported_magnitude


def fixed_action_contract(action_manager, asset):
    """Comparable actuator constants, independent of environment count/DR draws."""
    def canonical(value):
        if isinstance(value, torch.Tensor):
            return canonical(value.detach().cpu().tolist())
        if isinstance(value, Mapping):
            return tuple((str(key), canonical(item)) for key, item in sorted(value.items()))
        if isinstance(value, (tuple, list)):
            return tuple(canonical(item) for item in value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(f"Unsupported fixed actuator configuration value: {type(value).__name__}")

    required = ("joint_ids", "joint_names", "action_scaling", "default_joint_pos")
    missing = [name for name in required if not hasattr(action_manager, name)]
    if missing:
        raise ValueError(f"Physical model requires fixed JointPosition contract fields: {missing}")
    default = action_manager.default_joint_pos.detach()
    if default.ndim != 2 or default.shape[0] < 1:
        raise ValueError("JointPosition default_joint_pos must be [num_envs, num_joints]")
    if not torch.equal(default, default[:1].expand_as(default)):
        raise ValueError("Per-environment default_joint_pos is unsupported as a fixed actuator contract; "
                         "varying default poses must be represented explicitly in model context")
    actuator_configs = getattr(getattr(asset, "cfg", None), "actuators", {})
    limits = tuple((str(name), tuple((field, canonical(getattr(config, field, None))) for field in (
        "joint_names_expr", "effort_limit_sim", "effort_limit", "velocity_limit_sim", "velocity_limit",
    ))) for name, config in sorted(actuator_configs.items()))
    return (
        ("type", f"{type(action_manager).__module__}.{type(action_manager).__qualname__}"),
        ("joint_ids", canonical(action_manager.joint_ids)),
        ("joint_names", canonical(action_manager.joint_names)),
        ("action_scaling", canonical(action_manager.action_scaling)),
        ("default_joint_pos", canonical(default[0])),
        ("delay_range", (canonical(getattr(action_manager, "min_delay", None)),
                         canonical(getattr(action_manager, "max_delay", None)))),
        ("alpha_range", canonical(getattr(action_manager, "alpha_range", None))),
        ("configured_actuator_limits", limits),
    )


class DifferentiableKinematics(nn.Module):
    """USD joint-frame forward kinematics with world-frame spatial velocities.

    ``joints`` entries have name/parent/child/type (fixed, revolute, prismatic),
    local_pos0/local_pos1, local_quat0/local_quat1 (wxyz), and an axis vector.
    Intermediate fixed links may be absent from ``body_names``; the complete
    tree is evaluated and outputs follow the simulator's exact body ordering.
    Uniform per-environment scaling is supported. Nonuniform articulated scaling
    is rejected at construction instead of silently using incorrect kinematics.
    """

    def __init__(self, body_names, joint_names, joints, root_name=None):
        super().__init__()
        self.body_names = tuple(body_names)
        self.joint_names = tuple(joint_names)
        self.root_name = root_name or self.body_names[0]
        pending = list(joints)
        names = [self.root_name]
        ordered = []
        while pending:
            ready = [j for j in pending if j["parent"] in names]
            if not ready:
                missing = [j["name"] for j in pending]
                raise ValueError(f"Disconnected or cyclic USD joint tree: {missing}")
            for joint in ready:
                if joint["child"] in names:
                    raise ValueError(f"Multiple parents for USD body {joint['child']}")
                names.append(joint["child"])
                ordered.append(joint)
                pending.remove(joint)
        missing = set(self.body_names) - set(names)
        if missing:
            raise ValueError(f"Simulator bodies missing from USD tree: {sorted(missing)}")
        used = {j["name"] for j in ordered if j["type"] != "fixed"}
        if used != set(self.joint_names):
            raise ValueError(f"USD/simulator moving joints differ: {used ^ set(self.joint_names)}")
        self.parents = tuple(names.index(j["parent"]) for j in ordered)
        self.all_body_names = tuple(names)
        self.joint_indices = tuple(-1 if j["type"] == "fixed" else self.joint_names.index(j["name"]) for j in ordered)
        self.joint_types = tuple(j["type"] for j in ordered)
        self.output_indices = tuple(names.index(name) for name in self.body_names)
        self.schema = (self.body_names, self.joint_names, tuple(
            (j["name"], j["parent"], j["child"], j["type"]) for j in ordered
        ))
        for key, default in (("local_pos0", (0., 0., 0.)), ("local_pos1", (0., 0., 0.)),
                             ("local_quat0", (1., 0., 0., 0.)), ("local_quat1", (1., 0., 0., 0.)),
                             ("axis", (0., 0., 1.))):
            self.register_buffer(key, torch.tensor([j.get(key, default) for j in ordered], dtype=torch.float32).reshape(len(ordered), len(default)))
        self._build_level_plan()

    def _build_level_plan(self):
        """Batch independent links while keeping each ancestor calculation ordered.

        The constructor orders joints breadth first. A level therefore reads
        only the preceding level, and sibling links can share tensor kernels.
        These indices are derived from topology, not checkpoint state.
        """
        depths = [0]
        for parent in self.parents:
            depths.append(depths[parent] + 1)
        levels, previous = [], [0]
        for depth in range(1, max(depths) + 1):
            bodies = [i for i, value in enumerate(depths) if value == depth]
            joints = [i - 1 for i in bodies]
            if joints != list(range(joints[0], joints[-1] + 1)):
                raise ValueError("Kinematic joints must be ordered breadth first")
            parent_name, joint_name = f"_level_parents_{depth}", f"_level_joints_{depth}"
            self.register_buffer(parent_name, torch.tensor(
                [previous.index(self.parents[i]) for i in joints], dtype=torch.long,
                device=self.local_pos0.device), persistent=False)
            self.register_buffer(joint_name, torch.tensor(
                [max(0, self.joint_indices[i]) for i in joints], dtype=torch.long,
                device=self.local_pos0.device), persistent=False)
            kinds = tuple(self.joint_types[i] for i in joints)
            for kind in ("revolute", "prismatic"):
                self.register_buffer(f"_level_{kind}_{depth}", torch.tensor(
                    [value == kind for value in kinds], device=self.local_pos0.device).unsqueeze(-1),
                    persistent=False)
            levels.append((depth, joints[0], joints[-1] + 1, tuple(sorted(set(kinds)))))
            previous = bodies
        self._levels = tuple(levels)
        self.register_buffer("_output_index", torch.tensor(self.output_indices, dtype=torch.long,
                             device=self.local_pos0.device), persistent=False)

    def __setstate__(self, state):
        super().__setstate__(state)
        # Offline evidence contains trusted pickled modules predating this
        # execution plan. Their topology and persistent weights are unchanged.
        if not hasattr(self, "_levels"):
            self._build_level_plan()

    @classmethod
    def from_usd(cls, usd_path, body_names, joint_names, root_name=None):
        # pxr is supplied by Isaac Sim after AppLauncher initialization. Keeping
        # this import local allows all tensor utilities/tests to run on CPU.
        try:
            from pxr import Usd, UsdPhysics
        except ImportError as exc:
            raise RuntimeError("USD kinematic extraction requires initialized Isaac USD bindings (pxr)") from exc
        stage = Usd.Stage.Open(str(usd_path))
        if stage is None:
            raise ValueError(f"Cannot open USD asset: {usd_path}")
        joints = []
        def quaternion(value):
            return (float(value.GetReal()), *(float(x) for x in value.GetImaginary()))
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            if joint.GetJointEnabledAttr().Get() is False:
                continue
            parent, child = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
            if not parent or not child:
                # A world attachment is not a relative link joint. This model
                # still requires its modeled root to be a floating body.
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                kind = "revolute"
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                kind = "prismatic"
            elif prim.IsA(UsdPhysics.FixedJoint):
                kind = "fixed"
            else:
                raise ValueError(f"Unsupported USD joint type {prim.GetTypeName()} at {prim.GetPath()}")
            axis_name = prim.GetAttribute("physics:axis").Get() or "Z"
            joints.append(dict(
                name=prim.GetName(), parent=parent[0].name, child=child[0].name, type=kind,
                local_pos0=tuple(joint.GetLocalPos0Attr().Get()), local_pos1=tuple(joint.GetLocalPos1Attr().Get()),
                local_quat0=quaternion(joint.GetLocalRot0Attr().Get()), local_quat1=quaternion(joint.GetLocalRot1Attr().Get()),
                axis={"X": (1., 0., 0.), "Y": (0., 1., 0.), "Z": (0., 0., 1.)}[axis_name],
            ))
        return cls(body_names, joint_names, joints, root_name=root_name)

    def forward(self, root_pos, root_quat, joint_pos, root_link_velocity=None,
                joint_vel=None, scale=None, com_offsets=None):
        batch = root_pos.shape[:-1]
        if root_link_velocity is None:
            root_link_velocity = root_pos.new_zeros(*batch, 6)
        if joint_vel is None:
            joint_vel = torch.zeros_like(joint_pos)
        if scale is None:
            scale = root_pos.new_ones(*batch, 3)
        positions, rotations = [root_pos.unsqueeze(-2)], [quat_normalize(root_quat).unsqueeze(-2)]
        linear, angular = [root_link_velocity[..., None, :3]], [root_link_velocity[..., None, 3:]]
        scale = scale.unsqueeze(-2)
        for depth, start, stop, kinds in self._levels:
            parents = getattr(self, f"_level_parents_{depth}")
            p, q, v, w = (values[-1].index_select(-2, parents)
                          for values in (positions, rotations, linear, angular))
            local_axis = self.axis[start:stop]
            r0 = quat_rotate(q, self.local_pos0[start:stop] * scale)
            joint_origin = p + r0
            joint_rotation = quat_multiply(q, self.local_quat0[start:stop].expand_as(q))
            v_joint = v + torch.cross(w, r0, dim=-1)
            inverse_frame = quat_conjugate(self.local_quat1[start:stop]).expand_as(q)
            if kinds == ("fixed",):
                q_child = quat_multiply(joint_rotation, inverse_frame)
                w_child = w
            else:
                joint_indices = getattr(self, f"_level_joints_{depth}")
                angles = joint_pos.index_select(-1, joint_indices).unsqueeze(-1)
                velocities = joint_vel.index_select(-1, joint_indices).unsqueeze(-1)
                axis = quat_rotate(joint_rotation, local_axis.expand_as(p))
                q_child = (None if kinds == ("revolute",)
                           else quat_multiply(joint_rotation, inverse_frame))
                w_child = w
                if "revolute" in kinds:
                    relative = quat_exp(local_axis * angles)
                    revolute_quat = quat_multiply(quat_multiply(joint_rotation, relative), inverse_frame)
                    revolute_velocity = w + axis * velocities
                    if kinds == ("revolute",):
                        q_child, w_child = revolute_quat, revolute_velocity
                    else:
                        rotating = getattr(self, f"_level_revolute_{depth}")
                        q_child = torch.where(rotating, revolute_quat, q_child)
                        w_child = torch.where(rotating, revolute_velocity, w_child)
                if "prismatic" in kinds:
                    sliding = getattr(self, f"_level_prismatic_{depth}")
                    offset = torch.where(sliding, axis * angles, 0.)
                    joint_origin = joint_origin + offset
                    v_joint = v_joint + torch.cross(w, offset, dim=-1) + torch.where(sliding, axis * velocities, 0.)
            q_child = quat_normalize(q_child)
            r1 = quat_rotate(q_child, self.local_pos1[start:stop] * scale)
            positions.append(joint_origin - r1)
            rotations.append(q_child)
            angular.append(w_child)
            linear.append(v_joint - torch.cross(w_child, r1, dim=-1))
        pos, quat, lin, ang = (torch.cat(values, -2).index_select(-2, self._output_index)
                              for values in (positions, rotations, linear, angular))
        if com_offsets is not None:
            lin = lin + torch.cross(ang, quat_rotate(quat, com_offsets), dim=-1)
        return pos, quat, torch.cat((lin, ang), -1)


class PhysicalWorldModel(nn.Module):
    """Predict compact generalized dynamics and reconstruct physical context.

    The outer learner calls its observation adapter on this packed prediction,
    then calculates the original reward and termination formulas. Reference,
    geometry, histories and actor/critic observation vectors are not MLP outputs.
    ``kinematics`` is an optional prefix->DifferentiableKinematics mapping used
    by CPU fixtures; production extracts and checks each tree from the USD asset.
    ``training_loss_scales`` optionally overrides optimization units for root
    positions/rotations/velocities across assets, joint positions/velocities
    (robot substep positions follow ``joint_pos``). ``object_root_*`` keys
    override the root units of the ``object`` asset only, and ``object_joint_*``
    keys its articulated joint units. Validation units and contact losses stay fixed.
    ``object_dynamics='root'`` omits object joints and unused object torques from
    learned state and targets. Its wheel links are fixed-frame placeholders;
    this is a reduced physical model, not an exact Markov closure of wheels.
    ``contact_model=False`` (SuperMimic-style) removes contact sensor fields and
    end-effector contact forces from the network inputs, the learned outputs and
    the loss: the next state is predicted from generalized state, actuator state
    and domain parameters only. Contact fields are then carried unchanged from
    the rollout's start state, so only equations that do not read contact force
    or contact timers may consume predictions.
    ``frame='heading'`` makes the network invariant to horizontal translation
    and to rotation about gravity (as SuperMimic and NeRD). The robot root
    enters as height, body-frame gravity and body-frame twist. Other assets
    enter as their pose relative to the robot heading frame, and as their twist
    in their own heading frame. Predicted root increments are expressed in each
    asset's heading frame of the step's start state and rotated to the world
    before the unchanged substep integration and FK. Contact-force inputs, if
    any, stay in the world frame.
    ``object_body_features=True`` adds, for every non-robot asset, the position
    of every robot body link in that asset's root frame. These are exact FK
    quantities of the input state, not predicted outputs; they tell the network
    which body is near which part of the object, for any object shape.
    """

    def __init__(self, codec, transform, env, hidden_dims=(512, 512, 512), device="cpu", *,
                 kinematics=None, training_loss_scales=None, object_dynamics="articulated",
                 contact_model=True, frame="world", object_body_features=False):
        super().__init__()
        if not isinstance(object_body_features, bool):
            raise ValueError("object_body_features must be a bool")
        self.object_body_features = object_body_features
        if object_dynamics not in ("articulated", "root"):
            raise ValueError("object_dynamics must be 'articulated' or 'root'")
        if not isinstance(contact_model, bool):
            raise ValueError("contact_model must be a bool")
        if frame not in ("world", "heading"):
            raise ValueError("frame must be 'world' or 'heading'")
        self.object_dynamics = object_dynamics
        self.contact_model = contact_model
        self.frame = frame
        # Optimization units are intentionally separate from the fixed physical
        # validation units. These optional Python scalars add no checkpoint keys.
        # Generic keys apply to every asset (robot and object roots, robot
        # joints). object_root_* keys override only the object's root terms,
        # and object_joint_* only its articulated joints.
        allowed_scales = {"root_pos", "root_rotation", "root_velocity", "joint_pos", "joint_velocity",
                          "object_root_pos", "object_root_rotation", "object_root_velocity",
                          "object_joint_pos", "object_joint_velocity"}
        if training_loss_scales is None:
            training_loss_scales = {}
        if not isinstance(training_loss_scales, Mapping):
            raise ValueError("training_loss_scales must be a mapping of physical loss names to positive scales")
        unknown_scales = set(training_loss_scales) - allowed_scales
        if unknown_scales:
            raise ValueError(f"Unknown training_loss_scales keys: {sorted(unknown_scales, key=str)}")
        self.training_loss_scales = {}
        for name, value in training_loss_scales.items():
            try:
                scale = float(value)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(f"training_loss_scales[{name!r}] must be finite and positive") from error
            if isinstance(value, bool) or not isfinite(scale) or scale <= 0:
                raise ValueError(f"training_loss_scales[{name!r}] must be finite and positive")
            self.training_loss_scales[name] = scale
        self.codec = codec
        self.context_shapes = dict(transform.context_schema)
        context_start = codec.slices["wm_context"].start
        self.context_slices = {
            name: slice(context_start + section.start, context_start + section.stop)
            for name, section in transform.context_slices.items()
        }
        self.context_order = tuple(sorted(self.context_slices, key=lambda name: self.context_slices[name].start))
        self.context_chunk_widths, self.context_chunk_indices = [], {}
        cursor = 0
        for name in self.context_order:
            section = self.context_slices[name]
            if section.start > cursor:
                self.context_chunk_widths.append(section.start - cursor)
            self.context_chunk_indices[name] = len(self.context_chunk_widths)
            self.context_chunk_widths.append(section.stop - section.start)
            cursor = section.stop
        if cursor < codec.state_dim:
            self.context_chunk_widths.append(codec.state_dim - cursor)
        self.context_chunk_widths = tuple(self.context_chunk_widths)
        self.physics_dt = float(env.physics_dt)
        self.decimation = int(env.decimation)
        if self.decimation < 1 or self.physics_dt <= 0:
            raise ValueError("Physical model requires positive physics_dt and decimation")
        self.step_dt = self.physics_dt * self.decimation
        self.register_buffer("action_prefix_mask", torch.tril(torch.ones(self.decimation, self.decimation)).unsqueeze(-1))
        self.register_buffer("action_substep_onehot", torch.eye(self.decimation))
        self.action_dim = int(env.action_manager.action_dim)
        self.prefixes = tuple(name for name in ("robot", "object", "object2")
                              if f"{name}.root_link_pos_w" in self.context_slices)
        if "robot" not in self.prefixes or "object" not in self.prefixes:
            raise ValueError("Physical dynamics requires both robot and object context")
        self.kinematics = nn.ModuleDict()
        self.velocity_slices = {}
        offset = 0
        for prefix in self.prefixes:
            for field in ("root_link_pos_w", "root_link_quat_w", "root_com_vel_w", "body_link_pos_w", "body_link_quat_w", "body_com_vel_w"):
                if f"{prefix}.{field}" not in self.context_slices:
                    raise ValueError(f"Missing physical field: {prefix}.{field}")
            num_joints = prod(self.context_shapes.get(f"{prefix}.joint_pos", (0,)))
            predicted_joints = 0 if prefix == "object" and object_dynamics == "root" else num_joints
            self.velocity_slices[prefix] = slice(offset, offset + 6 + predicted_joints)
            offset += 6 + predicted_joints
            asset = getattr(env.command_manager, "asset" if prefix == "robot" else prefix)
            if kinematics is not None:
                tree = kinematics[prefix]
            elif num_joints:
                tree = DifferentiableKinematics.from_usd(
                    asset.cfg.spawn.usd_path, asset.body_names, asset.joint_names,
                )
            else:
                if len(asset.body_names) != 1:
                    raise ValueError("Rigid object with multiple bodies requires an explicit kinematic tree")
                tree = DifferentiableKinematics(asset.body_names, (), ())
            self.kinematics[prefix] = tree
            if prefix == "object" and object_dynamics == "root":
                # Root-only dynamics can reconstruct deck contact geometry only
                # when that tracked body is the floating root. Wheel links use
                # fixed zero-joint placeholders, never factual wheel state.
                body_id = getattr(env.command_manager, "object_body_id_asset", 0)
                if tree.body_names[0] != tree.root_name or body_id != 0:
                    raise ValueError("Root-only object dynamics requires the tracked contact body "
                                     "to be the first/root object body")
            if kinematics is None:
                self._verify_kinematics(prefix, asset, tree.to(device))
        self.velocity_dim = offset
        # Physical inputs include generalized state and actual actuator state,
        # contact state and domain parameters. They exclude thousands of derived
        # observation/reference/history channels that the adapter reconstructs.
        physical_suffixes = ("root_link_pos_w", "root_link_quat_w", "root_com_vel_w", "joint_pos", "joint_vel")
        self.input_keys = tuple(name for name in self.context_shapes if (
            any(name == f"{prefix}.{suffix}" for prefix in self.prefixes for suffix in physical_suffixes)
            or (name.startswith("action.") and name != "action.action_buf") or name.startswith("contact.")
            or ".physics." in name or name.endswith(("kinematic_scale", "joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff", "_custom_friction", "_custom_damping"))
            or name in ("command.eef_contact_forces_w", "command.eef2_contact_forces_w")
        ) and name not in ("contact.timestamp", "contact.timestamp_last_update"))
        if object_dynamics == "root":
            self.input_keys = tuple(name for name in self.input_keys
                                    if name not in ("object.joint_pos", "object.joint_vel"))
        if not contact_model:
            self.input_keys = tuple(name for name in self.input_keys if not is_contact_field(name))
        self.canonical_root_keys = ()
        self.canonical_dim = 0
        if frame == "heading":
            # World-frame root fields are replaced by their invariant features.
            self.canonical_root_keys = tuple(f"{prefix}.{field}" for prefix in self.prefixes
                                             for field in HEADING_ROOT_FIELDS)
            self.input_keys = tuple(name for name in self.input_keys if name not in self.canonical_root_keys)
            # Robot: height, gravity, linear and angular twist. Other assets:
            # relative position, x/y axes, linear and angular twist.
            self.canonical_dim = 10 + 15 * (len(self.prefixes) - 1)
        self.object_body_dim = 0
        if object_body_features:
            if len(self.prefixes) < 2:
                raise ValueError("object_body_features requires at least one non-robot asset")
            self.object_body_dim = (len(self.prefixes) - 1) * prod(self.context_shapes["robot.body_link_pos_w"])
        self.contact_timer_keys = ("contact.current_air_time", "contact.current_contact_time")
        self.contact_body_count = 0
        self.contact_history_length = 0
        self.contact_force_threshold = 1.
        if contact_model and any(key in self.context_slices for key in self.contact_timer_keys):
            required_contact = (*self.contact_timer_keys, "contact.last_air_time", "contact.last_contact_time",
                                "contact.net_forces_w_substeps", "contact.net_forces_w_history",
                                "contact.timestamp", "contact.timestamp_last_update")
            missing = [key for key in required_contact if key not in self.context_slices]
            if missing:
                raise ValueError(f"Exact substep contact dynamics requires context fields: {missing}")
            self.contact_body_count = prod(self.context_shapes[self.contact_timer_keys[0]])
            if self.context_shapes["contact.net_forces_w_substeps"] != (self.decimation, self.contact_body_count, 3):
                raise ValueError("Contact substep force layout must be [decimation, contact bodies, 3]")
            self.contact_history_length = self.context_shapes["contact.net_forces_w_history"][0]
            sensor = getattr(getattr(env, "scene", None), "sensors", {}).get("contact_forces")
            if sensor is not None:
                self.contact_force_threshold = float(sensor.cfg.force_threshold)
        if contact_model:
            schema_name = ("physical_causal_contact_substeps_root_v1" if object_dynamics == "root"
                           else "physical_causal_contact_substeps_v2")
        else:
            schema_name = ("physical_causal_nocontact_root_v1" if object_dynamics == "root"
                           else "physical_causal_nocontact_v1")
        if frame == "heading":
            schema_name += "_heading"
        if object_body_features:
            schema_name += "_objectbodies"
        self.schema = (
            schema_name, self.input_keys,
            self.action_dim, self.decimation, self.physics_dt,
            (self.contact_body_count, self.contact_history_length, self.contact_force_threshold),
            fixed_action_contract(env.action_manager, env.command_manager.asset),
            tuple((prefix, tree.schema) for prefix, tree in self.kinematics.items()),
        )
        self.auxiliary_keys = tuple(name for name in self.context_shapes if (
            name.startswith("contact.") or name in ("command.eef_contact_forces_w", "command.eef2_contact_forces_w")
            or name.endswith((".applied_torque", ".custom_torques"))
        ) and name not in getattr(transform, "exogenous_context_slices", {})
            and name not in ("contact.timestamp", "contact.timestamp_last_update"))
        if object_dynamics == "root":
            self.auxiliary_keys = tuple(name for name in self.auxiliary_keys
                                        if name not in ("object.applied_torque", "object.custom_torques"))
        if not contact_model:
            # Applied torques stay learned outputs: the teacher observes them.
            self.auxiliary_keys = tuple(name for name in self.auxiliary_keys if not is_contact_field(name))
        self.auxiliary_slices = {}
        width = 0
        for name in self.auxiliary_keys:
            size = prod(self.context_shapes[name])
            self.auxiliary_slices[name] = slice(width, width + size)
            width += size
        self.auxiliary_dim = width
        self.learned_auxiliary_keys = tuple(name for name in self.auxiliary_keys if not name.startswith("contact."))
        self.learned_auxiliary_slices = {}
        learned_width, learned_indices = 0, []
        for name in self.learned_auxiliary_keys:
            section = self.auxiliary_slices[name]
            size = section.stop - section.start
            self.learned_auxiliary_slices[name] = slice(learned_width, learned_width + size)
            learned_indices.extend(range(section.start, section.stop))
            learned_width += size
        self.register_buffer("learned_auxiliary_indices", torch.tensor(learned_indices, dtype=torch.long))
        for label, keys in (("input", self.input_keys), ("auxiliary", self.auxiliary_keys)):
            indices = [i for key in keys for i in range(self.context_slices[key].start, self.context_slices[key].stop)]
            indices = torch.tensor(indices, dtype=torch.long)
            self.register_buffer(f"{label}_indices", indices)
            # CPU replay statistics must gather compact fields before transfer
            # even when registered model buffers live on a CUDA device.
            setattr(self, f"_{label}_indices_cpu", indices.clone())
        self.state_stats = RunningMoments(self.canonical_dim + self.object_body_dim +
                                          sum(prod(self.context_shapes[name]) for name in self.input_keys))
        self.action_stats = RunningMoments(self.action_dim)
        self.auxiliary_stats = RunningMoments(max(1, width), min_std=0.05)
        self.contact_force_stats = RunningMoments(max(1, self.contact_body_count * 3), min_std=0.05)
        layers = []
        in_dim = self.state_stats.mean.numel() + self.decimation * self.action_dim + self.decimation
        for hidden in hidden_dims:
            layers.extend((nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.SiLU()))
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.velocity_head = nn.Linear(in_dim, self.velocity_dim)
        self.auxiliary_head = nn.Linear(in_dim, learned_width) if learned_width else None
        self.contact_presence_head = None
        self.contact_force_head = None
        self.register_buffer("contact_presence_counts", torch.ones(2))
        if self.contact_body_count:
            self.contact_presence_head = nn.Linear(in_dim, self.contact_body_count)
            self.contact_force_head = nn.Linear(in_dim, 3 * self.contact_body_count)
            nn.init.zeros_(self.contact_presence_head.weight)
            nn.init.zeros_(self.contact_presence_head.bias)
            nn.init.normal_(self.contact_force_head.weight, std=1e-3)
            nn.init.zeros_(self.contact_force_head.bias)
        nn.init.normal_(self.velocity_head.weight, std=1e-3)
        nn.init.zeros_(self.velocity_head.bias)
        if self.auxiliary_head is not None:
            nn.init.normal_(self.auxiliary_head.weight, std=1e-3)
            nn.init.zeros_(self.auxiliary_head.bias)
        scale = torch.ones(self.velocity_dim)
        for section in self.velocity_slices.values():
            scale[section.start + 3:section.start + 6] = 3.
            scale[section.start + 6:section.stop] = 10.
        # Fixed physical units prevent output rescaling as input moments evolve.
        self.register_buffer("velocity_scale", scale / self.decimation)
        self.exogenous_slices = tuple(
            self.context_slices[name] for name in getattr(transform, "exogenous_context_slices", {})
        )
        self.exogenous_context_keys = tuple(getattr(transform, "exogenous_context_slices", {}))
        self.reward_velocity_fields = []
        for group_name, group in getattr(env, "reward_groups", {}).items():
            for name, reward in group.funcs.items():
                key = f"reward.{group_name}.{name}.joint_vel"
                if key in self.context_slices and hasattr(reward, "joint_ids"):
                    self.reward_velocity_fields.append((key, tuple(reward.joint_ids)))
        self._prepare_reward_velocity_indices()
        self.to(device)

    def _prepare_reward_velocity_indices(self):
        """Keep fixed gathers on device; Python-list indexing copies during capture."""
        names = []
        for index, (_, indices) in enumerate(self.reward_velocity_fields):
            name = f"_reward_velocity_indices_{index}"
            self.register_buffer(name, torch.tensor(indices, dtype=torch.long,
                                                   device=self.velocity_scale.device), persistent=False)
            names.append(name)
        self._reward_velocity_index_names = tuple(names)

    def context(self, state, name):
        return state[..., self.context_slices[name]].reshape(state.shape[:-1] + self.context_shapes[name])

    def _unpack_context(self, state):
        chunks = torch.split(state, self.context_chunk_widths, dim=-1)
        return {name: chunks[index].reshape(state.shape[:-1] + self.context_shapes[name])
                for name, index in self.context_chunk_indices.items()}

    def set_context(self, state, updates):
        replacements = [(self.context_slices[name], updates[name])
                        for name in self.context_order if name in updates]
        return self._replace_sections(state, replacements)

    @staticmethod
    def _replace_sections(state, replacements):
        """One functional concatenation, without a chain of full-state copies.

        Slice assignment creates a CopySlices backward for every small field,
        each copying the entire packed state. Keeping unchanged adjacent spans
        intact makes reverse-mode cost proportional to the packed state size.
        ``replacements`` must be disjoint and ordered by their slice starts.
        """
        if not replacements:
            return state
        widths, replacements_by_index, cursor = [], {}, 0
        for section, value in replacements:
            if section.start > cursor:
                widths.append(section.start - cursor)
            replacements_by_index[len(widths)] = value.reshape(*state.shape[:-1], -1)
            widths.append(section.stop - section.start)
            cursor = section.stop
        if cursor < state.shape[-1]:
            widths.append(state.shape[-1] - cursor)
        # A single split also avoids a full-state SliceBackward scatter for
        # every unchanged span in the concatenation.
        pieces = list(torch.split(state, widths, dim=-1))
        for index, value in replacements_by_index.items():
            pieces[index] = value
        return torch.cat(pieces, -1)

    def _features(self, state):
        raw = state.index_select(-1, self._indices_for(state, "input"))
        derived = []
        if getattr(self, "frame", "world") == "heading":
            derived.append(self._canonical_root_features(state))
        if getattr(self, "object_body_features", False):
            derived.append(self._object_body_positions(state))
        return torch.cat((*derived, raw), -1) if derived else raw

    def _object_body_positions(self, state):
        """Every robot body link position in each non-robot asset's root frame."""
        bodies = self.context(state, "robot.body_link_pos_w")
        features = []
        for prefix in self.prefixes[1:]:
            position = self.context(state, f"{prefix}.root_link_pos_w").unsqueeze(-2)
            inverse = quat_conjugate(quat_normalize(self.context(state, f"{prefix}.root_link_quat_w"))).unsqueeze(-2)
            features.append(quat_rotate(inverse, bodies - position).flatten(-2))
        return torch.cat(features, -1)

    def _root_fields(self, state, prefix):
        pos, quat, velocity = (self.context(state, f"{prefix}.{field}") for field in HEADING_ROOT_FIELDS)
        return pos, quat_normalize(quat), velocity

    def _canonical_root_features(self, state):
        """Root state features invariant to horizontal translation and yaw."""
        pos, quat, velocity = self._root_fields(state, "robot")
        inverse = quat_conjugate(quat)
        features = [pos[..., 2:], quat_axes(quat)[2],
                    quat_rotate(inverse, velocity[..., :3]), quat_rotate(inverse, velocity[..., 3:])]
        robot_heading = heading_basis(quat)
        for prefix in self.prefixes[1:]:
            other_pos, other_quat, other_velocity = self._root_fields(state, prefix)
            x_axis, y_axis, _ = quat_axes(other_quat)
            own_heading = heading_basis(other_quat)
            features.extend((world_to_heading(robot_heading, other_pos - pos),
                             world_to_heading(robot_heading, x_axis), world_to_heading(robot_heading, y_axis),
                             world_to_heading(own_heading, other_velocity[..., :3]),
                             world_to_heading(own_heading, other_velocity[..., 3:])))
        return torch.cat(features, -1)

    def _increments_to_world(self, state, increments):
        """Rotate each asset's predicted root increments from its heading frame."""
        pieces = []
        for prefix in self.prefixes:
            basis = heading_basis(self._root_fields(state, prefix)[1]).unsqueeze(-2)
            section = self.velocity_slices[prefix]
            pieces.extend((heading_to_world(basis, increments[..., section.start:section.start + 3]),
                           heading_to_world(basis, increments[..., section.start + 3:section.start + 6]),
                           increments[..., section.start + 6:section.stop]))
        return torch.cat(pieces, -1)

    def _auxiliaries(self, state):
        return state.index_select(-1, self._indices_for(state, "auxiliary"))

    def _indices_for(self, state, label):
        indices = getattr(self, f"{label}_indices")
        if state.device == indices.device:
            return indices
        if state.device.type == "cpu":
            return getattr(self, f"_{label}_indices_cpu")
        return indices.to(state.device)

    @torch.no_grad()
    def update_stats(self, state, action, next_state, reward=None, valid=None):
        """Accumulate valid compact fields without copying masked full states.

        Chunked compact gathers bound CPU and transfer memory for full 4096-env
        rollouts. Action moments describe filtered applied actions, which are
        the actual network inputs, rather than newly issued queued actions.
        """
        state = state.reshape(-1, state.shape[-1])
        next_state = next_state.reshape(-1, next_state.shape[-1])
        action = action.reshape(-1, action.shape[-1])
        valid = None if valid is None else valid.reshape(-1).to(state.device)
        for start in range(0, state.shape[0], 16384):
            section = slice(start, start + 16384)
            current, following, issued = state[section], next_state[section], action[section]
            selection = slice(None) if valid is None else valid[section]
            self.state_stats.update(self._features(current)[selection])
            self.state_stats.update(self._features(following)[selection])
            _, applied = self._actuator_trajectory(current, issued)
            self.action_stats.update(applied[selection])
            if self.auxiliary_dim:
                self.auxiliary_stats.update(self._auxiliaries(following)[selection])
            if self.contact_presence_head is not None:
                forces = self.context(following, "contact.net_forces_w_substeps")[selection]
                self.contact_force_stats.update(forces.flatten(-2))
                active = forces.norm(dim=-1) > self.contact_force_threshold
                self.contact_presence_counts.add_(torch.bincount(active.long().reshape(-1), minlength=2)
                                                  .to(self.contact_presence_counts))

    def _com_offsets(self, state, prefix, context=None):
        name = f"{prefix}.physics.com"
        if name not in self.context_slices:
            return state.new_zeros(
                *state.shape[:-1], len(self.kinematics[prefix].body_names), 3)
        value = self.context(state, name) if context is None else context[name]
        return value[..., :3]

    @torch.no_grad()
    def _verify_kinematics(self, prefix, asset, tree):
        device = tree.local_pos0.device
        # The scene performs an initial simulator step before all IsaacLab data
        # timestamps advance. Cached root/dof/link fields can therefore refer
        # to different instants at construction. Compare FK against one direct
        # PhysX snapshot after its kinematic refresh, without stepping physics
        # or changing the environment's observation caches.
        view = asset.root_physx_view
        if tree.joint_names:
            asset._physics_sim_view.update_articulations_kinematic()
            root_pose = view.get_root_transforms().clone().to(device)
            qpos = view.get_dof_positions().clone().to(device)
            body_pose = view.get_link_transforms().clone().to(device)
        else:
            root_pose = view.get_transforms().clone().to(device)
            qpos = root_pose.new_zeros(root_pose.shape[0], 0)
            body_pose = root_pose.unsqueeze(-2)
        pos = root_pose[..., :3]
        quat = root_pose[..., [6, 3, 4, 5]]
        target_pos = body_pose[..., :3]
        target_quat = body_pose[..., [6, 3, 4, 5]]
        scale = getattr(asset.cfg.spawn, "scale", None)
        scale = torch.as_tensor((1., 1., 1.) if scale is None else scale, device=device, dtype=pos.dtype)
        scale = scale.expand(pos.shape[0], 3)
        if not torch.allclose(scale, scale[..., :1].expand_as(scale), rtol=1e-5, atol=1e-6):
            raise ValueError(f"{prefix}: nonuniform articulated scale is unsupported")
        predicted_pos, predicted_quat, _ = tree(pos, quat, qpos, scale=scale)
        pos_error = (predicted_pos - target_pos).abs().amax().item()
        quat_error = torch.minimum((predicted_quat - target_quat).norm(dim=-1),
                                   (predicted_quat + target_quat).norm(dim=-1)).amax().item()
        if pos_error > 2e-3 or quat_error > 2e-3:
            per_body = torch.minimum((predicted_quat - target_quat).norm(dim=-1),
                                     (predicted_quat + target_quat).norm(dim=-1)).amax(0)
            worst = per_body.topk(min(5, per_body.numel())).indices.tolist()
            details = [(tree.body_names[i], per_body[i].item(),
                        (predicted_pos[:, i] - target_pos[:, i]).abs().amax().item()) for i in worst]
            # Small diagnostic snapshot only on a fail-closed startup mismatch.
            # It permits offline CPU comparison without repeatedly starting Sim.
            diagnostic = Path("/tmp") / f"wm_fk_failure_{prefix}.pt"
            torch.save({"root_pos": pos.cpu(), "root_quat": quat.cpu(), "joint_pos": qpos.cpu(),
                        "scale": scale.cpu(), "body_pos": target_pos.cpu(),
                        "body_quat": target_quat.cpu(), "body_names": tree.body_names,
                        "joint_names": tree.joint_names, "usd_path": str(asset.cfg.spawn.usd_path)}, diagnostic)
            raise ValueError(f"{prefix}: USD FK does not match simulator (position={pos_error:.6g} m, quaternion={quat_error:.6g}, worst={details}, diagnostic={diagnostic}); refusing inconsistent dynamics")

    def _actuator_trajectory(self, state, action, context=None):
        """Exact JointPosition delayed queue and per-physics-step lerp."""
        batch = state.shape[:-1]
        context = self._unpack_context(state) if context is None else context
        action_buffer = context["action.action_buf"]
        shifted = torch.cat((action.unsqueeze(-1), action_buffer[..., :-1]), -1)
        applied = context["action.applied_action"]
        alpha = context["action.alpha"]
        delay = context["action.delay"].long()
        trajectory = []
        for substep in range(self.decimation):
            index = ((delay - substep + self.decimation - 1) // self.decimation).clamp(0, shifted.shape[-1] - 1)
            selected = shifted.gather(-1, index.unsqueeze(-2).expand(*batch, self.action_dim, 1)).squeeze(-1)
            applied = applied + alpha * (selected - applied)
            trajectory.append(applied)
        return shifted, torch.stack(trajectory, -2)

    def _integrate(self, state, action, increments, next_exogenous=None, *,
                   context=None, actuator_trajectory=None, return_updates=False):
        if not hasattr(self, "_reward_velocity_index_names"):
            # Trusted offline module dumps may predate these derived buffers.
            self._prepare_reward_velocity_indices()
        updates = {}
        batch = state.shape[:-1]
        context = self._unpack_context(state) if context is None else context
        shifted, applied_trajectory = (self._actuator_trajectory(state, action, context)
                                       if actuator_trajectory is None else actuator_trajectory)
        updates["action.action_buf"] = shifted
        updates["action.applied_action"] = applied_trajectory[..., -1, :]
        for name in ("episode_length", "command.t"):
            if name in self.context_slices:
                updates[name] = context[name] + 1
        for prefix in self.prefixes:
            pos = context[f"{prefix}.root_link_pos_w"]
            quat = quat_normalize(context[f"{prefix}.root_link_quat_w"])
            velocity = context[f"{prefix}.root_com_vel_w"]
            root_only = prefix == "object" and getattr(self, "object_dynamics", "articulated") == "root"
            num_joints = self.velocity_slices[prefix].stop - self.velocity_slices[prefix].start - 6
            if root_only:
                # These fields remain in the common replay/observation schema,
                # but are not predicted state, inputs or supervision targets.
                # Zero joint pose/velocity gives coherent fixed-frame FK whose
                # output cannot depend on an unmodeled initial wheel channel.
                joint_pos = state.new_zeros(*batch, len(self.kinematics[prefix].joint_names))
                joint_vel = torch.zeros_like(joint_pos)
            else:
                joint_pos = context[f"{prefix}.joint_pos"] if num_joints else state.new_zeros(*batch, 0)
                joint_vel = context[f"{prefix}.joint_vel"] if num_joints else state.new_zeros(*batch, 0)
            offsets = self._com_offsets(state, prefix, context)
            root_offset = offsets[..., 0, :]
            qpos_substeps, qvel_substeps = {}, {}
            for substep in range(self.decimation):
                delta = increments[..., substep, self.velocity_slices[prefix]]
                velocity = velocity + delta[..., :6]
                if not root_only:
                    joint_vel = joint_vel + delta[..., 6:]
                com_pos = pos + quat_rotate(quat, root_offset)
                quat = quat_normalize(quat_multiply(quat_exp(velocity[..., 3:] * self.physics_dt), quat))
                pos = com_pos + velocity[..., :3] * self.physics_dt - quat_rotate(quat, root_offset)
                if not root_only:
                    joint_pos = joint_pos + joint_vel * self.physics_dt
                    qpos_substeps[substep % 2] = joint_pos
                    qvel_substeps[substep % 2] = joint_vel
            updates[f"{prefix}.root_link_pos_w"] = pos
            updates[f"{prefix}.root_link_quat_w"] = quat
            updates[f"{prefix}.root_com_vel_w"] = velocity
            if root_only:
                for suffix in ("joint_pos", "joint_vel", "joint_pos_substeps", "joint_vel_substeps",
                               "applied_torque", "custom_torques"):
                    key = f"{prefix}.{suffix}"
                    if key in context:
                        updates[key] = torch.zeros_like(context[key])
            if num_joints:
                updates[f"{prefix}.joint_pos"] = joint_pos
                updates[f"{prefix}.joint_vel"] = joint_vel
                qp = torch.stack([qpos_substeps.get(i, joint_pos) for i in (0, 1)], -2)
                qv = torch.stack([qvel_substeps.get(i, joint_vel) for i in (0, 1)], -2)
                updates[f"{prefix}.joint_pos_substeps"] = qp
                updates[f"{prefix}.joint_vel_substeps"] = qv
                if prefix == "robot":
                    for (name, _), index_name in zip(self.reward_velocity_fields, self._reward_velocity_index_names):
                        updates[name] = qv.index_select(-1, getattr(self, index_name))
            scale_key = f"{prefix}.kinematic_scale"
            scale = context[scale_key] if scale_key in self.context_slices else pos.new_ones(*batch, 3)
            root_link_linear = velocity[..., :3] - torch.cross(velocity[..., 3:], quat_rotate(quat, root_offset), dim=-1)
            body_pos, body_quat, body_vel = self.kinematics[prefix](
                pos, quat, joint_pos, torch.cat((root_link_linear, velocity[..., 3:]), -1),
                joint_vel, scale, offsets,
            )
            updates[f"{prefix}.body_link_pos_w"] = body_pos
            updates[f"{prefix}.body_link_quat_w"] = body_quat
            updates[f"{prefix}.body_com_vel_w"] = body_vel
        if return_updates:
            return updates
        return self._pack_updates(state, updates, next_exogenous)

    def _pack_updates(self, state, updates, next_exogenous=None):
        if next_exogenous is not None:
            updates.update({name: self.context(next_exogenous, name) for name in self.exogenous_context_keys})
        replacements = [(self.context_slices[name], updates[name])
                        for name in self.context_order if name in updates]
        if next_exogenous is not None:
            replacements.extend((self.codec.slices[key], next_exogenous[..., self.codec.slices[key]])
                                for key in self.codec.exogenous_keys)
            replacements.sort(key=lambda item: item[0].start)
        return self._replace_sections(state, replacements)

    def _causal_network(self, state, action, *, all_substeps=False, applied=None):
        """Predict each increment from its available applied-action prefix.

        Raw queued actions never enter the MLP. This prevents both newly issued
        and older queued actions from affecting a physics substep before the
        exact delay/filter calculation makes them available. A lower-triangular
        prefix also prevents an earlier head from seeing later applied actions.
        """
        normalized_state = self.state_stats.normalize(self._features(state))
        if applied is None:
            _, applied = self._actuator_trajectory(state, action)
        normalized_applied = self.action_stats.normalize(applied)
        prefix = (normalized_applied.unsqueeze(-3) * self.action_prefix_mask).flatten(-2)
        substeps = self.action_substep_onehot.expand(*state.shape[:-1], self.decimation, self.decimation)
        control = torch.cat((prefix, substeps), -1)
        if len(self.trunk):
            # Project the large initial-state vector only once. The action
            # prefix and subsequent shared layers run as one batched tensor.
            first = self.trunk[0]
            state_width = normalized_state.shape[-1]
            features = F.linear(normalized_state, first.weight[:, :state_width], first.bias).unsqueeze(-2)
            features = features + F.linear(control, first.weight[:, state_width:])
            for layer in self.trunk[1:]:
                features = layer(features)
        else:
            expanded_state = normalized_state.unsqueeze(-2).expand(*state.shape[:-1], self.decimation, normalized_state.shape[-1])
            features = torch.cat((expanded_state, control), -1)
        increments = self.velocity_head(features)
        return increments, features if all_substeps else features[..., -1, :]

    def _contact_transition(self, state, forces, context=None):
        context = self._unpack_context(state) if context is None else context
        result = contact_sensor_transition(
            forces, context["contact.current_air_time"], context["contact.current_contact_time"],
            context["contact.last_air_time"], context["contact.last_contact_time"],
            context["contact.timestamp"], context["contact.timestamp_last_update"],
            self.physics_dt, self.contact_force_threshold, self.contact_history_length,
            context["contact.net_forces_w_history"],
        )
        return {f"contact.{key}": value for key, value in result.items()}

    def oracle_contact_transition(self, state, factual_next):
        """Reconstruct sensor bookkeeping from factual next substep forces."""
        if not self.contact_body_count:
            return {}
        return self._contact_transition(state, self.context(factual_next, "contact.net_forces_w_substeps"))

    def forward(self, state, action, next_exogenous=None, *, return_auxiliary=False):
        context = self._unpack_context(state)
        trajectory = self._actuator_trajectory(state, action, context)
        increments, features = self._causal_network(state, action, all_substeps=True, applied=trajectory[1])
        if getattr(self, "frame", "world") == "heading":
            increments = self._increments_to_world(state, increments)
        updates = self._integrate(state, action, increments * self.velocity_scale,
                                  context=context, actuator_trajectory=trajectory, return_updates=True)
        contact_logits, raw_forces = None, None
        if self.auxiliary_head is not None:
            auxiliary = self.auxiliary_head(features[..., -1, :])
            auxiliary = (auxiliary * self.auxiliary_stats.std.index_select(0, self.learned_auxiliary_indices)
                         + self.auxiliary_stats.mean.index_select(0, self.learned_auxiliary_indices))
            for name, section in self.learned_auxiliary_slices.items():
                value = auxiliary[..., section].reshape(state.shape[:-1] + self.context_shapes[name])
                updates[name] = value
        if self.contact_presence_head is not None:
            contact_logits = self.contact_presence_head(features)
            raw_forces = self.contact_force_stats.denormalize(self.contact_force_head(features))
            raw_forces = raw_forces.reshape(*state.shape[:-1], self.decimation, self.contact_body_count, 3)
            forces = contact_force_support(raw_forces, contact_logits > 0, self.contact_force_threshold)
            updates.update(self._contact_transition(state, forces, context))
        prediction = self._pack_updates(state, updates, next_exogenous)
        return (prediction, contact_logits, raw_forces) if return_auxiliary else prediction

    def baseline_prediction(self, state, action, next_exogenous=None):
        """Constant velocity/force baseline with exact actuator and sensor history."""
        increments = state.new_zeros(*state.shape[:-1], self.decimation, self.velocity_dim)
        prediction = self._integrate(state, action, increments, next_exogenous)
        if self.contact_body_count:
            last_force = self.context(state, "contact.net_forces_w_history")[..., :1, :, :]
            forces = last_force.expand(*state.shape[:-1], self.decimation, self.contact_body_count, 3)
            prediction = self.set_context(prediction, self._contact_transition(state, forces))
        return prediction

    def validation_errors(self, prediction, target, *, reporting=True):
        """Physical errors in fixed reporting units, independent of loss scales."""
        return self._physical_errors(prediction, target, reporting=reporting)

    def _physical_errors(self, prediction, target, *, reporting, training_scales=None):
        """Shared per-transition errors; optional units affect optimization only."""
        errors, scaled = {}, []
        predicted_context, target_context = self._unpack_context(prediction), self._unpack_context(target)
        for prefix in self.prefixes:
            for label, suffix, unit_scale in (("root_pos", "root_link_pos_w", .5),
                                               ("root_velocity", "root_com_vel_w", 2.5),
                                               ("joint_pos", "joint_pos", .25),
                                               ("joint_velocity", "joint_vel", 2.5)):
                key = f"{prefix}.{suffix}"
                if (prefix == "object" and getattr(self, "object_dynamics", "articulated") == "root"
                        and suffix in ("joint_pos", "joint_vel")):
                    continue
                if key not in self.context_slices:
                    continue
                error = (predicted_context[key] - target_context[key]).square().mean(-1)
                name = "joint_pos" if prefix == "robot" and label == "joint_pos" else f"{prefix}_{label}"
                if reporting:
                    errors[name] = error.clamp_min(0).sqrt()
                if training_scales:
                    unit_scale = training_scales.get(f"{prefix}_{label}", training_scales.get(label, unit_scale))
                scaled.append(error / unit_scale ** 2)
            angle = quaternion_error(predicted_context[f"{prefix}.root_link_quat_w"],
                                     target_context[f"{prefix}.root_link_quat_w"])
            if reporting:
                errors[f"{prefix}_root_rotation"] = angle
            rotation_scale = (training_scales.get(f"{prefix}_root_rotation", training_scales.get("root_rotation", .5))
                              if training_scales else .5)
            scaled.append(angle.square() / rotation_scale ** 2)
        errors["normalized_dynamics"] = torch.stack(scaled, -1).mean(-1)
        if self.auxiliary_dim:
            auxiliary_error = ((self._auxiliaries(prediction) - self._auxiliaries(target)) /
                               self.auxiliary_stats.std).square()
            errors["auxiliary"] = auxiliary_error.mean(-1)
            for label, matching in (() if not reporting else (
                ("contact_force", lambda key: "forces" in key),
                ("torque", lambda key: "torque" in key),
            )):
                sections = [section for key, section in self.auxiliary_slices.items() if matching(key)]
                if sections:
                    errors[label] = torch.cat([auxiliary_error[..., section] for section in sections], -1).mean(-1)
        if (reporting and getattr(self, "contact_model", True)
                and "contact.current_contact_time" in self.context_slices):
            duration_pred = predicted_context["contact.current_contact_time"]
            duration_true = target_context["contact.current_contact_time"]
            event_pred = (duration_pred > 0) & (duration_pred < self.step_dt + 1e-8)
            event_true = (duration_true > 0) & (duration_true < self.step_dt + 1e-8)
            errors["contact_event_error"] = (event_pred != event_true).float().mean(-1)
            errors["contact_event_true_count"] = event_true.sum(-1).float()
            errors["contact_event_miss_count"] = (event_true & ~event_pred).sum(-1).float()
            # Also expose recall-sensitive evidence: rare contact events can
            # otherwise be obscured by the many inactive sensor bodies.
            errors["contact_event_miss_rate"] = (errors["contact_event_miss_count"] /
                                                 errors["contact_event_true_count"].clamp_min(1))
        return errors

    def transition_loss(self, state, action, next_state, reward=None, terminated=None, discount=None,
                        group_weights=None, return_prediction=False):
        prediction, contact_logits, raw_forces = self(state, action, next_state, return_auxiliary=True)
        errors = self._physical_errors(prediction, next_state, reporting=False,
                                       training_scales=getattr(self, "training_loss_scales", None))
        losses = {"dynamics": errors["normalized_dynamics"],
                  "auxiliary": errors.get("auxiliary", torch.zeros_like(errors["normalized_dynamics"]))}
        if contact_logits is not None:
            target_forces = self.context(next_state, "contact.net_forces_w_substeps")
            active = (target_forces.norm(dim=-1) > self.contact_force_threshold).to(contact_logits.dtype)
            classification = F.binary_cross_entropy_with_logits(contact_logits, active, reduction="none").mean((-1, -2))
            force_error = ((raw_forces.flatten(-2) - target_forces.flatten(-2)) /
                           self.contact_force_stats.std).square().mean((-1, -2))
            losses["contact_presence"] = classification
            losses["contact_force_raw"] = force_error
            losses["auxiliary"] = losses["auxiliary"] + classification + force_error
        if "robot.joint_pos_substeps" in self.context_slices:
            # Substep joint positions share the robot joint-position unit.
            scales = getattr(self, "training_loss_scales", None) or {}
            unit = scales.get("joint_pos", .25)
            error = (self.context(prediction, "robot.joint_pos_substeps") -
                     self.context(next_state, "robot.joint_pos_substeps")).square().mean((-1, -2)) / unit ** 2
            losses["substep_joint_pos"] = error
            losses["dynamics"] = losses["dynamics"] + error
        return (losses, prediction) if return_prediction else losses
