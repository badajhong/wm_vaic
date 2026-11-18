"""Differentiable VAIC observations reconstructed from one physical state.

These tensor expressions mirror the existing skateboard teacher observations,
including their pre/post-command timing, noisy sensor histories, and action units.
No environment is instantiated and no simulator observation function is mutated.
"""

from __future__ import annotations

import math
import torch


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_multiply(a, b):
    a, b = torch.broadcast_tensors(a, b)
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv)), -1)


def quat_rotate(q, v):
    shape = torch.broadcast_shapes(q.shape[:-1], v.shape[:-1])
    q, v = q.expand(*shape, 4), v.expand(*shape, 3)
    t = 2 * torch.linalg.cross(q[..., 1:], v)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t)


def quat_inverse_rotate(q, v):
    return quat_rotate(quat_conjugate(q), v)


def yaw_angle(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def yaw_quaternion(q):
    half = yaw_angle(q) / 2
    zero = torch.zeros_like(half)
    return torch.stack((half.cos(), zero, zero, half.sin()), -1)


def rotation_matrix(q):
    w, x, y, z = q.unbind(-1)
    s = 2 / q.square().sum(-1).clamp_min(1e-12)
    return torch.stack((1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
                        s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
                        s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def _zero_height(value):
    return torch.cat((value[..., :2], torch.zeros_like(value[..., 2:3])), -1)


class VAICWorldModelAdapter:
    """Rebuild original observation groups from physical context and references.

    ``rebuild(predicted_state, next_template, action, previous_state=...)`` accepts
    packed codec tensors. Only reference and noise fields may come from the next
    template. Sensor history and cached physical inputs come from the previous
    imagined state, so this remains valid for counterfactual action sequences.
    """

    SUPPORTED = {
        "root_ang_vel_history", "projected_gravity_history", "joint_pos_history", "prev_actions",
        "ref_root_pos_future_b", "ref_root_ori_future_b", "diff_body_pos_future_local",
        "diff_body_ori_future_local", "diff_body_lin_vel_future_local", "diff_body_ang_vel_future_local",
        "root_linvel_b", "body_pos_b", "body_vel_b", "body_height", "applied_action", "applied_torque",
        "object_xy_b", "object_heading_b", "object_pos_b", "object_ori_b", "ref_contact_pos_b",
        "diff_object_pos_future", "diff_object_ori_future", "ref_object_contact_future", "diff_contact_pos_b",
        "ref_root_vel_future_b", "ref_contact_flag_future", "object_points",
        "ref_body_pos_future_local", "ref_joint_pos_future", "ref_motion_phase", "ref_joint_pos_action_policy",
    }

    def __init__(self, env, transform, codec):
        self.env = getattr(env, "base_env", env)
        self.transform, self.codec = transform, codec
        self.shapes = dict(transform.context_schema)
        self.context_slices = transform.context_slices
        self._context_names = tuple(self.context_slices)
        self._context_widths = tuple(section.stop - section.start for section in self.context_slices.values())
        context_section = codec.slices["wm_context"]
        self._packed_context_widths = (context_section.start, context_section.stop - context_section.start,
                                       codec.state_dim - context_section.stop)
        self._observation_widths = tuple(section.stop - section.start for section in codec.slices.values())
        self.groups = self.env.observation_funcs
        self.command = self.env.command_manager
        self._index_cache = {}
        self._constant_cache = {}
        if getattr(self.env.cfg, "terrain", "plane") != "plane":
            raise ValueError("The physical VAIC observation adapter currently supports plane terrain only.")
        if getattr(self.command, "object2", None) is not None:
            raise ValueError("The physical VAIC observation adapter currently supports one manipulated object.")
        for group_name, group in self.groups.items():
            if group_name not in codec.keys:
                continue
            for name, obs in group.funcs.items():
                if type(obs).__name__ not in self.SUPPORTED:
                    raise ValueError(f"Unsupported physical-WM observation {group_name}.{name}: {type(obs).__name__}")
                if type(obs).__name__ == "root_linvel_b" and torch.count_nonzero(obs.ema.gammas):
                    raise ValueError("Physical WM requires the baseline gamma=0 root-velocity EMA.")
        self.reference_groups = tuple(k for k in ("command", "command_", "ref_joint_pos_") if k in codec.keys)
        pure_reference = {"ref_body_pos_future_local", "ref_joint_pos_future", "ref_motion_phase",
                          "ref_joint_pos_action_policy"}
        for group_name in self.reference_groups:
            if any(type(obs).__name__ not in pure_reference for obs in self.groups[group_name].funcs.values()):
                raise ValueError(f"Group {group_name!r} contains a physical observation and cannot be teacher-forced.")
        # Do not change live observation/command objects. Materialize their
        # fixed indices once, including on the initial simulator device, so
        # advanced indexing performs no hidden CPU->GPU copies during capture.
        device = torch.device(getattr(self.env, "device", "cpu"))
        for name in ("tracking_body_indices_asset", "tracking_joint_indices_asset", "contact_eef_body_indices_asset"):
            self._index(self.command, name, device)
        for group in self.groups.values():
            for obs in group.funcs.values():
                for name in ("joint_ids", "history_steps", "body_indices", "body_ids"):
                    if hasattr(obs, name):
                        self._index(obs, name, device)

    def _index(self, owner, name, like):
        device = like.device if isinstance(like, torch.Tensor) else torch.device(like)
        key = (id(owner), name, device)
        if key not in self._index_cache:
            value = getattr(owner, name)
            self._index_cache[key] = (value if isinstance(value, (slice, int)) else
                                      torch.as_tensor(value, dtype=torch.long, device=device))
        return self._index_cache[key]

    def _constant(self, owner, name, like):
        key = (id(owner), name, like.device, like.dtype)
        if key not in self._constant_cache:
            self._constant_cache[key] = torch.as_tensor(getattr(owner, name), device=like.device,
                                                        dtype=like.dtype)
        return self._constant_cache[key]

    def context(self, state, name):
        offset = self.codec.slices["wm_context"].start
        section = self.context_slices[name]
        return state[..., offset + section.start:offset + section.stop].reshape(
            *state.shape[:-1], *self.shapes[name])

    def set_context(self, state, values):
        unknown = values.keys() - self.context_slices.keys()
        if unknown:
            raise KeyError(next(iter(unknown)))
        prefix, original, suffix = torch.split(state, self._packed_context_widths, dim=-1)
        original_parts = (None,) * len(self._context_names) if len(values) == len(self._context_names) else (
            torch.split(original, self._context_widths, dim=-1))
        # One concatenation per level avoids a CopySlices autograd node and
        # scatter kernel for every context field during imagined rollouts.
        context = torch.cat([
            values[name].reshape(*state.shape[:-1], -1) if name in values else original_part
            for name, original_part in zip(self._context_names, original_parts)
        ], dim=-1)
        return torch.cat((prefix, context, suffix), dim=-1)

    def _get_contexts(self, state):
        _, context, _ = torch.split(state, self._packed_context_widths, dim=-1)
        parts = torch.split(context, self._context_widths, dim=-1)
        return {name: part.reshape(*state.shape[:-1], *self.shapes[name])
                for name, part in zip(self._context_names, parts)}

    def rebuild(self, state, next_context_template, action, previous_state=None,
                rebuild_observations=True):
        """Update context and optionally reconstruct the original observation groups.

        ``rebuild_observations=False`` is a physical-model fitting optimization:
        it preserves the complete updated context, but leaves stale observation
        groups in the packed result. Such results must never feed an actor,
        critic or observation-parity check without full reconstruction.
        """
        context = self.prepare_context(state, next_context_template, action, previous_state)
        return self.pack_observations(state, next_context_template, context,
                                      rebuild_observations=rebuild_observations)

    def prepare_context(self, state, next_context_template, action, previous_state=None,
                        *, previous_context=None, template_context=None):
        """Update tensor views without constructing an intermediate packed state.

        Grounded dynamics can pass these same views to analytic termination and
        reward equations before packing once. Supplied dictionaries are read
        only; updates are made to a newly decoded context dictionary.
        """
        if next_context_template is None:
            next_context_template = state
        c = self._get_contexts(state)
        template = (self._get_contexts(next_context_template)
                    if template_context is None else template_context)
        old = previous_context
        if old is None and previous_state is not None:
            old = self._get_contexts(previous_state)
        for name in self.transform.exogenous_context_slices:
            c[name] = template[name]
        if old is not None:
            for name in self.transform.static_context_slices:
                c[name] = old[name]
            for name in c:
                if name.startswith("pre.") and not name.startswith("pre.ref_"):
                    source = "command." + name[4:]
                    if source in old:
                        c[name] = old[source]
            if "action.action_buf" in c:
                c["action.action_buf"] = torch.cat((action.unsqueeze(-1), old["action.action_buf"][..., :-1]), -1)
            c["episode_length"] = old["episode_length"] + 1
        self._update_command_physics(c)
        self._update_histories(c, old)
        return c

    def pack_observations(self, state, next_context_template, context, *, rebuild_observations=True):
        """Pack updated context and the unchanged original observation equations."""
        if next_context_template is None:
            next_context_template = state
        c = context
        if not rebuild_observations:
            return self.set_context(state, c)
        # Rebuild directly into the packed groups. Creating two TensorDicts and
        # an intermediate full packed state adds dispatch and copies at every
        # imagined step, while the complete context is already available here.
        widths = getattr(self, "_observation_widths", tuple(
            section.stop - section.start for section in self.codec.slices.values()))
        observations = dict(zip(self.codec.keys, torch.split(state, widths, dim=-1)))
        template_observations = dict(zip(self.codec.keys, torch.split(next_context_template, widths, dim=-1)))
        observations["wm_context"] = torch.cat([
            c[name].reshape(*state.shape[:-1], -1) for name in self._context_names], -1)
        for group_name, group in self.groups.items():
            if group_name not in self.codec.keys:
                continue
            if group_name in self.reference_groups:
                observations[group_name] = template_observations[group_name]
                continue
            if group_name == "object_geo_":
                continue
            parts = [self._observation(type(obs).__name__, obs, group_name, name, c)
                     .reshape(*state.shape[:-1], -1) for name, obs in group.funcs.items()]
            observations[group_name] = torch.cat(parts, -1)
        return torch.cat([observations[key] for key in self.codec.keys], -1)

    def _update_command_physics(self, c):
        command = self.command
        body_ids = self._index(command, "tracking_body_indices_asset", c["robot.body_link_pos_w"])
        joint_ids = self._index(command, "tracking_joint_indices_asset", c["robot.joint_pos"])
        replacements = {
            "robot_root_pos_w": c["robot.root_link_pos_w"],
            "robot_root_quat_w": c["robot.root_link_quat_w"],
            "robot_body_pos_w": c["robot.body_link_pos_w"][..., body_ids, :],
            "robot_body_quat_w": c["robot.body_link_quat_w"][..., body_ids, :],
            "robot_body_lin_vel_w": c["robot.body_com_vel_w"][..., body_ids, :3],
            "robot_body_ang_vel_w": c["robot.body_com_vel_w"][..., body_ids, 3:],
            "robot_joint_pos": c["robot.joint_pos"][..., joint_ids],
            "robot_joint_vel": c["robot.joint_vel"][..., joint_ids],
            "object_pos_w": c["object.root_link_pos_w"], "object_quat_w": c["object.root_link_quat_w"],
        }
        object_quat = c["object.body_link_quat_w"][..., command.object_body_id_asset, :]
        replacements["contact_target_pos_w"] = (
            c["object.body_link_pos_w"][..., command.object_body_id_asset, :].unsqueeze(-2)
            + quat_rotate(object_quat.unsqueeze(-2), c["command.contact_target_pos_offset"]))
        eef_ids = self._index(command, "contact_eef_body_indices_asset", c["robot.body_link_pos_w"])
        replacements["contact_eef_pos_w"] = (c["robot.body_link_pos_w"][..., eef_ids, :]
            + quat_rotate(c["robot.body_link_quat_w"][..., eef_ids, :], c["command.contact_eef_pos_offset"]))
        # A world model without contact prediction keeps start-state forces,
        # which no consumer reads; GroundedWorldModel turns this rotation off.
        if getattr(self, "rebuild_contact_forces", True) and "command.eef_contact_forces_b" in c:
            replacements["eef_contact_forces_b"] = quat_inverse_rotate(
                object_quat.unsqueeze(-2), c["command.eef_contact_forces_w"])
        for key, value in replacements.items():
            if "command." + key in c:
                c["command." + key] = value

    def _update_histories(self, c, old):
        rootq = c["robot.root_link_quat_w"]
        down = torch.zeros_like(c["robot.root_link_pos_w"])
        down[..., 2] = -1
        for group_name, group in self.groups.items():
            for name, obs in group.funcs.items():
                kind = type(obs).__name__
                history = f"obs_history.{group_name}.{name}"
                if history not in c:
                    continue
                noise = c.get(f"obs_noise.{group_name}.{name}", 0.)
                if kind == "root_ang_vel_history":
                    value = quat_inverse_rotate(rootq, c["robot.root_com_vel_w"][..., 3:]) + noise
                elif kind == "projected_gravity_history":
                    value = quat_inverse_rotate(rootq, down) + noise
                    if obs.noise_std > 0:
                        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                elif kind == "joint_pos_history":
                    value = c["robot.joint_pos_substeps"][..., self._index(obs, "joint_ids", rootq)].mean(-2) + noise
                elif kind == "root_linvel_b":
                    # Default task's gamma=0 EMA equals the final substep COM velocity.
                    if c[history].shape[-2] != 1:
                        raise ValueError("Physical WM currently requires the task's single root-velocity EMA.")
                    c[history] = c["robot.root_com_vel_w"][..., None, :3]
                    continue
                else:
                    continue
                tail = old[history][..., :-1, :] if old is not None else c[history][..., 1:, :]
                c[history] = torch.cat((value.unsqueeze(-2), tail), -2)

    def _observation(self, kind, obs, group, name, c):
        get = c.__getitem__
        rootp, rootq = get("robot.root_link_pos_w"), get("robot.root_link_quat_w")
        prep, preq = get("pre.robot_root_pos_w"), get("pre.robot_root_quat_w")
        objp, objq = get("object.root_link_pos_w"), get("object.root_link_quat_w")
        noise = c.get(f"obs_noise.{group}.{name}", 0.)
        if kind in ("root_ang_vel_history", "projected_gravity_history", "joint_pos_history"):
            value = get(f"obs_history.{group}.{name}")
            if kind == "joint_pos_history":
                value = (value - get("action.offset")[..., self._index(obs, "joint_ids", value)].unsqueeze(-2)) * self._constant(obs, "joint_mask", value)
            return value[..., self._index(obs, "history_steps", value), :]
        if kind == "prev_actions":
            value = get("action.action_buf")[..., :obs.steps]
            return value.transpose(-2, -1) if obs.permute else value
        if kind == "applied_action":
            return get("action.applied_action")
        if kind == "applied_torque":
            return get("robot.applied_torque")[..., self._index(obs, "joint_ids", rootq)]
        if kind == "root_linvel_b":
            q = yaw_quaternion(rootq) if obs.yaw_only else rootq
            return quat_inverse_rotate(q.unsqueeze(-2), get(f"obs_history.{group}.{name}"))
        if kind == "body_pos_b":
            return quat_inverse_rotate(yaw_quaternion(rootq).unsqueeze(-2),
                get("robot.body_link_pos_w")[..., self._index(obs, "body_indices", rootq), :] - _zero_height(rootp).unsqueeze(-2))
        if kind == "body_vel_b":
            q = yaw_quaternion(rootq) if obs.yaw_only else rootq
            return quat_inverse_rotate(q.unsqueeze(-2), get("robot.body_com_vel_w")[..., self._index(obs, "body_indices", rootq), :3])
        if kind == "body_height":
            return get("robot.body_link_pos_w")[..., self._index(obs, "body_ids", rootq), 2]
        if kind == "ref_root_pos_future_b":
            return quat_inverse_rotate(preq.unsqueeze(-2), get("pre.ref_root_pos_future_w") - prep.unsqueeze(-2))
        if kind == "ref_root_ori_future_b":
            q = quat_multiply(quat_conjugate(preq).unsqueeze(-2), get("pre.ref_root_quat_future_w"))
            return rotation_matrix(q)[..., :2, :]
        if kind.startswith("diff_body_"):
            rq = yaw_quaternion(get("pre.ref_root_quat_w")).unsqueeze(-2).unsqueeze(-2)
            pq = yaw_quaternion(preq).unsqueeze(-2)
            if kind == "diff_body_pos_future_local":
                reference = quat_inverse_rotate(rq, get("pre.ref_body_pos_future_w")
                    - _zero_height(get("pre.ref_root_pos_w")).unsqueeze(-2).unsqueeze(-2))
                actual = quat_inverse_rotate(pq, get("pre.robot_body_pos_w") - _zero_height(prep).unsqueeze(-2))
                return reference - actual.unsqueeze(-3)
            if kind == "diff_body_ori_future_local":
                reference = quat_multiply(quat_conjugate(rq), get("pre.ref_body_quat_future_w"))
                actual = quat_multiply(quat_conjugate(pq), get("pre.robot_body_quat_w")).unsqueeze(-3)
                return rotation_matrix(quat_multiply(quat_conjugate(actual), reference))[..., :2, :]
            vel = "lin" if kind == "diff_body_lin_vel_future_local" else "ang"
            reference = quat_inverse_rotate(rq, get(f"pre.ref_body_{vel}_vel_future_w"))
            actual = quat_inverse_rotate(pq, get(f"pre.robot_body_{vel}_vel_w"))
            return reference - actual.unsqueeze(-3)
        if kind == "object_xy_b":
            return quat_inverse_rotate(yaw_quaternion(preq), objp - prep)[..., :2] + noise
        if kind == "object_heading_b":
            yaw = torch.remainder(yaw_angle(objq) - yaw_angle(preq) + math.pi, 2 * math.pi) - math.pi
            yaw = yaw.unsqueeze(-1) + noise
            return torch.cat((yaw.cos(), yaw.sin()), -1)
        if kind == "object_pos_b":
            return quat_inverse_rotate(preq, objp - prep)
        if kind == "object_ori_b":
            return rotation_matrix(quat_multiply(quat_conjugate(preq), objq))
        if kind == "ref_contact_pos_b":
            q = yaw_quaternion(preq) if obs.yaw_only else preq
            return quat_inverse_rotate(q.unsqueeze(-2), get("pre.contact_target_pos_w") - prep.unsqueeze(-2)) + noise
        if kind == "diff_contact_pos_b":
            return quat_inverse_rotate(preq.unsqueeze(-2), get("pre.contact_target_pos_w") - get("pre.contact_eef_pos_w"))
        if kind == "diff_object_pos_future":
            return quat_inverse_rotate(objq.unsqueeze(-2), get("pre.ref_object_pos_future_w") - objp.unsqueeze(-2))
        if kind == "diff_object_ori_future":
            return rotation_matrix(quat_multiply(quat_conjugate(objq).unsqueeze(-2), get("pre.ref_object_quat_future_w")))
        if kind in ("ref_object_contact_future", "ref_contact_flag_future"):
            return get("command.ref_object_contact_future")
        if kind == "ref_root_vel_future_b":
            q = yaw_quaternion(rootq).unsqueeze(-2)
            lin = quat_inverse_rotate(q, get("command.ref_root_lin_vel_future_w"))
            ang = quat_inverse_rotate(q, get("command.ref_root_ang_vel_future_w"))
            return torch.cat((torch.round(lin[..., :1] / .2) * .2,
                              torch.round(lin[..., 1:2] / 5) * 5,
                              torch.round(ang[..., 2:3] / 5) * 5), -1)
        raise ValueError(f"Observation {group}.{name} ({kind}) cannot be reconstructed from physical state.")
