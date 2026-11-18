"""Tensor-only reward and termination oracle for the VAIC skateboard task.

The simulator's ordering is significant: body/joint penalties read the new
physics state, references and object/contact caches still belong to the previous
command update.  This module preserves that ordering.  Contact forces are inputs
to the formula, not an assertion that their learned dynamics have exact gradients.
Contact events and consecutive-error termination counters use hard comparisons;
there is deliberately no straight-through or sigmoid termination surrogate.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def quat_conjugate(q):
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def quat_mul(a, b):
    """Hamilton product, wxyz, with broadcasting."""
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat((aw * bw - (av * bv).sum(-1, keepdim=True),
                      aw * bv + bw * av + torch.linalg.cross(av, bv, dim=-1)), -1)


def quat_apply_inverse(q, v):
    xyz = q[..., 1:].expand_as(v)
    t = torch.linalg.cross(xyz, v, dim=-1) * 2
    return v - q[..., :1] * t + torch.linalg.cross(xyz, t, dim=-1)


def yaw_quat(q):
    w, x, y, z = q.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
    zero = torch.zeros_like(yaw)
    return F.normalize(torch.stack(((yaw / 2).cos(), zero, zero, (yaw / 2).sin()), -1), dim=-1)


def axis_angle_from_quat(q, eps=1e-6):
    """IsaacLab's shortest axis-angle, avoiding an inactive 0/0 autograd branch."""
    q = q * (1 - 2 * (q[..., :1] < 0).to(q.dtype))
    magnitude = q[..., 1:].norm(dim=-1)
    half_angle = torch.atan2(magnitude, q[..., 0])
    angle = 2 * half_angle
    safe_angle = torch.where(angle.abs() > eps, angle, torch.ones_like(angle))
    ratio = torch.where(angle.abs() > eps, half_angle.sin() / safe_angle,
                        0.5 - angle.square() / 48)
    return q[..., 1:] / ratio.unsqueeze(-1)


def orientation_error(reference, actual):
    return axis_angle_from_quat(quat_mul(quat_conjugate(reference), actual)).norm(dim=-1)


def local_positions(body, root_pos, root_quat):
    # Original local tracking removes root x/y, retaining absolute body height.
    root_xy = torch.cat((root_pos[..., :2], torch.zeros_like(root_pos[..., 2:])), -1)
    return quat_apply_inverse(yaw_quat(root_quat).unsqueeze(-2), body - root_xy.unsqueeze(-2))


class _Term(nn.Module):
    """Copy configuration only; never keep or mutate a live simulator reward."""

    def __init__(self, name, original):
        super().__init__()
        self.name = name
        self.kind = type(original).__name__
        self.enabled = bool(getattr(original, "enabled", True))
        self.weight = float(getattr(original, "weight", 1.0))
        for attribute in (
            "sigma", "tolerance", "pos_sigma", "pos_tolerance", "frc_sigma",
            "frc_thres", "gain", "thres", "soft_discount", "default_mass_total",
            "threshold", "min_steps", "pos_thres",
            "body_indices_asset", "body_indices_motion", "joint_indices_asset",
            "joint_indices_motion", "joint_ids", "body_ids", "articulation_body_ids",
        ):
            if not hasattr(original, attribute):
                continue
            value = getattr(original, attribute)
            if attribute in {"body_indices_asset", "body_indices_motion", "joint_indices_asset",
                             "joint_indices_motion", "joint_ids", "body_ids", "articulation_body_ids"}:
                # Python index lists allocate/copy a CPU index tensor at every
                # CUDA read, which is also illegal during CUDA graph capture.
                # These metadata buffers move with the model but never persist.
                if not isinstance(value, slice):
                    self.register_buffer(attribute, torch.as_tensor(value, dtype=torch.long).detach().clone(),
                                         persistent=False)
                    continue
            if isinstance(value, torch.Tensor):
                # These are live-environment metadata, not learned state. In
                # particular impact-force normalization captures randomized
                # env-0 mass after startup DR. A checkpoint must not overwrite
                # the freshly constructed environment's reward equation.
                self.register_buffer(attribute, value.detach().clone(), persistent=False)
            elif isinstance(value, (list, tuple)):
                setattr(self, attribute, list(value))
            else:
                setattr(self, attribute, value)


class VAICRewardModel(nn.Module):
    """Exact task formulas evaluated on real or imagined physical transitions.

    ``state`` and ``next_state`` normally use ObservationCodec's packed format.
    A mapping of context names to tensors is also accepted for standalone oracle
    tests.  ``next_reference`` can be a packed reference template or an explicit
    mapping containing reward-time ``ref_*`` fields.  Normally ``next_state.pre.ref_*``
    supplies these references; physical caches always come from ``state`` so a
    replay template cannot leak factual next poses/forces into imagination.
    """

    reward_kinds = frozenset({
        "keypoint_pos_tracking_product", "keypoint_pos_tracking_local_product",
        "keypoint_ori_tracking_product", "keypoint_ori_tracking_local_product",
        "keypoint_lin_vel_tracking_product", "keypoint_ang_vel_tracking_product",
        "joint_pos_tracking_product", "joint_vel_tracking_product",
        "object_pos_tracking", "object_ori_tracking", "eef_contact_exp",
        "action_rate_l2", "action_rate2_l2", "joint_vel_l2", "joint_pos_limits",
        "joint_torque_limits", "survival", "impact_force_l2", "feet_slip",
        "feet_air_time_skateboard",
    })
    # Terms whose value depends on the learned contact/torque heads rather
    # than on the predicted kinematic state. SuperMimic differentiates only
    # functions of the modeled kinematic state, so ``kinematic=True`` drops
    # these from imagined actor rewards (the contact kernel keeps its FK
    # position factor). Factual fitting and validation keep the exact terms.
    auxiliary_reward_kinds = frozenset({
        "feet_air_time_skateboard", "impact_force_l2", "feet_slip", "joint_torque_limits",
    })
    # Exact terms that read contact forces or contact timers. A world model
    # without contact prediction cannot reproduce them; see contact_free_groups.
    contact_reward_kinds = frozenset({
        "feet_air_time_skateboard", "impact_force_l2", "feet_slip", "eef_contact_exp",
    })

    def contact_free_groups(self):
        """Per reward group: True when no enabled exact term reads contact."""
        return tuple(all(self.reward_terms[index].kind not in self.contact_reward_kinds for index in indices)
                     for indices in self.group_term_indices)
    termination_kinds = frozenset({
        "cum_body_pos_error", "cum_body_z_error", "cum_body_ori_error",
        "cum_body_pos_error_local", "cum_body_ori_error_local", "cum_joint_pos_error",
        "cum_object_pos_error", "cum_object_ori_error", "cum_lost_contact_steps",
    })

    def __init__(self, env, transform, codec):
        super().__init__()
        if getattr(env.command_manager, "object2", None) is not None:
            raise ValueError("VAICRewardModel currently supports one manipulated object.")
        self.codec = codec
        self.context_shapes = dict(transform.context_schema)
        self.context_slices = dict(transform.context_slices)
        self.context_offset = codec.slices["wm_context"].start if codec is not None else 0
        widths, fields, offset = [], [], 0
        for name, region in sorted(self.context_slices.items(), key=lambda item: item[1].start):
            start, stop = self.context_offset + region.start, self.context_offset + region.stop
            if start > offset:
                widths.append(start - offset)
            fields.append((name, len(widths), self.context_shapes[name]))
            widths.append(stop - start)
            offset = stop
        self._context_decode_widths = tuple(widths)
        self._context_decode_fields = tuple(fields)
        self._context_decode_stop = offset
        self.step_dt = float(env.step_dt)
        self.reward_dt = self.step_dt if env.mult_dt else 1.0
        self.group_names = tuple(env.reward_groups)
        self.multiplicative = tuple(bool(group.multiplicative) for group in env.reward_groups.values())
        self.reward_terms = nn.ModuleList()
        self.group_term_indices = []
        for group_name, group in env.reward_groups.items():
            indices = []
            for reward_name, reward in group.funcs.items():
                kind = type(reward).__name__
                # Disabled terms are still executed by the real environment and
                # may change discount.  A nontrivial unsupported side effect must
                # not be silently ignored.
                if not reward.enabled:
                    if getattr(reward, "soft_discount", 1.0) != 1.0:
                        raise ValueError(f"Disabled reward {group_name}.{reward_name} changes discount.")
                    continue
                if kind not in self.reward_kinds:
                    raise ValueError(f"Unsupported enabled VAIC reward: {group_name}.{reward_name} ({kind}).")
                indices.append(len(self.reward_terms))
                self.reward_terms.append(_Term(f"{group_name}.{reward_name}", reward))
            self.group_term_indices.append(tuple(indices))
        self.termination_terms = nn.ModuleList()
        for name, term in env.termination_funcs.items():
            if type(term).__name__ not in self.termination_kinds:
                raise ValueError(f"Unsupported VAIC termination: {name} ({type(term).__name__}).")
            self.termination_terms.append(_Term(name, term))
        counter_keys = {
            "termination." + term.name + "." + suffix
            for term in self.termination_terms
            for suffix in ("error", "_cum_error_mixin__exceeded", "_cum_error_mixin__cum_steps")
        }
        # Keep untouched intervals as views and assemble the whole state once.
        # Repeated indexed assignments instead build a CopySlices autograd chain
        # whose backward copies this large packed state for every counter field.
        self._termination_context_layout = tuple(sorted(
            ((name, self.context_offset + region.start, self.context_offset + region.stop)
             for name, region in self.context_slices.items() if name in counter_keys),
            key=lambda item: item[1],
        ))

    def _has(self, state, name):
        return name in state if isinstance(state, Mapping) else name in self.context_slices

    def _decode_context(self, state):
        """Make one split view for all reads in this call, without graph caches.

        Independent indexing of the packed state in every reward term creates
        one full-state SliceBackward scatter per read. Splitting once makes
        those views share a single backward concatenation instead.
        """
        if state is None or isinstance(state, Mapping):
            return state
        widths = (*self._context_decode_widths, state.shape[-1] - self._context_decode_stop)
        parts = state.split(widths, dim=-1)
        batch_shape = state.shape[:-1]
        return {name: parts[index].reshape(*batch_shape, *shape)
                for name, index, shape in self._context_decode_fields}

    def _get(self, state, name):
        if isinstance(state, Mapping):
            if name not in state:
                raise KeyError(f"Analytic VAIC reward requires context {name!r}.")
            return state[name]
        if name not in self.context_slices:
            raise KeyError(f"Analytic VAIC reward requires context {name!r}; update the raw-context transform.")
        region = self.context_slices[name]
        region = slice(self.context_offset + region.start, self.context_offset + region.stop)
        return state[..., region].reshape(*state.shape[:-1], *self.context_shapes[name])

    def _reference(self, state, next_state, name, explicit=None):
        if explicit is not None:
            if isinstance(explicit, Mapping) and name in explicit:
                return explicit[name]
            if self._has(explicit, "pre." + name):
                return self._get(explicit, "pre." + name)
        if self._has(next_state, "pre." + name):
            return self._get(next_state, "pre." + name)
        return self._get(state, "command." + name)

    def _cached(self, state, name, physical=None):
        key = "command." + name
        if self._has(state, key):
            return self._get(state, key)
        if physical is None:
            raise KeyError(f"Analytic VAIC reward requires cached command field {name!r}.")
        return self._get(state, physical)

    def _contact_force(self, state):
        if self._has(state, "command.eef_contact_forces_b"):
            return self._get(state, "command.eef_contact_forces_b")
        # Scalar force thresholds only need its norm, but using the correct frame
        # also supports the original vector-threshold formula.
        force = self._get(state, "command.eef_contact_forces_w")
        quat = self._cached(state, "object_quat_w", "object.root_link_quat_w")
        return quat_apply_inverse(quat.unsqueeze(-2), force)

    def _first_contact(self, next_state):
        duration = self._get(next_state, "contact.current_contact_time")
        return (duration > 0) & (duration < self.step_dt + 1e-8)

    def _reward(self, term, state, next_state, action, reference, *, kinematic=False):
        kind = term.kind
        get = lambda name: self._get(next_state, name)
        ref = lambda name: self._reference(state, next_state, name, reference)
        if kind.startswith("keypoint_"):
            ai, ri = term.body_indices_asset, term.body_indices_motion
            if "pos_tracking" in kind:
                actual = get("robot.body_link_pos_w")[..., ai, :]
                target = ref("ref_body_pos_w")[..., ri, :]
                if "local" in kind:
                    actual = local_positions(actual,
                        self._cached(state, "robot_root_pos_w", "robot.root_link_pos_w"),
                        self._cached(state, "robot_root_quat_w", "robot.root_link_quat_w"))
                    target = local_positions(target, ref("ref_root_pos_w"), ref("ref_root_quat_w"))
                error = (target - actual).norm(dim=-1)
            elif "ori_tracking" in kind:
                actual = get("robot.body_link_quat_w")[..., ai, :]
                target = ref("ref_body_quat_w")[..., ri, :]
                if "local" in kind:
                    actual = quat_mul(quat_conjugate(yaw_quat(self._cached(
                        state, "robot_root_quat_w", "robot.root_link_quat_w"))).unsqueeze(-2), actual)
                    target = quat_mul(quat_conjugate(yaw_quat(ref("ref_root_quat_w"))).unsqueeze(-2), target)
                error = orientation_error(target, actual)
            else:
                linear = "lin_vel" in kind
                actual = get("robot.body_com_vel_w")[..., ai, :3 if linear else 6]
                if not linear:
                    actual = actual[..., 3:]
                target = ref("ref_body_lin_vel_w" if linear else "ref_body_ang_vel_w")[..., ri, :]
                error = (target - actual).norm(dim=-1)
            return (-(error - term.tolerance).clamp_min(0).mean(-1, keepdim=True) / term.sigma).exp()
        if kind in ("joint_pos_tracking_product", "joint_vel_tracking_product"):
            suffix = "pos" if kind == "joint_pos_tracking_product" else "vel"
            actual = get("robot.joint_" + suffix)[..., term.joint_indices_asset]
            target = ref("ref_joint_" + suffix)[..., term.joint_indices_motion]
            error = ((target - actual).abs() - term.tolerance).clamp_min(0)
            return (-error.mean(-1, keepdim=True) / term.sigma).exp()
        if kind == "object_pos_tracking":
            actual = self._cached(state, "object_pos_w", "object.root_link_pos_w")
            return (-(ref("ref_object_pos_w") - actual).norm(dim=-1, keepdim=True) / term.sigma).exp()
        if kind == "object_ori_tracking":
            actual = self._cached(state, "object_quat_w", "object.root_link_quat_w")
            return (-orientation_error(ref("ref_object_quat_w"), actual).unsqueeze(-1) / term.sigma).exp()
        if kind == "eef_contact_exp":
            error = (self._get(state, "command.contact_eef_pos_w")
                     - self._get(state, "command.contact_target_pos_w")).norm(dim=-1)
            error = (error - term.pos_tolerance).clamp_min(0)
            active = ref("ref_body_contact").to(action.dtype)
            if kinematic:
                return ((-error / term.pos_sigma).exp() * active * term.gain).mean(-1, keepdim=True)
            force = self._contact_force(state)
            if isinstance(term.frc_thres, torch.Tensor) and term.frc_thres.numel() > 1:
                deficit = (force.abs() - term.frc_thres).clamp_max(0).mean(-1)
            else:
                deficit = (force.norm(dim=-1) - term.frc_thres).clamp_max(0)
            return ((-error / term.pos_sigma).exp() * (deficit / term.frc_sigma).exp()
                    * active * term.gain).mean(-1, keepdim=True)
        if kind in ("action_rate_l2", "action_rate2_l2"):
            history = self._get(state, "action.action_buf")
            difference = action - history[..., 0]
            if kind == "action_rate2_l2":
                difference = action - 2 * history[..., 0] + history[..., 1]
            return -difference.square().sum(-1, keepdim=True)
        if kind == "joint_vel_l2":
            velocity = get("reward." + term.name + ".joint_vel").mean(-2)
            return -velocity.square().clamp_max(5).sum(-1, keepdim=True)
        if kind == "joint_pos_limits":
            position = get("robot.joint_pos")[..., term.joint_ids]
            limits = get("reward." + term.name + ".soft_limits")
            return -((limits[..., 0] - position).clamp_min(0)
                     + (position - limits[..., 1]).clamp_min(0)).sum(-1, keepdim=True)
        if kind == "joint_torque_limits":
            torque = get("robot.applied_torque")[..., term.joint_ids]
            limits = get("reward." + term.name + ".soft_limits")
            return -((torque / limits - 1).clamp_min(0)
                     + (-torque / limits - 1).clamp_min(0)).sum(-1, keepdim=True)
        if kind == "survival":
            return torch.ones_like(action[..., :1])
        if kind == "impact_force_l2":
            contact = self._first_contact(next_state)[..., term.body_ids]
            forces = get("contact.net_forces_w_history").norm(dim=-1).mean(-2)[..., term.body_ids]
            force = forces / term.default_mass_total
            return -((force.square() - 1).clamp_min(0) * contact).sum(-1, keepdim=True)
        if kind == "feet_slip":
            contact = get("contact.current_contact_time")[..., term.body_ids] > 0.02
            velocity = get("robot.body_com_vel_w")[..., term.articulation_body_ids, :2]
            speed = (velocity.norm(dim=-1) - term.tolerance).clamp(0, 1)
            return -(contact * speed).sum(-1, keepdim=True)
        if kind == "feet_air_time_skateboard":
            contact = self._first_contact(next_state)[..., term.body_ids]
            airtime = get("contact.last_air_time")[..., term.body_ids]
            standing = self._get(state, "command.is_standing_env").bool()
            return ((airtime - term.thres).clamp_max(0) * contact).sum(-1, keepdim=True) * (~standing)
        raise AssertionError(kind)

    def reward_terms_and_discount(self, state, next_state, action, next_reference=None, *, kinematic=False,
                                  skip_contact=False):
        """Weighted, dt-scaled enabled terms and the original soft discount.

        ``kinematic=True`` omits :attr:`auxiliary_reward_kinds` (and the
        contact-timer discount they carry) and evaluates the contact kernel by
        FK position only. ``skip_contact=True`` omits the exact
        :attr:`contact_reward_kinds` (a world model without contact prediction
        cannot evaluate them); it does not affect the kinematic contact kernel.
        """
        state, next_state, next_reference = (
            self._decode_context(value) for value in (state, next_state, next_reference))
        values = {}
        discount = torch.ones_like(action[..., :1])
        for term in self.reward_terms:
            if kinematic and term.kind in self.auxiliary_reward_kinds:
                continue
            if skip_contact and not kinematic and term.kind in self.contact_reward_kinds:
                continue
            values[term.name] = self._reward(term, state, next_state, action, next_reference,
                                             kinematic=kinematic) * term.weight * self.reward_dt
            if term.kind == "feet_air_time_skateboard":
                contact = self._first_contact(next_state)[..., term.body_ids]
                airtime = self._get(next_state, "contact.last_air_time")[..., term.body_ids]
                violation = ((airtime < term.thres) & contact).any(-1, keepdim=True)
                discount = torch.where(violation, torch.full_like(discount, term.soft_discount), discount)
        return values, discount

    def termination_errors(self, state, next_state, next_reference=None):
        """Task errors with the simulator's original state/reference timing."""
        state, next_state, next_reference = (
            self._decode_context(value) for value in (state, next_state, next_reference))
        errors = {}
        get = lambda name: self._get(next_state, name)
        ref = lambda name: self._reference(state, next_state, name, next_reference)
        for term in self.termination_terms:
            kind = term.kind
            if kind.startswith("cum_body_"):
                ai, ri = term.body_indices_asset, term.body_indices_motion
                if "ori" in kind:
                    actual = get("robot.body_link_quat_w")[..., ai, :]
                    target = ref("ref_body_quat_w")[..., ri, :]
                    if "local" in kind:
                        actual = quat_mul(quat_conjugate(yaw_quat(get("robot.root_link_quat_w"))).unsqueeze(-2), actual)
                        target = quat_mul(quat_conjugate(yaw_quat(ref("ref_root_quat_w"))).unsqueeze(-2), target)
                    error = orientation_error(target, actual)
                else:
                    actual = get("robot.body_link_pos_w")[..., ai, :]
                    target = ref("ref_body_pos_w")[..., ri, :]
                    if "local" in kind:
                        # Unlike the reward, termination uses the NEW root pose.
                        actual = local_positions(actual, get("robot.root_link_pos_w"), get("robot.root_link_quat_w"))
                        target = local_positions(target, ref("ref_root_pos_w"), ref("ref_root_quat_w"))
                    difference = target - actual
                    error = difference[..., 2].abs() if kind == "cum_body_z_error" else difference.norm(dim=-1)
                error = error.amax(-1)
            elif kind == "cum_joint_pos_error":
                error = (ref("ref_joint_pos")[..., term.joint_indices_motion]
                         - get("robot.joint_pos")[..., term.joint_indices_asset]).abs().amax(-1)
            elif kind == "cum_object_pos_error":
                error = (ref("ref_object_pos_w") - get("object.root_link_pos_w")).norm(dim=-1)
            elif kind == "cum_object_ori_error":
                error = orientation_error(get("object.root_link_quat_w"), ref("ref_object_quat_w"))
            elif kind == "cum_lost_contact_steps":
                distance = (self._get(state, "command.contact_eef_pos_w")
                            - self._get(state, "command.contact_target_pos_w")).norm(dim=-1)
                position = distance < term.pos_thres
                if (isinstance(term.frc_thres, (int, float)) and not isinstance(term.frc_thres, bool)
                        and term.frc_thres <= 0):
                    # A force norm always meets a nonpositive scalar threshold:
                    # the rule is position-only, so the force is not read.
                    force_ok = torch.ones_like(position)
                else:
                    force = self._contact_force(state)
                    if isinstance(term.frc_thres, torch.Tensor) and term.frc_thres.numel() > 1:
                        force_ok = (force.abs() >= term.frc_thres).all(-1)
                    else:
                        force_ok = force.norm(dim=-1) >= term.frc_thres
                active = ref("ref_body_contact").bool()
                error = (active & ~(position & force_ok)).any(-1).to(distance.dtype) * 2
            else:
                raise AssertionError(kind)
            errors[term.name] = error
        return errors

    def termination_details(self, state, next_state, next_reference=None):
        """Errors, next consecutive counters and exact hard termination per term."""
        errors = self.termination_errors(state, next_state, next_reference)
        state = self._decode_context(state)
        details = {}
        for term in self.termination_terms:
            error = errors[term.name]
            previous = self._get(state, "termination." + term.name + "._cum_error_mixin__cum_steps")
            exceeded = error >= term.threshold
            count = torch.where(exceeded, previous + 1, torch.zeros_like(previous))
            details[term.name] = {"error": error, "exceeded": exceeded, "count": count,
                                  "terminated": (count >= term.min_steps).unsqueeze(-1)}
        return details

    def update_termination_state(self, state, next_state, next_reference=None):
        """Write analytic counters into an imagined next state, without mutation."""
        details = self.termination_details(state, next_state, next_reference)
        updates = {}
        for name, values in details.items():
            for suffix, value in (("error", values["error"]),
                                  ("_cum_error_mixin__exceeded", values["exceeded"]),
                                  ("_cum_error_mixin__cum_steps", values["count"])):
                updates["termination." + name + "." + suffix] = value
        if isinstance(next_state, Mapping):
            result = dict(next_state)
            result.update(updates)
        else:
            widths, replacements, offset = [], [], 0
            for name, start, stop in self._termination_context_layout:
                if start > offset:
                    widths.append(start - offset)
                replacements.append((len(widths), name, stop - start))
                widths.append(stop - start)
                offset = stop
            widths.append(next_state.shape[-1] - offset)
            # A single split has one backward concatenation; many separate
            # slices would each scatter their gradient into a full-size tensor.
            parts = list(next_state.split(widths, dim=-1))
            for index, name, width in replacements:
                parts[index] = updates[name].reshape(*next_state.shape[:-1], width).to(next_state.dtype)
            result = torch.cat(parts, dim=-1)
        if details:
            terminated = torch.stack([x["terminated"] for x in details.values()], -1).any(-1)
        else:
            template = next(iter(result.values())) if isinstance(result, Mapping) else result
            if isinstance(result, Mapping):
                event_ndim = len(self.context_shapes.get(next(iter(result)), template.shape[1:]))
                batch_shape = template.shape[:-event_ndim] if event_ndim else template.shape
            else:
                batch_shape = result.shape[:-1]
            terminated = torch.zeros(*batch_shape, 1, dtype=torch.bool, device=template.device)
        return result, terminated

    def max_step_rewards(self, *, kinematic=False):
        """Per-group analytic ceiling of one step's reward, ``inf`` if unbounded."""
        from active_adaptation.utils.reward_bounds import group_ceiling
        ceilings = []
        for indices, multiplicative in zip(self.group_term_indices, self.multiplicative):
            terms = [(self.reward_terms[i].kind, self.reward_terms[i].weight,
                      getattr(self.reward_terms[i], "gain", 1.)) for i in indices
                     if not (kinematic and self.reward_terms[i].kind in self.auxiliary_reward_kinds)]
            ceilings.append(group_ceiling(terms, multiplicative) * self.reward_dt)
        return torch.tensor(ceilings or [0.], dtype=torch.float32)

    def forward(self, state, next_state, action, next_reference=None, *, compute_termination=True,
                kinematic=False, skip_contact=False):
        # Share each split across reward and termination computations below.
        state, next_state, next_reference = (
            self._decode_context(value) for value in (state, next_state, next_reference))
        values, discount = self.reward_terms_and_discount(state, next_state, action, next_reference,
                                                          kinematic=kinematic, skip_contact=skip_contact)
        groups = []
        for indices, multiplicative in zip(self.group_term_indices, self.multiplicative):
            group = [values[self.reward_terms[index].name] / self.reward_dt for index in indices
                     if self.reward_terms[index].name in values]
            if group:
                group = torch.cat(group, -1)
                group = group.prod(-1, keepdim=True) if multiplicative else group.sum(-1, keepdim=True)
            else:
                group = torch.ones_like(action[..., :1]) if multiplicative else torch.zeros_like(action[..., :1])
            groups.append(group * self.reward_dt)
        reward = torch.cat(groups, -1) if groups else torch.zeros_like(action[..., :1])
        if not compute_termination:
            # GroundedWorldModel already evaluated and stored the exact counters.
            # Avoid repeating those equations when their result is discarded.
            return reward, None, discount
        details = self.termination_details(state, next_state, next_reference)
        terminated = (torch.stack([x["terminated"] for x in details.values()], -1).any(-1)
                      if details else torch.zeros_like(action[..., :1], dtype=torch.bool))
        return reward, terminated, discount
