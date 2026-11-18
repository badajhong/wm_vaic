"""Advance VAIC motion references without borrowing factual replay outcomes.

The motion dataset is immutable. Only its tensors/configuration are copied here;
this provider never calls command.update(), resets, or other simulator methods.
References use the same origin-relative coordinates as wm_context. RobotTracking
increments t after reading body/joint frames, whereas RobotObjectTracking reads
contact flags after that increment. Both phases are intentionally retained.
"""

from __future__ import annotations

import torch
from torch import nn

from .adapter import quat_inverse_rotate, yaw_quaternion


class VAICImaginationReference(nn.Module):
    """Create the next exogenous template from a packed imagined current state.

    ``step(state)`` returns ``(next_template, truncated)``. The template has the
    same packed shape as state and is consumed by GroundedWorldModel.forward;
    its physical observations are not a predicted state. Truncation includes the
    ORIGINAL motion-end and episode-time-limit rules, not stored replay done.
    Task failures are evaluated separately by the analytic termination model.

    No new context fields or persistent checkpoint tensors are introduced.
    Replay does not separate episodic offsets from step noise: fields listed in
    ``frozen_noise_fields`` retain their combined realization during imagination.
    Pure step-noise fields can be resampled exactly. This is not full stochastic
    equivalence for observations containing an episodic offset.
    """

    def __init__(self, env, adapter):
        super().__init__()
        env = getattr(env, "base_env", env)
        command = env.command_manager
        self.adapter = adapter
        self.max_episode_length = int(env.max_episode_length)
        self.replay_motion = bool(getattr(command, "replay_motion", False))
        if self.replay_motion:
            raise ValueError("Independent WM imagination requires replay_motion=false.")
        required = {"command.motion_ids", "command.motion_len", "command.t", "episode_length"}
        missing = required - adapter.context_slices.keys()
        if missing:
            raise ValueError(f"Independent motion references require replay context {sorted(missing)}")
        dataset = command.dataset
        device = dataset.data.body_pos_w.device

        def buffer(name, value, *, dtype=None):
            self.register_buffer(name, torch.as_tensor(value, device=device, dtype=dtype).detach().clone(),
                                 persistent=False)

        buffer("motion_starts", dataset.starts, dtype=torch.long)
        buffer("motion_ends", dataset.ends, dtype=torch.long)
        buffer("future_steps", command.future_steps, dtype=torch.long)
        buffer("tracking_body_ids", command.tracking_body_indices_motion, dtype=torch.long)
        buffer("tracking_joint_ids", command.tracking_joint_indices_motion, dtype=torch.long)
        self.root_body_id = int(command.root_body_idx_motion)
        self.object_body_id = int(command.object_body_id_motion)
        for name in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
                     "joint_pos", "joint_vel"):
            buffer("motion_" + name, getattr(dataset.data, name))
        for name in ("body", "feet", "object"):
            buffer("contact_" + name, getattr(command, "_" + name + "_contact"))

        # Per-motion reset offsets already live in static replay context. The
        # residual teacher reference instead uses the fixed nominal joint pose.
        self._reference_observations = {}
        self._noise = []
        for group_name, group in adapter.groups.items():
            if group_name in adapter.reference_groups:
                observations = []
                for obs_name, obs in group.funcs.items():
                    kind = type(obs).__name__
                    constants = None
                    if kind == "ref_joint_pos_action_policy":
                        token = f"reference_{len(self._reference_observations)}_{len(observations)}"
                        defaults = torch.as_tensor(obs.default_joint_pos, device=device)
                        if defaults.ndim > 1:
                            if not torch.equal(defaults, defaults[:1].expand_as(defaults)):
                                raise ValueError("Residual reference needs an environment-independent nominal joint pose")
                            defaults = defaults[0]
                        constants = (token + "_ids", token + "_default", token + "_scale")
                        buffer(constants[0], obs.action_indices_motion, dtype=torch.long)
                        buffer(constants[1], defaults)
                        buffer(constants[2], obs.action_scaling)
                    if kind not in {"ref_body_pos_future_local", "ref_joint_pos_future", "ref_motion_phase",
                                    "ref_joint_pos_action_policy"}:
                        raise ValueError(f"Unsupported independent reference observation {group_name}.{obs_name}: {kind}")
                    observations.append((kind, constants))
                self._reference_observations[group_name] = observations
            for obs_name, obs in group.funcs.items():
                name = f"obs_noise.{group_name}.{obs_name}"
                if name in adapter.context_slices:
                    self._noise.append((name, type(obs).__name__, float(getattr(obs, "noise_std", 0.)),
                                        float(getattr(obs, "episodic_noise_std", 0.))))
        self.reference_context_names = tuple(
            name for name in adapter.context_slices
            if name.startswith(("pre.ref_", "command.ref_")) or name == "command.t")
        self.reference_observation_groups = tuple(self._reference_observations)
        self.frozen_noise_fields = tuple(name for name, _, _, episodic_std in self._noise if episodic_std)

    def _references(self, motion_ids, base_time):
        """The command after update entered with t=base_time (before increment)."""
        start = self.motion_starts[motion_ids].unsqueeze(-1)
        end = self.motion_ends[motion_ids].unsqueeze(-1) - 1
        indices = torch.minimum(start + base_time.unsqueeze(-1) + self.future_steps, end)
        body_pos = self.motion_body_pos_w[indices]
        body_quat = self.motion_body_quat_w[indices]
        body_lin = self.motion_body_lin_vel_w[indices]
        body_ang = self.motion_body_ang_vel_w[indices]
        joint_pos = self.motion_joint_pos[indices]
        joint_vel = self.motion_joint_vel[indices]
        future = {
            "ref_body_pos_future_w": body_pos.index_select(-2, self.tracking_body_ids),
            "ref_body_quat_future_w": body_quat.index_select(-2, self.tracking_body_ids),
            "ref_body_lin_vel_future_w": body_lin.index_select(-2, self.tracking_body_ids),
            "ref_body_ang_vel_future_w": body_ang.index_select(-2, self.tracking_body_ids),
            "ref_root_pos_future_w": body_pos[..., self.root_body_id, :],
            "ref_root_quat_future_w": body_quat[..., self.root_body_id, :],
            "ref_root_lin_vel_future_w": body_lin[..., self.root_body_id, :],
            "ref_root_ang_vel_future_w": body_ang[..., self.root_body_id, :],
            "ref_object_pos_future_w": body_pos[..., self.object_body_id, :],
            "ref_object_quat_future_w": body_quat[..., self.object_body_id, :],
            "ref_joint_pos_future_": joint_pos.index_select(-1, self.tracking_joint_ids),
            "ref_joint_vel_future_": joint_vel.index_select(-1, self.tracking_joint_ids),
        }
        refs = dict(future)
        for name in ("body_pos", "body_quat", "body_lin_vel", "body_ang_vel", "root_pos", "root_quat",
                     "object_pos", "object_quat"):
            refs[f"ref_{name}_w"] = future[f"ref_{name}_future_w"].select(1, 0)
        refs["ref_joint_pos"] = future["ref_joint_pos_future_"][:, 0]
        refs["ref_joint_vel"] = future["ref_joint_vel_future_"][:, 0]
        refs["current_joint_pos"] = joint_pos[:, 0]
        # Contact indices deliberately use the incremented command t.
        contact_indices = torch.minimum(start + (base_time + 1).unsqueeze(-1) + self.future_steps, end)
        for name in ("body", "feet", "object"):
            value = getattr(self, "contact_" + name)[contact_indices]
            refs[f"ref_{name}_contact_future"] = value
            refs[f"ref_{name}_contact"] = value[:, 0]
        return refs

    def _reference_observation(self, kind, constants, before, after, next_time, motion_len):
        if kind == "ref_body_pos_future_local":
            root = before["ref_root_pos_w"]
            root = torch.cat((root[..., :2], torch.zeros_like(root[..., 2:])), -1)
            return quat_inverse_rotate(yaw_quaternion(before["ref_root_quat_w"])[:, None, None],
                                       before["ref_body_pos_future_w"] - root[:, None, None])
        if kind == "ref_joint_pos_future":
            return after["ref_joint_pos_future_"]
        if kind == "ref_motion_phase":
            return (next_time / motion_len).unsqueeze(-1)
        if kind == "ref_joint_pos_action_policy":
            indices, default, scale = (getattr(self, name) for name in constants)
            return (after["current_joint_pos"].index_select(-1, indices) - default) / scale
        raise AssertionError(kind)

    @torch.no_grad()
    def step(self, state, *, resample_noise=False):
        """Advance reference time once; no recorded next state/done is read."""
        batch_shape = state.shape[:-1]
        flat = state.reshape(-1, state.shape[-1])
        context = self.adapter._get_contexts(flat)
        motion_ids = context["command.motion_ids"].long().reshape(-1)
        time = context["command.t"].long().reshape(-1)
        motion_len = context["command.motion_len"].reshape(-1)
        before, after = self._references(motion_ids, time - 1), self._references(motion_ids, time)
        updates = {}
        for name in self.reference_context_names:
            if name == "command.t":
                value = time + 1
            elif name.startswith("pre."):
                value = before[name[4:]]
            else:
                value = after[name[8:]]
            updates[name] = value.to(flat.dtype).reshape_as(context[name])
        episode_length = context["episode_length"] + 1
        updates["episode_length"] = episode_length
        if resample_noise:
            for name, kind, std, episodic_std in self._noise:
                # Existing replay captures combined step+episodic noise. Keep
                # that realization fixed when it contains a latent episode
                # offset; do not silently resample the offset at each step.
                if episodic_std:
                    continue
                noise = torch.randn_like(context[name]).clamp(-3., 3.) * std
                if kind == "ref_contact_pos_b":
                    noise = noise + torch.randn_like(noise).clamp(-1., 1.) * std
                updates[name] = noise
        template = self.adapter.set_context(flat, updates)
        observations = dict(zip(self.adapter.codec.keys,
                                torch.split(template, self.adapter._observation_widths, -1)))
        for group_name, items in self._reference_observations.items():
            observations[group_name] = torch.cat([
                self._reference_observation(kind, constants, before, after, time + 1, motion_len)
                    .reshape(flat.shape[0], -1).to(flat.dtype)
                for kind, constants in items], -1)
        template = torch.cat([observations[key] for key in self.adapter.codec.keys], -1)
        truncated = ((time + 1 >= motion_len) |
                     (episode_length.reshape(-1) >= self.max_episode_length)).unsqueeze(-1)
        return template.reshape(*batch_shape, -1), truncated.reshape(*batch_shape, 1)
