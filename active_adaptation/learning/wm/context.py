"""Training-only simulator context and observation snapshots for world models.

The teacher policy still receives its original observation groups.  This transform
must precede VecNorm: its nested copies are raw data, in a fixed coordinate system,
for fitting a dynamics model independently of changing normalization statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence

import torch
from tensordict import TensorDictBase
from torchrl.data import Unbounded
from torchrl.envs.transforms import Transform


@dataclass(frozen=True)
class _ContextField:
    name: str
    owner: object
    attribute: str
    shape: tuple[int, ...]
    relative_position: bool = False
    static_in_episode: bool = False
    exogenous: bool = False


class RawWorldModelObservation(Transform):
    """Copy raw observations and append a simulator-only context vector.

    Context fields are selected once, with an explicit allowlist.  Changes in the
    schema during a run fail rather than silently changing the model's inputs.
    Physics properties randomized at startup are read once from PhysX and cached;
    episode-varying action delay, filtering and wheel friction remain live fields.
    No simulator state is changed, and none of these fields are actor/critic inputs.
    """

    raw_key = "wm_observation"
    context_key = "wm_context"

    def __init__(self, base_env, observation_keys: Sequence[str]):
        keys = list(observation_keys)
        if not keys or any(not isinstance(key, str) for key in keys):
            raise ValueError("World-model raw observation keys must be nonempty top-level strings.")
        if len(set(keys)) != len(keys) or self.context_key in keys or self.raw_key in keys:
            raise ValueError("World-model raw observation keys must be unique and exclude reserved keys.")
        super().__init__(in_keys=keys, out_keys=[(self.raw_key, key) for key in keys])
        # An EnvBase is an nn.Module: do not register it as a transform child.
        object.__setattr__(self, "_base_env", base_env)
        self._num_envs = int(base_env.num_envs)
        self._context_device = torch.device(base_env.device)
        self._fields: list[_ContextField] = []
        self._static_context: dict[str, torch.Tensor] = {}
        self._pre_command_sources = []
        self._observation_noise_sources = []

        for key in keys:
            spec = base_env.observation_spec.get(key, None)
            if spec is None or not spec.dtype.is_floating_point:
                raise ValueError(f"Raw world-model observation {key!r} must have a floating-point spec.")

        scene = base_env.scene
        origins = getattr(scene, "env_origins", None)
        if not isinstance(origins, torch.Tensor) or origins.shape != (self._num_envs, 3):
            raise ValueError("World-model context requires scene.env_origins with shape [num_envs, 3].")

        command = base_env.command_manager
        robot = getattr(command, "asset", None)
        if robot is None:
            raise ValueError("World-model context requires command_manager.asset (the robot).")
        self._add_asset("robot", robot, require_joints=True)
        obj = getattr(command, "object", None)
        if obj is None:
            raise ValueError("Object-manipulation world-model context requires command_manager.object.")
        articulated = getattr(command, "object_joint_idx_asset", None) is not None
        articulated |= getattr(command, "object_asset_name", None) == "skateboard"
        self._add_asset("object", obj, require_joints=articulated)
        obj2 = getattr(command, "object2", None)
        if obj2 is not None:
            self._add_asset("object2", obj2,
                            require_joints=getattr(command, "object2_joint_idx_asset", None) is not None)

        for name in ("t", "ref_body_contact", "ref_feet_contact", "ref_object_contact"):
            self._add_field(f"command.{name}", command, name, exogenous=True)
        for name in ("motion_ids", "motion_len", "contact_target_pos_offset", "contact_eef_pos_offset"):
            self._add_field(f"command.{name}", command, name, static_in_episode=True)
        for name in ("contact_target_pos_w", "contact_eef_pos_w"):
            self._add_field(f"command.{name}", command, name, relative_position=True)
        self._add_field("command.eef_contact_forces_w", command, "eef_contact_forces_w", required=True)
        self._add_field("command.eef_contact_forces_b", command, "eef_contact_forces_b")
        self._add_field("command.is_standing_env", command, "is_standing_env", exogenous=True)
        if obj2 is not None:
            for name in ("eef2_contact_forces_w", "ref_body2_contact", "ref_object2_contact"):
                self._add_field(f"command.{name}", command, name)
            for name in ("contact2_target_pos_w", "contact2_eef_pos_w"):
                self._add_field(f"command.{name}", command, name, relative_position=True)
            for name in ("contact2_target_pos_offset", "contact2_eef_pos_offset"):
                self._add_field(f"command.{name}", command, name, static_in_episode=True)

        # Only the known robot contact sensor is read; no camera/ray data or
        # arbitrary scene properties are enumerated.
        contact = getattr(scene, "sensors", {}).get("contact_forces")
        if contact is not None:
            # History already contains the current force in its first slot.
            force_key = "net_forces_w_history" if getattr(contact.data, "net_forces_w_history", None) is not None else "net_forces_w"
            self._add_field(f"contact.{force_key}", contact.data, force_key)
            for name in ("current_air_time", "last_air_time",
                         "current_contact_time", "last_contact_time"):
                self._add_field(f"contact.{name}", contact.data, name)
            self._add_contact_substeps(contact)

        action = base_env.action_manager
        for name in ("action_buf", "applied_action"):
            self._add_field(f"action.{name}", action, name, required=True)
        for name in ("alpha", "delay", "offset"):
            self._add_field(f"action.{name}", action, name, required=True, static_in_episode=True)
        self._add_field("episode_length", base_env, "episode_length_buf", required=True)

        # These private names belong to the repository's _cum_error_mixin.  Use
        # exact names rather than introspecting unrelated termination internals.
        for name, term in sorted(getattr(base_env, "termination_funcs", {}).items()):
            for field in ("error", "_cum_error_mixin__cum_steps", "_cum_error_mixin__exceeded"):
                self._add_field(f"termination.{name}.{field}", term, field)
        for group_name, group in sorted(getattr(base_env, "reward_groups", {}).items()):
            for reward_name, reward in sorted(group.funcs.items()):
                # Stateful clearance rewards store the last touchdown height.
                self._add_field(f"reward.{group_name}.{reward_name}.last_feet_contact_height",
                                reward, "last_feet_contact_height")
                self._add_field(f"reward.{group_name}.{reward_name}.joint_vel", reward, "joint_vel")
                if (type(reward).__name__ == "joint_vel_l2"
                        and len(reward.joint_ids) == command.asset.data.joint_pos.shape[-1]
                        and not any(f.name == "robot.joint_vel_substeps" for f in self._fields)):
                    self._add_field("robot.joint_vel_substeps", reward, "joint_vel")
                self._add_field(f"reward.{group_name}.{reward_name}.soft_limits", reward, "soft_limits",
                                static_in_episode=True)

        self._add_reconstruction_fields(command)

        self.context_schema = tuple((field.name, field.shape) for field in self._fields)
        self.context_slices = {}
        start = 0
        for field in self._fields:
            width = 1
            for size in field.shape:
                width *= size
            self.context_slices[field.name] = slice(start, start + width)
            start += width
        self.context_dim = start
        # Validate every chosen field immediately, before a large rollout buffer
        # is allocated, and retain the exact spec shape for TorchRL.
        self._context_shape = self.snapshot_context().shape

    @property
    def static_context_slices(self) -> dict[str, slice]:
        """Fields to copy unchanged through imagination, keyed like context_slices.

        This includes episode-constant live simulator tensors as well as cached
        startup physics.  Real resets still refresh their collected values.
        """
        return {field.name: self.context_slices[field.name]
                for field in self._fields if field.static_in_episode}

    @property
    def exogenous_context_slices(self) -> dict[str, slice]:
        """Reference trajectories and recorded sensor-noise draws, never physical targets."""
        return {field.name: self.context_slices[field.name]
                for field in self._fields if field.exogenous}

    def _add_field(self, name, owner, attribute, *, required=False, relative_position=False,
                   static_in_episode=False, exogenous=False):
        value = getattr(owner, attribute, None)
        if value is None and not required:
            return
        if not isinstance(value, torch.Tensor) or value.ndim < 1 or value.shape[0] != self._num_envs:
            raise ValueError(f"World-model context {name!r} requires a tensor with leading dimension "
                             f"{self._num_envs}; got {type(value).__name__} "
                             f"{getattr(value, 'shape', '')}.")
        if relative_position and value.shape[-1] != 3:
            raise ValueError(f"World-model position context {name!r} must end in three coordinates.")
        if value.numel() == 0:
            if required:
                raise ValueError(f"Required world-model context {name!r} is empty (missing object joints?).")
            return
        self._fields.append(_ContextField(name, owner, attribute, tuple(value.shape[1:]),
                                         relative_position, static_in_episode, exogenous))

    def _add_contact_substeps(self, sensor):
        """Observe already refreshed contact data after each physics substep.

        ContactSensor with history enabled updates in scene.update, before the
        existing post-step callbacks. Reading its private data here deliberately
        avoids the public property, which can trigger another sensor update.
        The float32 timestamps matter: adding nominal dt to a timer instead of
        subtracting sensor timestamps can change the original first-contact test.
        """
        if not hasattr(sensor, "_timestamp"):
            return  # Non-Isaac test fixtures; the physical model requires clocks.
        env = self._base_env
        data = sensor._data
        if int(sensor.cfg.history_length) < 1:
            raise ValueError("Physical WM contact capture requires per-substep sensor history updates")
        self._add_field("contact.timestamp", sensor, "_timestamp", required=True)
        self._add_field("contact.timestamp_last_update", sensor, "_timestamp_last_update", required=True)
        holder = SimpleNamespace(value=torch.zeros(
            (self._num_envs, int(env.decimation), *data.net_forces_w.shape[1:]),
            device=data.net_forces_w.device, dtype=data.net_forces_w.dtype))
        self._add_field("contact.net_forces_w_substeps", holder, "value", required=True)

        def capture(substep):
            holder.value[:, substep].copy_(sensor._data.net_forces_w)

        def reset(env_ids):
            holder.value[env_ids] = 0

        env._post_step_callbacks.append(capture)
        env._reset_callbacks.append(reset)

    def _add_reconstruction_fields(self, command):
        """Capture the exact two reference phases used by the existing env.

        Cached Observation.update and reward inputs use the command before its
        update; direct Observation.compute uses the updated command.  A WM-only
        callback records the former without changing simulator callback order.
        """
        future = ("ref_body_pos_future_w", "ref_body_quat_future_w", "ref_body_lin_vel_future_w",
                  "ref_body_ang_vel_future_w", "ref_root_pos_future_w", "ref_root_quat_future_w",
                  "ref_object_pos_future_w", "ref_object_quat_future_w")
        current = ("ref_body_pos_w", "ref_body_quat_w", "ref_body_lin_vel_w", "ref_body_ang_vel_w",
                   "ref_joint_pos", "ref_joint_vel", "ref_root_pos_w", "ref_root_quat_w",
                   "ref_object_pos_w", "ref_object_quat_w", "ref_body_contact", "ref_feet_contact",
                   "ref_object_contact")
        physical = ("robot_root_pos_w", "robot_root_quat_w", "robot_body_pos_w", "robot_body_quat_w",
                    "robot_body_lin_vel_w", "robot_body_ang_vel_w", "robot_joint_pos", "robot_joint_vel",
                    "object_pos_w", "object_quat_w", "contact_target_pos_w", "contact_eef_pos_w",
                    "eef_contact_forces_w", "eef_contact_forces_b")
        positions = {name for name in future + current + physical if "pos" in name and "joint" not in name}
        for name in future + current + physical:
            value = getattr(command, name, None)
            if not isinstance(value, torch.Tensor):
                continue
            holder = SimpleNamespace(value=value.detach().clone())
            self._pre_command_sources.append((holder, command, name))
            self._add_field(f"pre.{name}", holder, "value", relative_position=name in positions,
                            exogenous=name.startswith("ref_"))
        for name in current + physical + ("ref_root_lin_vel_future_w", "ref_root_ang_vel_future_w",
                                            "ref_object_contact_future"):
            if f"command.{name}" in {f.name for f in self._fields}:
                continue
            self._add_field(f"command.{name}", command, name, relative_position=name in positions,
                            exogenous=name.startswith("ref_"))
        # Complete sensor histories are carried through imagination. Only their
        # newly sampled noise is exogenous, never factual next physical history.
        for group_name, group in getattr(self._base_env, "observation_funcs", {}).items():
            for obs_name, obs in group.funcs.items():
                kind = type(obs).__name__
                if kind in ("root_ang_vel_history", "projected_gravity_history", "joint_pos_history"):
                    self._add_field(f"obs_history.{group_name}.{obs_name}", obs, "buffer")
                    if kind == "joint_pos_history" and not any(f.name == "robot.joint_pos_substeps" for f in self._fields):
                        if len(obs.joint_ids) != self._base_env.command_manager.asset.data.joint_pos.shape[-1]:
                            raise ValueError("Physical WM requires one full robot joint-position history.")
                        self._add_field("robot.joint_pos_substeps", obs, "joint_pos")
                if kind == "root_linvel_b":
                    self._add_field(f"obs_history.{group_name}.{obs_name}", obs.ema, "ema")
                if getattr(obs, "noise_std", 0.0) > 0 or getattr(obs, "episodic_noise_std", 0.0) > 0:
                    if kind in ("root_ang_vel_history", "projected_gravity_history", "joint_pos_history"):
                        value = torch.zeros_like(obs.buffer[:, 0])
                    elif kind == "object_xy_b":
                        value = torch.zeros_like(obs.object_xy_b)
                    elif kind == "object_heading_b":
                        value = torch.zeros_like(obs.object_yaw_b)
                    elif kind == "ref_contact_pos_b":
                        value = torch.zeros_like(obs.ref_contact_pos_b)
                    else:
                        raise ValueError(f"Physical WM does not support noisy observation {kind}.")
                    holder = SimpleNamespace(value=value)
                    self._observation_noise_sources.append((holder, obs, kind))
                    self._add_field(f"obs_noise.{group_name}.{obs_name}", holder, "value", exogenous=True)
        if hasattr(self._base_env, "_update_callbacks"):
            self._base_env._update_callbacks.append(self._capture_pre_command)

    def _capture_pre_command(self):
        for holder, command, name in self._pre_command_sources:
            holder.value = getattr(command, name).detach().clone()

    def _capture_observation_noise(self):
        from .adapter import quat_inverse_rotate, yaw_quaternion
        robot = self._base_env.command_manager.asset.data
        pre = {name: holder.value for holder, _, name in self._pre_command_sources}
        for holder, obs, kind in self._observation_noise_sources:
            if kind == "root_ang_vel_history":
                nominal = quat_inverse_rotate(robot.root_link_quat_w, robot.root_com_vel_w[..., 3:])
                holder.value = obs.buffer[:, 0] - nominal
            elif kind == "projected_gravity_history":
                down = torch.zeros_like(robot.root_link_pos_w)
                down[:, 2] = -1
                holder.value = obs.buffer[:, 0] - quat_inverse_rotate(robot.root_link_quat_w, down)
            elif kind == "joint_pos_history":
                holder.value = obs.buffer[:, 0] - obs.joint_pos.mean(1)
            elif kind in ("object_xy_b", "object_heading_b"):
                holder.value = obs.step_noise + obs.episodic_noise
            elif kind == "ref_contact_pos_b":
                quat = pre["robot_root_quat_w"][:, None]
                if obs.yaw_only:
                    quat = yaw_quaternion(quat)
                nominal = quat_inverse_rotate(quat, pre["contact_target_pos_w"] - pre["robot_root_pos_w"][:, None])
                holder.value = obs.ref_contact_pos_b - nominal

    def _add_asset(self, prefix, asset, *, require_joints):
        data = asset.data
        for name in ("root_link_pos_w", "root_link_quat_w", "root_com_vel_w",
                     "body_link_pos_w", "body_link_quat_w", "body_com_vel_w"):
            self._add_field(f"{prefix}.{name}", data, name, required=True,
                            relative_position=name.endswith("pos_w"))
        for name in ("joint_pos", "joint_vel"):
            self._add_field(f"{prefix}.{name}", data, name, required=require_joints)
        for name in ("applied_torque",):
            self._add_field(f"{prefix}.{name}", data, name)
        for name in ("joint_stiffness", "joint_damping", "joint_armature", "joint_friction_coeff"):
            self._add_field(f"{prefix}.{name}", data, name, static_in_episode=True)
        for name in ("_custom_friction", "_custom_damping"):
            self._add_field(f"{prefix}.{name}", asset, name, static_in_episode=True)
        self._add_field(f"{prefix}.custom_torques", asset, "custom_torques")
        spawn = getattr(getattr(asset, "cfg", None), "spawn", None)
        scale = getattr(spawn, "scale", None)
        if scale is None:
            scale = torch.ones(self._num_envs, 3, device=self._context_device)
        else:
            scale = torch.as_tensor(scale, device=self._context_device, dtype=torch.float32)
            scale = scale.expand(self._num_envs, 3).clone()
        self._add_field(f"{prefix}.kinematic_scale", _StaticTensor(scale), "value", static_in_episode=True)

        # All these properties are randomized only at startup in the baseline
        # skateboard task.  Capture sampled values, never default_* tensors.
        view = getattr(asset, "root_physx_view", None)
        for name, method in (("mass", "get_masses"), ("inertia", "get_inertias"),
                             ("com", "get_coms"), ("materials", "get_material_properties")):
            getter = getattr(view, method, None)
            if getter is not None:
                value = getter()
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"World-model {prefix} physics property {name!r} is not a tensor.")
                cached = value.detach().to(device=self._context_device, dtype=torch.float32).clone()
                key = f"{prefix}.physics.{name}"
                self._static_context[key] = cached
                # The immutable snapshot is held by a tiny dedicated container;
                # no PhysX getter is called again during collection.
                holder = _StaticTensor(cached)
                self._add_field(key, holder, "value", required=True, static_in_episode=True)

    def snapshot_context(self) -> torch.Tensor:
        """Return an owned [num_envs, context_dim] float32 snapshot without resets."""
        origins = self._base_env.scene.env_origins
        parts = []
        for field in self._fields:
            value = getattr(field.owner, field.attribute, None)
            expected = (self._num_envs, *field.shape)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise RuntimeError(f"World-model context schema changed at {field.name!r}: "
                                   f"expected {expected}, got {getattr(value, 'shape', None)}.")
            value = value.detach().to(device=self._context_device, dtype=torch.float32)
            if field.relative_position:
                value = value - origins.reshape(self._num_envs, *([1] * (value.ndim - 2)), 3)
            parts.append(value.reshape(self._num_envs, -1))
        return torch.cat(parts, dim=-1)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._capture_observation_noise()
        for key in self.in_keys:
            value = tensordict.get(key, None)
            if value is None or not value.is_floating_point():
                raise KeyError(f"Missing floating-point raw world-model observation {key!r}.")
            tensordict.set((self.raw_key, key), value.detach().clone())
        tensordict.set((self.raw_key, self.context_key), self.snapshot_context())
        return tensordict

    def _step(self, tensordict, next_tensordict):
        return self._call(next_tensordict)

    def _reset(self, tensordict, tensordict_reset):
        return self._call(tensordict_reset)

    def forward(self, tensordict):
        return self._call(tensordict)

    def transform_observation_spec(self, observation_spec):
        result = observation_spec.clone()
        for key in self.in_keys:
            result[self.raw_key, key] = observation_spec[key].clone()
        result[self.raw_key, self.context_key] = Unbounded(
            shape=self._context_shape, dtype=torch.float32, device=self._context_device,
        )
        return result


@dataclass(frozen=True)
class _StaticTensor:
    value: torch.Tensor
