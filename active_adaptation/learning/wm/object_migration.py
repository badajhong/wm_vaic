"""Explicit articulated-object -> root-only physical-model warm start.

The WM checkpoint loader must first verify exact observation and context schemas.
This helper handles physical weights only. It is deliberately not an optimizer
migration: the caller must reset the world-model optimizer, while retaining the
teacher/critic modules and their independent optimizers.
"""

from math import prod

import torch


LEGACY_SCHEMA = "physical_causal_contact_substeps_v2"
ROOT_SCHEMA = "physical_causal_contact_substeps_root_v1"
REMOVED_INPUTS = frozenset(("object.joint_pos", "object.joint_vel"))
REMOVED_AUXILIARIES = frozenset(("object.applied_torque", "object.custom_torques"))


def _sections(keys, shapes):
    result, offset = {}, 0
    for key in keys:
        if key not in shapes:
            raise ValueError(f"Object-root warm start has unknown context field {key!r}")
        width = prod(shapes[key])
        result[key] = slice(offset, offset + width)
        offset += width
    return result, offset


def _retained_columns(source_keys, target_keys, shapes):
    sections, width = _sections(source_keys, shapes)
    if len(set(source_keys)) != len(source_keys) or len(set(target_keys)) != len(target_keys):
        raise ValueError("Object-root warm start requires unique context field names")
    columns = []
    for key in target_keys:
        if key not in sections:
            raise ValueError(f"Object-root warm start cannot initialize new field {key!r}")
        columns.extend(range(sections[key].start, sections[key].stop))
    return columns, width


def _select(value, dimension, indices):
    return value.index_select(dimension, torch.tensor(indices, dtype=torch.long, device=value.device))


def migrate_object_root_physics(source_state, source_schema, target_physics):
    """Return ``(fully_initialized_state_dict, metadata)`` or raise on mismatch.

    Preconditions: the caller checked the outer checkpoint's ``wm_schema`` and
    ``wm_context_schema`` for exact equality. This function additionally checks
    physical contracts, absolute packed-field indices, kinematic geometry and all
    tensor shapes. Every trainable target element comes from an explicit source
    element. Target-generated index buffers retain the target representation.

    Removing input columns is a warm start, not function equivalence for arbitrary
    old wheel states: it corresponds to zero *normalized* removed features.
    """
    target_schema = target_physics.schema
    if (not isinstance(source_schema, (tuple, list)) or len(source_schema) != 8
            or source_schema[0] != LEGACY_SCHEMA):
        raise ValueError("Object-root warm start requires an articulated physical v2 source checkpoint")
    if len(target_schema) != 8 or target_schema[0] != ROOT_SCHEMA:
        raise ValueError("Object-root warm start requires an explicitly root-only target model")
    if source_schema[2:] != target_schema[2:]:
        raise ValueError("Object-root warm start action, contact, timestep or kinematic schema mismatch")
    shapes = target_physics.context_shapes
    source_inputs = tuple(source_schema[1])
    expected_inputs = tuple(key for key in source_inputs if key not in REMOVED_INPUTS)
    if tuple(target_physics.input_keys) != expected_inputs or tuple(target_schema[1]) != expected_inputs:
        raise ValueError("Object-root warm start input schema differs beyond object joint removal")
    input_columns, source_input_width = _retained_columns(source_inputs, expected_inputs, shapes)

    # Old v2 checkpoints did not serialize auxiliary names in their schema.
    # Reconstruct the exact old selection from the already-matched context.
    exogenous = set(target_physics.exogenous_context_keys)
    source_auxiliary = tuple(key for key in shapes if (
        key.startswith("contact.") or key in ("command.eef_contact_forces_w", "command.eef2_contact_forces_w")
        or key.endswith((".applied_torque", ".custom_torques"))
    ) and key not in exogenous and key not in ("contact.timestamp", "contact.timestamp_last_update"))
    expected_auxiliary = tuple(key for key in source_auxiliary if key not in REMOVED_AUXILIARIES)
    if tuple(target_physics.auxiliary_keys) != expected_auxiliary:
        raise ValueError("Object-root warm start auxiliary schema differs beyond object torque removal")
    auxiliary_columns, source_auxiliary_width = _retained_columns(source_auxiliary, expected_auxiliary, shapes)
    source_learned = tuple(key for key in source_auxiliary if not key.startswith("contact."))
    target_learned = tuple(key for key in expected_auxiliary if not key.startswith("contact."))
    learned_rows, source_learned_width = _retained_columns(source_learned, target_learned, shapes)

    target_state = target_physics.state_dict()
    missing = set(target_state) - set(source_state)
    extra = set(source_state) - set(target_state)
    allowed_extra = {"auxiliary_head.weight", "auxiliary_head.bias"} if not target_learned else set()
    if missing or extra - allowed_extra:
        raise ValueError(f"Object-root warm start state keys mismatch: missing={sorted(missing)}, extra={sorted(extra - allowed_extra)}")
    for label, keys in (("input", source_inputs), ("auxiliary", source_auxiliary)):
        expected_indices = [index for key in keys for index in range(
            target_physics.context_slices[key].start, target_physics.context_slices[key].stop)]
        actual = source_state[f"{label}_indices"]
        if actual.tolist() != expected_indices:
            raise ValueError(f"Object-root warm start {label} packed context indices mismatch")
    old_auxiliary_sections, _ = _sections(source_auxiliary, shapes)
    expected_learned_indices = [index for key in source_learned for index in range(
        old_auxiliary_sections[key].start, old_auxiliary_sections[key].stop)]
    if source_state["learned_auxiliary_indices"].tolist() != expected_learned_indices:
        raise ValueError("Object-root warm start learned auxiliary indices mismatch")
    if source_state["state_stats.mean"].numel() != source_input_width:
        raise ValueError("Object-root warm start source input moment width mismatch")
    if source_state["auxiliary_stats.mean"].numel() != max(1, source_auxiliary_width):
        raise ValueError("Object-root warm start source auxiliary moment width mismatch")

    velocity_rows, source_velocity_width = [], 0
    for prefix in target_physics.prefixes:
        old_width = 6 + prod(shapes.get(f"{prefix}.joint_pos", (0,)))
        section = target_physics.velocity_slices[prefix]
        new_width = section.stop - section.start
        if new_width != (6 if prefix == "object" else old_width):
            raise ValueError(f"Object-root warm start unexpected {prefix} velocity layout")
        velocity_rows.extend(range(source_velocity_width, source_velocity_width + new_width))
        source_velocity_width += old_width
    if source_state["velocity_head.bias"].numel() != source_velocity_width:
        raise ValueError("Object-root warm start source velocity width mismatch")

    derived = {"input_indices", "auxiliary_indices", "learned_auxiliary_indices", "velocity_scale"}
    projected = {}
    for name, destination in target_state.items():
        value = source_state[name]
        if name in ("state_stats.mean", "state_stats.var"):
            value = _select(value, 0, input_columns)
        elif name in ("auxiliary_stats.mean", "auxiliary_stats.var"):
            if not auxiliary_columns:
                raise ValueError("Object-root warm start with no remaining auxiliary channels is unsupported")
            value = _select(value, 0, auxiliary_columns)
        elif name == "trunk.0.weight":
            control_width = target_physics.decimation * (target_physics.action_dim + 1)
            if value.shape[1] != source_input_width + control_width:
                raise ValueError("Object-root warm start first-layer input/control width mismatch")
            value = _select(value, 1, input_columns + list(range(source_input_width, value.shape[1])))
        elif name in ("velocity_head.weight", "velocity_head.bias"):
            value = _select(value, 0, velocity_rows)
        elif name in ("auxiliary_head.weight", "auxiliary_head.bias"):
            if value.shape[0] != source_learned_width:
                raise ValueError("Object-root warm start learned auxiliary head width mismatch")
            value = _select(value, 0, learned_rows)
        elif name in derived:
            if name == "velocity_scale":
                expected = _select(value, 0, velocity_rows).to(destination.device)
                if not torch.equal(expected, destination):
                    raise ValueError("Object-root warm start velocity units mismatch")
            value = destination
        if value.shape != destination.shape or value.dtype != destination.dtype:
            raise ValueError(f"Object-root warm start tensor mismatch for {name}: {tuple(value.shape)} vs {tuple(destination.shape)}")
        if (name.startswith("kinematics.") or name in ("action_prefix_mask", "action_substep_onehot")):
            if not torch.equal(value.to(destination.device), destination):
                raise ValueError(f"Object-root warm start immutable geometry/action buffer mismatch: {name}")
        projected[name] = value.detach().to(destination.device).clone()
    metadata = {"source_schema": source_schema[0], "target_schema": target_schema[0],
                "input_features_before": source_input_width, "input_features_after": len(input_columns),
                "velocity_outputs_before": source_velocity_width, "velocity_outputs_after": len(velocity_rows),
                "auxiliary_channels_before": source_auxiliary_width, "auxiliary_channels_after": len(auxiliary_columns),
                "removed_input_fields": sorted(set(source_inputs) - set(expected_inputs)),
                "removed_auxiliary_fields": sorted(set(source_auxiliary) - set(expected_auxiliary)),
                "world_model_optimizer_reset_required": True,
                "initialization": "All target parameters copied or projected from source; no random parameter initialization."}
    return projected, metadata
