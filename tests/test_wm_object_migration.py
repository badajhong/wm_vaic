"""Explicit root-object warm start projects every parameter without randomness."""

import copy

import pytest
import torch

from active_adaptation.learning.wm.object_migration import migrate_object_root_physics
from test_wm_physics import make_model


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def models():
    source, state = make_model(object_torques=True)
    target, _ = make_model(object_dynamics="root", object_torques=True)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.copy_(torch.sin(torch.arange(parameter.numel()).reshape(parameter.shape) * .013) * .03)
        for stats in (source.state_stats, source.action_stats, source.auxiliary_stats, source.contact_force_stats):
            stats.mean.copy_(torch.arange(stats.mean.numel()) * .003)
            stats.var.copy_(1. + torch.arange(stats.var.numel()) * .002)
            stats.count.fill_(123)
    return source, target, state


def test_projects_all_weights_moments_and_retains_target_indices_without_mutating_source():
    source, target, _ = models()
    before = copy.deepcopy(source.state_dict())
    original_target = copy.deepcopy(target.state_dict())
    result, metadata = migrate_object_root_physics(source.state_dict(), source.schema, target)
    assert metadata["velocity_outputs_before"] - metadata["velocity_outputs_after"] == 1
    assert metadata["input_features_before"] - metadata["input_features_after"] == 2
    assert metadata["auxiliary_channels_before"] - metadata["auxiliary_channels_after"] == 2
    assert metadata["world_model_optimizer_reset_required"]
    assert set(result) == set(original_target)
    target.load_state_dict(result, strict=True)
    source_index = {value: index for index, value in enumerate(source.input_indices.tolist())}
    kept = [source_index[value] for value in target.input_indices.tolist()]
    torch.testing.assert_close(target.state_stats.mean, source.state_stats.mean[kept], rtol=0, atol=0)
    torch.testing.assert_close(target.state_stats.var, source.state_stats.var[kept], rtol=0, atol=0)
    controls = list(range(source.state_stats.mean.numel(), source.trunk[0].weight.shape[1]))
    torch.testing.assert_close(target.trunk[0].weight, source.trunk[0].weight[:, kept + controls], rtol=0, atol=0)
    source_aux_index = {value: index for index, value in enumerate(source.auxiliary_indices.tolist())}
    kept_aux = [source_aux_index[value] for value in target.auxiliary_indices.tolist()]
    torch.testing.assert_close(target.auxiliary_stats.mean, source.auxiliary_stats.mean[kept_aux], rtol=0, atol=0)
    torch.testing.assert_close(target.auxiliary_stats.var, source.auxiliary_stats.var[kept_aux], rtol=0, atol=0)
    for name in ("input_indices", "auxiliary_indices", "learned_auxiliary_indices", "velocity_scale"):
        torch.testing.assert_close(result[name], original_target[name], rtol=0, atol=0)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    # Returning owned tensors makes later checkpoint mutation harmless.
    result["trunk.0.bias"].zero_()
    torch.testing.assert_close(source.trunk[0].bias, before["trunk.0.bias"], rtol=0, atol=0)


def test_projected_network_matches_original_at_zero_normalized_removed_inputs():
    source, target, state = models()
    result, _ = migrate_object_root_physics(source.state_dict(), source.schema, target)
    target.load_state_dict(result)
    offset, replacements = 0, {}
    for key in source.input_keys:
        width = source.context_slices[key].stop - source.context_slices[key].start
        if key in ("object.joint_pos", "object.joint_vel"):
            replacements[key] = source.state_stats.mean[offset:offset + width].expand(len(state), width)
        offset += width
    state = source.set_context(state, replacements)
    action = torch.full((len(state), 1), .12)
    before, source_features = source._causal_network(state, action, all_substeps=True)
    after, target_features = target._causal_network(state, action, all_substeps=True)
    torch.testing.assert_close(target_features, source_features, rtol=2e-5, atol=2e-6)
    keep_velocity = list(range(source.velocity_slices["robot"].stop)) + list(range(
        source.velocity_slices["object"].start, source.velocity_slices["object"].start + 6))
    torch.testing.assert_close(after, before[..., keep_velocity], rtol=2e-5, atol=2e-6)
    source_aux = source.auxiliary_head(source_features[..., -1, :])
    target_aux = target.auxiliary_head(target_features[..., -1, :])
    keep_rows = [index for key in target.learned_auxiliary_keys for index in range(
        source.learned_auxiliary_slices[key].start, source.learned_auxiliary_slices[key].stop)]
    torch.testing.assert_close(target_aux, source_aux[..., keep_rows], rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(target.contact_presence_head(target_features), source.contact_presence_head(source_features), rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("change", ["action", "kinematics", "indices", "learned_indices", "shape", "extra", "source_mode"])
def test_rejects_unrelated_checkpoint_changes(change):
    source, target, _ = models()
    state = copy.deepcopy(source.state_dict())
    schema = source.schema
    if change == "action":
        schema = (*schema[:2], schema[2] + 1, *schema[3:])
    elif change == "kinematics":
        state["kinematics.object.local_pos0"] += .1
    elif change == "indices":
        state["input_indices"][0] += 1
    elif change == "learned_indices":
        state["learned_auxiliary_indices"][0] += 1
    elif change == "shape":
        state["trunk.3.weight"] = state["trunk.3.weight"][:-1]
    elif change == "extra":
        state["unknown_parameter"] = torch.ones(2)
    elif change == "source_mode":
        schema = target.schema
    with pytest.raises(ValueError, match="warm start"):
        migrate_object_root_physics(state, schema, target)


def test_requires_root_target_and_rejects_new_random_input_fields():
    source, target, _ = models()
    with pytest.raises(ValueError, match="root-only target"):
        migrate_object_root_physics(source.state_dict(), source.schema, source)
    target.input_keys = (*target.input_keys, "object.joint_pos")
    with pytest.raises(ValueError, match="input schema"):
        migrate_object_root_physics(source.state_dict(), source.schema, target)
