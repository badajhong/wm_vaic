"""Imagined soft failure preserves consecutive grace and hard task equations."""

import itertools
from types import SimpleNamespace

import pytest
import torch

from active_adaptation.learning.wm.reward import VAICRewardModel
from active_adaptation.learning.wm.termination import SoftTerminationModel
from test_wm_reward import fixture, term


def counter_key(name):
    return "termination." + name + "._cum_error_mixin__cum_steps"


def cohort_fixture(horizon, limits, counts):
    terms = [SimpleNamespace(name=str(i), kind="test", threshold=1., min_steps=limit)
             for i, limit in enumerate(limits)]
    reward = SimpleNamespace(termination_terms=terms, _decode_context=lambda value: value,
                             _get=lambda state, name: state[name], context_shapes={})
    model = SoftTerminationModel(reward, horizon)
    state = {counter_key(str(i)): counts[..., i] for i in range(len(limits))}
    return model, model.initialize(state)


@pytest.mark.parametrize("limit", [1, 2, 3, 5000])
def test_binary_cohorts_reproduce_every_short_hard_violation_sequence(limit):
    horizon = 5
    patterns = torch.tensor(list(itertools.product([0., 1.], repeat=horizon)), dtype=torch.double)
    initial = torch.tensor([0, max(0, limit - 2), limit - 1], dtype=torch.double)
    probabilities = patterns[:, None, :].expand(-1, len(initial), -1).reshape(-1, horizon)
    initial = initial.expand(len(patterns), -1).reshape(-1)
    model, carry = cohort_fixture(horizon, [limit], initial[:, None])
    count, alive = initial.clone(), torch.ones_like(initial, dtype=torch.bool)
    for step in range(horizon):
        p = probabilities[:, step]
        count = torch.where(p.bool(), count + 1, 0.)
        expected = alive & (count >= limit)
        hazard, carry = model.step_probabilities(p[:, None], carry)
        torch.testing.assert_close(hazard[:, 0], expected.double(), rtol=0, atol=0)
        alive &= ~expected
        assert torch.isfinite(carry.cohorts).all()
    assert carry.cohorts.shape[-1] == horizon + 1
    with pytest.raises(ValueError, match="horizon"):
        model.step_probabilities(probabilities[:, -1:], carry)


def enumerated_survival(probabilities, limit, initial_count):
    survival = probabilities.new_zeros(())
    for sequence in itertools.product([0, 1], repeat=len(probabilities)):
        count, failed, weight = initial_count, False, probabilities.new_ones(())
        for p, violated in zip(probabilities, sequence):
            weight = weight * (p if violated else 1 - p)
            count = count + 1 if violated else 0
            failed |= count >= limit
        if not failed:
            survival = survival + weight
    return survival


@pytest.mark.parametrize("limit,initial", [(1, 0), (2, 0), (3, 1), (5000, 4998)])
def test_soft_cohort_survival_and_gradients_match_exhaustive_bernoulli_enumeration(limit, initial):
    probabilities = torch.tensor([.23, .67, .41, .84], dtype=torch.double, requires_grad=True)
    model, carry = cohort_fixture(4, [limit], probabilities.new_tensor([[initial]]))
    survival = probabilities.new_ones(())
    for p in probabilities:
        hazard, carry = model.step_probabilities(p.reshape(1, 1), carry)
        survival = survival * (1 - hazard.squeeze())
    expected = enumerated_survival(probabilities, limit, initial)
    torch.testing.assert_close(survival, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(survival, probabilities, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, probabilities)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)


def test_different_terms_combine_as_conditional_independent_survival():
    probabilities = torch.tensor([[.2, .7], [.6, .4], [.9, .3], [.1, .8]], dtype=torch.double)
    model, carry = cohort_fixture(4, [2, 3], torch.tensor([[0., 1.]], dtype=torch.double))
    survival = torch.ones((), dtype=torch.double)
    for p in probabilities:
        hazard, carry = model.step_probabilities(p[None], carry)
        survival *= 1 - hazard.squeeze()
    expected = enumerated_survival(probabilities[:, 0], 2, 0) * enumerated_survival(probabilities[:, 1], 3, 1)
    torch.testing.assert_close(survival, expected, rtol=1e-12, atol=1e-12)


def test_exact_grace_has_zero_hazard_and_gradient_before_failure_is_possible():
    probabilities = torch.full((16, 2), .8, dtype=torch.double, requires_grad=True)
    model, carry = cohort_fixture(16, [25, 5000], torch.zeros(1, 2, dtype=torch.double))
    hazards = []
    for p in probabilities:
        hazard, carry = model.step_probabilities(p[None], carry)
        hazards.append(hazard)
    total = torch.stack(hazards).sum()
    assert total.item() == 0
    assert torch.count_nonzero(torch.autograd.grad(total, probabilities)[0]) == 0
    assert carry.cohorts.numel() == 2 * 17  # Independent of the 5000-step grace.


def test_conditional_cohorts_stay_finite_after_near_certain_or_complete_absorption():
    probabilities = torch.tensor([[1.], [.999999], [0.], [.5]], requires_grad=True)
    model, carry = cohort_fixture(4, [1], torch.zeros(1, 1))
    hazards = []
    for p in probabilities:
        hazard, carry = model.step_probabilities(p[None], carry)
        hazards.append(hazard)
        assert torch.isfinite(carry.cohorts).all()
    values = torch.stack(hazards).flatten()
    torch.testing.assert_close(values, torch.tensor([1., 0., 0., 0.]), rtol=0, atol=0)
    assert torch.isfinite(torch.autograd.grad(values.sum(), probabilities)[0]).all()


def real_reward_fixture(kind, *, threshold=.5, min_steps=3, **kwargs):
    _, state, nxt, action, _, env = fixture("survival")
    env.termination_funcs[kind] = term(kind, threshold=threshold, min_steps=min_steps, **kwargs)
    state[counter_key(kind)] = torch.full((3,), min_steps - 1., dtype=torch.double)
    reward = VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None)
    return reward, state, nxt, action


def test_soft_object_boundary_has_continuous_gradient_without_changing_hard_equations():
    reward, state, nxt, action = real_reward_fixture("cum_object_pos_error")
    target = nxt["pre.ref_object_pos_w"]
    offset = torch.tensor([[.49, 0, 0], [.5, 0, 0], [.51, 0, 0]], dtype=torch.double, requires_grad=True)
    nxt["object.root_link_pos_w"] = target + offset
    original_state = {key: value.clone() for key, value in state.items()}
    hard_before = reward(state, nxt, action)
    model = SoftTerminationModel(reward, 2)
    hazard, carry = model.step(state, nxt, model.initialize(state))
    torch.testing.assert_close(hazard[:, 0], torch.sigmoid((offset[:, 0] / .5 - 1) / .05))
    gradient = torch.autograd.grad(hazard.sum(), offset)[0]
    assert (gradient[:, 0] > 0).all() and torch.isfinite(gradient).all()
    hard_after = reward(state, nxt, action)
    for actual, expected in zip(hard_after, hard_before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name in state:
        torch.testing.assert_close(state[name], original_state[name], rtol=0, atol=0)
    assert not carry.initial_counts.requires_grad


@pytest.mark.parametrize("vector_threshold", [False, True])
def test_contact_surrogate_uses_continuous_distance_and_force_with_original_cache_timing(vector_threshold):
    force_threshold = torch.tensor([1., 1., 0.], dtype=torch.double) if vector_threshold else 1.
    reward, state, nxt, _ = real_reward_fixture("cum_lost_contact_steps", threshold=1.,
                                                pos_thres=.2, frc_thres=force_threshold)
    distance = torch.tensor([.19, .2, .21], dtype=torch.double, requires_grad=True)
    force = torch.tensor([.9, 1., 1.1], dtype=torch.double, requires_grad=True)
    zeros = torch.zeros_like(distance)
    state["command.contact_target_pos_w"] = torch.zeros(3, 2, 3, dtype=torch.double)
    state["command.contact_eef_pos_w"] = torch.stack((distance, zeros, zeros), -1)[:, None].expand(-1, 2, -1)
    components = (force, force, zeros) if vector_threshold else (force, zeros, zeros)
    state["command.eef_contact_forces_b"] = torch.stack(components, -1)[:, None].expand(-1, 2, -1)
    nxt["pre.ref_body_contact"] = torch.tensor([[True, False]]).expand(3, -1)
    model = SoftTerminationModel(reward, 2)
    carry = model.initialize(state)
    hazard, _ = model.step(state, nxt, carry)
    distance_grad, force_grad = torch.autograd.grad(hazard.sum(), (distance, force), retain_graph=True)
    assert (distance_grad > 0).all() and (force_grad < 0).all()
    assert torch.isfinite(distance_grad).all() and torch.isfinite(force_grad).all()
    wrong_next = dict(nxt)
    wrong_next["command.contact_eef_pos_w"] = torch.full((3, 2, 3), 100., dtype=torch.double)
    wrong_next["command.eef_contact_forces_b"] = torch.zeros(3, 2, 3, dtype=torch.double)
    torch.testing.assert_close(model.step(state, wrong_next, carry)[0], hazard, rtol=0, atol=0)
    inactive = {"ref_body_contact": torch.zeros(3, 2, dtype=torch.bool)}
    torch.testing.assert_close(model.step(state, nxt, carry, inactive)[0], torch.zeros_like(hazard), rtol=0, atol=0)


def test_contact_unreachable_threshold_and_zero_force_threshold_follow_original_support():
    reward, state, nxt, _ = real_reward_fixture("cum_lost_contact_steps", threshold=3., pos_thres=.2, frc_thres=0.)
    model = SoftTerminationModel(reward, 1)
    hazard, _ = model.step(state, nxt, model.initialize(state))
    assert torch.count_nonzero(hazard) == 0
    reward.termination_terms[0].threshold = 0.
    model = SoftTerminationModel(reward, 1)
    hazard, _ = model.step(state, nxt, model.initialize(state))
    torch.testing.assert_close(hazard, torch.ones_like(hazard), rtol=0, atol=0)


def test_no_termination_terms_give_zero_hazard_and_no_random_draws():
    reward, state, nxt, _, _, _ = fixture("survival")
    model = SoftTerminationModel(reward, 3)
    rng = torch.random.get_rng_state().clone()
    carry = model.initialize(state)
    hazard, carry = model.step(state, nxt, carry)
    assert hazard.shape == (3, 1) and carry.cohorts.shape == (3, 0, 4)
    assert torch.count_nonzero(hazard) == 0
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("vector_threshold", [False, True])
def test_position_only_contact_ignores_predicted_force(vector_threshold):
    force_threshold = torch.tensor([1., 1., 0.], dtype=torch.double) if vector_threshold else 1.
    reward, state, nxt, _ = real_reward_fixture("cum_lost_contact_steps", threshold=1.,
                                                pos_thres=.2, frc_thres=force_threshold)
    distance = torch.tensor([.1, .2, .3], dtype=torch.double, requires_grad=True)
    zeros = torch.zeros_like(distance)
    state["command.contact_target_pos_w"] = torch.zeros(3, 2, 3, dtype=torch.double)
    state["command.contact_eef_pos_w"] = torch.stack((distance, zeros, zeros), -1)[:, None].expand(-1, 2, -1)
    force = torch.zeros(3, 2, 3, dtype=torch.double, requires_grad=True)
    state["command.eef_contact_forces_b"] = force
    nxt["pre.ref_body_contact"] = torch.tensor([[True, False]]).expand(3, -1)
    model = SoftTerminationModel(reward, 2, contact_force=False)
    hazard, _ = model.step(state, nxt, model.initialize(state))
    # Zero predicted force no longer reads as lost contact for a planted foot.
    expected = torch.sigmoid(-(1 - distance.detach() / .2) / model.softness)
    torch.testing.assert_close(hazard[:, 0], expected)
    assert hazard[0, 0] < 1e-3 and hazard[2, 0] > .99
    distance_grad, force_grad = torch.autograd.grad(hazard.sum(), (distance, force), allow_unused=True)
    assert (distance_grad > 0).all()
    assert force_grad is None or torch.count_nonzero(force_grad) == 0
    forceful = dict(state)
    forceful["command.eef_contact_forces_b"] = torch.full((3, 2, 3), 50., dtype=torch.double)
    torch.testing.assert_close(model.step(forceful, nxt, model.initialize(state))[0], hazard)
    with pytest.raises(ValueError, match="bool"):
        SoftTerminationModel(reward, 2, contact_force=1)


@pytest.mark.parametrize("frc_thres", [0., 0, -1.])
def test_nonpositive_scalar_force_threshold_is_position_only_and_never_reads_force(frc_thres):
    reward, state, nxt, _ = real_reward_fixture("cum_lost_contact_steps", threshold=1., pos_thres=.2,
                                                frc_thres=frc_thres)
    distance = torch.tensor([.19, .2, .21], dtype=torch.double)
    zeros = torch.zeros_like(distance)
    state["command.contact_target_pos_w"] = torch.zeros(3, 2, 3, dtype=torch.double)
    state["command.contact_eef_pos_w"] = torch.stack((distance, zeros, zeros), -1)[:, None].expand(-1, 2, -1)
    state["command.eef_contact_forces_b"] = torch.zeros(3, 2, 3, dtype=torch.double)
    nxt["pre.ref_body_contact"] = torch.tensor([[True, False]]).expand(3, -1)
    def forbidden(*args, **kwargs):
        raise AssertionError("A nonpositive scalar threshold must not read contact force")
    reward._contact_force = forbidden
    errors = reward.termination_errors(state, nxt, nxt)
    # Zero force still counts as in contact: only the 0.2 m position rule decides.
    torch.testing.assert_close(errors["cum_lost_contact_steps"],
                               torch.tensor([0., 2., 2.], dtype=torch.double))
