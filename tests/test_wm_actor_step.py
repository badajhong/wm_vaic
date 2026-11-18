"""Independent checks of transactional, one-backward actor step control."""

import copy
import math

import pytest
import torch

from active_adaptation.learning.wm.actor_step import ActorStepController


def system(*, lr=.1, encoder_lr=.03, max_lr=None, **kwargs):
    actor = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64))
    encoder = torch.nn.Parameter(torch.tensor([.3, .4], dtype=torch.float64))
    height = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=torch.float64))
    parameters = [actor, encoder, height]
    optimizer = torch.optim.Adam([{"params": [p]} for p in parameters], lr=3e-4)
    controller = ActorStepController(optimizer, actor_lr=lr, encoder_lr=encoder_lr,
                                     max_actor_lr=max_lr or lr, **kwargs)
    inputs = torch.tensor([[1., 2.], [2., 1.], [.5, -1.]], dtype=torch.float64)

    def mean():
        return inputs * actor + encoder + height

    return parameters, optimizer, controller, mean


def set_gradients(parameters):
    for index, parameter in enumerate(parameters):
        parameter.grad = torch.tensor([.3 + index, -.7 - index], dtype=parameter.dtype)


def assert_tree_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_accepted_update_matches_adam_and_exact_masked_gaussian_kl():
    parameters, optimizer, controller, mean = system(lr=.001, encoder_lr=.0003)
    set_gradients(parameters)
    before = mean().detach().clone()
    std = torch.tensor([[.2, .6], [.7, .1], [.3, .8]], dtype=torch.float64)
    mask = torch.tensor([False, True, True])
    result = controller.step(mean, before, std, mask)
    change = (mean().detach() - before)[mask]
    expected_kl = (.5 * (change / std[mask]).square().sum(-1)).mean().item()
    assert result["actor/step_accepted"] == 1
    assert result["actor/backtracks"] == 0
    assert result["actor/update_mean_kl"] == pytest.approx(expected_kl)
    assert result["actor/action_delta_rms"] == pytest.approx(change.square().mean().sqrt().item())
    assert result["actor/lr"] == .001
    assert all(optimizer.state[p]["step"] == 1 for p in parameters)


def test_oversized_proposal_backtracks_all_groups_but_updates_adam_moments_once(monkeypatch):
    parameters, optimizer, controller, mean = system()
    set_gradients(parameters)
    old = [p.detach().clone() for p in parameters]
    before = mean().detach().clone()
    reference_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    reference_optimizer = torch.optim.Adam([
        {"params": [p], "lr": group["lr"]}
        for p, group in zip(reference_parameters, optimizer.param_groups)
    ])
    set_gradients(reference_parameters)
    reference_optimizer.step()
    calls = 0
    original_step = optimizer.step

    def counted_step():
        nonlocal calls
        calls += 1
        return original_step()

    monkeypatch.setattr(optimizer, "step", counted_step)
    result = controller.step(mean, before, torch.full_like(before, .05), torch.ones(3, dtype=torch.bool))
    assert calls == 1
    assert result["actor/proposed_mean_kl"] > controller.max_kl
    assert 0 < result["actor/step_scale"] < 1
    assert 0 < result["actor/update_mean_kl"] <= controller.max_kl
    assert result["actor/backtracks"] >= 1
    for p, start, proposal in zip(parameters, old, reference_parameters):
        torch.testing.assert_close(p, torch.lerp(start, proposal, result["actor/step_scale"]))
    for p, reference in zip(parameters, reference_parameters):
        assert_tree_equal(optimizer.state[p], reference_optimizer.state[reference])
    assert optimizer.param_groups[1]["lr"] / optimizer.param_groups[0]["lr"] == pytest.approx(.3)
    assert optimizer.param_groups[2]["lr"] == optimizer.param_groups[1]["lr"]


def test_nonlinear_output_is_rechecked_after_interpolation():
    parameters, optimizer, controller, base_mean = system(lr=1., encoder_lr=.3, max_backtracks=8)
    set_gradients(parameters)
    before_parameters = parameters[0].detach().clone()
    evaluations = []

    def mean():
        # A steep nonlinear mean makes a guessed quadratic scale insufficient.
        displacement = parameters[0] - before_parameters
        output = 100 * torch.sign(displacement) * torch.abs(displacement).sqrt()
        evaluations.append(output.detach().clone())
        return output.expand(3, -1)

    initial = mean().detach().clone()
    result = controller.step(mean, initial, torch.ones_like(initial), torch.ones(3, dtype=torch.bool))
    assert result["actor/backtracks"] > 1
    assert result["actor/step_accepted"] == 1
    assert result["actor/update_mean_kl"] <= controller.max_kl
    assert len(evaluations) == 2 + result["actor/backtracks"]


@pytest.mark.parametrize("initialized", [False, True])
def test_full_rejection_restores_parameters_and_complete_optimizer_state(initialized):
    parameters, optimizer, controller, mean = system(max_backtracks=3)
    set_gradients(parameters)
    if initialized:
        optimizer.step()
    old = [p.detach().clone() for p in parameters]
    old_state = {p: copy.deepcopy(value) for p, value in optimizer.state.items()}
    before = mean().detach().clone()
    result = controller.step(lambda: torch.full_like(before, float("nan")), before,
                             torch.ones_like(before), torch.ones(3, dtype=torch.bool))
    assert result["actor/step_accepted"] == 0
    assert result["actor/step_scale"] == 0
    assert result["actor/update_mean_kl"] == 0
    assert result["actor/lr"] == 0
    assert math.isnan(result["actor/proposed_mean_kl"])
    for p, start in zip(parameters, old):
        torch.testing.assert_close(p, start, rtol=0, atol=0)
    assert_tree_equal(dict(optimizer.state), old_state)
    assert controller.scale == .5


def test_callback_exception_is_transactional():
    parameters, optimizer, controller, mean = system()
    set_gradients(parameters)
    optimizer.step()
    old = [p.detach().clone() for p in parameters]
    old_state = copy.deepcopy(optimizer.state_dict())
    before = mean().detach().clone()

    def fail():
        raise RuntimeError("teacher failed")

    with pytest.raises(RuntimeError, match="teacher failed"):
        controller.step(fail, before, torch.ones_like(before), torch.ones(3, dtype=torch.bool))
    for p, start in zip(parameters, old):
        torch.testing.assert_close(p, start, rtol=0, atol=0)
    assert_tree_equal(optimizer.state_dict(), old_state)
    assert controller.scale == 1


def test_invalid_reset_rows_are_excluded_even_when_their_outputs_are_nan():
    parameters, optimizer, controller, original_mean = system(lr=.001, encoder_lr=.0003)
    set_gradients(parameters)

    def mean():
        result = original_mean().clone()
        result[0] = float("nan")
        return result

    before = mean().detach().clone()
    result = controller.step(mean, before, torch.ones_like(before), torch.tensor([[False], [True], [True]]))
    assert result["actor/step_accepted"] == 1
    assert math.isfinite(result["actor/update_mean_kl"])


def test_no_valid_starts_does_not_advance_adam_or_call_teacher():
    parameters, optimizer, controller, mean = system()
    set_gradients(parameters)
    before = mean().detach().clone()

    def forbidden():
        raise AssertionError("No valid starts must not update/evaluate the actor")

    result = controller.step(forbidden, before, torch.ones_like(before), torch.zeros(3, dtype=torch.bool))
    assert result["actor/step_accepted"] == 0
    assert not optimizer.state
    assert controller.scale == 1


def test_adaptive_growth_is_bounded_and_keeps_encoder_height_lr_ratio():
    parameters, optimizer, controller, mean = system(lr=1e-5, encoder_lr=3e-6, max_lr=2e-5)
    for _ in range(10):
        set_gradients(parameters)
        before = mean().detach().clone()
        result = controller.step(mean, before, torch.ones_like(before), torch.ones(3, dtype=torch.bool))
        assert result["actor/update_mean_kl"] <= .02
        assert optimizer.param_groups[0]["lr"] <= 2e-5
        assert optimizer.param_groups[1]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"] * .3)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-5)
    assert result["actor/lr"] == pytest.approx(2e-5)


def test_new_checkpoint_resume_preserves_next_step_and_legacy_load_overrides_unsafe_lr():
    parameters, optimizer, controller, mean = system()
    set_gradients(parameters)
    before = mean().detach().clone()
    controller.step(mean, before, torch.full_like(before, .05), torch.ones(3, dtype=torch.bool))
    saved_parameters = [p.detach().clone() for p in parameters]
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    saved_controller = controller.state_dict()
    p2, opt2, controller2, mean2 = system()
    with torch.no_grad():
        for p, value in zip(p2, saved_parameters):
            p.copy_(value)
    opt2.load_state_dict(saved_optimizer)
    controller2.load_state_dict(saved_controller)
    assert controller2.scale == controller.scale
    set_gradients(parameters)
    set_gradients(p2)
    r1 = controller.step(mean, mean().detach().clone(), torch.ones(3, 2), torch.ones(3, dtype=torch.bool))
    r2 = controller2.step(mean2, mean2().detach().clone(), torch.ones(3, 2), torch.ones(3, dtype=torch.bool))
    assert r1 == r2
    for p, other in zip(parameters, p2):
        torch.testing.assert_close(p, other, rtol=0, atol=0)
    assert_tree_equal(optimizer.state_dict(), opt2.state_dict())

    # Loading optimizer state first restores its old rates; migration then
    # overrides only rates, without dropping learned Adam moments or groups.
    for group in saved_optimizer["param_groups"]:
        group["lr"] = 3e-4
    opt2.load_state_dict(saved_optimizer)
    moments = {p: copy.deepcopy(value) for p, value in opt2.state.items()}
    safe = ActorStepController(opt2)
    safe.load_state_dict(None)
    assert [group["lr"] for group in opt2.param_groups] == [1.5e-5, 5e-6, 5e-6]
    assert_tree_equal(dict(opt2.state), moments)


@pytest.mark.parametrize("overrides", [
    {"actor_lr": 0}, {"encoder_lr": float("nan")},
    {"target_kl": .03}, {"max_actor_lr": 1e-6},
    {"min_scale": 0}, {"max_backtracks": -1},
])
def test_invalid_control_configuration_is_rejected(overrides):
    optimizer = torch.optim.Adam([{"params": [torch.nn.Parameter(torch.zeros(1))]} for _ in range(2)])
    with pytest.raises(ValueError):
        ActorStepController(optimizer, **overrides)


# --- SuperMimic-style fixed-rate actor step -------------------------------

from active_adaptation.learning.wm.actor_step import FixedRateActorStep


def fixed_system(*, lr=.1, beta2=.95, clip_value=.5, ratios=(1., .3, .3)):
    actor = torch.nn.Parameter(torch.tensor([.2, -.1], dtype=torch.float64))
    encoder = torch.nn.Parameter(torch.tensor([.3, .4], dtype=torch.float64))
    height = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=torch.float64))
    parameters = [actor, encoder, height]
    optimizer = torch.optim.Adam([{"params": [p]} for p in parameters], lr=3e-4)
    step = FixedRateActorStep(optimizer, lr=lr, beta2=beta2, clip_value=clip_value,
                              group_lr_ratios=ratios)
    inputs = torch.tensor([[1., 2.], [2., 1.], [.5, -1.]], dtype=torch.float64)

    def mean():
        return inputs * actor + encoder + height

    return parameters, optimizer, step, mean


def test_fixed_step_matches_clipped_adam_with_configured_rates_and_beta2():
    parameters, optimizer, step, mean = fixed_system()
    reference = [p.detach().clone().requires_grad_() for p in parameters]
    expected = torch.optim.Adam([{"params": [p], "lr": rate} for p, rate in zip(reference, (.1, .1 * .3, .1 * .3))],
                                betas=(.9, .95))
    set_gradients(parameters)
    set_gradients(reference)
    for parameter in reference:
        parameter.grad.clamp_(-.5, .5)
    initial = mean().detach()
    std = torch.full_like(initial, .2)
    valid = torch.tensor([True, False, True])
    metrics = step.step(mean, initial, std, valid)
    expected.step()
    for actual, wanted in zip(parameters, reference):
        torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([.1, .03, .03])
    assert all(group["betas"] == (.9, .95) for group in optimizer.param_groups)
    change = (mean() - initial)[valid]
    assert metrics["actor/step_accepted"] == 1 and metrics["actor/backtracks"] == 0
    assert metrics["actor/lr"] == .1
    assert metrics["actor/update_mean_kl"] == pytest.approx(
        (.5 * (change / .2).square().sum(-1)).mean().item(), rel=1e-12)
    assert metrics["actor/action_delta_rms"] == pytest.approx(change.square().mean().sqrt().item(), rel=1e-12)


def test_fixed_step_has_no_kl_limit_and_skips_batches_without_valid_starts():
    parameters, optimizer, step, mean = fixed_system(lr=10., clip_value=0.)
    set_gradients(parameters)
    initial = mean().detach()
    before = [p.detach().clone() for p in parameters]
    metrics = step.step(mean, initial, torch.full_like(initial, 1e-3), torch.zeros(3, dtype=torch.bool))
    assert metrics["actor/step_accepted"] == 0 and not optimizer.state
    for parameter, value in zip(parameters, before):
        torch.testing.assert_close(parameter, value, rtol=0, atol=0)
    metrics = step.step(mean, initial, torch.full_like(initial, 1e-3), torch.ones(3, dtype=torch.bool))
    assert metrics["actor/step_accepted"] == 1 and metrics["actor/update_mean_kl"] > 1e3


def test_fixed_step_checkpoint_is_interchangeable_with_the_kl_controller():
    _, optimizer, step, _ = fixed_system()
    optimizer.param_groups[0]["lr"] = 3e-4
    optimizer.param_groups[0]["betas"] = (.9, .999)
    step.load_state_dict(None)
    assert optimizer.param_groups[0]["lr"] == .1 and optimizer.param_groups[0]["betas"] == (.9, .95)
    _, _, controller, _ = system()
    controller.load_state_dict(step.state_dict())
    assert controller.scale == 1
    step.load_state_dict(controller.state_dict())
    with pytest.raises(ValueError, match="group count"):
        step.load_state_dict({"version": 1, "base_lrs": (1.,)})


@pytest.mark.parametrize("overrides", [
    {"lr": 0.}, {"lr": float("nan")}, {"beta2": 1.}, {"beta2": 0.},
    {"clip_value": -1.}, {"ratios": (1., 1.)}, {"ratios": (1., 0., 1.)},
])
def test_invalid_fixed_step_configuration_is_rejected(overrides):
    with pytest.raises(ValueError):
        fixed_system(**overrides)
