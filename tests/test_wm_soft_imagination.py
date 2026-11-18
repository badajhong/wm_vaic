"""Scalar checks of soft-survival BPTT and the SAME rollout's critic targets.

These fixtures have no robot, encoder, simulator, replay future, or CUDA path.
Expected returns/gradients are computed independently of the learner's helpers.
"""

from types import MethodType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.wm.algorithm import WM, frozen_parameters
from active_adaptation.learning.wm.actor_step import ActorStepController


DTYPE = torch.float64


class ScalarActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(.2, dtype=DTYPE))

    def distribution(self, state):
        mean = self.theta.expand_as(state[..., :1])
        return SimpleNamespace(mean=mean, stddev=torch.ones_like(mean))


class ScalarCritic(torch.nn.Module):
    in_keys = ["position"]

    def __init__(self, slope=0., bias=2.):
        super().__init__()
        self.slope = torch.nn.Parameter(torch.tensor(slope, dtype=DTYPE))
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=DTYPE))

    def forward(self, observation):
        observation["state_value"] = self.bias + self.slope * observation["position"]
        return observation


class ScalarDynamics(torch.nn.Module):
    """Action affects only failure probability, never reward or next position."""

    def __init__(self, probability=.2, *, action_hazard=False, timeout=None, discount=1.):
        super().__init__()
        self.hazard_bias = torch.nn.Parameter(torch.tensor(-.4, dtype=DTYPE))
        self.probability = probability
        self.action_hazard = action_hazard
        self.timeout = timeout
        self.discount = discount

    def _transition(self, state, action):
        prediction = state + 1
        reward = torch.ones_like(state[..., :1])
        probability = (torch.sigmoid(action + self.hazard_bias) if self.action_hazard
                       else torch.full_like(reward, self.probability))
        truncated = (torch.zeros_like(reward, dtype=torch.bool) if self.timeout is None
                     else prediction[..., :1] >= self.timeout)
        return prediction, reward, probability, torch.full_like(reward, self.discount), truncated

    def imagine(self, state, action):
        return self._transition(state, action)


class CarryDynamics(ScalarDynamics):
    """Exercise the new API and reject accidental use of the legacy hard path."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.token = object()
        self.initializations = []
        self.steps = []

    def begin_soft_imagination(self, state, horizon):
        self.initializations.append((state.detach().clone(), horizon))
        return self.token, 0

    def imagine(self, state, action):
        raise AssertionError("The actor must use imagine_soft when it is available")

    def imagine_soft(self, state, action, termination_model, termination_state):
        assert termination_model is self.token
        assert termination_state == len(self.steps)
        self.steps.append(termination_state)
        return (*self._transition(state, action), termination_state + 1)


def make_scalar_policy(*, probability=.2, action_hazard=False, timeout=None,
                       discount=1., slope=0., soft_api=False):
    actor, critic = ScalarActor(), ScalarCritic(slope=slope)
    dynamics = (CarryDynamics if soft_api else ScalarDynamics)(
        probability, action_hazard=action_hazard, timeout=timeout, discount=discount)
    policy = SimpleNamespace(
        actor=actor, critic=critic, world_model=dynamics, device=torch.device("cpu"),
        cfg=SimpleNamespace(gamma=.9, lmbda=.5, max_grad_norm=1e6,
                            imagination_horizon=3, actor_microbatch_size=2,
                            cuda_graph_actor=False, profile_timing=False,
                            critic_epochs=1, critic_num_minibatches=1,
                            critic_microbatch_size=2, schedule_mode="epochs"),
        reward_scales=torch.ones(1, dtype=DTYPE),
        _teacher_dist=actor.distribution,
        _critic_observation=lambda state: TensorDict({"position": state[..., :1]}, state.shape[:-1]),
    )

    def value(state, target=False):
        assert not target, "Soft imagination must bootstrap from the current critic"
        return critic(policy._critic_observation(state))["state_value"]

    policy._value = value
    policy._imagine_microbatch = MethodType(WM._imagine_microbatch, policy)
    policy._actor_microbatch_backward = MethodType(WM._actor_microbatch_backward, policy)
    policy.opt_policy = torch.optim.Adam([{"params": actor.parameters(), "name": "actor"}], lr=1e-6)
    policy._actor_step = ActorStepController(policy.opt_policy, actor_lr=1e-6, max_actor_lr=1e-6)
    return policy


def make_starts(count=1):
    return TensorDict({"state": torch.zeros(count, 1, dtype=DTYPE),
                       "is_init": torch.zeros(count, 1, dtype=torch.bool)}, [count])


def scalar_return(probability, *, steps, gamma=.9, discount=1., final_value=2.):
    """Closed-form geometric objective, not the learner's rollout recurrence."""
    continuation = gamma * discount * (1 - probability)
    return sum(continuation ** t for t in range(steps)) + continuation ** steps * final_value


@pytest.mark.parametrize("soft_api", [False, True])
def test_fractional_hazard_continues_states_and_matches_return_and_detached_gae(soft_api):
    policy = make_scalar_policy(soft_api=soft_api, slope=.3)
    loss, payload, metrics, action, std = policy._imagine_microbatch(make_starts(), 3, 1)
    # H=3, p=.2: reward weights are [1, .72, .5184]. The final value is V(3).
    expected = 1 + .72 + .5184 + .373248 * 2.9
    assert loss.item() == pytest.approx(-expected, abs=1e-12)
    assert metrics["model_return"].item() == pytest.approx(expected, abs=1e-12)
    assert metrics["effective_horizon"].item() == pytest.approx(1 + .8 + .64, abs=1e-12)
    assert payload["valid"].tolist() == [[True, True, True]]
    torch.testing.assert_close(payload["observation", "position"][0, :, 0],
                               torch.tensor([0., 1., 2.], dtype=DTYPE))
    # Independently expand GAE(lambda=.5) using next values [2.3, 2.6, 2.9].
    last = 1 + .9 * .8 * 2.9
    middle = 1 + .9 * .8 * (.5 * 2.6 + .5 * last)
    first = 1 + .9 * .8 * (.5 * 2.3 + .5 * middle)
    torch.testing.assert_close(payload["target"][0, :, 0],
                               torch.tensor([first, middle, last], dtype=DTYPE))
    assert all(not value.requires_grad for _, value in payload.items(True, True))
    torch.testing.assert_close(action, torch.tensor([[.2]], dtype=DTYPE))
    torch.testing.assert_close(std, torch.ones_like(action))
    assert metrics["soft_termination"].item() == float(soft_api)
    if soft_api:
        assert len(policy.world_model.initializations) == 1
        assert policy.world_model.initializations[0][1] == 3
        assert policy.world_model.steps == [0, 1, 2]


def test_hazard_only_actor_gradient_matches_closed_form_and_freezes_model_and_critic():
    policy = make_scalar_policy(action_hazard=True, soft_api=True)
    assert not hasattr(policy, "encoder_priv")
    theta = policy.actor.theta.detach().clone().requires_grad_()
    probability = torch.sigmoid(theta + policy.world_model.hazard_bias.detach())
    expected_loss = -scalar_return(probability, steps=3)
    expected_gradient = torch.autograd.grad(expected_loss, theta)[0]
    before = {name: parameter.detach().clone() for name, parameter in policy.world_model.named_parameters()}
    critic_before = {name: parameter.detach().clone() for name, parameter in policy.critic.named_parameters()}
    # Exercise production ownership/freeze logic, not just a test-side freeze.
    metrics = WM._imagine(policy, make_starts(), train_critic=False)
    assert metrics["actor/policy_loss"] == pytest.approx(expected_loss.item(), abs=1e-12)
    torch.testing.assert_close(policy.actor.theta.grad, expected_gradient, rtol=1e-12, atol=1e-12)
    assert abs(policy.actor.theta.grad.item()) > .1
    assert metrics["actor/step_accepted"] == 1
    assert metrics["actor/encoder_priv_grad_norm"] == 0
    for module, original in ((policy.world_model, before), (policy.critic, critic_before)):
        for name, parameter in module.named_parameters():
            assert parameter.grad is None
            assert parameter.requires_grad  # Original flags are restored after BPTT.
            torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    # A finite difference of the independent closed form catches sign/order mistakes.
    def objective(at):
        return -scalar_return(torch.sigmoid(at + policy.world_model.hazard_bias.detach()), steps=3)
    epsilon = 1e-6
    finite_difference = (objective(theta.detach() + epsilon) - objective(theta.detach() - epsilon)) / (2 * epsilon)
    torch.testing.assert_close(policy.actor.theta.grad, finite_difference, rtol=1e-8, atol=1e-9)


def test_timeout_is_the_only_hard_cut_and_keeps_soft_weighted_bootstrap():
    policy = make_scalar_policy(timeout=2, slope=.3)
    loss, payload, metrics, _, _ = policy._imagine_microbatch(make_starts(), 4, 1)
    # Timeout after transition 2 bootstraps at V(2), with survival at both steps.
    expected = 1 + .72 + .5184 * 2.6
    assert loss.item() == pytest.approx(-expected, abs=1e-12)
    assert metrics["bootstrap_return"].item() == pytest.approx(.5184 * 2.6, abs=1e-12)
    assert metrics["effective_horizon"].item() == pytest.approx(1.8, abs=1e-12)
    assert payload["valid"].tolist() == [[True, True, False, False]]
    torch.testing.assert_close(payload["observation", "position"][0, :, 0],
                               torch.tensor([0., 1., 2., 2.], dtype=DTYPE))
    last = 1 + .9 * .8 * 2.6
    first = 1 + .9 * .8 * (.5 * 2.3 + .5 * last)
    torch.testing.assert_close(payload["target"][0, :2, 0], torch.tensor([first, last], dtype=DTYPE))


@pytest.mark.parametrize("probability", [0., 1.])
def test_probability_endpoints_change_survival_without_hard_stopping_critic_rows(probability):
    policy = make_scalar_policy(probability=probability)
    loss, payload, metrics, _, _ = policy._imagine_microbatch(make_starts(), 3, 1)
    assert loss.item() == pytest.approx(-scalar_return(probability, steps=3), abs=1e-12)
    assert payload["valid"].all()
    assert payload["observation", "position"][0, -1, 0].item() == 2
    assert metrics["effective_horizon"].item() == (3. if probability == 0 else 1.)
    if probability == 1:
        torch.testing.assert_close(payload["target"], torch.ones_like(payload["target"]))
        assert metrics["bootstrap_return"].item() == 0


def test_discount_and_probability_both_weight_survival_and_critic_targets():
    policy = make_scalar_policy(probability=.2, discount=.7)
    loss, payload, metrics, _, _ = policy._imagine_microbatch(make_starts(), 3, 1)
    assert loss.item() == pytest.approx(-scalar_return(.2, steps=3, discount=.7), abs=1e-12)
    assert metrics["effective_horizon"].item() == pytest.approx(1 + .56 + .56 ** 2, abs=1e-12)
    last = 1 + .9 * .56 * 2
    middle = 1 + .9 * .56 * (.5 * 2 + .5 * last)
    first = 1 + .9 * .56 * (.5 * 2 + .5 * middle)
    torch.testing.assert_close(payload["target"][0, :, 0], torch.tensor([first, middle, last], dtype=DTYPE))


def test_invalid_reset_start_contributes_neither_actor_objective_nor_critic_samples():
    policy = make_scalar_policy(action_hazard=True)
    starts = make_starts(2)
    starts["is_init"][1] = True
    with frozen_parameters(policy.world_model, policy.critic):
        loss, payload, _, _, _ = policy._imagine_microbatch(starts, 3, 1)
        loss.backward()
    theta = policy.actor.theta.detach().clone().requires_grad_()
    expected = -scalar_return(torch.sigmoid(theta + policy.world_model.hazard_bias.detach()), steps=3)
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(policy.actor.theta.grad, torch.autograd.grad(expected, theta)[0])
    assert payload["valid"].tolist() == [[True, True, True], [False, False, False]]


def test_critic_fit_uses_all_valid_imagined_states_without_survival_weighting():
    policy = make_scalar_policy(probability=.8, slope=.3)
    with frozen_parameters(policy.world_model, policy.critic):
        _, payload, _, _, _ = policy._imagine_microbatch(make_starts(), 3, 1)
    # One SGD step makes the intended unweighted three-row MSE directly testable.
    parameters = tuple(policy.critic.parameters())
    before = [parameter.detach().clone() for parameter in parameters]
    estimates = policy.critic(payload["observation"].clone(False))["state_value"]
    expected_loss = (estimates - payload["target"]).square().mean()
    expected_gradients = torch.autograd.grad(expected_loss, parameters)
    policy.opt_critic = torch.optim.SGD(parameters, lr=.01)
    result = WM._fit_imagined_critic(policy, payload, cache_metrics={})
    assert result["critic/updates"] == 1
    assert result["critic/sample_passes"] == 3
    assert result["critic/value_loss"] == pytest.approx(expected_loss.item(), abs=1e-7)
    for parameter, original, gradient in zip(parameters, before, expected_gradients):
        torch.testing.assert_close(parameter, original - .01 * gradient, rtol=1e-12, atol=1e-12)
    assert policy.actor.theta.grad is None
    assert all(parameter.grad is None for parameter in policy.world_model.parameters())


def test_imagined_contact_termination_defaults_to_position_only():
    from active_adaptation.learning.wm.algorithm import WMConfig
    assert WMConfig().imagination_contact_force_termination is False
