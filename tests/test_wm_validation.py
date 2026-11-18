"""Independent checks for held-out admission and deterministic actor learning."""

from types import SimpleNamespace, MethodType

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.wm.algorithm import WM, lambda_returns
from active_adaptation.learning.wm.actor_step import ActorStepController
from test_wm_training import make_policy, make_rollout, parameters


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    yield
    torch.set_num_threads(previous)


LIMITS = {"joint_pos": .15, "robot_root_pos": .03, "robot_root_rotation": .15,
          "robot_root_velocity": 1., "object_root_pos": .03,
          "object_root_rotation": .15, "object_root_velocity": 1.}


class ValidationModel:
    """Controlled model errors, independent of the gate implementation."""

    def __init__(self):
        self.overrides = {}
        self.baseline_overrides = {}
        self.reward_error = 0.
        self.termination_wrong = False
        self.calls = 0

    def __call__(self, state, action, reference):
        self.calls += 1
        prediction = reference.clone()
        prediction[..., 1] = 1  # distinguish learned prediction from baseline
        reward = torch.as_tensor(self.reward_error, device=state.device, dtype=state.dtype)
        reward = reward.expand_as(state[..., :1]) if reward.ndim < 2 else reward.expand(state.shape[0], -1)
        termination = torch.full_like(state[..., :1], self.termination_wrong)
        return prediction, reward, termination, torch.ones_like(termination)

    def baseline_prediction(self, state, action, reference):
        prediction = reference.clone()
        prediction[..., 1] = 2
        return prediction

    def validation_errors(self, prediction, target):
        step = int(target[0, 0].item())
        baseline = bool((prediction[..., 1] == 2).all())
        values = {name: limit * .5 for name, limit in LIMITS.items()}
        values["normalized_dynamics"] = 1. if baseline else .5
        values["auxiliary"] = 1. if baseline else .5
        values["contact_event_error"] = .2 if baseline else .01
        values["contact_event_true_count"] = 1.
        values["contact_event_miss_count"] = 1. if baseline else 0.
        values.update((self.baseline_overrides if baseline else self.overrides).get(step, {}))
        return {name: torch.as_tensor(value, device=prediction.device, dtype=prediction.dtype)
                    .expand_as(prediction[..., 0]) for name, value in values.items()}


class ValidationReplay:
    def __init__(self, batch):
        self.batch = batch
        self.samples = 0

    def sample(self, batch_size, length, device):
        assert batch_size == self.batch.shape[0]
        self.samples += 1
        return self.batch[:, :length].to(device)


def validation_harness(*, good_checks=2, horizon=8):
    batch_size = 32
    times = torch.arange(1, horizon + 1).reshape(1, horizon, 1).expand(batch_size, -1, -1)
    target = torch.cat((times, torch.zeros_like(times)), -1).float()
    scalar = torch.zeros(batch_size, horizon, 1)
    batch = TensorDict({
        "state": torch.zeros_like(target), "action": scalar.clone(),
        "is_init": scalar.bool(), "next": TensorDict({
            "state": target, "reward": scalar.clone(), "terminated": scalar.bool(),
            "done": scalar.bool(), "discount": torch.ones_like(scalar),
        }, [batch_size, horizon]),
    }, [batch_size, horizon])
    model = ValidationModel()
    harness = SimpleNamespace(
        cfg=SimpleNamespace(imagination_horizon=horizon, validation_horizon=8,
            train_every=horizon, validation_batch_size=batch_size,
            validation_good_checks=good_checks, validation_max_error_ratio=.95,
            validation_max_reward_rmse=.04, validation_max_auxiliary_nmse=.75,
            validation_max_contact_event_error=.1, validation_max_contact_event_miss_rate=.5,
            validation_min_contact_events=4, gamma=.99, lmbda=.95),
        world_model=model, validation_replay=ValidationReplay(batch), device="cpu",
        _oracle_checked=True, _good_checks={h: 0 for h in (2, 4, 8) if h <= horizon},
        _validated_horizon=0, reward_scales=torch.ones(1),
    )
    def real_value(state):
        # Learned/baseline predictions have markers 1/2. Critic diagnostics must
        # use the real state trajectory, never imagined states from the gate.
        assert (state[..., 1] == 0).all()
        return torch.full_like(state[..., :1], .25)
    harness._value = real_value
    return harness, model, batch


def validate(harness):
    return WM._validate_world_model(harness)


def test_gate_needs_consecutive_heldout_checks_and_closes_after_regression():
    harness, model, _ = validation_harness()
    assert validate(harness)["validation/approved_horizon"] == 0
    assert validate(harness)["validation/approved_horizon"] == 8
    model.reward_error = .1
    assert validate(harness)["validation/approved_horizon"] == 0
    assert all(value == 0 for value in harness._good_checks.values())
    model.reward_error = 0.
    assert validate(harness)["validation/approved_horizon"] == 0
    assert validate(harness)["validation/approved_horizon"] == 8
    assert harness.validation_replay.samples == 5


def test_later_horizon_cannot_hide_an_earlier_physical_error_regression():
    harness, model, _ = validation_harness(good_checks=1)
    assert validate(harness)["validation/approved_horizon"] == 8
    model.overrides[2] = {"object_root_pos": .5}
    assert validate(harness)["validation/approved_horizon"] == 0


def test_nonfinite_normalized_error_cannot_pass_via_tiny_absolute_errors():
    harness, model, _ = validation_harness(good_checks=1)
    tiny = {key: 0. for key in LIMITS}
    for step in (1, 2, 4, 8):
        model.overrides[step] = {**tiny, "normalized_dynamics": float("nan")}
    assert validate(harness)["validation/approved_horizon"] == 0


@pytest.mark.parametrize("failure", ["too_few_valid", "termination", "baseline_worse"])
def test_gate_rejects_insufficient_samples_wrong_termination_and_unhelpful_model(failure):
    harness, model, batch = validation_harness(good_checks=1)
    if failure == "too_few_valid":
        batch["is_init"][:28] = True
    elif failure == "termination":
        model.termination_wrong = True
    else:
        for step in (1, 2, 4, 8):
            model.overrides[step] = {"normalized_dynamics": 1.1}
    assert validate(harness)["validation/approved_horizon"] == 0


@pytest.mark.parametrize("failure", ["auxiliary_absolute", "auxiliary_baseline"])
def test_continuous_force_and_torque_auxiliary_failures_close_otherwise_accurate_gate(failure):
    harness, model, _ = validation_harness(good_checks=1)
    assert validate(harness)["validation/approved_horizon"] == 8
    for step in (1, 2, 4, 8):
        if failure == "auxiliary_absolute":
            # Better than baseline, but still fails the absolute accuracy bar.
            model.overrides[step] = {"auxiliary": .8}
        elif failure == "auxiliary_baseline":
            # Below the absolute bar, but worse than simply carrying old forces.
            model.overrides[step] = {"auxiliary": .6}
            model.baseline_overrides[step] = {"auxiliary": .5}
    assert validate(harness)["validation/approved_horizon"] == 0


def test_discrete_contact_errors_cap_actor_at_two_steps_without_claiming_contact_readiness():
    harness, model, _ = validation_harness(good_checks=1)
    assert validate(harness)["validation/approved_horizon"] == 8
    for step in (1, 2, 4, 8):
        model.overrides[step] = {"contact_event_error": .2}
    metrics = validate(harness)
    assert metrics["validation/approved_horizon"] == 2
    assert metrics["validation/short_pathwise_ready"] == 1
    assert metrics["validation/contact_ready"] == 0
    for step in (1, 2, 4, 8):
        assert metrics[f"validation/h{step}/continuous_ok"] == 1
        assert metrics[f"validation/h{step}/contact_ok"] == 0


def test_critic_diagnostic_uses_real_heldout_transitions_and_preserves_gate():
    harness, _, batch = validation_harness(good_checks=1)
    metrics = validate(harness)
    # The current imagined-trained critic is evaluated against an independent
    # real replay Bellman target. The obsolete target critic is not consulted.
    carry = .25
    targets = []
    for _ in reversed(range(batch.shape[1])):
        carry = .99 * (.05 * .25 + .95 * carry)
        targets.append(carry)
    expected = torch.tensor([(.25 - value) ** 2 for value in targets]).mean().sqrt().item()
    assert metrics["validation/critic_rmse"] == pytest.approx(expected, abs=1e-7)
    assert metrics["validation/approved_horizon"] == 8


@pytest.mark.parametrize("horizon", [16, 32])
def test_extended_validation_reports_physical_reward_and_termination_errors_without_expanding_gate(horizon):
    harness, model, batch = validation_harness(good_checks=1, horizon=horizon)
    harness.cfg.validation_horizon = horizon
    batch["next", "reward"][:, 8:] = .25
    batch["next", "terminated"][:, 8:] = True
    model.overrides[horizon] = {"object_root_pos": .7, "contact_event_error": .9}
    metrics = validate(harness)
    assert model.calls == horizon
    assert metrics[f"validation/h{horizon}/object_root_pos"] == pytest.approx(.7)
    assert metrics[f"validation/h{horizon}/reward_rmse"] == pytest.approx(.25)
    assert metrics[f"validation/h{horizon}/termination_error"] == 1.
    assert metrics[f"validation/h{horizon}/continuous_ok"] == 0.
    assert metrics[f"validation/h{horizon}/contact_ok"] == 0.
    assert metrics["validation/approved_horizon"] == 8
    assert metrics["validation/contact_ready"] == 1.
    assert set(harness._good_checks) == {2, 4, 8}


def test_extended_validation_preserves_h8_critic_target_and_default_scope():
    harness, model, batch = validation_harness(good_checks=1, horizon=32)
    legacy = validate(harness)
    assert model.calls == 8
    assert "validation/h16/reward_rmse" not in legacy
    assert "validation/h32/reward_rmse" not in legacy
    # Large later rewards must not change the established <=H8 critic metric.
    batch["next", "reward"][:, 8:] = 100.
    batch["next", "terminated"][:, 8:] = True
    harness.cfg.validation_horizon = 32
    extended = validate(harness)
    assert extended["validation/critic_rmse"] == legacy["validation/critic_rmse"]
    assert extended["validation/h32/reward_rmse"] == pytest.approx(100.)


@pytest.mark.parametrize("collection_steps,actor_horizon,expected", [(16, 32, 16), (32, 16, 16)])
def test_extended_validation_caps_at_collection_and_actor_horizons(collection_steps, actor_horizon, expected):
    harness, model, _ = validation_harness(good_checks=1, horizon=32)
    harness.cfg.validation_horizon = 32
    harness.cfg.train_every = collection_steps
    harness.cfg.imagination_horizon = actor_horizon
    metrics = validate(harness)
    assert model.calls == expected
    assert f"validation/h{expected}/reward_rmse" in metrics
    assert "validation/h32/reward_rmse" not in metrics


def test_extended_validation_masks_factual_terminal_and_reset_boundaries():
    harness, model, batch = validation_harness(good_checks=1, horizon=32)
    harness.cfg.validation_horizon = 32
    batch["next", "done"][:8, 10] = True
    batch["is_init"][8:16, 0] = True
    errors = torch.full((32,), .01)
    errors[:16] = 100.
    model.overrides[32] = {"object_root_pos": errors}
    metrics = validate(harness)
    assert metrics["validation/h32/valid_samples"] == 16
    assert metrics["validation/h32/object_root_pos"] == pytest.approx(.01)


def test_rare_first_contact_events_require_conditional_recall_for_long_horizons():
    harness, model, _ = validation_harness(good_checks=1)
    events = torch.zeros(32)
    events[:4] = 1.
    for step in (1, 2, 4, 8):
        model.overrides[step] = {"contact_event_true_count": events,
                                "contact_event_miss_count": torch.zeros_like(events)}
    accurate = validate(harness)
    assert accurate["validation/approved_horizon"] == 8
    assert accurate["validation/h1/contact_event_true_count"] == 4
    assert accurate["validation/h1/contact_event_support"] == 1
    assert accurate["validation/h1/contact_event_miss_rate"] == 0
    for step in (1, 2, 4, 8):
        # An all-negative predictor misses four events across 32*30 body
        # comparisons: its overall error is tiny, but its recall is zero.
        model.overrides[step].update({"contact_event_miss_count": events,
                                     "contact_event_error": events / 30})
    missed = validate(harness)
    assert missed["validation/h1/contact_event_error"] < .01
    assert missed["validation/h1/contact_event_miss_rate"] == 1
    assert missed["validation/approved_horizon"] == 2
    assert missed["validation/short_pathwise_ready"] == 1
    assert missed["validation/contact_ready"] == 0
    assert harness._good_checks[2] > 0
    assert all(value == 0 for horizon, value in harness._good_checks.items() if horizon > 2)


@pytest.mark.parametrize("event_count", [0, 3, 4])
def test_first_contact_support_counts_only_valid_transitions(event_count):
    harness, model, batch = validation_harness(good_checks=1)
    batch["is_init"][:8] = True  # These contacts must not count toward support.
    events = torch.zeros(32)
    events[:8] = 2.
    events[8:8 + event_count] = 1.
    for step in (1, 2, 4, 8):
        model.overrides[step] = {"contact_event_true_count": events,
                                "contact_event_miss_count": torch.zeros_like(events)}
    metrics = validate(harness)
    assert metrics["validation/h1/contact_event_true_count"] == event_count
    assert metrics["validation/h1/contact_event_support"] == float(event_count >= 4)
    assert metrics["validation/approved_horizon"] == (8 if event_count >= 4 else 2)


def test_first_contact_recall_threshold_and_minimum_support_are_configurable():
    harness, model, _ = validation_harness(good_checks=1)
    events, misses = torch.zeros(32), torch.zeros(32)
    events[:4], misses[:2] = 1., 1.
    for step in (1, 2, 4, 8):
        model.overrides[step] = {"contact_event_true_count": events,
                                "contact_event_miss_count": misses}
    assert validate(harness)["validation/approved_horizon"] == 8  # Exactly 50% recall.
    harness.cfg.validation_max_contact_event_miss_rate = .49
    assert validate(harness)["validation/approved_horizon"] == 2
    harness.cfg.validation_max_contact_event_miss_rate = .5
    harness.cfg.validation_min_contact_events = 5
    assert validate(harness)["validation/approved_horizon"] == 2


def test_short_pathwise_training_requires_three_continuous_checks_and_still_closes_on_regression():
    harness, model, _ = validation_harness(good_checks=3)
    for step in (1, 2, 4, 8):
        model.overrides[step] = {"contact_event_true_count": 0., "contact_event_miss_count": 0.}
    for expected in (0, 0, 2):
        metrics = validate(harness)
        assert metrics["validation/approved_horizon"] == expected
        assert metrics["validation/short_pathwise_ready"] == float(expected == 2)
        assert metrics["validation/contact_ready"] == 0
    model.overrides[1]["robot_root_pos"] = .031
    metrics = validate(harness)
    assert metrics["validation/approved_horizon"] == 0
    assert metrics["validation/short_pathwise_ready"] == 0
    model.overrides[1].pop("robot_root_pos")
    assert [validate(harness)["validation/approved_horizon"] for _ in range(3)] == [0, 0, 2]


def test_long_horizon_contact_prefix_must_pass_earlier_steps_and_recover_consecutively():
    harness, model, _ = validation_harness(good_checks=2)
    model.overrides[1] = {"contact_event_miss_count": 1.}
    assert validate(harness)["validation/approved_horizon"] == 0
    assert validate(harness)["validation/approved_horizon"] == 2
    # Accurate h2/h4/h8 contact predictions cannot hide bad h1 contact prediction.
    model.overrides.clear()
    assert validate(harness)["validation/approved_horizon"] == 2
    assert validate(harness)["validation/approved_horizon"] == 8
    model.overrides[4] = {"contact_event_miss_count": 1.}
    metrics = validate(harness)
    assert metrics["validation/approved_horizon"] == 2
    assert metrics["validation/short_pathwise_ready"] == 1
    # Contact recovery never licenses an independently inaccurate later physics step.
    model.overrides[4] = {"object_root_rotation": .16}
    assert validate(harness)["validation/approved_horizon"] == 2


def test_pooled_reward_rmse_exposes_rare_large_errors_and_ignores_reset_rows():
    harness, model, batch = validation_harness(good_checks=1)
    error = torch.zeros(32, 1)
    error[0] = .3
    model.reward_error = error
    metrics = validate(harness)
    # The former mean of row RMSEs was .009375 and would have passed .04.
    assert metrics["validation/h1/reward_rmse"] == pytest.approx(.3 / 32 ** .5)
    assert metrics["validation/h1/reward_max_group_rmse"] == pytest.approx(.3 / 32 ** .5)
    assert metrics["validation/approved_horizon"] == 0
    batch["is_init"][0] = True
    metrics = validate(harness)
    assert metrics["validation/h1/reward_rmse"] == 0
    assert metrics["validation/approved_horizon"] == 8


def test_reward_diagnostics_distinguish_group_errors_from_actor_weighted_cancellation():
    harness, model, batch = validation_harness(good_checks=1)
    model.reward_error = torch.tensor([[.01, -.02]])
    batch["next", "reward"] = torch.zeros(32, 8, 2)
    harness.reward_scales = torch.tensor([.2, .8])
    metrics = validate(harness)
    assert metrics["validation/h1/reward_rmse"] == pytest.approx(((.01 ** 2 + .02 ** 2) / 2) ** .5)
    assert metrics["validation/h1/reward_group_0_rmse"] == pytest.approx(.01)
    assert metrics["validation/h1/reward_group_1_rmse"] == pytest.approx(.02)
    assert metrics["validation/h1/reward_max_group_rmse"] == pytest.approx(.02)
    assert metrics["validation/h1/reward_weighted_rmse"] == pytest.approx(.014)


def test_fixed_environment_split_keeps_all_data_disjoint_and_excludes_validation_from_moments():
    policy = make_policy(validation_env_stride=2)
    for collection in range(2):
        rollout = make_rollout(policy).clone()
        markers = (torch.arange(policy.env.num_envs) + collection * 10).reshape(-1, 1)
        rollout["wm_observation", "wm_context"][..., -1] = markers
        rollout["next", "wm_observation", "wm_context"][..., -1] = markers
        policy._add_replay(rollout)
    context_last = policy.codec.slices["wm_context"].stop - 1
    def markers_in(replay):
        return set(torch.cat([chunk[0]["state"][..., context_last].reshape(-1)
                              for chunk in replay._chunks]).tolist())
    assert markers_in(policy.replay) == {1., 3., 11., 13.}
    assert markers_in(policy.validation_replay) == {0., 2., 10., 12.}
    assert len(policy.replay) + len(policy.validation_replay) == 2 * rollout.numel()
    # Only training states and targets enter physical model moments.
    assert policy.world_model.state_stats.count == 2 * 2 * 2 * policy.cfg.train_every
    assert policy.world_model.state_stats.mean[context_last] == pytest.approx(7.)


@pytest.mark.parametrize("optimizer", ["adam", "kl"])
def test_bptt_uses_mean_has_finite_gradients_and_never_trains_noise_or_critic(monkeypatch, optimizer):
    policy = make_policy(actor_optimizer=optimizer)
    policy._add_replay(make_rollout(policy))
    policy._fit_world_model(policy._sample(2))
    policy.opt_world_model.zero_grad(set_to_none=True)
    sample = policy._sample(2)
    sample["is_init"][0, 0] = True  # Diagnostics must exclude invalid starts too.
    std = policy.actor.module[0][2].module.actor_std
    with torch.no_grad():
        std.copy_(torch.linspace(.1, .6, std.numel()).reshape_as(std))
    std_before = std.detach().clone()
    critic_before = parameters(policy.critic)
    actor_before = parameters(policy.actor)
    original_dist = policy._teacher_dist
    distributions = []
    distribution_stds, distribution_states, grad_modes = [], [], []

    def mean_only(state):
        distribution = original_dist(state)
        class MeanOnly:
            mean = distribution.mean
            stddev = distribution.stddev
            def rsample(self, *args, **kwargs):
                raise AssertionError("Deterministic BPTT cannot sample policy actions")
            def sample(self, *args, **kwargs):
                raise AssertionError("Actor diagnostics cannot sample policy actions")
            def entropy(self):
                raise AssertionError("WM actor cannot optimize Gaussian entropy")
        distributions.append(distribution.mean.detach().clone())
        distribution_stds.append(distribution.stddev.detach().clone())
        distribution_states.append(state.detach().clone())
        grad_modes.append(torch.is_grad_enabled())
        return MeanOnly()

    def forbid_critic(*args, **kwargs):
        raise AssertionError("Imagined trajectories cannot train the critic")

    monkeypatch.setattr(policy, "_teacher_dist", mean_only)
    monkeypatch.setattr(policy, "_fit_imagined_critic", forbid_critic)
    result = policy._imagine(sample, train_critic=False)
    checks = 1 + result["actor/backtracks"]
    assert len(distributions) == sample.shape[1] + checks
    assert grad_modes == [True] * sample.shape[1] + [False] * checks
    for start_state in distribution_states[sample.shape[1]:]:
        torch.testing.assert_close(start_state, distribution_states[0], rtol=0, atol=0)
    start_valid = ~sample["is_init"][:, 0].squeeze(-1)
    change = (distributions[-1] - distributions[0])[start_valid]
    expected_rms = change.square().mean().sqrt()
    expected_kl = .5 * (change / distribution_stds[0][start_valid]).square().sum(-1).mean()
    assert torch.isfinite(expected_rms) and expected_rms > 0
    assert torch.isfinite(expected_kl) and expected_kl > 0
    assert result["actor/action_delta_rms"] == pytest.approx(expected_rms.item())
    assert result["actor/update_mean_kl"] == pytest.approx(expected_kl.item())
    assert result["actor/step_accepted"] == 1
    if optimizer == "kl":
        # Only the legacy trust region bounds the measured change.
        assert result["actor/update_mean_kl"] <= policy.cfg.actor_max_kl
    else:
        assert result["actor/lr"] == policy.cfg.actor_adam_lr and result["actor/backtracks"] == 0
    assert torch.isfinite(torch.tensor(result["actor/grad_norm"]))
    assert result["actor/grad_norm"] > 0 and result["actor/entropy_coef"] == 0
    assert not std.requires_grad and std.grad is None
    torch.testing.assert_close(std, std_before, rtol=0, atol=0)
    assert all(torch.equal(value, critic_before[name]) for name, value in policy.critic.named_parameters())
    assert any(not torch.equal(value, actor_before[name]) for name, value in policy.actor.named_parameters())
    assert all(p.grad is None for p in policy.world_model.parameters())


@pytest.mark.parametrize("stage", ["actor", "critic", "world_model"])
def test_nonfinite_objective_stops_before_optimizer_update_and_restores_frozen_flags(monkeypatch, stage):
    policy = make_policy()
    policy._add_replay(make_rollout(policy))
    sample = policy._sample(2)
    module = getattr(policy, {"actor": "actor", "critic": "critic", "world_model": "world_model"}[stage])
    before = parameters(module)
    if stage == "actor":
        original = policy.world_model.forward
        def corrupt(*args, **kwargs):
            state, reward, term, discount = original(*args, **kwargs)
            return state, reward * float("nan"), term, discount
        monkeypatch.setattr(policy.world_model, "forward", corrupt)
        operation = lambda: policy._imagine(sample)
    elif stage == "critic":
        # The production critic fits detached imagined targets.
        payload = TensorDict({"observation": policy._critic_observation(sample["state"]),
                              "target": sample["next", "reward"] * float("nan"),
                              "valid": torch.ones(sample.batch_size, dtype=torch.bool)}, sample.batch_size)
        operation = lambda: policy._fit_imagined_critic(payload)
    else:
        original = policy.world_model.transition_loss
        def corrupt(*args, **kwargs):
            losses, prediction = original(*args, **kwargs)
            losses["dynamics"] = losses["dynamics"] * float("nan")
            return losses, prediction
        monkeypatch.setattr(policy.world_model, "transition_loss", corrupt)
        operation = lambda: policy._fit_world_model(sample)
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        operation()
    assert all(torch.equal(value, before[name]) for name, value in module.named_parameters())
    assert all(p.requires_grad for p in policy.world_model.parameters())
    assert not policy.actor.module[0][2].module.actor_std.requires_grad


def test_bptt_exact_return_and_gradient_ignore_factual_done_and_respect_modeled_boundaries():
    """Actual learner versus an independent scalar return/gradient recurrence."""
    theta = torch.nn.Parameter(torch.tensor(.2, dtype=torch.double))

    class ToyCritic(torch.nn.Module):
        in_keys = ["position"]
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 2, bias=False, dtype=torch.double)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor([[2.], [4.]], dtype=torch.double))
        def forward(self, observation):
            observation["state_value"] = self.linear(observation["position"])
            return observation

    class ToyDynamics(torch.nn.Module):
        def imagine(self, state, action):
            position = state[..., :1] + action
            row, step = state[..., 1:2], state[..., 2:3]
            prediction = torch.cat((position, row, step + 1), -1)
            reward = torch.cat((.2 * position, .4 * position), -1)
            terminated = ((row == 1) & (step == 1)).to(state.dtype)
            discount = torch.where((row == 2) & (step == 0), torch.full_like(row, .8), torch.ones_like(row))
            truncated = (row == 2) & (step == 1)
            return prediction, reward, terminated, discount, truncated

    n, horizon = 5, 4
    states = torch.zeros(n, horizon, 3, dtype=torch.double)
    states[:, 0, 0] = states[:, 0, 1] = torch.arange(n, dtype=torch.double)
    is_init = torch.zeros(n, horizon, 1, dtype=torch.bool)
    is_init[3, 0] = True
    batch = TensorDict({"state": states, "is_init": is_init,
        "next": TensorDict({"state": torch.full_like(states, 100000.),
                            "done": torch.ones_like(is_init),
                            "terminated": torch.ones_like(is_init)}, [n, horizon])}, [n, horizon])
    critic = ToyCritic()
    policy = SimpleNamespace(
        cfg=SimpleNamespace(gamma=.9, lmbda=.95, max_grad_norm=1e6,
                            imagination_horizon=horizon, actor_microbatch_size=2),
        world_model=ToyDynamics(), encoder_priv=torch.nn.Identity(), critic=critic,
        reward_scales=torch.tensor([.5, .5], dtype=torch.double),
        opt_policy=torch.optim.Adam([{"params": [theta]}, {"params": []}], lr=1e-8),
        _teacher_dist=lambda state: SimpleNamespace(mean=theta * (1 + state[..., :1])),
        _critic_observation=lambda state: TensorDict({"position": state[..., :1]}, state.shape[:-1]),
    )
    def value(state, target=False):
        assert not target, "The actor must use the same current critic fitted on imagined states"
        return critic(policy._critic_observation(state))["state_value"]
    policy._value = value
    policy._imagine_microbatch = MethodType(WM._imagine_microbatch, policy)
    policy._actor_microbatch_backward = MethodType(WM._actor_microbatch_backward, policy)
    policy._actor_step = ActorStepController(policy.opt_policy, actor_lr=1e-8,
                                            encoder_lr=1e-8, max_actor_lr=1e-8)
    expected_theta = theta.detach().clone().requires_grad_()
    immediate, bootstrap = [], []
    for row in (0, 1, 2, 4):
        position, weight = states[row, 0, 0], 1.
        row_immediate, row_bootstrap = 0., 0.
        for t in range(horizon):
            position = position + expected_theta * (1 + position)
            row_immediate = row_immediate + weight * .3 * position
            weight = weight * .9 * (.8 if row == 2 and t == 0 else 1.)
            if row == 1 and t == 1:
                row_bootstrap = position * 0
                break
            if (row == 2 and t == 1) or t == horizon - 1:
                row_bootstrap = weight * 3 * position
                break
        immediate.append(row_immediate)
        bootstrap.append(row_bootstrap)
    expected_immediate = torch.stack(immediate).mean()
    expected_bootstrap = torch.stack(bootstrap).mean()
    expected_return = expected_immediate + expected_bootstrap
    expected_gradient = torch.autograd.grad(-expected_return, expected_theta)[0]
    # Independently verify the exact SAME rollout's detached critic GAE, not
    # merely the actor return. A unit failure probability removes continuation
    # without truncating the imagined state sequence; timeout retains bootstrap.
    _, payload, _, _, _ = policy._imagine_microbatch(batch[:, 0], horizon, torch.tensor(4))
    expected_valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 0, 0],
                                   [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    torch.testing.assert_close(payload["valid"], expected_valid)
    for row in (0, 1, 2, 4):
        length = 2 if row == 2 else horizon
        position = torch.tensor(float(row), dtype=torch.double)
        positions = []
        for _ in range(length):
            position = position + theta.detach() * (1 + position)
            positions.append(position)
        value_scale = torch.tensor([2., 4.], dtype=torch.double)
        reward_scale = torch.tensor([.2, .4], dtype=torch.double)
        carry = positions[-1] * value_scale
        targets = []
        for step in reversed(range(length)):
            next_value = positions[step] * value_scale
            continuation = 0. if row == 1 and step == 1 else (.8 if row == 2 and step == 0 else 1.)
            carry = positions[step] * reward_scale + .9 * continuation * (.05 * next_value + .95 * carry)
            targets.append(carry)
        torch.testing.assert_close(payload["target"][row, :length], torch.stack(targets[::-1]),
                                   rtol=1e-10, atol=1e-10)
    assert not payload["target"].requires_grad
    metrics = WM._imagine(policy, batch, train_critic=False)
    assert metrics["actor/model_return"] == pytest.approx(expected_return.item(), rel=1e-10)
    assert metrics["actor/immediate_return"] == pytest.approx(expected_immediate.item(), rel=1e-10)
    assert metrics["actor/bootstrap_return"] == pytest.approx(expected_bootstrap.item(), rel=1e-10)
    assert metrics["actor/policy_loss"] == pytest.approx(-expected_return.item(), rel=1e-10)
    torch.testing.assert_close(theta.grad, expected_gradient, rtol=1e-10, atol=1e-10)
    assert all(parameter.grad is None for parameter in critic.parameters())


def test_contact_free_validation_separates_unmodeled_contact_groups_from_comparable_metrics():
    harness, model, batch = validation_harness(good_checks=1)
    model.reward_error = torch.tensor([[.01, .5, .02]])
    batch["next", "reward"] = torch.zeros(32, 8, 3)
    harness.reward_scales = torch.tensor([.2, .5, .3])
    model.reward_model = SimpleNamespace(contact_free_groups=lambda: (True, False, True))
    contact = validate(harness)
    # With contact prediction every group is modeled; the contact-free aggregate
    # is still reported so both modes share one comparable reward metric.
    assert contact["validation/h1/reward_group_1_rmse"] == pytest.approx(.5)
    assert contact["validation/h1/reward_max_group_rmse"] == pytest.approx(.5)
    free_rmse = ((.01 ** 2 + .02 ** 2) / 2) ** .5
    assert contact["validation/h1/reward_contact_free_rmse"] == pytest.approx(free_rmse)
    model.reward_loss_group_mask = torch.tensor([1., 0., 1.])
    free = validate(harness)
    assert "validation/h1/reward_group_1_rmse" not in free
    assert free["validation/h1/reward_contact_group_1_rmse"] == pytest.approx(.5)
    assert free["validation/h1/reward_group_0_rmse"] == pytest.approx(.01)
    assert free["validation/h1/reward_rmse"] == pytest.approx(free_rmse)
    assert free["validation/h1/reward_contact_free_rmse"] == pytest.approx(free_rmse)
    assert free["validation/h1/reward_max_group_rmse"] == pytest.approx(.02)
    assert free["validation/h1/reward_weighted_rmse"] == pytest.approx(.2 * .01 + .3 * .02)
