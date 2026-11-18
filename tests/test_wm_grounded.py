"""Validation follows the sensor bodies used by the unchanged task equations."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from active_adaptation.learning.wm.grounded import GroundedWorldModel


class _Physics(nn.Module):
    step_dt = .02

    def context(self, state, key):
        assert key == "contact.current_contact_time"
        return state

    def validation_errors(self, predicted, target):
        event = lambda value: (value > 0) & (value < self.step_dt + 1e-8)
        p, t = event(predicted), event(target)
        return {"contact_event_error": (p != t).float().mean(-1),
                "contact_event_true_count": t.sum(-1).float(),
                "contact_event_miss_count": (t & ~p).sum(-1).float(),
                "auxiliary": torch.tensor([.25]), "contact_force": torch.tensor([.4])}


def test_first_event_gate_uses_configured_reward_bodies_and_keeps_all_sensor_evidence():
    reward = SimpleNamespace(reward_terms=[
        SimpleNamespace(kind="impact_force_l2", body_ids=[2]),
        SimpleNamespace(kind="feet_air_time_skateboard", body_ids=torch.tensor([1, 2])),
        SimpleNamespace(kind="survival"),
    ])
    model = GroundedWorldModel(_Physics(), nn.Identity(), reward)
    assert model.reward_contact_body_ids.tolist() == [1, 2]
    # Index0 is an unused wrist timer, indices1/2 the configured ankles.
    target = torch.tensor([[.005, .005, .005]])
    result = model.validation_errors(torch.tensor([[.1, .005, .005]]), target)
    assert result["contact_event_true_count"].item() == 2
    assert result["contact_event_miss_count"].item() == 0
    assert result["all_sensor_contact_event_true_count"].item() == 3
    assert result["all_sensor_contact_event_miss_count"].item() == 1
    assert result["auxiliary"].item() == .25
    torch.testing.assert_close(result["contact_force"], torch.tensor([.4]))
    # Correct wrist timing cannot hide either missed ankle event.
    result = model.validation_errors(torch.tensor([[.005, .1, .1]]), target)
    assert result["contact_event_miss_count"].item() == 2
    assert result["contact_event_error"].item() == 1
    assert "reward_contact_body_ids" not in model.state_dict()


def test_task_without_timer_rewards_retains_physical_contact_diagnostics():
    model = GroundedWorldModel(_Physics(), nn.Identity(), SimpleNamespace(reward_terms=[]))
    result = model.validation_errors(torch.tensor([[0., 0.]]), torch.tensor([[.005, .005]]))
    assert "contact_event_true_count" not in result
    assert result["all_sensor_contact_event_miss_count"].item() == 2
    assert "auxiliary" in result and "contact_force" in result


def test_finish_evaluates_termination_once_with_exact_reward_counter_and_gradient_parity(monkeypatch):
    from test_wm_reward import fixture, term
    from active_adaptation.learning.wm.reward import VAICRewardModel
    _, state, physical, action, _, env = fixture("keypoint_pos_tracking_local_product")
    kind = "cum_object_pos_error"
    env.termination_funcs[kind] = term(kind, threshold=.5, min_steps=3)
    state["termination." + kind + "._cum_error_mixin__cum_steps"] = torch.tensor(
        [0., 2., 4.], dtype=torch.double)
    reward = VAICRewardModel(env, SimpleNamespace(context_schema=(), context_slices={}), None).double()
    for fields in (state, physical):
        for value in fields.values():
            if value.is_floating_point():
                value.requires_grad_()
    class Adapter(nn.Module):
        def rebuild(self, predicted, reference, action, **kwargs):
            return dict(predicted)
    model = GroundedWorldModel(_Physics(), Adapter(), reward)
    calls = []
    original_details = reward.termination_details
    def details(*args, **kwargs):
        calls.append(1)
        return original_details(*args, **kwargs)
    monkeypatch.setattr(reward, "termination_details", details)
    # Previous implementation: update counters, then evaluate full reward again.
    expected_state, expected_term = reward.update_termination_state(state, physical)
    expected_reward, _, expected_discount = reward(state, expected_state, action, physical)
    assert len(calls) == 2
    calls.clear()
    actual_state, actual_reward, actual_term, actual_discount = model._finish(
        state, physical, action, physical)
    assert len(calls) == 1
    for name in expected_state:
        torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0, atol=0)
    torch.testing.assert_close(actual_reward, expected_reward, rtol=0, atol=0)
    torch.testing.assert_close(actual_term.bool(), expected_term, rtol=0, atol=0)
    torch.testing.assert_close(actual_discount, expected_discount, rtol=0, atol=0)
    inputs = (state["command.robot_root_pos_w"], state["command.robot_root_quat_w"],
              physical["robot.body_link_pos_w"], physical["object.root_link_pos_w"])
    def objective(values, rewards):
        return rewards.square().sum() + values["termination." + kind + ".error"].square().sum()
    actual_grad = torch.autograd.grad(objective(actual_state, actual_reward), inputs, retain_graph=True)
    expected_grad = torch.autograd.grad(objective(expected_state, expected_reward), inputs)
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sum(grad.abs().sum() for grad in actual_grad) > 0


def _fused_finish_fixture():
    """Actual observation/reward adapters with history, contacts and counters."""
    from collections import OrderedDict
    from test_wm_adapter import _fixture
    from test_wm_reward import term
    from active_adaptation.learning.wm.adapter import VAICWorldModelAdapter
    from active_adaptation.learning.wm.context import RawWorldModelObservation
    from active_adaptation.learning.wm.core import ObservationCodec
    from active_adaptation.learning.wm.reward import VAICRewardModel

    env, _, _, _, _, _ = _fixture(references=True, noisy=True, contact_substeps=True)
    env.step_dt, env.mult_dt = .02, True
    env.command_manager.ref_object_pos_w = env.scene.env_origins + torch.tensor([1., 2., 3.])
    env.command_manager.is_standing_env = torch.zeros(2, 1, dtype=torch.bool)
    env.reward_groups = OrderedDict(main=SimpleNamespace(multiplicative=False, funcs=OrderedDict(
        action=term("action_rate_l2"), tracking=term("object_pos_tracking"),
        contact=term("feet_air_time_skateboard", body_ids=[0, 1], thres=.2))))
    env.termination_funcs = {"cum_object_pos_error": term(
        "cum_object_pos_error", threshold=.02, min_steps=2, error=torch.zeros(2),
        _cum_error_mixin__cum_steps=torch.zeros(2),
        _cum_error_mixin__exceeded=torch.zeros(2, dtype=torch.bool))}
    transform = RawWorldModelObservation(env, ["policy", "object_", "priv"])
    spec = transform.transform_observation_spec(env.observation_spec)["wm_observation"]
    codec = ObservationCodec(spec, list(spec.keys()))
    adapter = VAICWorldModelAdapter(env, transform, codec)
    reward = VAICRewardModel(env, transform, codec)
    state = codec.pack(transform(env.observation_spec.zero())["wm_observation"])
    return GroundedWorldModel(_Physics(), adapter, reward), state


def test_soft_imagination_preserves_hard_physics_reward_and_counter_path():
    model, state = _fused_finish_fixture()
    adapter = model.adapter
    class Reference:
        def step(self, state, *, resample_noise=False):
            assert resample_noise
            return state, torch.zeros(*state.shape[:-1], 1, dtype=torch.bool)
    model.set_imagination_reference(Reference())
    def physics(state, action, reference):
        target = adapter.context(reference, "command.ref_object_pos_w")
        displacement = torch.cat((action[..., :1], torch.zeros_like(action[..., :2])), -1)
        return adapter.set_context(state, {"object.root_link_pos_w": target + displacement})
    model.physics.forward = physics
    initial, original = state, state.clone()
    action = torch.zeros(len(state), 5, requires_grad=True)
    with torch.no_grad():
        action[:, 0] = .02  # The configured object-position error threshold.
    helper, carry = model.begin_soft_imagination(state, 2)
    assert model.begin_soft_imagination(state, 2)[0] is helper
    for step in range(2):
        hard = model.imagine(state, action)
        soft = model.imagine_soft(state, action, helper, carry)
        for index in (0, 1, 3, 4):
            torch.testing.assert_close(soft[index], hard[index], rtol=0, atol=0)
        hazard, carry = soft[2], soft[-1]
        if step == 0:
            torch.testing.assert_close(hazard, torch.zeros_like(hazard))
        else:
            # Two independent near-threshold violations, not instant failure.
            torch.testing.assert_close(hazard, torch.full_like(hazard, .25), atol=2e-5, rtol=0)
            gradient = torch.autograd.grad(hazard.sum(), action)[0]
            assert torch.isfinite(gradient).all() and (gradient[:, 0] > 0).all()
        state = soft[0]
    torch.testing.assert_close(initial, original, rtol=0, atol=0)


@pytest.mark.parametrize("horizon", [1, 2, 16])
@pytest.mark.parametrize("rebuild_observations", [False, True])
def test_fused_finish_preserves_multistep_values_gradients_rng_and_input_ownership(
        horizon, rebuild_observations):
    model, initial = _fused_finish_fixture()
    adapter = model.adapter
    torch.manual_seed(815)
    controls = torch.randn(horizon, 2, 5) * .2
    gain_value = torch.tensor([.3, .2, .1, .04, .05])
    rng = torch.get_rng_state()
    checkpoint_keys = tuple(model.state_dict())

    def rollout(fused):
        torch.set_rng_state(rng)
        model.use_fused_finish = fused
        initial_leaf = initial.detach().clone().requires_grad_()
        actions = controls.detach().clone().requires_grad_()
        gain = gain_value.detach().clone().requires_grad_()
        state, records, loss = initial_leaf, [], 0.
        for step in range(horizon):
            # A differentiable physical predictor exercises state, action and
            # model-parameter derivatives, including feedback through histories.
            action = actions[step] + state[..., :5] * .01
            object_pos = adapter.context(state, "object.root_link_pos_w") + action[..., :3] * gain[:3]
            joint_pos = adapter.context(state, "robot.joint_pos_substeps") + (action * gain).unsqueeze(-2)
            contact = state.new_tensor([[.005, .025, 0.], [.03, .005, 0.]])
            if step % 2:
                contact = contact.flip(-1)
            physical = adapter.set_context(state, {
                "object.root_link_pos_w": object_pos,
                "robot.joint_pos_substeps": joint_pos,
                "robot.root_com_vel_w": torch.cat((action[..., :3], action[..., :3] * .1), -1),
                "contact.current_contact_time": contact,
                "contact.last_air_time": state.new_full((2, 3), .1),
            })
            reference = adapter.set_context(state.detach(), {
                "obs_noise.policy.root_ang_vel_history": torch.randn(2, 3) * .1,
            })
            before = (state.detach().clone(), physical.detach().clone(), reference.clone())
            prediction, reward, terminated, discount = model._finish(
                state, physical, action, reference, rebuild_observations=rebuild_observations)
            for original, expected in zip((state, physical, reference), before):
                torch.testing.assert_close(original, expected, rtol=0, atol=0)
            records.append(tuple(x.detach().clone() for x in (prediction, reward, terminated, discount)))
            loss = loss + reward.square().sum() + prediction.square().mean() * 1e-4
            state = prediction
        gradients = torch.autograd.grad(loss, (initial_leaf, actions, gain))
        assert all(torch.isfinite(gradient).all() and gradient.abs().sum() > 0 for gradient in gradients)
        assert tuple(model.state_dict()) == checkpoint_keys
        return records, gradients, torch.get_rng_state()

    expected, expected_gradients, expected_rng = rollout(False)
    actual, actual_gradients, actual_rng = rollout(True)
    for actual_step, expected_step in zip(actual, expected):
        for actual_value, expected_value in zip(actual_step, expected_step):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-6, atol=2e-7)
    assert torch.equal(actual_rng, expected_rng)
    assert any((record[3] < 1).any() for record in actual)
    if horizon > 1:
        assert any(record[2].bool().any() for record in actual)


def test_soft_contact_force_switch_reaches_each_cached_termination_model():
    from test_wm_soft_termination import real_reward_fixture
    reward, state, _, _ = real_reward_fixture("cum_lost_contact_steps", threshold=1., pos_thres=.2, frc_thres=1.)
    model = GroundedWorldModel(_Physics(), nn.Identity(), reward)
    # Without the switch, the helper keeps the real env's force condition.
    assert model.begin_soft_imagination(state, 2)[0].contact_force is True
    model.set_soft_contact_force(False)
    assert model.begin_soft_imagination(state, 2)[0].contact_force is False
    assert model.begin_soft_imagination(state, 3)[0].contact_force is False
    with pytest.raises(ValueError, match="bool"):
        model.set_soft_contact_force(0)


@pytest.mark.parametrize("enabled", [False, True])
def test_kinematic_reward_switch_reaches_only_imagined_rollouts(enabled):
    model, state = _fused_finish_fixture()
    class Reference:
        def step(self, state, *, resample_noise=False):
            return state, torch.zeros(*state.shape[:-1], 1, dtype=torch.bool)
    model.set_imagination_reference(Reference())
    model.physics.forward = lambda state, action, reference: state
    seen = []
    original = model.reward_model.forward
    def forward(*args, kinematic=False, **kwargs):
        seen.append(kinematic)
        return original(*args, kinematic=kinematic, **kwargs)
    model.reward_model.forward = forward
    model.set_kinematic_imagination_reward(enabled)
    action = torch.zeros(len(state), 5)
    model(state, action, state)
    model.imagine(state, action)
    helper, carry = model.begin_soft_imagination(state, 1)
    model.imagine_soft(state, action, helper, carry)
    # Factual forward stays exact; both imagination paths follow the switch.
    assert seen == [False, enabled, enabled]
    with pytest.raises(ValueError, match="bool"):
        model.set_kinematic_imagination_reward(1)


class _ContactFreePhysics(_Physics):
    contact_model = False


def test_contact_free_physics_rejects_force_and_exact_imagined_reward_switches():
    model = GroundedWorldModel(_ContactFreePhysics(), nn.Identity(), SimpleNamespace(reward_terms=[]))
    with pytest.raises(ValueError, match="wm_contact_model"):
        model.set_soft_contact_force(True)
    with pytest.raises(ValueError, match="wm_contact_model"):
        model.set_kinematic_imagination_reward(False)
    model.set_soft_contact_force(False)
    model.set_kinematic_imagination_reward(True)
    assert model._imagined_reward_options() == {"kinematic_reward": True}


@pytest.mark.parametrize("contact_model", [True, False])
def test_reward_loss_excludes_contact_groups_only_without_contact_prediction(contact_model):
    physics = _Physics() if contact_model else _ContactFreePhysics()
    reward = SimpleNamespace(reward_terms=[], contact_free_groups=lambda: (True, False, True))
    model = GroundedWorldModel(physics, nn.Identity(), reward)
    predicted = torch.tensor([[1., 5., 3.], [0., 7., 0.]])
    target = torch.zeros(2, 3)
    physics.transition_loss = lambda *args, **kwargs: ({"dynamics": torch.zeros(2)}, None)
    model._finish = lambda *args, **kwargs: (None, predicted, None, None)
    losses = model.transition_loss(None, None, None, target, None, None)
    expected = (predicted.square().mean(-1) if contact_model
                else predicted[:, [0, 2]].square().mean(-1))
    torch.testing.assert_close(losses["reward"], expected)


def test_contact_free_world_model_skips_exact_contact_terms_and_force_rotation():
    class Adapter(nn.Module):
        def prepare_context(self, *args, **kwargs):
            raise AssertionError("unused")
    contact_adapter, free_adapter = Adapter(), Adapter()
    contact = GroundedWorldModel(_Physics(), contact_adapter, SimpleNamespace(reward_terms=[]))
    free = GroundedWorldModel(_ContactFreePhysics(), free_adapter, SimpleNamespace(reward_terms=[]))
    assert getattr(contact_adapter, "rebuild_contact_forces", True) is True
    assert free_adapter.rebuild_contact_forces is False
    # Contact-predicting models keep the previous reward call exactly.
    assert contact._reward_options(False) == {"kinematic": False}
    assert free._reward_options(False) == {"kinematic": False, "skip_contact": True}
    assert free._reward_options(True) == {"kinematic": True, "skip_contact": True}
