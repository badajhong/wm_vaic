"""SuperMimic-style imagined rewards and critic value ceiling."""

import math
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

import active_adaptation.learning.wm.reward as wm
from active_adaptation.utils.reward_bounds import environment_step_ceiling, group_ceiling, term_ceiling
from test_wm_reward import fixture, term


@pytest.mark.parametrize("kind", sorted(wm.VAICRewardModel.auxiliary_reward_kinds))
def test_kinematic_mode_drops_terms_that_read_learned_contact_or_torque(kind):
    model, state, nxt, action, _, _ = fixture(kind)
    exact, _, _ = model(state, nxt, action)
    kinematic, _, discount = model(state, nxt, action, kinematic=True)
    assert torch.count_nonzero(exact[:, 1:]) > 0
    torch.testing.assert_close(kinematic, torch.zeros_like(exact), rtol=0, atol=0)
    # The contact-timer discount is a contact-head quantity as well.
    torch.testing.assert_close(discount, torch.ones_like(discount), rtol=0, atol=0)


def test_kinematic_contact_kernel_keeps_fk_position_and_ignores_force():
    model, state, nxt, action, reward, _ = fixture("eef_contact_exp")
    error = ((state["command.contact_eef_pos_w"] - state["command.contact_target_pos_w"]).norm(dim=-1)
             - reward.pos_tolerance).clamp_min(0)
    active = state["command.ref_body_contact"].to(action.dtype)
    expected = ((-error / reward.pos_sigma).exp() * active * reward.gain).mean(-1, keepdim=True)
    kinematic, _, _ = model(state, nxt, action, kinematic=True)
    torch.testing.assert_close(kinematic[:, 1:], expected * reward.weight * .02)
    state["command.eef_contact_forces_b"] = torch.zeros_like(state["command.eef_contact_forces_b"])
    torch.testing.assert_close(model(state, nxt, action, kinematic=True)[0], kinematic)
    # Exact mode still applies the force deficit.
    assert not torch.allclose(model(state, nxt, action)[0], kinematic)


@pytest.mark.parametrize("kind", ["keypoint_pos_tracking_local_product", "object_pos_tracking",
                                  "action_rate_l2", "joint_pos_limits", "survival"])
def test_kinematic_mode_leaves_state_functions_unchanged(kind):
    model, state, nxt, action, _, _ = fixture(kind)
    torch.testing.assert_close(model(state, nxt, action, kinematic=True)[0], model(state, nxt, action)[0],
                               rtol=0, atol=0)


def test_reward_ceilings_follow_kernel_bounds_and_penalties():
    assert term_ceiling("survival", 2.) == 2.
    assert term_ceiling("eef_contact_exp", 1., 5.) == 5.
    assert term_ceiling("action_rate_l2", .1) == 0.
    assert term_ceiling("action_rate_l2", -.1) == math.inf
    assert term_ceiling("unknown_kind", 1.) == math.inf
    assert group_ceiling([("survival", 1., 1.), ("object_pos_tracking", .5, 1.)]) == 1.5
    assert group_ceiling([("survival", 1., 1.), ("object_pos_tracking", .5, 1.)], multiplicative=True) == .5
    assert group_ceiling([("survival", 1., 1.), ("feet_slip", .5, 1.)], multiplicative=True) == math.inf
    model, *_ = fixture("eef_contact_exp")
    torch.testing.assert_close(model.max_step_rewards(), torch.tensor([0., .7 * 5. * .02]))


def test_environment_ceiling_sums_enabled_groups_with_dt():
    groups = OrderedDict([
        ("track", SimpleNamespace(funcs={"a": term("object_pos_tracking", weight=1.),
                                         "off": term("survival", weight=9., enabled=False)},
                                  multiplicative=False)),
        ("loco", SimpleNamespace(funcs={"s": term("survival", weight=1.),
                                        "r": term("action_rate_l2", weight=.1)}, multiplicative=False)),
    ])
    env = SimpleNamespace(reward_groups=groups, step_dt=.02, mult_dt=True)
    assert environment_step_ceiling(env) == pytest.approx(2 * .02)


def test_critic_ceiling_clamps_bootstrap_and_lambda_targets():
    from test_wm_training import make_policy, make_rollout
    policy = make_policy()
    policy.train_op(make_rollout(policy))
    starts = policy._sample_actor_starts(4)
    starts["is_init"] = torch.zeros_like(starts["is_init"])
    groups = policy.critic(policy._policy_observation(starts["state"]))["state_value"].shape[-1]
    policy._value_ceiling = torch.full((groups,), -1e3, device=starts["state"].device)
    _, payload, metrics, _, _ = policy._imagine_microbatch(starts, 2, torch.tensor(4.))
    clamped = metrics["bootstrap_clamped_value"].item()
    assert clamped > 0
    # The clamped bootstrap is exactly the ceiling: the objective lost raw - ceiling.
    assert metrics["bootstrap_return"].item() < 0
    assert (payload["target"] <= -1e3).all()
    policy._value_ceiling = torch.full((groups,), math.inf, device=starts["state"].device)
    _, _, metrics, _, _ = policy._imagine_microbatch(starts, 2, torch.tensor(4.))
    assert metrics["bootstrap_clamped_value"].item() == 0.


@pytest.mark.parametrize("kind", sorted(wm.VAICRewardModel.contact_reward_kinds))
def test_contact_reward_kinds_mark_their_group_as_contact_dependent(kind):
    model, *_ = fixture(kind)
    assert model.contact_free_groups() == (True, False)


@pytest.mark.parametrize("kind", ["keypoint_pos_tracking_local_product", "joint_torque_limits", "survival"])
def test_kinematic_and_torque_terms_keep_their_group_contact_free(kind):
    model, *_ = fixture(kind)
    assert model.contact_free_groups() == (True, True)


@pytest.mark.parametrize("kind", sorted(wm.VAICRewardModel.contact_reward_kinds))
def test_skip_contact_omits_exact_contact_terms_but_keeps_kinematic_kernel(kind):
    model, state, nxt, action, _, _ = fixture(kind)
    exact, _, _ = model(state, nxt, action)
    skipped, _, discount = model(state, nxt, action, skip_contact=True)
    assert torch.count_nonzero(exact[:, 1:]) > 0
    torch.testing.assert_close(skipped, torch.zeros_like(exact), rtol=0, atol=0)
    torch.testing.assert_close(discount, torch.ones_like(discount), rtol=0, atol=0)
    # The kinematic contact kernel is a function of FK position, not contact.
    torch.testing.assert_close(model(state, nxt, action, kinematic=True, skip_contact=True)[0],
                               model(state, nxt, action, kinematic=True)[0], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["keypoint_pos_tracking_local_product", "joint_torque_limits", "survival"])
def test_skip_contact_leaves_non_contact_terms_exact(kind):
    model, state, nxt, action, _, _ = fixture(kind)
    torch.testing.assert_close(model(state, nxt, action, skip_contact=True)[0], model(state, nxt, action)[0],
                               rtol=0, atol=0)
