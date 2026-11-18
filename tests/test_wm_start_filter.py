"""Actor BPTT starts exclude replay states that are already failing."""

from types import MethodType, SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.wm.algorithm import WM
from test_wm_training import make_policy, make_rollout


class CounterReward:
    """Reads one consecutive-violation counter per term from fixed state columns."""

    def __init__(self, columns, kinds=None):
        self.columns = dict(columns)
        kinds = kinds or {}
        self.termination_terms = [SimpleNamespace(name=name, kind=kinds.get(name, "cum_body_pos_error"))
                                  for name in self.columns]

    def _get(self, state, name):
        term = name.removeprefix("termination.").removesuffix("._cum_error_mixin__cum_steps")
        return state[..., self.columns[term]]


def stub_policy(tolerance, contact_kind="cum_lost_contact_steps"):
    reward = CounterReward({"body": 0, "contact": 1}, {"contact": contact_kind})
    policy = SimpleNamespace(cfg=SimpleNamespace(actor_start_violation_tolerance=tolerance),
                             world_model=SimpleNamespace(reward_model=reward))
    policy._filter_failing_starts = MethodType(WM._filter_failing_starts, policy)
    return policy


def starts(counts, is_init=None):
    state = torch.cat((torch.tensor(counts, dtype=torch.float32), torch.randn(len(counts), 3)), -1)
    if is_init is None:
        is_init = [False] * len(counts)
    return TensorDict({"state": state, "is_init": torch.tensor(is_init).unsqueeze(-1)}, [len(counts)])


def test_starts_beyond_violation_tolerance_become_invalid_without_mutating_replay():
    original = starts([[0, 0], [2, 1], [3, 0], [0, 25], [0, 0]], is_init=[False, False, False, False, True])
    is_init_before = original["is_init"].clone()
    filtered, fraction = stub_policy(2, contact_kind="cum_body_ori_error")._filter_failing_starts(original)
    assert filtered["is_init"].squeeze(-1).tolist() == [False, False, True, True, True]
    assert fraction == pytest.approx(2 / 4)
    assert filtered["state"] is original["state"]
    torch.testing.assert_close(original["is_init"], is_init_before, rtol=0, atol=0)


def test_lost_contact_counters_never_filter_a_start():
    batch = starts([[0, 0], [0, 24], [3, 24]])
    filtered, fraction = stub_policy(2)._filter_failing_starts(batch)
    assert filtered["is_init"].squeeze(-1).tolist() == [False, False, True]
    assert fraction == pytest.approx(2 / 3)
    policy = stub_policy(0)
    policy.world_model.reward_model = CounterReward({"contact": 0}, {"contact": "cum_lost_contact_steps"})
    assert policy._filter_failing_starts(batch) == (batch, None)


def test_filter_is_off_by_default():
    from active_adaptation.learning.wm.algorithm import WMConfig
    assert WMConfig().actor_start_violation_tolerance < 0


def test_filter_is_disabled_by_negative_tolerance_or_missing_counters():
    batch = starts([[9, 9], [0, 0]])
    assert stub_policy(-1)._filter_failing_starts(batch) == (batch, None)
    policy = stub_policy(0)
    policy.world_model = SimpleNamespace()
    assert policy._filter_failing_starts(batch) == (batch, None)


def test_all_failing_minibatch_keeps_its_starts_and_reports_zero():
    batch = starts([[5, 0], [0, 7]])
    filtered, fraction = stub_policy(0, contact_kind="cum_body_ori_error")._filter_failing_starts(batch)
    assert filtered is batch and fraction == 0.


def test_imagined_update_excludes_failing_starts_and_reports_their_share():
    policy = make_policy()
    policy.cfg.actor_start_violation_tolerance = 2
    policy.train_op(make_rollout(policy))
    column = policy.codec.slices["wm_context"].start
    policy.world_model.reward_model = CounterReward({"body": column})
    batch = policy._sample_actor_starts(4)
    batch["is_init"] = torch.zeros_like(batch["is_init"])
    batch["state"][:, column] = torch.tensor([0., 1., 30., 0.])
    info = policy._imagine(batch)
    assert info["actor/healthy_start_fraction"] == pytest.approx(3 / 4)
    assert info["actor/valid_starts"] == 3
