"""Fixed replay pools are reused across epochs without materializing the pool."""

import pytest
import torch

from active_adaptation.learning.wm.epoch import EpochReplayPlan
from active_adaptation.learning.wm.replay import CompactSequenceReplay, ReplayPrefetcher
from test_wm_replay import batch


def replay_with_boundaries():
    chunks = [batch(n=2, t=5), batch(n=3, t=8, offset=10000)]
    replay = CompactSequenceReplay(80)
    for data in chunks:
        data["next", "done"] = torch.zeros(*data.shape, 1, dtype=torch.bool)
        data["next", "done"][0, 1] = True
        data["is_init"][0, 2] = True
        data["next", "state"][0, 1] += 777  # Exact terminal target, not reset observation.
        replay.add(data)
    return replay, chunks


def test_one_index_pool_reused_exactly_once_per_epoch_with_different_permutations(monkeypatch):
    replay, _ = replay_with_boundaries()
    original = torch.randint
    calls = []
    def randint(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, "randint", randint)
    plan = EpochReplayPlan(replay, 24, 4, 3, sequence_length=3,
                           generator=torch.Generator().manual_seed(23))
    assert len(calls) == 1
    for order in plan._orders:
        torch.testing.assert_close(order.sort().values, torch.arange(24), rtol=0, atol=0)
    assert not torch.equal(plan._orders[0], plan._orders[1])
    for epoch in range(3):
        found = torch.cat([plan.draws(epoch, index) for index in range(plan.num_batches)])
        torch.testing.assert_close(found.sort().values, plan._pool_draws.sort().values, rtol=0, atol=0)
        for index in range(plan.num_batches):
            plan.sample(epoch, index)
    assert len(calls) == 1  # Reusing a pool must not secretly draw fresh windows.
    found = plan.draws(0, 0)
    expected = found.clone()
    found.zero_()
    torch.testing.assert_close(plan.draws(0, 0), expected, rtol=0, atol=0)


def test_model_samples_preserve_exact_windows_terminal_states_and_owned_autograd_storage():
    replay, chunks = replay_with_boundaries()
    with torch.inference_mode():
        plan = EpochReplayPlan(replay, 16, 4, 3, sequence_length=3,
                               generator=torch.Generator().manual_seed(7))
        actual = plan.sample(1, 2)
    all_windows = [chunk[env, begin:begin + 3] for chunk in chunks
                   for env in range(chunk.shape[0]) for begin in range(chunk.shape[1] - 2)]
    expected = torch.stack([all_windows[index] for index in plan.draws(1, 2).tolist()])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.batch_size == (4, 3)
    assert all(not value.is_inference() for value in actual.values(True, True))
    assert all(value.device.type == "cpu" for value in actual.values(True, True))
    parameter = torch.nn.Parameter(torch.ones(17, 2))
    (actual["state"] @ parameter).sum().backward()
    assert torch.isfinite(parameter.grad).all()
    actual.zero_()
    torch.testing.assert_close(plan.sample(1, 2), expected, rtol=0, atol=0)


def test_actor_pool_includes_tails_and_terminal_starts_but_excludes_reset_rows():
    replay, chunks = replay_with_boundaries()
    plan = EpochReplayPlan(replay, 128, 8, 3, starts_only=True,
                           generator=torch.Generator().manual_seed(11))
    valid_rows = torch.stack([chunk[env, time].select("state", "is_init") for chunk in chunks
                             for env in range(chunk.shape[0]) for time in range(chunk.shape[1])
                             if not bool(chunk["is_init"][env, time])])
    sampled = []
    for index in range(plan.num_batches):
        sample = plan.sample(0, index)
        torch.testing.assert_close(sample, valid_rows[plan.draws(0, index)], rtol=0, atol=0)
        assert not sample["is_init"].any()
        sampled.append(sample["state"][:, 0])
    identities = torch.cat(sampled)
    assert any(bool((identities == chunk["state"][0, -1, 0]).any()) for chunk in chunks)
    assert any(bool((identities == chunk["state"][0, 1, 0]).any()) for chunk in chunks)
    with pytest.raises(ValueError, match="every original observation"):
        plan.sample(0, 0, state_columns=slice(2, 5))


def test_requested_full_size_plans_store_only_small_cpu_index_tensors(monkeypatch):
    replay, _ = replay_with_boundaries()
    def forbidden(*args, **kwargs):
        raise AssertionError("Constructing a plan must not gather any state batch")
    monkeypatch.setattr(replay, "sample_compact", forbidden)
    monkeypatch.setattr(replay, "sample_starts", forbidden)
    model = EpochReplayPlan(replay, 16 * 1024, 1024, 3, sequence_length=3)
    actor = EpochReplayPlan(replay, 4096, 512, 3, starts_only=True)
    assert (model.num_batches, actor.num_batches) == (16, 8)
    assert model.bytes + actor.bytes == 655360
    for plan in (model, actor):
        tensors = [value for value in vars(plan).values() if isinstance(value, torch.Tensor)]
        assert len(tensors) == 2
        assert all(value.device.type == "cpu" and value.dtype == torch.int64 for value in tensors)


def test_prefetch_reuses_planned_draws_without_rng_use_and_can_skip_capture_batch():
    replay, _ = replay_with_boundaries()
    plan = EpochReplayPlan(replay, 24, 4, 3, sequence_length=3,
                           generator=torch.Generator().manual_seed(41))
    section = slice(3, 12)
    first = plan.sample(0, 0, state_columns=section)
    expected = [plan.sample(0, index, state_columns=section) for index in range(1, plan.num_batches)]
    rng = torch.get_rng_state().clone()
    with plan.prefetch(0, "cpu", state_columns=section, start_batch=1) as prefetched:
        actual = list(prefetched)
    assert torch.equal(torch.get_rng_state(), rng)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert prefetched.transferred_bytes > 0
    assert prefetched.transferred_bytes < prefetched.materialized_bytes
    first.zero_()
    torch.testing.assert_close(actual[-1], expected[-1], rtol=0, atol=0)
    with plan.prefetch(0, "cpu", start_batch=plan.num_batches) as prefetched:
        assert list(prefetched) == []


def test_plan_and_prefetch_reject_replay_mutation_before_rows_can_be_reinterpreted():
    replay, _ = replay_with_boundaries()
    plan = EpochReplayPlan(replay, 8, 4, 3, sequence_length=3)
    prefetched = plan.prefetch(0, "cpu")
    replay.add(batch(n=2, t=5, offset=20000))
    with pytest.raises(RuntimeError, match="Replay changed"):
        plan.draws(0, 0)
    with pytest.raises(RuntimeError, match="Replay changed"):
        plan.sample(0, 0)
    with pytest.raises(RuntimeError, match="Replay changed"):
        prefetched.__enter__()


def test_seeded_plans_reproduce_and_fixed_draw_prefetch_owns_its_index_tensor():
    replay, _ = replay_with_boundaries()
    first = EpochReplayPlan(replay, 12, 4, 3, sequence_length=3,
                            generator=torch.Generator().manual_seed(99))
    second = EpochReplayPlan(replay, 12, 4, 3, sequence_length=3,
                             generator=torch.Generator().manual_seed(99))
    for epoch in range(3):
        for index in range(3):
            torch.testing.assert_close(first.draws(epoch, index), second.draws(epoch, index), rtol=0, atol=0)
    draws = torch.stack([first.draws(0, index) for index in range(3)])
    expected = [replay.sample(4, 3, draws=value) for value in draws]
    iterator = ReplayPrefetcher(replay, 4, 3, 3, "cpu", draw_batches=draws)
    draws.zero_()
    with iterator:
        for actual, wanted in zip(iterator, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("settings", [dict(pool_size=7), dict(epochs=0), dict(sequence_length=0), dict(minibatch_size=True)])
def test_invalid_or_partial_batches_fail_before_sampling(settings):
    replay, _ = replay_with_boundaries()
    kwargs = dict(pool_size=8, minibatch_size=4, epochs=3, sequence_length=3)
    kwargs.update(settings)
    with pytest.raises(ValueError):
        EpochReplayPlan(replay, **kwargs)
