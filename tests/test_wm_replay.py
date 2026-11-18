import torch
import pytest
from tensordict import TensorDict

from active_adaptation.learning.wm.replay import CompactSequenceReplay, ReplayPrefetcher


def batch(n=5, t=8, d=17, offset=0):
    state = torch.arange(n * (t + 1) * d).reshape(n, t + 1, d).float() + offset
    return TensorDict({"state": state[:, :-1], "action": state[:, :-1, :2],
                      "is_init": torch.zeros(n, t, 1, dtype=torch.bool),
                      "next": TensorDict({"state": state[:, 1:].clone(),
                                          "reward": state[:, 1:, :1]}, [n, t])}, [n, t])


def test_lossless_next_states_including_reset_exceptions_and_buffer_overwrite():
    source = batch()
    source["next", "state"][1, 3] += 101
    source["next", "state"][4, 1] *= -1
    expected = source.clone()
    replay = CompactSequenceReplay(80)
    replay.add(source)
    source.zero_()
    sample = replay.sample(200, 4)
    for row in sample:
        identity = row["state"][0, 0]
        location = (expected["state"][:, :, 0] == identity).nonzero()[0]
        env, start = location.tolist()
        torch.testing.assert_close(row, expected[env, start:start + 4], rtol=0, atol=0)
    assert replay.bytes < expected.bytes()


def test_capacity_keeps_complete_collections_and_rejects_silent_env_subsampling():
    replay = CompactSequenceReplay(80)
    for i in range(3):
        replay.add(batch(offset=i * 10000))
    assert len(replay) == 80
    assert replay.sample(200, 2)["state"].min() >= 10000
    with pytest.raises(ValueError, match="complete collection"):
        CompactSequenceReplay(10).add(batch())


@pytest.mark.parametrize("length", [1, 3, 4])
def test_mixed_chunk_sampling_preserves_draw_order_windows_dtypes_and_boundaries(monkeypatch, length):
    chunks = [batch(n=2, t=4), batch(n=3, t=6, offset=10000)]
    for source in chunks:
        n, t = source.batch_size
        source["action"] = source["action"].double()
        source["counter"] = torch.arange(n * t).reshape(n, t)
        source["matrix"] = torch.arange(n * t * 6).reshape(n, t, 3, 2).transpose(-1, -2)
        source["empty"] = torch.empty(n, t, 0)
        source["next", "done"] = torch.zeros(n, t, 1, dtype=torch.bool)
        source["next", "done"][0, 1] = True
        source["is_init"][0, 2] = True
        source["next", "state"][0, 1] += 999  # Exact terminal observation.
    replay = CompactSequenceReplay(sum(source.numel() for source in chunks))
    for source in chunks:
        replay.add(source)
    first_count = chunks[0].shape[0] * (chunks[0].shape[1] - length + 1)
    total = first_count + chunks[1].shape[0] * (chunks[1].shape[1] - length + 1)
    draws = torch.tensor([total - 1, 0, first_count, first_count - 1, 1, total - 2, first_count])

    def fixed_draws(high, shape):
        assert high == total and shape == (draws.numel(),)
        return draws.clone()

    monkeypatch.setattr(torch, "randint", fixed_draws)
    rows = []
    for draw in draws.tolist():
        chunk = chunks[0] if draw < first_count else chunks[1]
        local = draw if draw < first_count else draw - first_count
        starts = chunk.shape[1] - length + 1
        env, begin = divmod(local, starts)
        rows.append(chunk[env, begin:begin + length])
    expected = torch.stack(rows)
    actual = replay.sample(draws.numel(), length)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual["action"].dtype == torch.float64
    assert actual["counter"].dtype == torch.int64
    # No output leaf may alias replay or another independently sampled batch.
    actual.zero_()
    torch.testing.assert_close(replay.sample(draws.numel(), length), expected, rtol=0, atol=0)


def test_sampling_returns_owned_autograd_tensors_even_when_called_in_inference_mode():
    replay = CompactSequenceReplay(80)
    with torch.inference_mode():
        replay.add(batch())
        sample = replay.sample(4, 3)
    for value in sample.values(include_nested=True, leaves_only=True):
        assert not value.is_inference()
    weight = torch.nn.Parameter(torch.ones(17, 2))
    (sample["state"] @ weight + sample["next", "state"] @ weight).sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_short_chunks_are_ineligible_and_single_chunk_output_keeps_tail(monkeypatch):
    replay = CompactSequenceReplay(100)
    replay.add(batch(n=2, t=2))
    source = batch(n=2, t=5, offset=10000)
    # The eligible chunk can include scalar or empty event dimensions.
    source["empty"] = torch.empty(2, 5, 0)
    source["next", "state"][1, -1].fill_(-123)
    replay.add(source)
    monkeypatch.setattr(torch, "randint", lambda high, shape: torch.tensor([3, 0, 2, 1]))
    actual = replay.sample(4, 4)
    expected = torch.stack([source[1, 1:], source[0, :4], source[1, :4], source[0, 1:]])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prefetch_matches_sequential_draws_values_rng_and_owned_autograd_tensors():
    replay = CompactSequenceReplay(80)
    for offset in (0, 10000):
        source = batch(offset=offset)
        source["next", "state"][1, 3] += 31
        replay.add(source)
    torch.manual_seed(782)
    expected = [replay.sample(9, 4) for _ in range(5)]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(782)
    with torch.inference_mode(), ReplayPrefetcher(replay, 9, 4, 5, "cpu") as prefetched:
        actual = list(prefetched)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for found, wanted in zip(actual, expected):
        torch.testing.assert_close(found, wanted, rtol=0, atol=0)
        assert all(not leaf.is_inference() for leaf in found.values(True, True))
    actual[0].zero_()
    for found, wanted in zip(actual[1:], expected[1:]):
        torch.testing.assert_close(found, wanted, rtol=0, atol=0)
    parameter = torch.nn.Parameter(torch.ones(17, 2))
    (actual[-1]["state"] @ parameter).sum().backward()
    assert torch.isfinite(parameter.grad).all()


def test_prefetch_reserves_random_draws_on_caller_and_overlaps_only_one_sample(monkeypatch):
    import threading
    replay = CompactSequenceReplay(80)
    replay.add(batch())
    caller = threading.get_ident()
    original_randint, original_sample = torch.randint, replay.sample
    random_threads, samples = [], []
    started, release = threading.Event(), threading.Event()
    def randint(*args, **kwargs):
        random_threads.append(threading.get_ident())
        return original_randint(*args, **kwargs)
    def sample(*args, **kwargs):
        samples.append(threading.get_ident())
        if len(samples) == 2:
            started.set()
            assert release.wait(2)
        return original_sample(*args, **kwargs)
    monkeypatch.setattr(torch, "randint", randint)
    monkeypatch.setattr(replay, "sample", sample)
    try:
        with ReplayPrefetcher(replay, 4, 3, 4, "cpu") as prefetched:
            first = next(prefetched)
            assert started.wait(2)
            assert len(samples) == 2
            assert random_threads == [caller, caller]
            assert all(thread != caller for thread in samples)
            release.set()
            remaining = list(prefetched)
    finally:
        release.set()
    assert len(remaining) == 3 and first.batch_size == (4, 3)
    assert random_threads == [caller] * 4
    assert prefetched._future is None and prefetched._host_pending is None


def test_prefetch_closes_on_training_or_worker_error_and_empty_phase(monkeypatch):
    replay = CompactSequenceReplay(80)
    replay.add(batch())
    with ReplayPrefetcher(replay, 4, 3, 0, "cpu") as empty:
        assert list(empty) == []
    prefetched = ReplayPrefetcher(replay, 4, 3, 2, "cpu")
    with pytest.raises(RuntimeError, match="training failed"):
        with prefetched:
            next(prefetched)
            raise RuntimeError("training failed")
    assert prefetched._closed and prefetched._future is None
    def fail(*args, **kwargs):
        raise RuntimeError("sample failed")
    monkeypatch.setattr(replay, "sample", fail)
    prefetched = ReplayPrefetcher(replay, 4, 3, 2, "cpu")
    with pytest.raises(RuntimeError, match="sample failed"):
        with prefetched:
            next(prefetched)
    assert prefetched._closed and prefetched._future is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA transfer regression")
@pytest.mark.parametrize("compact,section", [(False, None), (True, None), (True, slice(4, 13))])
def test_cuda_prefetch_exact_values_rng_and_stream_lifetime(compact, section):
    replay = CompactSequenceReplay(80)
    replay.add(batch())
    torch.manual_seed(91)
    expected = [replay.sample(7, 4) for _ in range(4)]
    if section is not None:
        for item in expected:
            for key in ("state", ("next", "state")):
                item[key][..., :section.start] = 0
                item[key][..., section.stop:] = 0
    expected_rng = torch.get_rng_state()
    torch.manual_seed(91)
    parameter = torch.nn.Parameter(torch.ones(17, 2, device="cuda"))
    with ReplayPrefetcher(replay, 7, 4, 4, "cuda", compact=compact, state_columns=section) as prefetched:
        for found, wanted in zip(prefetched, expected):
            torch.testing.assert_close(found.cpu(), wanted, rtol=0, atol=0)
            assert not found["state"].is_inference()
            (found["state"] @ parameter).sum().backward()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert torch.isfinite(parameter.grad).all()


@pytest.mark.parametrize("length", [1, 3, 6])
def test_compact_transport_matches_every_draw_including_all_exception_positions(length, monkeypatch):
    sources = [batch(n=2, t=6), batch(n=3, t=8, offset=10000)]
    replay = CompactSequenceReplay(100)
    for source in sources:
        # Sparse exceptions at first, interior and last transitions, plus dense
        # exceptions in one environment. Neither terminal nor noise assumptions.
        source["next", "state"][0, 0] += 55
        source["next", "state"][0, 3] *= -1
        source["next", "state"][0, -1].fill_(-721)
        source["next", "state"][1] += 901
        source["matrix"] = torch.randn(*source.batch_size, 2, 3).transpose(-1, -2)
        source["empty"] = torch.empty(*source.batch_size, 0)
        source["next", "done"] = torch.zeros(*source.batch_size, 1, dtype=torch.bool)
        replay.add(source)
    total = replay._eligible(length)[2]
    draws = torch.arange(total - 1, -1, -1)
    expected = replay.sample(total, length, draws=draws)
    # The compact path must never first allocate an ordinary duplicate batch.
    monkeypatch.setattr(replay, "sample", lambda *a, **kw: pytest.fail("full sample called"))
    with torch.inference_mode():
        transport = replay.sample_transport(total, length, draws=draws)
        actual = transport.materialize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert all(not value.is_inference() for value in actual.values(True, True))
    assert transport.materialized_bytes == expected.bytes()
    saved_targets = actual["next", "state"].clone()
    actual["state"].zero_()
    torch.testing.assert_close(actual["next", "state"], saved_targets, rtol=0, atol=0)
    actual.zero_()
    torch.testing.assert_close(replay.sample_compact(total, length, draws=draws), expected, rtol=0, atol=0)


def test_compact_transport_reports_actual_bytes_and_backpropagates_identical_gradients():
    replay = CompactSequenceReplay(80)
    replay.add(batch())
    draws = torch.tensor([0, 1, 4, 2])
    transport = replay.sample_transport(4, 8, draws=draws)
    assert transport.frames.shape == (4, 9, 17)
    assert transport.exception_values.numel() == 0
    assert transport.bytes < transport.materialized_bytes
    expected = replay.sample(4, 8, draws=draws)
    weight = torch.nn.Parameter(torch.randn(17, 2))
    def gradient(sample):
        loss = ((sample["state"] @ weight) - sample["next", "state"][..., :2]).square().mean()
        return torch.autograd.grad(loss, weight)[0]
    torch.testing.assert_close(gradient(transport.materialize()), gradient(expected), rtol=0, atol=0)


def test_start_only_sampling_uniform_valid_domain_tail_ownership_and_cache_invalidation():
    sources = [batch(n=2, t=4), batch(n=3, t=3, offset=10000)]
    replay = CompactSequenceReplay(32)
    rows = []
    for source in sources:
        source["is_init"][0, 0] = True
        source["is_init"][1, 2] = True
        source["next", "done"] = torch.ones(*source.batch_size, 1, dtype=torch.bool)
        replay.add(source)
        rows.extend(source.select("state", "is_init").reshape(-1)[~source["is_init"].reshape(-1)].unbind(0))
    draws = torch.arange(len(rows) - 1, -1, -1)
    expected = torch.stack(rows[::-1])
    with torch.inference_mode():
        actual = replay.sample_starts(len(rows), draws=draws)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert set(actual.keys()) == {"state", "is_init"}
    assert not actual["is_init"].any()
    assert not actual["state"].is_inference()
    actual.zero_()
    torch.testing.assert_close(replay.sample_starts(len(rows), draws=draws), expected, rtol=0, atol=0)
    replay.add(batch(n=5, t=5, offset=20000))  # Evicts both earlier chunks.
    assert replay.sample_starts(50)["state"].min() >= 20000
    empty = CompactSequenceReplay(8)
    initial = batch(n=2, t=4)
    initial["is_init"].fill_(True)
    empty.add(initial)
    with pytest.raises(ValueError, match="non-reset"):
        empty.sample_starts(1)


def test_compact_prefetch_preserves_sampling_rng_and_reports_transport_volume():
    replay = CompactSequenceReplay(80)
    replay.add(batch())
    torch.manual_seed(413)
    expected = [replay.sample(9, 4) for _ in range(4)]
    final_rng = torch.get_rng_state()
    torch.manual_seed(413)
    with ReplayPrefetcher(replay, 9, 4, 4, "cpu", compact=True) as prefetched:
        actual = list(prefetched)
    assert torch.equal(torch.get_rng_state(), final_rng)
    assert prefetched.transferred_bytes < prefetched.materialized_bytes
    assert prefetched.materialized_bytes == sum(item.bytes() for item in expected)
    for found, wanted in zip(actual, expected):
        torch.testing.assert_close(found, wanted, rtol=0, atol=0)


@pytest.mark.parametrize("section", [slice(0, 8), slice(4, 13), slice(9, 17)])
def test_explicit_projection_retains_exact_selected_targets_and_zeros_only_omitted_columns(section):
    source = batch(n=3, t=5)
    source["next", "state"][0, 1] += 99
    source["next", "state"][1, -1] *= -1
    replay = CompactSequenceReplay(20)
    replay.add(source)
    draws = torch.arange(replay._eligible(3)[2])
    expected = replay.sample(len(draws), 3, draws=draws)
    projected = replay.sample_transport(len(draws), 3, draws=draws, state_columns=section)
    assert projected.frames.shape[-1] == section.stop - section.start
    actual = projected.materialize()
    for key in ("state", ("next", "state")):
        wanted = torch.zeros_like(expected[key])
        wanted[..., section] = expected[key][..., section]
        torch.testing.assert_close(actual[key], wanted, rtol=0, atol=0)
    assert projected.materialized_bytes == expected.bytes()
    for key in expected.keys(True, True):
        if key not in ("state", ("next", "state")):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    with ReplayPrefetcher(replay, 4, 3, 2, "cpu", compact=True, state_columns=section) as prefetch:
        for item in prefetch:
            assert not item["state"][..., :section.start].any()
            assert not item["next", "state"][..., section.stop:].any()


def test_projection_is_explicit_and_rejects_invalid_or_ambiguous_columns():
    replay = CompactSequenceReplay(40)
    replay.add(batch())
    for section in (slice(None, 10), slice(2, None), slice(-1, 2), slice(0, 99),
                    slice(3, 3), slice(0, 4, 2), torch.tensor([0, 2])):
        with pytest.raises(ValueError, match="contiguous state slice"):
            replay.sample_transport(2, 3, state_columns=section)
    with pytest.raises(ValueError, match="requires compact"):
        ReplayPrefetcher(replay, 2, 3, 2, "cpu", state_columns=slice(0, 4))


def test_context_projection_preserves_autoregressive_physical_losses_and_all_parameter_gradients():
    from test_wm_physics import make_model
    torch.manual_seed(713)
    model, initial = make_model()
    section = model.codec.slices["wm_context"]
    actions = torch.randn(4, 3, 1)
    states, targets = [], []
    state = initial.clone()
    with torch.no_grad():
        for t in range(3):
            state[..., :section.start] = torch.randn_like(state[..., :section.start]) * 100
            states.append(state.clone())
            state = model.baseline_prediction(state, actions[:, t])
            targets.append(state.clone())
    source = TensorDict({"state": torch.stack(states, 1), "action": actions,
                         "is_init": torch.zeros(4, 3, 1, dtype=torch.bool),
                         "next": TensorDict({"state": torch.stack(targets, 1)}, [4, 3])}, [4, 3])
    replay = CompactSequenceReplay(12)
    replay.add(source)
    draws = torch.arange(4)
    full = replay.sample(4, 3, draws=draws)
    compact = replay.sample_compact(4, 3, draws=draws, state_columns=section)
    def evaluate(sample):
        state, all_losses = sample["state"][:, 0], []
        for t in range(3):
            values, state = model.transition_loss(state, sample["action"][:, t],
                                                 sample["next", "state"][:, t], return_prediction=True)
            all_losses.append(values)
        total = sum(value[key].mean() for value in all_losses for key in ("dynamics", "auxiliary"))
        gradients = torch.autograd.grad(total, tuple(model.parameters()), allow_unused=True)
        return all_losses, gradients
    expected_losses, expected_gradients = evaluate(full)
    actual_losses, actual_gradients = evaluate(compact)
    for actual, expected in zip(actual_losses, expected_losses):
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    for actual, expected in zip(actual_gradients, expected_gradients):
        if expected is None:
            assert actual is None
        else:
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
