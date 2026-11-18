"""Independent tests of logical BPTT/critic batches and real-frame scheduling."""

import copy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.learning.wm.algorithm import WM, frozen_parameters
from test_wm_training import make_policy, make_rollout, assert_tree_equal


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(161)
    yield
    torch.set_num_threads(previous)


def starts_for(policy, count=7):
    policy._add_replay(make_rollout(policy))
    starts = policy.replay.sample_starts(count, device="cpu")
    starts["is_init"][2] = True
    return starts


def test_actor_uneven_microbatches_equal_full_logical_gradient_and_detached_critic_data():
    policy = make_policy(imagination_horizon=5)
    starts = starts_for(policy)
    denominator = (~starts["is_init"]).sum()
    parameters = [p for group in policy.opt_policy.param_groups for p in group["params"]]
    with frozen_parameters(policy.world_model, policy.critic):
        policy.opt_policy.zero_grad(set_to_none=True)
        full_loss, expected_payload, _, _, _ = policy._imagine_microbatch(starts, 5, denominator)
        full_loss.backward()
        expected_gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
        policy.opt_policy.zero_grad(set_to_none=True)
        losses, payloads = [], []
        for micro in starts.split(2):
            loss, payload, _, _, _ = policy._imagine_microbatch(micro, 5, denominator)
            loss.backward()
            losses.append(loss.detach())
            payloads.append(payload)
    torch.testing.assert_close(torch.stack(losses).sum(), full_loss.detach())
    for actual, expected in zip(parameters, expected_gradients):
        if expected is None:
            assert actual.grad is None
        else:
            torch.testing.assert_close(actual.grad, expected, rtol=1e-4, atol=2e-7)
    payload = torch.cat(payloads)
    for key, value in expected_payload.items(True, True):
        torch.testing.assert_close(payload[key], value)
        assert not payload[key].requires_grad
    assert not payload["valid"][2].any()
    assert "wm_context" not in payload["observation"].keys()
    assert all(p.grad is None for p in policy.world_model.parameters())
    assert all(p.grad is None for p in policy.critic.parameters())


def test_captured_callback_preserves_every_critic_payload_tensor_and_mean_action():
    policy = make_policy(imagination_horizon=5)
    starts = starts_for(policy)
    denominator = (~starts["is_init"]).sum()
    policy._actor_backward_horizon = 5
    with frozen_parameters(policy.world_model, policy.critic):
        expected_loss, payload, metrics, action, std = policy._imagine_microbatch(starts, 5, denominator)
        actual_loss, outputs = policy._actor_loss_tensors(starts["state"], starts["is_init"], denominator)
    torch.testing.assert_close(actual_loss, expected_loss)
    for key, tensor in payload.items(True, True):
        path = (key,) if isinstance(key, str) else key
        torch.testing.assert_close(outputs["payload/" + "/".join(path)], tensor)
    for key, tensor in metrics.items():
        torch.testing.assert_close(outputs["metric/" + key], tensor)
    torch.testing.assert_close(outputs["action"], action)
    torch.testing.assert_close(outputs["std"], std)


@pytest.mark.parametrize("epochs,minibatches", [(1, 1), (5, 4)])
def test_critic_microbatching_preserves_updates_sample_passes_and_weights(epochs, minibatches):
    policy = make_policy(critic_epochs=epochs, critic_num_minibatches=minibatches, critic_microbatch_size=3)
    starts = starts_for(policy, 13)
    with frozen_parameters(policy.world_model, policy.critic):
        _, payload, _, _, _ = policy._imagine_microbatch(starts, 3, (~starts["is_init"]).sum())
    other = make_policy(critic_epochs=epochs, critic_num_minibatches=minibatches, critic_microbatch_size=1000)
    # Double precision isolates sample weighting from GEMM accumulation order
    # across Adam updates (tiny FP32 gradients amplify rounding).
    policy.critic.double()
    other.critic.double()
    payload = payload.apply(lambda value: value.double() if value.is_floating_point() else value)
    other.critic.load_state_dict(copy.deepcopy(policy.critic.state_dict()))
    actor_before = copy.deepcopy(policy.actor.state_dict())
    dynamics_before = copy.deepcopy(policy.world_model.state_dict())
    torch.manual_seed(1234)
    actual = policy._fit_imagined_critic(payload)
    torch.manual_seed(1234)
    expected = other._fit_imagined_critic(payload)
    assert actual["critic/updates"] == expected["critic/updates"] == epochs * minibatches
    assert actual["critic/sample_passes"] == expected["critic/sample_passes"] == int(payload["valid"].sum()) * epochs
    assert actual["critic/value_loss"] == pytest.approx(expected["critic/value_loss"], rel=1e-4)
    for first, second in zip(policy.critic.parameters(), other.critic.parameters()):
        torch.testing.assert_close(first, second, rtol=1e-9, atol=1e-10)
    assert_tree_equal(policy.actor.state_dict(), actor_before)
    assert_tree_equal(policy.world_model.state_dict(), dynamics_before)


def test_frame_mode_uses_actual_collected_transitions_and_interleaves_model_actor(monkeypatch):
    policy = make_policy(schedule_mode="frames", wm_predictions_per_frame=1.,
                         actor_starts_per_frame=.5, warmup_model_updates=1,
                         imagination_horizon=5, require_model_validation=False)
    order = []
    for method, event in (("_fit_world_model", "model"), ("_validate_world_model", "validation"),
                          ("_imagine", "actor")):
        original = getattr(policy, method)
        def record(*args, _original=original, _event=event, **kwargs):
            order.append(_event)
            return _original(*args, **kwargs)
        monkeypatch.setattr(policy, method, record)
    rollout = make_rollout(policy)
    metrics = policy.train_op(rollout)
    assert rollout.numel() == 16
    assert order == ["model", "validation", "actor", "model", "actor"]
    assert metrics["wm/model_prediction_samples"] == 16
    assert metrics["wm/actor_starts"] == 8
    assert metrics["wm/actor_imagined_steps"] == 40
    assert metrics["wm/prediction_credit"] == metrics["wm/actor_start_credit"] == 0
    assert metrics["critic/updates"] == 2


def test_frame_warmup_retains_actor_budget_and_repaid_budget_is_checkpointed():
    policy = make_policy(schedule_mode="frames", wm_predictions_per_frame=.5,
                         actor_starts_per_frame=.25, warmup_model_updates=3,
                         require_model_validation=False)
    for iteration in range(3):
        info = policy.train_op(make_rollout(policy))
        assert info["wm/model_updates"] == 1
        assert info["wm/actor_update_attempts"] == (0 if iteration < 2 else 3)
        assert info["wm/actor_start_credit"] == (4 * (iteration + 1) if iteration < 2 else 0)
    policy._frame_schedule.accrue(1)
    saved = policy.state_dict()
    restored = make_policy(schedule_mode="frames")
    restored.load_state_dict(saved)
    assert restored._frame_schedule.state_dict() == policy._frame_schedule.state_dict()
    assert restored._optimizer_model_updates_since_resume == 0
    del saved["wm_frame_schedule"]
    restored.load_state_dict(saved)
    assert restored._frame_schedule.frames == 0
    assert restored._frame_schedule.actor_credit == 0


def test_normalizer_snapshot_preserves_captured_tensor_addresses_between_updates():
    policy = make_policy()
    from torchrl.envs.transforms import VecNorm
    normalizer = VecNorm(in_keys=["policy"], decay=0.)
    calibration = make_rollout(policy)[:, 0].clone()
    normalizer(calibration)
    policy.set_observation_normalizer(normalizer, "train")
    policy._snapshot_normalizer()
    original = tuple(t.data_ptr() for t in policy._norm_snapshot["policy"])
    old_loc = policy._norm_snapshot["policy"][0].clone()
    calibration["policy"] += 10
    normalizer(calibration)
    policy._snapshot_normalizer()
    assert tuple(t.data_ptr() for t in policy._norm_snapshot["policy"]) == original
    assert not torch.equal(policy._norm_snapshot["policy"][0], old_loc)


def test_collection_actor_drift_logging_preserves_training_rng_and_optimizer_results():
    disabled = make_policy(actor_drift_batch_size=0, actor_updates=3)
    rollout = make_rollout(disabled)
    enabled = make_policy(actor_drift_batch_size=7, actor_updates=3)
    enabled.load_state_dict(copy.deepcopy(disabled.state_dict()))
    torch.manual_seed(932)
    expected = disabled.train_op(rollout)
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(932)
    actual = enabled.train_op(rollout)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert_tree_equal(enabled.state_dict(), disabled.state_dict())
    for key, value in expected.items():
        if not key.startswith("performance/"):
            assert actual[key] == value, key
    assert "actor/collection_mean_kl" not in expected
    assert actual["actor/collection_mean_kl"] > 0
    assert actual["actor/collection_action_delta_rms"] > 0
    assert actual["actor/collection_anchor_samples"] == 7


def test_collection_actor_drift_uses_fixed_anchor_std_without_mutating_modules_or_rng(monkeypatch):
    policy = make_policy(actor_drift_batch_size=5)
    starts_for(policy)
    snapshot = copy.deepcopy(policy.state_dict())
    original = policy._teacher_dist
    mean_offset = [0.]
    def noisy_distribution(state):
        distribution = original(state)
        # Deliberately consume CPU RNG to test both capture and evaluation
        # isolation independently of the current deterministic teacher.
        torch.rand(7)
        return SimpleNamespace(mean=distribution.mean + mean_offset[0], stddev=distribution.stddev)
    monkeypatch.setattr(policy, "_teacher_dist", noisy_distribution)
    rng = torch.get_rng_state().clone()
    anchor = policy._capture_collection_actor_anchor()
    assert torch.equal(torch.get_rng_state(), rng)
    assert policy._collection_actor_drift(anchor)["actor/collection_mean_kl"] == 0
    mean_offset[0] = .1
    actual = policy._collection_actor_drift(anchor)
    assert torch.equal(torch.get_rng_state(), rng)
    expected = (.5 * (.1 / anchor[2]).square().sum(-1)).mean().item()
    assert actual["actor/collection_mean_kl"] == pytest.approx(expected)
    assert actual["actor/collection_action_delta_rms"] == pytest.approx(.1)
    assert_tree_equal(policy.state_dict(), snapshot)


@pytest.mark.parametrize("free,mode,device,expected", [
    (1023, "auto", "cuda:0", False), (1024, "auto", "cuda:0", True),
    (100000, "cpu", "cuda:0", False), (100000, "auto", "cpu", False),
])
def test_critic_gpu_cache_requires_payload_space_plus_full_memory_reserve(monkeypatch, free, mode, device, expected):
    payload = TensorDict({"feature": torch.zeros(64)}, [64])
    moved = []
    # Exercise allocation policy on CPU; the actual CUDA equivalence test
    # below separately verifies device transfers and persistent pointer reuse.
    def simulated_transfer(source, destination):
        moved.append(destination)
        return source.clone()
    monkeypatch.setattr(TensorDict, "to", simulated_transfer)
    policy = SimpleNamespace(device=device, cfg=SimpleNamespace(
        critic_cache_device=mode, critic_cache_reserve_gib=768 / 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free, 100000))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result, metrics = WM._cache_imagined_critic(policy, payload)
    assert (result is not payload) == expected
    assert metrics["performance/critic_cache_bytes"] == 256
    assert metrics["performance/critic_gpu_cache"] == float(expected)
    assert moved == ([device] if expected else [])


def test_persistent_critic_cache_refreshes_all_keys_and_only_rechecks_reserve_on_incompatible_payload(monkeypatch):
    policy = SimpleNamespace(device="cuda:0", cfg=SimpleNamespace(
        critic_cache_device="auto", critic_cache_reserve_gib=1.))
    free = [10 * 1024 ** 3]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[0], 20 * 1024 ** 3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    # A CPU copy stands in for H2D here; allocation/reuse logic is unchanged.
    monkeypatch.setattr(TensorDict, "to", lambda source, destination: source.clone())
    payload = TensorDict({"observation": TensorDict({"x": torch.arange(12.).reshape(3, 4)}, [3]),
                          "target": torch.ones(3, 2), "valid": torch.tensor([True, False, True])}, [3])
    cache, first = WM._cache_imagined_critic(policy, payload)
    pointers = {key: value.data_ptr() for key, value in cache.items(True, True)}
    assert first["performance/critic_cache_reused"] == 0
    refreshed = payload.apply(lambda value: value + 9 if value.is_floating_point() else ~value)
    free[0] = 0  # Already-owned cache must remain usable despite allocator reserves.
    again, second = WM._cache_imagined_critic(policy, refreshed)
    assert again is cache
    assert second["performance/critic_cache_reused"] == second["performance/critic_gpu_cache"] == 1
    assert second["performance/critic_cache_fallback"] == 0
    for key, value in refreshed.items(True, True):
        assert again[key].data_ptr() == pointers[key]
        assert again[key].data_ptr() != value.data_ptr()
        torch.testing.assert_close(again[key], value, rtol=0, atol=0)
    # A dtype or shape change needs an allocation and must respect free memory.
    changed = refreshed.apply(lambda value: value.double() if value.is_floating_point() else value)
    fallback, info = WM._cache_imagined_critic(policy, changed)
    assert fallback is changed and info["performance/critic_cache_fallback"] == 1
    assert policy._critic_payload_cache is policy._critic_payload_signature is None
    free[0] = 10 * 1024 ** 3
    resized = refreshed[:2].clone()
    replacement, info = WM._cache_imagined_critic(policy, resized)
    assert replacement.batch_size == torch.Size([2])
    assert info["performance/critic_gpu_cache"] == 1 and info["performance/critic_cache_reused"] == 0
    policy.cfg.critic_cache_device = "cpu"
    uncached, info = WM._cache_imagined_critic(policy, resized)
    assert uncached is resized
    assert policy._critic_payload_cache is policy._critic_payload_signature is None
    assert info["performance/critic_gpu_cache"] == info["performance/critic_cache_fallback"] == 0


def test_checkpoint_load_clears_transient_critic_payload_cache():
    policy = make_policy()
    checkpoint = copy.deepcopy(policy.state_dict())
    policy._critic_payload_cache = TensorDict({"valid": torch.ones(2, dtype=torch.bool)}, [2])
    policy._critic_payload_signature = ("transient",)
    policy._critic_cache_reclaim_signatures = {("transient",)}
    policy.load_state_dict(checkpoint)
    assert policy._critic_payload_cache is policy._critic_payload_signature is None
    assert not policy._critic_cache_reclaim_signatures


def test_critic_cache_acquisition_uses_full_logical_shape_without_copying_template(monkeypatch):
    policy = SimpleNamespace(device="cuda:0", cfg=SimpleNamespace(
        critic_cache_device="auto", critic_cache_reserve_gib=0.))
    template = TensorDict({"observation": TensorDict({"x": torch.ones(2, 3, 4)}, [2, 3]),
                           "target": torch.ones(2, 3, 2), "valid": torch.ones(2, 3, dtype=torch.bool)}, [2, 3])
    free = [10_000]
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[0], 10_000))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    original_empty = torch.empty
    def simulated_allocation(*args, **kwargs):
        assert str(kwargs.pop("device")) == "cuda:0"
        return original_empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", simulated_allocation)
    monkeypatch.setattr(TensorDict, "to", lambda *_args, **_kwargs: pytest.fail("Acquisition must not copy a template"))
    cache, metrics = WM._cache_imagined_critic(policy, template, batch_size=(5, 3))
    assert cache.batch_size == torch.Size([5, 3])
    assert metrics["performance/critic_cache_bytes"] == 5 * 3 * (4 * 4 + 2 * 4 + 1)
    assert metrics["performance/critic_gpu_cache"] == 1
    for key, value in cache.items(True, True):
        assert value.data_ptr() != template[key].data_ptr()
        value.fill_(7)
    free[0] = 0
    reused, metrics = WM._cache_imagined_critic(policy, template, batch_size=(5, 3))
    assert reused is cache and metrics["performance/critic_cache_reused"] == 1
    for key, value in cache.items(True, True):
        assert (value == (True if value.dtype == torch.bool else 7)).all()
    rejected, metrics = WM._cache_imagined_critic(policy, template, batch_size=(6, 3))
    assert rejected is None and metrics["performance/critic_cache_fallback"] == 1


@pytest.mark.parametrize("recovered_free", [1023, 1024])
def test_critic_cache_reclaims_unused_memory_once_without_relaxing_reserve(monkeypatch, recovered_free):
    policy = SimpleNamespace(device="cuda:0", cfg=SimpleNamespace(
        critic_cache_device="auto", critic_cache_reserve_gib=768 / 1024 ** 3))
    payload = TensorDict({"feature": torch.arange(64.)}, [64])  # 256 bytes + 768-byte reserve.
    free, releases = [100], []
    def release_unused():
        releases.append(True)
        free[0] = recovered_free
    monkeypatch.setattr(torch.cuda, "empty_cache", release_unused)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (free[0], 100000))
    monkeypatch.setattr(TensorDict, "to", lambda source, destination: source.clone())
    first, metrics = WM._cache_imagined_critic(policy, payload)
    expected_gpu = recovered_free >= 1024
    assert metrics["performance/critic_cache_reclaim_attempted"] == 1
    assert metrics["performance/critic_cache_reclaimed_free_bytes"] == recovered_free - 100
    assert metrics["performance/critic_cache_free_bytes"] == recovered_free
    assert metrics["performance/critic_gpu_cache"] == float(expected_gpu)
    assert metrics["performance/critic_cache_fallback"] == float(not expected_gpu)
    assert (first is not payload) == expected_gpu
    torch.testing.assert_close(first["feature"], payload["feature"], rtol=0, atol=0)
    free[0] = 0
    for _ in range(3):
        result, metrics = WM._cache_imagined_critic(policy, payload)
        assert metrics["performance/critic_cache_reclaim_attempted"] == 0
        assert metrics["performance/critic_gpu_cache"] == float(expected_gpu)
        assert result is (first if expected_gpu else payload)
    assert len(releases) == 1
    # A new signature is a separate bounded allocation attempt, not a retry
    # on every ordinary actor update after a genuine memory shortage.
    _, changed = WM._cache_imagined_critic(policy, payload[:63])
    assert changed["performance/critic_cache_reclaim_attempted"] == 1
    assert len(releases) == 2


def test_direct_microbatch_writes_preserve_reused_outputs_rng_updates_and_cpu_fallback(monkeypatch):
    streamed = make_policy(imagination_horizon=3, actor_microbatch_size=2,
                           critic_epochs=2, critic_num_minibatches=2, critic_cache_device="cpu")
    starts = starts_for(streamed, 7)
    direct = make_policy(imagination_horizon=3, actor_microbatch_size=2,
                         critic_epochs=2, critic_num_minibatches=2, critic_cache_device="cpu")
    direct.load_state_dict(copy.deepcopy(streamed.state_dict()))
    captured_payloads = {}
    for name, policy in (("streamed", streamed), ("direct", direct)):
        original_fit = policy._fit_imagined_critic
        def capture(payload, *, cache_metrics=None, _name=name, _fit=original_fit):
            captured_payloads[_name] = payload.clone()
            return _fit(payload, cache_metrics=cache_metrics)
        monkeypatch.setattr(policy, "_fit_imagined_critic", capture)
    torch.manual_seed(742)
    expected = streamed._imagine(starts)
    expected_rng = torch.get_rng_state().clone()

    # Stand in for owned CUDA storage while exercising the production direct
    # slice writes, including a final uneven microbatch, entirely on CPU.
    def acquire(template, *, batch_size=None):
        assert batch_size == (len(starts), 3)
        storage = TensorDict({key: torch.empty((*batch_size, *value.shape[2:]), dtype=value.dtype)
                              for key, value in template.items(True, True)}, batch_size)
        return storage, {"performance/critic_gpu_cache": 1.}
    monkeypatch.setattr(direct, "_cache_imagined_critic", acquire)
    original_backward = direct._actor_microbatch_backward
    output_storage = {}
    def reused_outputs(*args, **kwargs):
        payload, metrics, action, std, captured = original_backward(*args, **kwargs)
        shape = tuple(payload.batch_size)
        if shape not in output_storage:
            output_storage[shape] = payload.clone()
        else:
            output_storage[shape].update_(payload)
        return output_storage[shape], metrics, action, std, captured
    monkeypatch.setattr(direct, "_actor_microbatch_backward", reused_outputs)
    original_to = TensorDict.to
    def forbid_payload_roundtrip(source, *args, **kwargs):
        if source.ndim == 2 and source.shape[1] == 3 and args and args[0] == "cpu":
            pytest.fail("Direct cache must not stage critic microbatch payloads on CPU")
        return original_to(source, *args, **kwargs)
    monkeypatch.setattr(TensorDict, "to", forbid_payload_roundtrip)
    torch.manual_seed(742)
    actual = direct._imagine(starts)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for key, value in captured_payloads["streamed"].items(True, True):
        torch.testing.assert_close(captured_payloads["direct"][key], value, rtol=0, atol=0)
    for name in ("actor", "critic"):
        assert_tree_equal(getattr(direct, name).state_dict(), getattr(streamed, name).state_dict())
    for name in ("actor/model_return", "actor/grad_norm", "critic/value_loss", "critic/updates", "critic/sample_passes"):
        assert actual[name] == expected[name]
    assert actual["critic/updates"] == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA critic cache")
def test_gpu_critic_cache_and_cpu_streaming_fit_identical_minibatches_and_parameters(monkeypatch):
    streamed = make_policy(critic_epochs=2, critic_num_minibatches=4, critic_microbatch_size=3,
                          critic_cache_device="cpu")
    starts = starts_for(streamed, 13)
    with frozen_parameters(streamed.world_model, streamed.critic):
        _, payload, _, _, _ = streamed._imagine_microbatch(starts, 3, (~starts["is_init"]).sum())
    cached = make_policy(critic_epochs=2, critic_num_minibatches=4, critic_microbatch_size=3,
                         critic_cache_device="auto", critic_cache_reserve_gib=0.)
    cached.critic.load_state_dict(copy.deepcopy(streamed.critic.state_dict()))
    for policy in (streamed, cached):
        policy.critic.cuda()
        policy.device = torch.device("cuda")
    torch.manual_seed(654)
    first = streamed._fit_imagined_critic(payload)
    torch.manual_seed(654)
    second = cached._fit_imagined_critic(payload)
    assert first["performance/critic_gpu_cache"] == 0
    assert second["performance/critic_gpu_cache"] == 1
    assert first["critic/updates"] == second["critic/updates"] == 8
    assert first["critic/sample_passes"] == second["critic/sample_passes"]
    assert first["critic/value_loss"] == pytest.approx(second["critic/value_loss"], rel=1e-6)
    for actual, expected in zip(streamed.critic.parameters(), cached.critic.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    pointers = {key: value.data_ptr() for key, value in cached._critic_payload_cache.items(True, True)}
    payload = payload.apply(lambda value: value + .03 if value.is_floating_point() else value)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (0, 32 * 1024 ** 3))
    torch.manual_seed(876)
    first = streamed._fit_imagined_critic(payload)
    torch.manual_seed(876)
    second = cached._fit_imagined_critic(payload)
    assert second["performance/critic_gpu_cache"] == second["performance/critic_cache_reused"] == 1
    assert second["performance/critic_cache_fallback"] == 0
    for key, value in cached._critic_payload_cache.items(True, True):
        assert value.data_ptr() == pointers[key]
        torch.testing.assert_close(value.cpu(), payload[key], rtol=0, atol=0)
    assert first["critic/value_loss"] == pytest.approx(second["critic/value_loss"], rel=1e-6)
    for actual, expected in zip(streamed.critic.parameters(), cached.critic.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires direct CUDA critic cache writes")
def test_direct_cuda_payload_writes_preserve_cpu_permutation_rng_and_critic_parameters():
    streamed = make_policy(critic_epochs=2, critic_num_minibatches=4, critic_microbatch_size=3,
                           critic_cache_device="cpu")
    starts = starts_for(streamed, 13)
    with frozen_parameters(streamed.world_model, streamed.critic):
        _, payload, _, _, _ = streamed._imagine_microbatch(starts, 3, (~starts["is_init"]).sum())
    direct = make_policy(critic_epochs=2, critic_num_minibatches=4, critic_microbatch_size=3,
                         critic_cache_device="auto", critic_cache_reserve_gib=0.)
    direct.critic.load_state_dict(copy.deepcopy(streamed.critic.state_dict()))
    for policy in (streamed, direct):
        policy.critic.cuda()
        policy.device = torch.device("cuda")
    source = payload[:3].cuda()
    cache, cache_metrics = direct._cache_imagined_critic(source, batch_size=payload.batch_size)
    for begin in range(0, len(payload), 3):
        part = payload[begin:begin + 3].cuda()
        for key, value in part.items(True, True):
            cache[key][begin:begin + len(part)].copy_(value)
            value.zero_()  # Source storage can be reused immediately.
    for key, value in payload.items(True, True):
        torch.testing.assert_close(cache[key].cpu(), value, rtol=0, atol=0)
    torch.manual_seed(987)
    expected = streamed._fit_imagined_critic(payload)
    expected_cpu_rng, expected_cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
    torch.manual_seed(987)
    actual = direct._fit_imagined_critic(cache, cache_metrics=cache_metrics)
    assert torch.equal(torch.get_rng_state(), expected_cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), expected_cuda_rng)
    assert actual["critic/sample_passes"] == expected["critic/sample_passes"]
    assert actual["critic/value_loss"] == expected["critic/value_loss"]
    for actual, expected in zip(streamed.critic.parameters(), direct.critic.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
