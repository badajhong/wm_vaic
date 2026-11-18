"""CUDA capture preserves the ordinary model loss, gradients and optimizer."""

import copy
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from active_adaptation.learning.wm.execution import CUDAGraphBackward


def test_execution_rejects_cpu_parameters_and_empty_model():
    with pytest.raises(ValueError, match="CUDA device"):
        CUDAGraphBackward(lambda x: x, nn.Linear(2, 1).parameters())
    with pytest.raises(ValueError, match="requires model parameters"):
        CUDAGraphBackward(lambda x: x, [])


def test_accumulation_preserves_replayed_aliases_eviction_unused_gradients_and_clipping_on_cpu():
    # Exercise the real run/cache/accumulation code with a CPU stand-in for
    # CUDA replay. This does not claim to test CUDA capture on a CPU device.
    parameter, unused = nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(2))
    execution = object.__new__(CUDAGraphBackward)
    execution.parameters = (parameter, unused)
    execution.device = torch.device("cpu")
    execution._cache = OrderedDict()
    execution._signature = execution._parameter_signature()
    execution.max_cached_graphs = 1
    execution.captures = execution.replays = 0
    def capture(inputs):
        static = tuple(value.clone() for value in inputs)
        gradient, loss, payload = torch.zeros(3), torch.zeros(()), torch.zeros_like(static[0])
        @torch.no_grad()
        def replay():
            gradient.copy_(static[0].sum(0))
            loss.copy_((gradient * parameter).sum())
            payload.copy_(static[0])
        execution.captures += 1
        return SimpleNamespace(inputs=static, gradients=(gradient, None), loss=loss,
                               metrics={"payload": payload}, graph=SimpleNamespace(replay=replay))
    execution._capture = capture
    first = torch.arange(6.).reshape(2, 3)
    _, metrics = execution.run((first,))
    first_payload = metrics["payload"]
    assert parameter.grad is next(iter(execution._cache.values())).gradients[0]
    parameter.grad = parameter.grad.detach()  # Storage alias, different Python tensor.
    unused.grad = torch.full_like(unused, 7)
    second = torch.full((2, 3), 5.)
    execution.run((second,), accumulate=True)
    expected = first.sum(0) + second.sum(0)
    torch.testing.assert_close(parameter.grad, expected, rtol=0, atol=0)
    torch.testing.assert_close(unused.grad, torch.full_like(unused, 7))
    torch.testing.assert_close(first_payload, second)  # Returned metrics are reused.
    parameter.grad.mul_(.25)
    third = torch.full((4, 3), 2.)  # Forces capture eviction.
    execution.run((third,), accumulate=True)
    torch.testing.assert_close(parameter.grad, expected * .25 + third.sum(0), rtol=0, atol=0)
    previous = parameter.grad.clone()
    execution.clear()
    torch.testing.assert_close(parameter.grad, previous)
    parameter.grad = unused.grad = None
    execution.run((first,), accumulate=True)
    assert unused.grad is None
    torch.testing.assert_close(parameter.grad, first.sum(0))
    assert parameter.grad.data_ptr() != next(iter(execution._cache.values())).gradients[0].data_ptr()


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graph execution")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(7, 16), nn.LayerNorm(16), nn.SiLU(), nn.Linear(16, 3))
        self.unused = nn.Parameter(torch.ones(2))
        self.register_buffer("mean", torch.zeros(7))

    def loss(self, x, target):
        prediction = self.layers(x - self.mean)
        error = (prediction - target).square().mean()
        return error, {"mse": error, "mean_prediction": prediction.mean()}


@cuda
def test_cuda_graph_matches_optimizer_steps_dynamic_buffers_and_shape_cache():
    torch.manual_seed(12)
    eager = _Model().cuda()
    captured = copy.deepcopy(eager)
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=.01)
                  for model in (eager, captured)]
    execution = CUDAGraphBackward(captured.loss, captured.parameters(), max_cached_graphs=1)
    for step, batch in enumerate((4, 4, 9, 9, 4)):
        x = torch.randn(7, batch, device="cuda").T  # Deliberately noncontiguous.
        target = torch.randn(batch, 3, device="cuda")
        with torch.no_grad():
            eager.mean.add_(.01)
            captured.mean.copy_(eager.mean)
        optimizers[0].zero_grad(set_to_none=True)
        expected_loss, expected_metrics = eager.loss(x, target)
        expected_loss.backward()
        optimizers[1].zero_grad(set_to_none=True)
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        actual_loss, actual_metrics = execution.run((x, target))
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
        torch.testing.assert_close(actual_loss, expected_loss)
        for key in actual_metrics:
            torch.testing.assert_close(actual_metrics[key], expected_metrics[key])
        for actual, expected in zip(captured.parameters(), eager.parameters()):
            if expected.grad is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, expected.grad)
        # Clipping mutates .grad; replay must overwrite those buffers fully.
        for model in (eager, captured):
            torch.nn.utils.clip_grad_norm_(model.parameters(), .05)
        for optimizer in optimizers:
            optimizer.step()
        for actual, expected in zip(captured.parameters(), eager.parameters()):
            torch.testing.assert_close(actual, expected)
        assert execution.cached_graphs == 1
    assert execution.captures == 3 and execution.replays == 5
    assert execution.has_cache
    execution.clear()
    assert not execution.has_cache


@cuda
def test_cuda_graph_returns_owned_dense_gradients_for_constants_and_input_aliases():
    constant = nn.Parameter(torch.ones(5, device="cuda"))
    alias = nn.Parameter(torch.ones(5, device="cuda"))
    def loss(x):
        value = constant.sum() + (alias * x).sum()
        return value, {}
    execution = CUDAGraphBackward(loss, [constant, alias])
    x = torch.arange(1., 6., device="cuda")
    for _ in range(2):
        execution.run((x,))
        torch.testing.assert_close(constant.grad, torch.ones_like(constant))
        torch.testing.assert_close(alias.grad, x)
        assert constant.grad.is_contiguous() and alias.grad.is_contiguous()
        constant.grad.mul_(.1)
        alias.grad.zero_()
        torch.testing.assert_close(x, torch.arange(1., 6., device="cuda"))


@cuda
def test_cuda_graph_rng_warmup_is_invisible_and_replay_draws_fresh_randomness():
    parameter = nn.Parameter(torch.ones(5, device="cuda"))
    def loss(x):
        noise = torch.rand_like(x)
        value = (x * noise * parameter).square().mean()
        return value, {"noise": noise}
    execution = CUDAGraphBackward(loss, [parameter])
    x = torch.arange(1., 6., device="cuda")
    torch.cuda.manual_seed(32)
    expected = []
    for _ in range(3):
        parameter.grad = None
        value, metrics = loss(x)
        value.backward()
        expected.append((value.detach().clone(), metrics["noise"].clone(), parameter.grad.clone()))
    # A live eager autograd graph owns its leaf accumulation stream. Only the
    # detached reference results may survive when those same leaves are used
    # to capture a new backward on a CUDA capture stream.
    del value, metrics
    expected_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(32)
    for expected_value, expected_noise, expected_gradient in expected:
        value, metrics = execution.run((x,))
        torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
        torch.testing.assert_close(metrics["noise"], expected_noise, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
    assert torch.equal(torch.cuda.get_rng_state(), expected_state)


@cuda
def test_cuda_graph_rejects_input_gradients_and_recaptures_replaced_parameter_storage():
    model = _Model().cuda()
    execution = CUDAGraphBackward(model.loss, model.parameters())
    x, y = torch.randn(4, 7, device="cuda"), torch.randn(4, 3, device="cuda")
    with pytest.raises(ValueError, match="must not require gradients"):
        execution.run((x.requires_grad_(), y))
    x = x.detach()
    execution.run((x, y))
    parameter = model.layers[0].weight
    parameter.data = parameter.data.clone()
    loss, _ = execution.run((x, y))
    expected, _ = model.loss(x, y)
    torch.testing.assert_close(loss, expected)
    assert execution.captures == 2


@cuda
def test_cuda_graph_physical_autoregressive_loss_and_gradients_match_eager():
    from test_wm_physics import make_model
    torch.manual_seed(18)
    eager, state = make_model(hidden=(32, 32))
    eager = eager.cuda()
    state = state.cuda()
    captured = copy.deepcopy(eager)
    def make_loss(model):
        def loss(state, action, target):
            first_loss, first = model.transition_loss(state, action[:, 0], target[:, 0], return_prediction=True)
            second_loss = model.transition_loss(first, action[:, 1], target[:, 1])
            value = sum(values[key].mean() for values in (first_loss, second_loss)
                        for key in ("dynamics", "auxiliary"))
            return value, {"loss": value}
        return loss
    execution = CUDAGraphBackward(make_loss(captured), captured.parameters())
    for _ in range(2):
        action = torch.randn(4, 2, 1, device="cuda")
        target = state[:, None].expand(-1, 2, -1).clone()
        eager.zero_grad(set_to_none=True)
        expected, _ = make_loss(eager)(state, action, target)
        expected.backward()
        value, _ = execution.run((state, action, target))
        torch.testing.assert_close(value, expected)
        for actual, reference in zip(captured.parameters(), eager.parameters()):
            if reference.grad is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, reference.grad, atol=1e-6, rtol=1e-5)


@cuda
def test_cuda_graph_accumulated_microbatches_match_logical_batch_and_optimizer():
    torch.manual_seed(409)
    # Use FP64 to isolate accumulation math from shape-dependent TF32 GEMM
    # rounding (one B13 GEMM versus B4/B1 GEMMs). Same-shape FP32 capture parity
    # is covered independently above and in the physical-model regression.
    eager = _Model().cuda().double()
    captured = copy.deepcopy(eager)
    batch = 13
    def micro_loss(x, y):
        prediction = captured.layers(x - captured.mean)
        loss = (prediction - y).square().sum() / (batch * y.shape[-1])
        return loss, {"trajectory": prediction, "sample_error": (prediction - y).square().sum(-1)}
    execution = CUDAGraphBackward(micro_loss, captured.parameters(), max_cached_graphs=2)
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3) for model in (eager, captured)]
    for step in range(2):
        x = torch.randn(batch, 7, device="cuda", dtype=torch.float64)
        y = torch.randn(batch, 3, device="cuda", dtype=torch.float64)
        with torch.no_grad():
            eager.mean.add_(.01)
            captured.mean.copy_(eager.mean)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        expected, _ = eager.loss(x, y)
        expected.backward()
        actual_losses, trajectory = [], []
        for begin in range(0, batch, 4):
            loss, metrics = execution.run((x[begin:begin + 4], y[begin:begin + 4]), accumulate=True)
            actual_losses.append(loss.clone())
            trajectory.append(metrics["trajectory"].clone())
            assert not metrics["trajectory"].requires_grad
        torch.testing.assert_close(sum(actual_losses), expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(torch.cat(trajectory), eager.layers(x - eager.mean), atol=1e-6, rtol=1e-5)
        for actual, reference in zip(captured.parameters(), eager.parameters()):
            if reference.grad is None:
                assert actual.grad is None
            else:
                torch.testing.assert_close(actual.grad, reference.grad, atol=2e-6, rtol=2e-5)
        for model, optimizer in zip((eager, captured), optimizers):
            torch.nn.utils.clip_grad_norm_(model.parameters(), .2)
            optimizer.step()
        for actual, reference in zip(captured.parameters(), eager.parameters()):
            torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    assert execution.captures == 2


@cuda
def test_cuda_accumulation_restores_capture_rng_and_reuses_metrics_without_losing_gradients():
    parameter = nn.Parameter(torch.ones(5, device="cuda"))
    def loss(x):
        noise = torch.rand_like(x)
        value = (x * parameter * noise).square().sum()
        return value, {"noise_trajectory": noise}
    execution = CUDAGraphBackward(loss, [parameter])
    inputs = [torch.arange(1., 6., device="cuda") + index for index in range(3)]
    torch.cuda.manual_seed(701)
    parameter.grad = None
    expected = []
    for x in inputs:
        value, metrics = loss(x)
        value.backward()
        expected.append((value.detach().clone(), metrics["noise_trajectory"].clone(), parameter.grad.clone()))
    del value, metrics
    final_rng = torch.cuda.get_rng_state()
    parameter.grad = None
    torch.cuda.manual_seed(701)
    payload_pointer = None
    for x, (expected_loss, expected_noise, expected_gradient) in zip(inputs, expected):
        value, metrics = execution.run((x,), accumulate=True)
        noise = metrics["noise_trajectory"]
        torch.testing.assert_close(value, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(noise, expected_noise, rtol=0, atol=0)
        torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)
        if payload_pointer is not None:
            assert noise.data_ptr() == payload_pointer
        payload_pointer = noise.data_ptr()
    assert torch.equal(torch.cuda.get_rng_state(), final_rng)
