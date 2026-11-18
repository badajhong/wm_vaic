"""Eager CUDA-graph execution of an unchanged model loss and backward pass.

This is an execution cache, not a module: it adds no model/checkpoint state and
does not compile kernels, step an optimizer, clip gradients, or update moments.
The callback must contain only capture-safe tensor operations, must not mutate
its inputs/model buffers, and must return ``(scalar_loss, tensor_metrics)``.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Iterable

import torch


@dataclass
class _BackwardCapture:
    graph: torch.cuda.CUDAGraph
    inputs: tuple[torch.Tensor, ...]
    gradients: tuple[torch.Tensor | None, ...]
    loss: torch.Tensor
    metrics: dict[str, torch.Tensor]


class CUDAGraphBackward:
    """Cache fixed-shape forward/backward graphs with ordinary optimizer use.

    ``run(inputs)`` copies fresh replay data into owned static inputs, replays
    exactly one loss/backward, and leaves ``parameter.grad`` ready for the
    caller's finite checks, clipping and optimizer step. Warmup/capture never
    updates parameters and restores CPU/default-device RNG states. PyTorch's
    graph-aware default CUDA generator handles random operations during replay;
    custom generators must be registered separately and are not supported here.

    Returned tensors reference capture storage and must be consumed before the
    next run. Call ``clear`` after checkpoint loads or other external execution
    changes; recreate this object if the actual Parameter objects are replaced.
    In-place updates to existing weights and moment buffers are visible to a
    graph. No gradients with respect to replay input tensors are requested.
    ``run(inputs, accumulate=True)`` adds this call's gradients to existing
    ``.grad`` values, as ordinary microbatch backward does. The callback owns
    loss scaling by the logical batch denominator; this helper never rescales.
    Accumulated buffers do not alias graph outputs, so replay, clipping and
    cache eviction cannot overwrite an earlier microbatch's contribution.
    Do not retain a live eager autograd graph on these same parameters across
    the initial capture; detached metrics and ordinary optimizer state are safe.
    """

    def __init__(self, loss_fn: Callable, parameters: Iterable[torch.nn.Parameter], *,
                 warmup_steps: int = 2, max_cached_graphs: int = 1):
        self.loss_fn = loss_fn
        self.parameters = tuple(dict.fromkeys(parameters))
        if not self.parameters:
            raise ValueError("CUDA graph backward requires model parameters")
        devices = {parameter.device for parameter in self.parameters}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            raise ValueError("CUDA graph parameters must share one CUDA device")
        if warmup_steps < 1 or max_cached_graphs < 1:
            raise ValueError("Warmup steps and graph cache capacity must be positive")
        self.device = next(iter(devices))
        self.warmup_steps = warmup_steps
        self.max_cached_graphs = max_cached_graphs
        self._cache = OrderedDict()
        self._signature = self._parameter_signature()
        self.captures = 0
        self.replays = 0

    @property
    def has_cache(self):
        return bool(self._cache)

    @property
    def cached_graphs(self):
        return len(self._cache)

    def _parameter_signature(self):
        return tuple((id(parameter), parameter.data_ptr(), parameter.device,
                      parameter.dtype, tuple(parameter.shape), parameter.requires_grad)
                     for parameter in self.parameters)

    def _input_key(self, inputs):
        if not inputs or any(not isinstance(value, torch.Tensor) for value in inputs):
            raise ValueError("CUDA graph inputs must be a nonempty tuple of tensors")
        if any(value.device != self.device for value in inputs):
            raise ValueError("CUDA graph inputs and parameters must share one device")
        if any(value.requires_grad for value in inputs):
            raise ValueError("Replay inputs must not require gradients")
        return tuple((tuple(value.shape), value.dtype, value.device) for value in inputs)

    def _evaluate(self, inputs):
        loss, metrics = self.loss_fn(*inputs)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not loss.requires_grad:
            raise ValueError("Loss callback must return a differentiable scalar tensor")
        if not isinstance(metrics, dict) or any(
            not isinstance(key, str) or not isinstance(value, torch.Tensor)
            for key, value in metrics.items()
        ):
            raise ValueError("Loss callback metrics must be a string-to-tensor dictionary")
        return loss, metrics

    def _capture(self, inputs):
        stream = torch.cuda.Stream(device=self.device)
        current = torch.cuda.current_stream(self.device)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        trainable = tuple(parameter for parameter in self.parameters if parameter.requires_grad)
        with torch.enable_grad(), torch.random.fork_rng(devices=[self.device.index]):
            with torch.cuda.stream(stream):
                static_inputs = tuple(value.detach().clone(memory_format=torch.contiguous_format)
                                      for value in inputs)
                for _ in range(self.warmup_steps):
                    loss, warmup_metrics = self._evaluate(static_inputs)
                    warmup_gradients = torch.autograd.grad(loss, trainable, allow_unused=True)
                # The optimizer may modify .grad (e.g. clipping). It must not
                # modify an input alias or expanded constant returned by grad.
                gradient_buffers = tuple(None if value is None else torch.empty_like(
                    value, memory_format=torch.contiguous_format) for value in warmup_gradients)
                # Actor callbacks may return an entire imagined critic rollout.
                # Do not retain its warmup payload alongside captured storage.
                del warmup_gradients, loss, warmup_metrics
            current.wait_stream(stream)
            with torch.cuda.graph(graph, stream=stream):
                loss, metrics = self._evaluate(static_inputs)
                # Return gradients instead of accumulating into .grad. This
                # overwrites graph-owned outputs on every replay without
                # accumulating into pre-existing .grad. Unused parameters
                # remain None, preserving optimizer
                # weight-decay/momentum semantics.
                computed = torch.autograd.grad(loss, trainable, allow_unused=True)
                for target, source in zip(gradient_buffers, computed):
                    if target is not None:
                        target.copy_(source)
            current.wait_stream(stream)
        by_parameter = dict(zip(trainable, gradient_buffers))
        gradients = tuple(by_parameter.get(parameter) for parameter in self.parameters)
        self.captures += 1
        return _BackwardCapture(graph, static_inputs, gradients,
                                loss.detach(), {key: value.detach() for key, value in metrics.items()})

    def run(self, inputs, *, accumulate=False):
        if torch.is_inference_mode_enabled():
            raise RuntimeError("CUDA graph backward cannot run inside inference_mode")
        inputs = tuple(inputs)
        key = self._input_key(inputs)
        if accumulate:
            # Preserve old gradients BEFORE a capture/cache eviction/replay can
            # overwrite its output storage. This also supports switching from
            # replacement mode to accumulation and detached output aliases.
            self._preserve_accumulated_gradients()
        signature = self._parameter_signature()
        if signature != self._signature:
            self.clear()
            devices = {parameter.device for parameter in self.parameters}
            if devices != {self.device}:
                raise ValueError("Recreate CUDA execution cache after moving the model")
            self._signature = signature
        entry = self._cache.get(key)
        if entry is None:
            while len(self._cache) >= self.max_cached_graphs:
                _, evicted = self._cache.popitem(last=False)
                self._release_gradients(evicted)
                del evicted
            entry = self._capture(inputs)
            self._cache[key] = entry
        else:
            self._cache.move_to_end(key)
        for target, source in zip(entry.inputs, inputs):
            target.copy_(source)
        if not accumulate:
            for parameter, gradient in zip(self.parameters, entry.gradients):
                parameter.grad = gradient
        entry.graph.replay()
        if accumulate:
            self._accumulate_gradients(entry.gradients)
        self.replays += 1
        return entry.loss, entry.metrics

    @torch.no_grad()
    def _preserve_accumulated_gradients(self):
        graph_storages = {gradient.untyped_storage().data_ptr()
                          for entry in self._cache.values() for gradient in entry.gradients
                          if gradient is not None}
        for parameter in self.parameters:
            gradient = parameter.grad
            if gradient is not None and gradient.untyped_storage().data_ptr() in graph_storages:
                parameter.grad = gradient.clone(memory_format=torch.contiguous_format)

    @torch.no_grad()
    def _accumulate_gradients(self, gradients):
        targets, sources = [], []
        for parameter, gradient in zip(self.parameters, gradients):
            if gradient is None:
                # This microbatch contributes nothing. Earlier microbatches
                # may still have used the parameter; do not erase their grads.
                continue
            if parameter.grad is None:
                parameter.grad = gradient.clone(memory_format=torch.contiguous_format)
            else:
                targets.append(parameter.grad)
                sources.append(gradient)
        if targets:
            torch._foreach_add_(targets, sources)

    def _release_gradients(self, entry):
        for parameter, gradient in zip(self.parameters, entry.gradients):
            if gradient is not None and parameter.grad is gradient:
                parameter.grad = None

    def clear(self):
        """Release capture pools and their gradients; keep lifetime counters."""
        for entry in self._cache.values():
            self._release_gradients(entry)
        self._cache.clear()
