"""Rollout allocation independent of the simulator and policy implementation."""

import torch
from tensordict import TensorDict


def allocate_rollout_buffer(template, steps, *, device, contiguous_steps=False):
    """Allocate logical [environment, time] storage with owned tensor leaves.

    With ``contiguous_steps``, each complete simulation step is contiguous in
    memory. This avoids strided CPU destinations during WM's per-step GPU-to-CPU
    collection. The public batch axes and leaf dtypes remain unchanged. This
    does not pin memory or perform asynchronous transfers.
    """
    if template.ndim != 1 or steps < 1:
        raise ValueError("Rollout allocation requires one environment axis and positive steps")
    environments = template.shape[0]
    result = TensorDict({}, batch_size=[environments, steps], device=device)
    for key, value in template.items(include_nested=True, leaves_only=True):
        tail = value.shape[1:]
        if contiguous_steps:
            storage = torch.empty((steps, environments, *tail), dtype=value.dtype, device=device)
            storage = storage.transpose(0, 1)
        else:
            storage = torch.empty((environments, steps, *tail), dtype=value.dtype, device=device)
        result.set(key, storage)
    return result
