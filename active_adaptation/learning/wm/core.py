"""Observation codec and running moments shared by the world-model learner."""

from math import prod

import torch
from torch import nn
from tensordict import TensorDict, TensorDictBase


class ObservationCodec:
    """Losslessly flatten named observation groups, preserving event shapes.

    ``spec.shape`` is the environment batch shape, when present. Each leaf's
    remaining dimensions are its event shape. TensorDict batch dimensions may
    differ from the spec, for example replay samples with shape ``[B, T]``.
    Static and exogenous keys are exact names, not prefix matches.
    """

    def __init__(
        self,
        spec,
        keys,
        static_keys=("object_geo_",),
        exogenous_keys=("command", "command_", "ref_joint_pos_"),
    ):
        self.keys = tuple(keys)
        if not self.keys or len(set(self.keys)) != len(self.keys):
            raise ValueError("Observation keys must be nonempty and unique")
        self.shapes = {}
        self.slices = {}
        batch_ndim = len(getattr(spec, "shape", ()))
        offset = 0
        for key in self.keys:
            leaf = spec[key]
            if not leaf.dtype.is_floating_point:
                raise TypeError(f"World-model observation {key!r} must be floating point")
            shape = torch.Size(leaf.shape[batch_ndim:])
            size = prod(shape)
            if size == 0:
                raise ValueError(f"World-model observation {key!r} has no features")
            self.shapes[key] = shape
            self.slices[key] = slice(offset, offset + size)
            offset += size
        self.state_dim = offset
        self.static_keys = tuple(key for key in self.keys if key in static_keys)
        self.exogenous_keys = tuple(key for key in self.keys if key in exogenous_keys)
        if set(self.static_keys) & set(self.exogenous_keys):
            raise ValueError("A group cannot be both static and exogenous")
        self.dynamic_keys = tuple(
            key for key in self.keys
            if key not in self.static_keys and key not in self.exogenous_keys
        )
        if not self.dynamic_keys:
            raise ValueError("The world model needs at least one dynamic observation group")

    def pack(self, observations: TensorDictBase) -> torch.Tensor:
        parts = []
        for key in self.keys:
            value = observations[key]
            expected = observations.batch_size + self.shapes[key]
            if value.shape != expected:
                raise ValueError(f"Observation {key!r}: expected {expected}, got {value.shape}")
            parts.append(value.reshape(*observations.batch_size, -1))
        return torch.cat(parts, dim=-1)

    def unpack(self, state: torch.Tensor, out=None) -> TensorDictBase:
        if state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected {self.state_dim} state features, got {state.shape[-1]}")
        batch_shape = state.shape[:-1]
        if out is None:
            out = TensorDict({}, batch_size=batch_shape, device=state.device)
        elif out.batch_size != batch_shape:
            raise ValueError("Output TensorDict batch shape does not match packed state")
        for key in self.keys:
            out[key] = state[..., self.slices[key]].reshape(batch_shape + self.shapes[key])
        return out


class RunningMoments(nn.Module):
    """Persistent per-feature population moments, combined using Welford's rule."""

    def __init__(self, size, min_std=1e-3):
        super().__init__()
        self.min_std = min_std
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))

    @property
    def std(self):
        return self.var.clamp_min(self.min_std ** 2).sqrt()

    @torch.no_grad()
    def update(self, value):
        value = value.detach().reshape(-1, self.mean.numel()).to(self.mean)
        if value.numel() == 0:
            return
        # Corrupt transitions must not permanently poison the running moments.
        value = value[torch.isfinite(value).all(dim=-1)]
        if value.numel() == 0:
            return
        batch_var, batch_mean = torch.var_mean(value, dim=0, correction=0)
        batch_count = value.shape[0]
        total = self.count + batch_count
        fraction = (batch_count / total).to(self.mean.dtype)
        delta = batch_mean - self.mean
        combined_var = (
            self.var * (1 - fraction)
            + batch_var * fraction
            + delta.square() * fraction * (1 - fraction)
        )
        self.mean.add_(delta * fraction)
        self.var.copy_(combined_var.clamp_min(0))
        self.count.copy_(total)

    def normalize(self, value):
        return (value - self.mean) / self.std

    def denormalize(self, value):
        return value * self.std + self.mean
