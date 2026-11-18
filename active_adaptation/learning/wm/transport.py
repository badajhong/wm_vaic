"""Lossless replay transport; materialize duplicate next states on the receiver."""

from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass
class ReplayTransport:
    """Owned state frames plus exact non-adjacent next-state exceptions.

    ``frames`` is [B, L+1, D]. The final frame is already the factual final
    target; exceptions only override interior next states. No quantization or
    numerical reconstruction is involved. Other transition leaves keep [B, L].
    An explicit ``state_columns`` projection retains only that exact section;
    omitted columns become zero on materialization for a context-only WM loss.
    """

    frames: torch.Tensor
    fields: TensorDict
    exception_indices: torch.Tensor
    exception_values: torch.Tensor
    state_columns: slice | None = None
    full_state_width: int | None = None

    @property
    def bytes(self):
        return (self.fields.bytes() + self.frames.numel() * self.frames.element_size()
                + self.exception_indices.numel() * self.exception_indices.element_size()
                + self.exception_values.numel() * self.exception_values.element_size())

    @property
    def materialized_bytes(self):
        b, frames, width = self.frames.shape
        width = self.full_state_width or width
        return self.fields.bytes() + 2 * b * (frames - 1) * width * self.frames.element_size()

    @torch.inference_mode(False)
    def to(self, device, *, non_blocking=False):
        return ReplayTransport(
            self.frames.to(device, non_blocking=non_blocking),
            self.fields.to(device, non_blocking=non_blocking),
            self.exception_indices.to(device, non_blocking=non_blocking),
            self.exception_values.to(device, non_blocking=non_blocking),
            self.state_columns,
            self.full_state_width,
        )

    @torch.inference_mode(False)
    def materialize(self):
        # Current states reference exclusively owned transport storage. Targets
        # get separate storage, so changing either leaf cannot change the other.
        if self.state_columns is None:
            state = self.frames[:, :-1]
            target = self.frames[:, 1:].clone(memory_format=torch.contiguous_format)
            exception_target = target.view(-1, target.shape[-1])
        else:
            # Explicit WM-fit-only projection: reconstruct the established
            # packed schema on the receiver, retaining every selected bit.
            # Unselected observation groups must not be read by this objective.
            shape = (*self.fields.batch_size, self.full_state_width)
            state, target = self.frames.new_zeros(shape), self.frames.new_zeros(shape)
            state[..., self.state_columns].copy_(self.frames[:, :-1])
            target[..., self.state_columns].copy_(self.frames[:, 1:])
            exception_target = target.view(-1, self.full_state_width)[:, self.state_columns]
        if self.exception_indices.numel():
            exception_target.index_copy_(0, self.exception_indices, self.exception_values)
        return TensorDict({**dict(self.fields.items()), "state": state,
                           "next": self.fields.get("next", TensorDict({}, self.fields.batch_size)).clone(False)},
                          self.fields.batch_size, device=self.frames.device).set(("next", "state"), target)
