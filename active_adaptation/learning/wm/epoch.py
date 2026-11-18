"""Index-only replay pools reused, with a new permutation for each epoch."""

import torch

from .replay import ReplayPrefetcher


class EpochReplayPlan:
    """Draw one fixed pool, then visit every pool slot once in every epoch.

    Only CPU int64 indices are retained, not [pool, horizon, state] tensors.
    Sampling is with replacement, matching ordinary replay; duplicate sampled
    transitions are distinct pool slots and are each visited once per epoch.
    Every minibatch has the requested fixed shape for CUDA graph reuse.

    A plan belongs to one replay revision. Do not collect/insert new transitions
    until it is consumed. Model plans use contiguous replay windows; actor plans
    use only non-reset starts and impose no factual future-done restriction.
    """

    @torch.inference_mode(False)
    def __init__(self, replay, pool_size, minibatch_size, epochs, sequence_length=8,
                 *, starts_only=False, generator=None):
        values = (pool_size, minibatch_size, epochs, sequence_length)
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("Epoch pool, minibatch, epochs and sequence length must be positive integers")
        if pool_size % minibatch_size:
            raise ValueError("Epoch pool must divide into complete fixed-shape minibatches")
        if replay.device.type != "cpu":
            raise ValueError("Epoch plans require CPU replay storage")
        if generator is not None and generator.device.type != "cpu":
            raise ValueError("Epoch plans require a CPU random generator")
        self.replay = replay
        self.pool_size = pool_size
        self.minibatch_size = minibatch_size
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.starts_only = bool(starts_only)
        self.num_batches = pool_size // minibatch_size
        self._revision = replay.revision
        total = (replay._eligible_starts()[2] if starts_only else replay._eligible(sequence_length)[2])
        self._pool_draws = torch.randint(total, (pool_size,), generator=generator)
        self._orders = torch.stack([torch.randperm(pool_size, generator=generator) for _ in range(epochs)])

    @property
    def bytes(self):
        return self._pool_draws.numel() * self._pool_draws.element_size() + self._orders.numel() * self._orders.element_size()

    def _check(self, epoch, batch=None):
        if self.replay.revision != self._revision:
            raise RuntimeError("Replay changed after this epoch sample pool was created")
        if not isinstance(epoch, int) or isinstance(epoch, bool) or not 0 <= epoch < self.epochs:
            raise IndexError("Epoch index is outside this replay plan")
        if batch is not None and (not isinstance(batch, int) or isinstance(batch, bool)
                                  or not 0 <= batch < self.num_batches):
            raise IndexError("Minibatch index is outside this replay plan")

    @torch.inference_mode(False)
    def draws(self, epoch, batch):
        """Owned CPU int64 sample indices for one full minibatch."""
        self._check(epoch, batch)
        begin = batch * self.minibatch_size
        return self._pool_draws.index_select(0, self._orders[epoch, begin:begin + self.minibatch_size])

    def sample(self, epoch, batch, device=None, *, state_columns=None, pin_memory=False):
        draws = self.draws(epoch, batch)
        if self.starts_only:
            if state_columns is not None:
                raise ValueError("Actor start plans must preserve every original observation field")
            return self.replay.sample_starts(self.minibatch_size, device=device,
                                             pin_memory=pin_memory, draws=draws)
        return self.replay.sample_compact(self.minibatch_size, self.sequence_length, device=device,
                                          pin_memory=pin_memory, draws=draws, state_columns=state_columns)

    def prefetch(self, epoch, device, *, state_columns=None, start_batch=0):
        """Bounded one-batch model prefetch, optionally skipping a capture batch.

        Consume this context only after any CUDA graph capture for these shapes
        has completed. ``start_batch=1`` permits eager/capture of batch zero with
        no active sampling worker, followed by prefetch of the remaining pool.
        """
        self._check(epoch)
        if self.starts_only:
            raise ValueError("Actor start plans use sample(); model prefetch expects sequence windows")
        if not isinstance(start_batch, int) or isinstance(start_batch, bool) or not 0 <= start_batch <= self.num_batches:
            raise IndexError("Prefetch start_batch is outside this replay plan")
        draw_batches = (torch.stack([self.draws(epoch, batch) for batch in range(start_batch, self.num_batches)])
                        if start_batch < self.num_batches else torch.empty(0, self.minibatch_size, dtype=torch.long))
        return ReplayPrefetcher(self.replay, self.minibatch_size, self.sequence_length,
                               self.num_batches - start_batch, device, compact=True,
                               state_columns=state_columns, draw_batches=draw_batches)
