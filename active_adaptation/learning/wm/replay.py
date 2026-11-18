"""Lossless sequence replay that stores adjacent physical states only once."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch
from tensordict import TensorDict

from .transport import ReplayTransport


class CompactSequenceReplay:
    """Keep all environments and exact terminal states, with fixed capacity.

    Usually next_state[t] == state[t+1]. Store that tensor only once and keep
    sparse exceptions (auto-reset boundaries, plus any other discrepancies).
    This is lossless, including observation noise; no half-precision conversion
    or assumption about the environment's reset ordering is involved.
    """

    def __init__(self, capacity, device="cpu"):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self._chunks = deque()
        self._size = 0
        self._start_cache = None
        self._revision = 0

    def __len__(self):
        return self._size

    @property
    def revision(self):
        """Invalidate index-only sample plans when insertion/eviction changes rows."""
        return self._revision

    @property
    def bytes(self):
        return sum(data.bytes() + tail.numel() * tail.element_size()
                   + lookup.numel() * lookup.element_size() + exceptions.numel() * exceptions.element_size()
                   for data, tail, lookup, exceptions in self._chunks)

    def add(self, transitions):
        self._start_cache = None
        if transitions.ndim != 2:
            raise ValueError("Replay insertions need [environments, time] batches")
        if transitions.numel() > self.capacity:
            raise ValueError("Replay capacity must hold one complete collection batch")
        self._revision += 1
        # Evict before allocating the incoming collection. At 4096 environments
        # one collection is several GB; temporary over-capacity doubles RAM.
        while self._size + transitions.numel() > self.capacity:
            self._size -= self._chunks.popleft()[0].numel()
        # Clone on transfer: collection buffers are overwritten on the next step.
        state, target = transitions["state"], transitions["next", "state"]
        mismatches = []
        # Limit temporary GPU comparison allocations for large environment sets.
        for start in range(0, state.shape[0], 64):
            mismatch = (state[start:start + 64, 1:] != target[start:start + 64, :-1]).any(-1)
            rows, cols = mismatch.nonzero(as_tuple=True)
            mismatches.append(torch.stack((rows + start, cols), -1))
        indices = torch.cat(mismatches) if mismatches else torch.empty((0, 2), dtype=torch.long, device=state.device)
        # Advanced indexing already owns its storage. A device transfer needs
        # no second clone; the remaining leaves each need exactly one copy.
        exceptions = target[indices[:, 0], indices[:, 1]].detach().to(self.device)
        data = TensorDict({
            key: value.detach().to(self.device, copy=True, memory_format=torch.contiguous_format)
            for key, value in transitions.items(include_nested=True, leaves_only=True)
            if key != ("next", "state")
        }, transitions.batch_size, device=self.device)
        tail = target[:, -1].detach().to(self.device, copy=True, memory_format=torch.contiguous_format)
        # The sparse values stay compact; this small integer map is reused by
        # every sample rather than rebuilt for the complete chunk each time.
        lookup = torch.empty(0, device=self.device, dtype=torch.long)
        if indices.numel():
            indices = indices.to(self.device)
            lookup = torch.full((data.numel(),), -1, device=self.device, dtype=torch.long)
            lookup[indices[:, 0] * data.shape[1] + indices[:, 1]] = torch.arange(len(indices), device=self.device)
        self._chunks.append((data, tail, lookup, exceptions))
        self._size += transitions.numel()

    @torch.inference_mode(False)
    def sample(self, batch_size, sequence_length, device=None, *, pin_memory=False, draws=None):
        """Return owned, ordinary tensors, safe for autograd without a clone.

        Neither insertion under inference mode nor the caller's current mode
        changes this contract. Modifying a sample cannot alter replay storage.
        """
        if min(batch_size, sequence_length) < 1:
            raise ValueError("Batch size and sequence length must be positive")
        eligible, cumulative, total = self._eligible(sequence_length)
        if draws is None:
            draws = torch.randint(total, (batch_size,))
        elif draws.device.type != "cpu" or draws.dtype != torch.long or draws.shape != (batch_size,):
            raise ValueError("Replay draws must be a CPU int64 vector matching batch_size")
        elif (draws < 0).any() or (draws >= total).any():
            raise ValueError("Replay draw is outside eligible sequence range")
        if pin_memory and self.device.type != "cpu":
            raise ValueError("Pinned replay samples require CPU replay storage")
        chunk_ids = torch.bucketize(draws, torch.tensor(cumulative), right=True)
        # Each result leaf owns one allocation in the original random-draw
        # order. Avoid collecting large per-chunk TensorDicts, concatenating
        # them, and then making another full copy to restore that order.
        result = None
        offsets = torch.arange(sequence_length, device=self.device)
        def allocate(value, *shape):
            return torch.empty(*shape, dtype=value.dtype, device=self.device, pin_memory=pin_memory)
        for index, ((data, tail, lookup, exceptions), starts) in enumerate(eligible):
            positions = torch.where(chunk_ids == index)[0]
            if not positions.numel():
                continue
            local = draws[positions] - (cumulative[index - 1] if index else 0)
            env_ids = (local // starts).to(self.device)
            times = (local % starts).to(self.device)[:, None] + offsets
            flat_ids = (env_ids[:, None] * data.shape[1] + times).reshape(-1)
            if result is None:
                result = TensorDict({
                    key: allocate(value, batch_size, sequence_length, *value.shape[2:])
                    for key, value in data.items(include_nested=True, leaves_only=True)
                }, [batch_size, sequence_length], device=self.device)
                result["next", "state"] = allocate(
                    data["state"], batch_size, sequence_length, *data["state"].shape[2:])
            positions = positions.to(self.device)
            whole_batch = positions.numel() == batch_size
            for key, value in data.items(include_nested=True, leaves_only=True):
                source = value.reshape(data.numel(), *value.shape[2:])
                destination = result[key]
                if whole_batch:
                    torch.index_select(source, 0, flat_ids,
                                       out=destination.view(batch_size * sequence_length, *value.shape[2:]))
                else:
                    selected = source.index_select(0, flat_ids).view(
                        positions.numel(), sequence_length, *value.shape[2:])
                    destination.index_copy_(0, positions, selected)
            next_times = (times + 1).clamp_max(data.shape[1] - 1)
            source = data["state"].reshape(data.numel(), *data["state"].shape[2:])
            next_ids = (env_ids[:, None] * data.shape[1] + next_times).reshape(-1)
            if whole_batch:
                target = result["next", "state"]
                torch.index_select(source, 0, next_ids,
                                   out=target.view(batch_size * sequence_length, *source.shape[1:]))
            else:
                target = source.index_select(0, next_ids).view(
                    positions.numel(), sequence_length, *source.shape[1:])
            tail_rows, tail_times = (times == data.shape[1] - 1).nonzero(as_tuple=True)
            target[tail_rows, tail_times] = tail[env_ids[tail_rows]]
            if lookup.numel():
                selected = lookup[flat_ids].view_as(times)
                mask = selected >= 0
                target[mask] = exceptions[selected[mask]]
            if not whole_batch:
                result["next", "state"].index_copy_(0, positions, target)
        return result.to(device or self.device)

    @staticmethod
    def _draws(batch_size, total, draws):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if draws is None:
            return torch.randint(total, (batch_size,))
        if draws.device.type != "cpu" or draws.dtype != torch.long or draws.shape != (batch_size,):
            raise ValueError("Replay draws must be a CPU int64 vector matching batch_size")
        if (draws < 0).any() or (draws >= total).any():
            raise ValueError("Replay draw is outside eligible sequence range")
        return draws

    def _eligible_starts(self):
        if self._start_cache is None:
            chunks, cumulative, total = [], [], 0
            for data, *_ in self._chunks:
                initial = data.get("is_init", None)
                valid = (torch.ones(data.numel(), dtype=torch.bool, device=self.device)
                         if initial is None else ~initial.reshape(data.numel(), -1).any(-1))
                indices = valid.nonzero(as_tuple=True)[0]
                if indices.numel():
                    chunks.append((data, indices))
                    total += indices.numel()
                    cumulative.append(total)
            self._start_cache = chunks, cumulative, total
        if not self._start_cache[2]:
            raise ValueError("Replay has no non-reset initial states")
        return self._start_cache

    @torch.inference_mode(False)
    def sample_starts(self, batch_size, device=None, *, pin_memory=False, draws=None):
        """Sample only state/is_init, uniformly over all non-reset transitions.

        Future done flags and the remaining chunk length do not affect eligibility.
        Every output owns storage, independently of replay and other samples.
        """
        eligible, cumulative, total = self._eligible_starts()
        draws = self._draws(batch_size, total, draws)
        if pin_memory and self.device.type != "cpu":
            raise ValueError("Pinned replay samples require CPU replay storage")
        chunk_ids = torch.bucketize(draws, torch.tensor(cumulative), right=True)
        result = None
        for index, (data, valid_indices) in enumerate(eligible):
            positions = torch.where(chunk_ids == index)[0]
            if not positions.numel():
                continue
            local = draws[positions] - (cumulative[index - 1] if index else 0)
            selected = valid_indices[local.to(self.device)]
            values = {"state": data["state"]}
            if "is_init" in data.keys():
                values["is_init"] = data["is_init"]
            if result is None:
                result = TensorDict({key: torch.empty(batch_size, *value.shape[2:], dtype=value.dtype,
                                                     device=self.device, pin_memory=pin_memory)
                                     for key, value in values.items()}, [batch_size], device=self.device)
            for key, value in values.items():
                source = value.reshape(data.numel(), *value.shape[2:])
                if positions.numel() == batch_size:
                    torch.index_select(source, 0, selected, out=result[key])
                else:
                    result[key].index_copy_(0, positions.to(self.device), source.index_select(0, selected))
        return result.to(device or self.device)

    @torch.inference_mode(False)
    def sample_transport(self, batch_size, sequence_length, *, pin_memory=False, draws=None, state_columns=None):
        """Gather directly into deduplicated frames; never allocate CPU next_state.

        Draw distribution and ordering are identical to ``sample``. Exact reset,
        noise and other adjacency exceptions are transported without assumptions.
        ``state_columns`` explicitly selects one contiguous WM-only section;
        omitted columns become zero after materialization. Never use this option
        for actor/critic/validation batches that read the omitted observations.
        """
        if sequence_length < 1:
            raise ValueError("Sequence length must be positive")
        eligible, cumulative, total = self._eligible(sequence_length)
        full_width = eligible[0][0][0]["state"].shape[-1]
        if state_columns is not None:
            if (not isinstance(state_columns, slice) or state_columns.step not in (None, 1)
                    or state_columns.start is None or state_columns.stop is None
                    or not 0 <= state_columns.start < state_columns.stop <= full_width):
                raise ValueError("state_columns must be an explicit nonempty contiguous state slice")
            state_columns = slice(state_columns.start, state_columns.stop)
        draws = self._draws(batch_size, total, draws)
        if pin_memory and self.device.type != "cpu":
            raise ValueError("Pinned replay samples require CPU replay storage")
        chunk_ids = torch.bucketize(draws, torch.tensor(cumulative), right=True)
        frames, fields, sparse = None, None, []
        offsets = torch.arange(sequence_length + 1, device=self.device)
        def allocate(value, *shape):
            return torch.empty(*shape, dtype=value.dtype, device=self.device, pin_memory=pin_memory)
        for index, ((data, tail, lookup, exceptions), starts) in enumerate(eligible):
            positions = torch.where(chunk_ids == index)[0]
            if not positions.numel():
                continue
            local = draws[positions] - (cumulative[index - 1] if index else 0)
            env_ids = (local // starts).to(self.device)
            times = (local % starts).to(self.device)[:, None] + offsets
            positions = positions.to(self.device)
            source = data["state"].reshape(data.numel(), full_width)
            selected_tail, selected_exceptions = tail, exceptions
            if state_columns is not None:
                source = source[:, state_columns]
                selected_tail = tail[:, state_columns]
                selected_exceptions = exceptions[:, state_columns]
            width = source.shape[-1]
            if frames is None:
                frames = allocate(data["state"], batch_size, sequence_length + 1, width)
                fields = TensorDict({key: allocate(value, batch_size, sequence_length, *value.shape[2:])
                                     for key, value in data.items(True, True) if key != "state"},
                                    [batch_size, sequence_length], device=self.device)
            frame_ids = (env_ids[:, None] * data.shape[1] + times.clamp_max(data.shape[1] - 1)).reshape(-1)
            whole_batch = positions.numel() == batch_size
            if whole_batch:
                selected_frames = frames
                torch.index_select(source, 0, frame_ids, out=frames.view(-1, width))
            else:
                selected_frames = source.index_select(0, frame_ids).view(-1, sequence_length + 1, width)
            tail_rows = (times[:, -1] == data.shape[1]).nonzero(as_tuple=True)[0]
            selected_frames[tail_rows, -1] = selected_tail[env_ids[tail_rows]]
            flat_ids = (env_ids[:, None] * data.shape[1] + times[:, :-1]).reshape(-1)
            if lookup.numel():
                exception_ids = lookup[flat_ids].view(-1, sequence_length)
                last = exception_ids[:, -1]
                last_rows = (last >= 0).nonzero(as_tuple=True)[0]
                selected_frames[last_rows, -1] = selected_exceptions[last[last_rows]]
                rows, steps = (exception_ids[:, :-1] >= 0).nonzero(as_tuple=True)
                if rows.numel():
                    sparse.append((positions[rows] * sequence_length + steps,
                                   selected_exceptions, exception_ids[rows, steps]))
            if not whole_batch:
                frames.index_copy_(0, positions, selected_frames)
            for key, value in data.items(True, True):
                if key == "state":
                    continue
                source = value.reshape(data.numel(), *value.shape[2:])
                if whole_batch:
                    torch.index_select(source, 0, flat_ids,
                                       out=fields[key].view(batch_size * sequence_length, *value.shape[2:]))
                else:
                    selected = source.index_select(0, flat_ids).view(positions.numel(), sequence_length, *value.shape[2:])
                    fields[key].index_copy_(0, positions, selected)
        count = sum(indices.numel() for indices, _, _ in sparse)
        exception_indices = torch.empty(count, dtype=torch.long, device=self.device, pin_memory=pin_memory)
        exception_values = allocate(frames, count, frames.shape[-1])
        begin = 0
        for indices, source, selection in sparse:
            end = begin + indices.numel()
            exception_indices[begin:end].copy_(indices)
            torch.index_select(source, 0, selection, out=exception_values[begin:end])
            begin = end
        return ReplayTransport(frames, fields, exception_indices, exception_values, state_columns, full_width)

    @torch.inference_mode(False)
    def sample_compact(self, batch_size, sequence_length, device=None, *, pin_memory=False, draws=None, state_columns=None):
        """Compact transfer, optionally projecting an explicit WM-only section."""
        return self.sample_transport(batch_size, sequence_length, pin_memory=pin_memory,
                                     draws=draws, state_columns=state_columns).to(device or self.device).materialize()

    def _eligible(self, sequence_length):
        eligible, cumulative, total = [], [], 0
        for chunk in self._chunks:
            data = chunk[0]
            starts = data.shape[1] - sequence_length + 1
            if starts > 0:
                eligible.append((chunk, starts))
                total += data.shape[0] * starts
                cumulative.append(total)
        if not total:
            raise ValueError(f"Replay has no sequence of length {sequence_length}")
        return eligible, cumulative, total


class ReplayPrefetcher:
    """Overlap one upcoming replay sample/transfer with the current update.

    Use as a context manager for a complete training phase. Replay must not be
    mutated while this iterator is open. Random draws are reserved by the caller
    thread, in the same order and number as ordinary sampling; the worker never
    touches the random generator. The next draw is reserved before the current
    update, so phases must not interleave other CPU random draws. Closing early
    can consume the one already reserved upcoming draw.

    Only the consumed sample and one upcoming sample are resident internally.
    Each owns its tensors; callers may retain samples, at their own memory cost.
    CPU replay stays pageable. Only these sample buffers are pinned for CUDA.
    """

    def __init__(self, replay, batch_size, sequence_length, count, device, *, compact=False, state_columns=None,
                 draw_batches=None):
        if min(batch_size, sequence_length) < 1 or count < 0:
            raise ValueError("Prefetch sizes must be positive and count nonnegative")
        if replay.device.type != "cpu":
            raise ValueError("Replay prefetch requires CPU replay storage")
        self.replay = replay
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.count = count
        self.device = torch.device(device)
        self.compact = bool(compact)
        self.state_columns = state_columns
        self._revision = replay.revision
        if draw_batches is not None:
            if (not isinstance(draw_batches, torch.Tensor) or draw_batches.device.type != "cpu"
                    or draw_batches.dtype != torch.long or draw_batches.shape != (count, batch_size)):
                raise ValueError("Prefetch draw_batches must be a CPU int64 [count, batch_size] tensor")
            # Only small integer plans are copied, never sampled states.
            draw_batches = draw_batches.detach().clone()
        self._draw_batches = draw_batches
        if state_columns is not None and not compact:
            raise ValueError("state_columns requires compact replay transport")
        self.transferred_bytes = 0
        self.materialized_bytes = 0
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Replay prefetch supports CPU and CUDA destinations")
        self._submitted = 0
        self._future = None
        self._executor = None
        self._stream = None
        self._host_pending = None
        self._closed = False

    def __enter__(self):
        if self._closed or self._executor is not None:
            raise RuntimeError("ReplayPrefetcher cannot be entered twice")
        self._check_revision()
        if self.device.type == "cuda":
            self._stream = torch.cuda.Stream(device=self.device)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wm-replay")
        try:
            self._submit()
        except BaseException:
            self.close()
            raise
        return self

    def __iter__(self):
        return self

    def _submit(self):
        self._check_revision()
        if self._submitted >= self.count:
            self._future = None
            return
        total = self.replay._eligible(self.sequence_length)[2]
        draws = (torch.randint(total, (self.batch_size,)) if self._draw_batches is None
                 else self._draw_batches[self._submitted])
        self._future = self._executor.submit(self._sample, draws)
        self._submitted += 1

    @torch.inference_mode(False)
    def _sample(self, draws):
        self._check_revision()
        if self.compact:
            host = self.replay.sample_transport(self.batch_size, self.sequence_length,
                                                pin_memory=self.device.type == "cuda", draws=draws,
                                                state_columns=self.state_columns)
            self.transferred_bytes += host.bytes
            self.materialized_bytes += host.materialized_bytes
        else:
            host = self.replay.sample(self.batch_size, self.sequence_length, device="cpu",
                                     pin_memory=self.device.type == "cuda", draws=draws)
            self.transferred_bytes += host.bytes()
            self.materialized_bytes += host.bytes()
        if self._stream is None:
            return host.materialize() if self.compact else host, None, None
        with torch.cuda.device(self.device), torch.cuda.stream(self._stream):
            sample = host.to(self.device, non_blocking=True)
            if self.compact:
                sample = sample.materialize()
            event = torch.cuda.Event()
            event.record(self._stream)
        return sample, host, event

    def _release_host(self):
        if self._host_pending is not None:
            # The sample is consumed before the next call, so this normally
            # completes immediately. It also makes exceptional cleanup safe.
            self._host_pending[1].synchronize()
            self._host_pending = None

    def __next__(self):
        if self._executor is None:
            raise RuntimeError("Use ReplayPrefetcher inside a with block")
        if self._closed or self._future is None:
            raise StopIteration
        self._check_revision()
        sample, host, event = self._future.result()
        self._release_host()
        if event is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(event)
            for value in sample.values(include_nested=True, leaves_only=True):
                value.record_stream(stream)
            self._host_pending = (host, event)
        self._submit()
        return sample

    def _check_revision(self):
        if self.replay.revision != self._revision:
            raise RuntimeError("Replay changed after this prefetch plan was created")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
        self._release_host()
        if self._future is not None and self._future.done() and not self._future.cancelled():
            # Surface worker errors at consumption, without masking a separate
            # training exception during context-manager cleanup.
            if self._future.exception() is None:
                _, _, event = self._future.result()
                if event is not None:
                    event.synchronize()
        self._future = None

    def __exit__(self, exc_type, exc, traceback):
        self.close()
