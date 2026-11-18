"""Return-based RSI reset sampling.

Ported from SuperMimic's ``supermimic/runners/reset_sampling.py`` (Copyright
2026 The SuperMimic Authors, Apache License 2.0), with the mjlab bin layout
(one bin per 0.2 s of the concatenated motion timeline).

The motion timeline is cut into bins. ReturnSampler measures V[b], the
discounted return the policy earns from bin b on its own rollouts, and turns it
into the reset distribution: a uniform explore share, a softmax focus on the
bins with low return, and a keep share on the bins the focus has concentrated
on, so a hard motion segment that is learned keeps some practice.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import torch


@dataclass
class ReturnSamplerCfg:
    gamma: float = 0.99
    """Discount of the measured return; sets v_max = max_step_reward / (1 - gamma)."""
    horizon: int = 20
    """Steps per update. Each env's last `horizon` steps turn into returns that
    bootstrap from V where the window ends."""
    memory: int = 256
    """Most samples a bin's V remembers, so V follows the current policy."""
    steepness: float = 25.0
    """Softmax gain on V / v_max."""
    explore: float = 0.5
    """Share of resets spread uniformly over all bins."""
    keep: float = 0.1
    """Share of resets on bins in proportion to the summed square of the focus
    each received."""
    bin_seconds: float = 0.2
    """Bin width on the motion timeline (mjlab's adaptive-sampling bins)."""
    max_step_reward: float | None = None
    """Per-step reward ceiling; None derives it analytically from the task."""

    @classmethod
    def from_dict(cls, values):
        values = dict(values or {})
        names = {field.name for field in fields(cls)}
        unknown = set(values) - names
        if unknown:
            raise ValueError(f"Unknown return_sampler options: {sorted(unknown)}")
        return cls(**values)


class ReturnSampler:
    def __init__(self, cfg: ReturnSamplerCfg, bin_count: int, frame_count: int,
                 num_envs: int, max_step_reward: float, device="cpu"):
        if not 0.0 < cfg.gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {cfg.gamma}")
        if cfg.explore < 0 or cfg.keep < 0 or cfg.explore + cfg.keep >= 1.0:
            raise ValueError("explore and keep must be nonnegative and leave room for focus")
        if cfg.horizon < 1 or cfg.memory < 1:
            raise ValueError("horizon and memory must be positive")
        if not (max_step_reward > 0 and max_step_reward < float("inf")):
            raise ValueError("Return sampling needs a finite positive max_step_reward; "
                             "set return_sampler.max_step_reward for this task")
        self.cfg = cfg
        self.bin_count = bin_count
        self.frame_count = frame_count
        self.v_max = max_step_reward / (1.0 - cfg.gamma)
        # A bin reads v_max until it is first measured.
        self.value = torch.full((bin_count,), self.v_max, device=device)
        self.count = torch.zeros(bin_count, device=device)
        # Summed over the whole run, so float64 keeps the small per-bin terms exact.
        self.effort = torch.zeros(bin_count, dtype=torch.float64, device=device)
        shape = (num_envs, cfg.horizon)
        self._frames = torch.zeros(shape, dtype=torch.long, device=device)
        self._rewards = torch.zeros(shape, device=device)
        self._terminated = torch.zeros(shape, dtype=torch.bool, device=device)
        self._clip_end = torch.zeros(shape, dtype=torch.bool, device=device)
        self._timed_out = torch.zeros(shape, dtype=torch.bool, device=device)
        self._step = 0

    def bins(self, frames: torch.Tensor) -> torch.Tensor:
        """Global frame -> bin."""
        return (frames * self.bin_count // self.frame_count).clamp(0, self.bin_count - 1)

    def record(self, frames, rewards, terminated, clip_end, timed_out):
        """Buffer one env step: each env's frame, its reward, and how its
        episode ended, if it did (a fall terminates; reaching the clip's last
        frame and a time-out truncate). Every `horizon` steps, update V and
        return the new reset distribution; otherwise None."""
        col = self._step
        self._frames[:, col] = frames
        self._rewards[:, col] = rewards
        self._terminated[:, col] = terminated
        self._clip_end[:, col] = clip_end
        self._timed_out[:, col] = timed_out
        self._step += 1
        if self._step < self.cfg.horizon:
            return None
        self._step = 0
        self._update()
        return self._distribution()

    def _dones(self):
        return self._terminated | self._clip_end | self._timed_out

    def _returns(self):
        """Discounted return of every buffered step (lambda 1). Where an episode
        ends, and at the buffer's last step, the return continues from the
        value of the state after the step: none after a fall; the mean of V
        after the clip's last frame; V at the next frame's bin otherwise."""
        gamma = self.cfg.gamma
        next_frames = (self._frames + 1).clamp(max=self.frame_count - 1)
        next_value = torch.where(
            self._terminated, 0.0,
            torch.where(self._clip_end, self.value.mean(), self.value[self.bins(next_frames)]))
        dones = self._dones()
        returns = torch.empty_like(self._rewards)
        next_return = next_value[:, -1]
        for t in reversed(range(self.cfg.horizon)):
            next_return = torch.where(dones[:, t], next_value[:, t], next_return)
            returns[:, t] = self._rewards[:, t] + gamma * next_return
            next_return = returns[:, t]
        return returns

    def _update(self):
        """Fold the buffer into V. Each env's visits to one bin within one
        episode give one sample, their mean return; a bin's earlier samples
        count at most `memory` times."""
        returns = self._returns()
        dones = self._dones()
        num_envs, horizon = dones.shape
        episode = torch.cumsum(dones.long(), dim=1) - dones.long()
        envs = torch.arange(num_envs, device=dones.device).unsqueeze(1)
        key = (envs * horizon + episode) * self.bin_count + self.bins(self._frames)
        keys, inverse, visits = torch.unique(key.reshape(-1), return_inverse=True, return_counts=True)
        samples = torch.zeros(keys.numel(), device=returns.device)
        samples.scatter_add_(0, inverse, returns.reshape(-1))
        samples /= visits
        bins = keys % self.bin_count
        new = torch.zeros_like(self.count).scatter_add_(0, bins, torch.ones_like(samples))
        total = torch.zeros_like(self.value).scatter_add_(0, bins, samples)
        self.count = self.count.clamp(max=self.cfg.memory) + new
        self.value = torch.where(new > 0, self.value + (total - new * self.value) / self.count, self.value)

    def _distribution(self):
        cfg = self.cfg
        focus = torch.softmax(-cfg.steepness * self.value / self.v_max, dim=0)
        self.effort += focus.double().square()
        keep = (self.effort / self.effort.sum()).float()
        explore = cfg.explore / self.bin_count
        return explore + (1.0 - cfg.explore - cfg.keep) * focus + cfg.keep * keep

    def sample_frames(self, probs, count, generator=None):
        """Draw global frames: a bin from ``probs``, then a frame uniformly in it."""
        bins = torch.multinomial(probs, count, replacement=True, generator=generator)
        lo = (bins * self.frame_count) // self.bin_count
        span = (((bins + 1) * self.frame_count) // self.bin_count - lo).clamp_min(1)
        offset = (torch.rand(count, device=probs.device, generator=generator) * span).long()
        return (lo + offset.clamp_max(span - 1)).clamp_max(self.frame_count - 1)
