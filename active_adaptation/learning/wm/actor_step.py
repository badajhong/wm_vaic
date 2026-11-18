"""Bound deterministic BPTT actor updates using their measured action change.

The expensive world-model backward pass is performed by the caller exactly
once. This controller takes one Adam proposal and, when needed, interpolates
that proposal without another backward or optimizer step. For an accepted
interpolation, Adam's moments advance once: they describe the same observed
gradient, independently of the learning rate. A rejected proposal restores both
the parameters and all optimizer state, including lazily initialized state.

The Gaussian KL uses a fixed collection standard deviation and fixed real
replay starts. It controls the local size of an update; it is neither a PPO
objective nor a guarantee about return or unseen states.
"""

import copy
import math
from collections.abc import Callable

import torch


class ActorStepController:
    """Control named raw-teacher groups or legacy actor/encoder/height groups."""

    def __init__(self, optimizer, *, actor_lr=1.5e-5, encoder_lr=5e-6,
                 max_actor_lr=1e-4, target_kl=.01, max_kl=.02,
                 min_scale=1e-3, max_backtracks=8):
        if not isinstance(optimizer, torch.optim.Adam):
            raise TypeError("ActorStepController requires the existing Adam optimizer")
        if len(optimizer.param_groups) not in (1, 2, 3):
            raise ValueError("Expected actor and optional encoder/height optimizer groups")
        if not all(math.isfinite(x) and x > 0 for x in (actor_lr, encoder_lr, target_kl, max_kl)):
            raise ValueError("Actor learning rates and KL limits must be finite and positive")
        if target_kl > max_kl:
            raise ValueError("actor_target_kl must not exceed actor_max_kl")
        if not math.isfinite(max_actor_lr) or max_actor_lr < actor_lr:
            raise ValueError("max_actor_lr must be finite and at least actor_lr")
        if not 0 < min_scale <= 1 or not math.isfinite(min_scale):
            raise ValueError("min_scale must be finite and in (0, 1]")
        if not isinstance(max_backtracks, int) or max_backtracks < 0:
            raise ValueError("max_backtracks must be a nonnegative integer")
        self.optimizer = optimizer
        self.group_names = tuple(group.get("name", ("actor", "encoder_priv", "height")[i])
                                 for i, group in enumerate(optimizer.param_groups))
        self.base_lrs = (float(actor_lr),) + (float(encoder_lr),) * (len(optimizer.param_groups) - 1)
        self.target_kl = float(target_kl)
        self.max_kl = float(max_kl)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_actor_lr / actor_lr)
        self.max_backtracks = max_backtracks
        self.scale = 1.
        self._set_lrs()

    def _set_lrs(self):
        for group, maximum in zip(self.optimizer.param_groups, self.base_lrs):
            rate = maximum * self.scale
            if torch.is_tensor(group["lr"]):
                group["lr"].fill_(rate)
            else:
                group["lr"] = rate

    def state_dict(self):
        return {"version": 1, "scale": self.scale, "base_lrs": self.base_lrs,
                "max_scale": self.max_scale,
                "target_kl": self.target_kl, "max_kl": self.max_kl}

    def load_state_dict(self, state):
        """Call AFTER optimizer.load_state_dict; None migrates a legacy WM.

        New checkpoints preserve the controller scale. Explicitly configured
        maxima/limits remain authoritative, so resuming can never reintroduce a
        legacy 3e-4 actor rate. Adam parameter groups and moments are untouched.
        """
        if state is None:
            self.scale = 1.
        else:
            if state.get("version") != 1:
                raise ValueError("Unsupported WM actor step controller checkpoint version")
            if len(state.get("base_lrs", ())) != len(self.base_lrs):
                raise ValueError("WM actor optimizer group count changed on resume")
            scale = float(state["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid WM actor learning-rate scale in checkpoint")
            self.scale = min(self.max_scale, max(self.min_scale, scale))
        self._set_lrs()

    @torch.no_grad()
    def step(self, evaluate_mean: Callable[[], torch.Tensor], initial_mean,
             initial_std, valid_mask):
        """Apply already-computed/clipped gradients and return scalar metrics.

        ``evaluate_mean`` must use the SAME replay starts and observation
        normalizer as ``initial_mean`` without mutating policy buffers. It may
        be called several times for inexpensive forward-only backtracking.
        Invalid reset rows are excluded by indexing, including any NaNs there.
        """
        initial_mean = initial_mean.detach()
        initial_std = torch.broadcast_to(initial_std.detach(), initial_mean.shape)
        mask = valid_mask.to(device=initial_mean.device, dtype=torch.bool)
        if mask.shape == initial_mean.shape[:-1] + (1,):
            mask = mask.squeeze(-1)
        if mask.shape != initial_mean.shape[:-1]:
            raise ValueError("Actor step valid_mask must match the replay-start batch shape")
        if not mask.any():
            return self._metrics(0., 0., 0., 0., False, 0, self.scale)
        reference, std = initial_mean[mask], initial_std[mask]
        if not torch.isfinite(reference).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError("Actor step starts require finite action means and positive finite std")

        def measure():
            updated = evaluate_mean()
            if updated.shape != initial_mean.shape:
                raise ValueError("Actor action shape changed while evaluating its update")
            change = updated[mask] - reference
            rms = change.square().mean().sqrt().item()
            kl = (.5 * (change / std).square().sum(-1)).mean().item()
            return rms, kl

        # Clone values rather than deepcopying the state dictionary's parameter
        # keys: Adam state must remain keyed by the live Parameter objects.
        params = [p for group in self.optimizer.param_groups for p in group["params"]]
        before = [p.detach().clone() for p in params]
        state_before = {p: copy.deepcopy(value) for p, value in self.optimizer.state.items()}
        previous_scale = self.scale

        def restore():
            for parameter, value in zip(params, before):
                parameter.copy_(value)
            self.optimizer.state.clear()
            self.optimizer.state.update(state_before)

        try:
            self.optimizer.step()
            action_rms, kl = measure()
            proposed_kl = kl
            step_scale, backtracks = 1., 0
            if not (math.isfinite(kl) and math.isfinite(action_rms) and kl <= self.max_kl):
                proposal = [p.detach().clone() for p in params]
                for backtracks in range(1, self.max_backtracks + 1):
                    reduction = .9 * math.sqrt(self.target_kl / kl) if math.isfinite(kl) and kl > 0 else .5
                    step_scale *= min(.8, reduction)
                    for parameter, old, new in zip(params, before, proposal):
                        parameter.copy_(torch.lerp(old, new, step_scale))
                    action_rms, kl = measure()
                    if math.isfinite(kl) and math.isfinite(action_rms) and kl <= self.max_kl:
                        break
            accepted = math.isfinite(kl) and math.isfinite(action_rms) and kl <= self.max_kl
            if accepted:
                # Preserve actor/encoder rate ratio; growth is gentle and cannot
                # exceed configured maxima. Backtracking already used
                # the same gradient, so moments must not be updated again.
                adjustment = min(1.2, max(.5, math.sqrt(self.target_kl / max(kl, 1e-16))))
                self.scale = min(self.max_scale, max(self.min_scale, previous_scale * step_scale * adjustment))
            else:
                restore()
                step_scale, action_rms, kl = 0., 0., 0.
                self.scale = max(self.min_scale, previous_scale * .5)
            self._set_lrs()
        except BaseException:
            restore()
            self.scale = previous_scale
            self._set_lrs()
            raise
        return self._metrics(action_rms, kl, proposed_kl, step_scale, accepted, backtracks, previous_scale)

    def _metrics(self, rms, kl, proposed_kl, step_scale, accepted, backtracks, previous_scale):
        encoder_rate = next((rate for name, rate in zip(self.group_names, self.base_lrs)
                             if name == "encoder_priv"), 0.)
        return {"actor/action_delta_rms": rms, "actor/update_mean_kl": kl,
                "actor/proposed_mean_kl": proposed_kl, "actor/step_scale": step_scale,
                "actor/step_accepted": float(accepted), "actor/backtracks": backtracks,
                "actor/lr": self.base_lrs[0] * previous_scale * step_scale,
                "actor/encoder_lr": encoder_rate * previous_scale * step_scale,
                "actor/next_lr": self.base_lrs[0] * self.scale,
                "actor/next_encoder_lr": encoder_rate * self.scale}


class FixedRateActorStep:
    """SuperMimic-style BPTT actor step: one fixed-rate Adam step per minibatch.

    The caller clips the global gradient norm; this step then clips every
    gradient element (SuperMimic ``grad_clip_value``) and runs Adam with a short
    second-moment EMA (``beta2``) so the rate recovers quickly after gradient
    spikes. There is no KL trust region, so the decaying exploration std no
    longer shrinks the update. The action change on the same replay starts is
    still measured and reported under the controller's metric names, as a
    diagnostic only.

    ``state_dict`` uses the controller's checkpoint format with ``scale=1``, so a
    checkpoint stays loadable by either update rule.
    """

    def __init__(self, optimizer, *, lr=2e-4, beta2=.95, clip_value=.1, group_lr_ratios=None):
        if not isinstance(optimizer, torch.optim.Adam):
            raise TypeError("FixedRateActorStep requires the existing Adam optimizer")
        if not (math.isfinite(lr) and lr > 0):
            raise ValueError("actor_adam_lr must be finite and positive")
        if not (math.isfinite(beta2) and 0 < beta2 < 1):
            raise ValueError("actor_adam_beta2 must be in (0, 1)")
        if not (math.isfinite(clip_value) and clip_value >= 0):
            raise ValueError("actor_grad_clip_value must be finite and nonnegative (zero disables)")
        count = len(optimizer.param_groups)
        ratios = (1.,) * count if group_lr_ratios is None else tuple(float(r) for r in group_lr_ratios)
        if len(ratios) != count or not all(math.isfinite(r) and r > 0 for r in ratios):
            raise ValueError("Expected one finite positive learning-rate ratio per optimizer group")
        self.optimizer = optimizer
        self.group_names = tuple(group.get("name", ("actor", "encoder_priv", "height")[i])
                                 for i, group in enumerate(optimizer.param_groups))
        self.lrs = tuple(float(lr) * ratio for ratio in ratios)
        self.beta2 = float(beta2)
        self.clip_value = float(clip_value)
        self.scale = 1.
        self._apply()

    def _apply(self):
        for group, rate in zip(self.optimizer.param_groups, self.lrs):
            if torch.is_tensor(group["lr"]):
                group["lr"].fill_(rate)
            else:
                group["lr"] = rate
            group["betas"] = (group["betas"][0], self.beta2)

    def state_dict(self):
        return {"version": 1, "mode": "fixed", "scale": 1., "base_lrs": self.lrs}

    def load_state_dict(self, state):
        """Call AFTER optimizer.load_state_dict, which restores saved rates/betas.

        Configured rates and beta2 stay authoritative; Adam moments are kept.
        ``None`` (or a legacy KL-controller state) migrates without error.
        """
        if state is not None:
            if state.get("version") != 1:
                raise ValueError("Unsupported WM actor step checkpoint version")
            if len(state.get("base_lrs", ())) != len(self.lrs):
                raise ValueError("WM actor optimizer group count changed on resume")
        self._apply()

    @torch.no_grad()
    def step(self, evaluate_mean: Callable[[], torch.Tensor], initial_mean,
             initial_std, valid_mask):
        """Apply already norm-clipped gradients; return controller-compatible metrics."""
        initial_mean = initial_mean.detach()
        initial_std = torch.broadcast_to(initial_std.detach(), initial_mean.shape)
        mask = valid_mask.to(device=initial_mean.device, dtype=torch.bool)
        if mask.shape == initial_mean.shape[:-1] + (1,):
            mask = mask.squeeze(-1)
        if mask.shape != initial_mean.shape[:-1]:
            raise ValueError("Actor step valid_mask must match the replay-start batch shape")
        if not mask.any():
            # No valid start contributed a gradient; do not let Adam momentum move.
            return self._metrics(0., 0., False)
        if self.clip_value > 0:
            params = [p for group in self.optimizer.param_groups for p in group["params"]
                      if p.grad is not None]
            if params:
                torch.nn.utils.clip_grad_value_(params, self.clip_value)
        self.optimizer.step()
        change = evaluate_mean()[mask] - initial_mean[mask]
        std = initial_std[mask].clamp_min(1e-8)
        rms = change.square().mean().sqrt().item()
        kl = (.5 * (change / std).square().sum(-1)).mean().item()
        return self._metrics(rms, kl, True)

    def _metrics(self, rms, kl, accepted):
        rate = self.lrs[0] if accepted else 0.
        encoder_rate = next((lr for name, lr in zip(self.group_names, self.lrs)
                             if name == "encoder_priv"), 0.)
        return {"actor/action_delta_rms": rms, "actor/update_mean_kl": kl,
                "actor/proposed_mean_kl": kl, "actor/step_scale": float(accepted),
                "actor/step_accepted": float(accepted), "actor/backtracks": 0,
                "actor/lr": rate, "actor/encoder_lr": encoder_rate if accepted else 0.,
                "actor/next_lr": self.lrs[0], "actor/next_encoder_lr": encoder_rate}
