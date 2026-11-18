"""Analytic soft termination for imagined VAIC rollouts only.

The real reward/termination model stays hard. Threshold probabilities below are
surrogates, not calibrated failure probabilities. Within each termination term,
run-length cohorts enforce the exact consecutive-violation grace period. The
OR across different soft terms assumes conditional independence; with binary
violations it reproduces the original hard OR exactly.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping
from typing import NamedTuple

import torch


def _product(values, keepdim=False):
    """Product over the last dimension that is safe inside CUDA-graph capture.

    ``torch.prod``'s backward calls ``nonzero()`` (a host synchronization) to
    handle zero factors, which aborts the captured actor backward. The few
    term/contact factors here are multiplied sequentially instead: same value,
    and an ordinary elementwise backward.
    """
    if values.shape[-1] == 0:
        result = values.new_ones(values.shape[:-1])
    else:
        result = functools.reduce(torch.mul, values.unbind(-1))
    return result.unsqueeze(-1) if keepdim else result


class SoftTerminationCarry(NamedTuple):
    # Conditional-on-survival mass: slots 0:H are runs of those lengths;
    # slot H is the initial factual run, whose age is initial_counts + elapsed.
    cohorts: torch.Tensor
    initial_counts: torch.Tensor
    elapsed: int


class SoftTerminationModel:
    """Parameterless imagined-termination helper with O(terms * horizon) state.

    Construct once per imagined rollout. ``initialize`` only reads factual
    counters; subsequent steps use their own differentiable carry, never the
    hard counters written into the imagined physical state. No simulator state,
    model parameter, optimizer, normalizer or random generator is mutated.
    """

    softness = .05

    def __init__(self, reward_model, horizon, *, contact_force=True):
        """``contact_force=False`` judges lost contact by FK position alone.

        The learned contact-force head misses most contact events at multi-step
        horizons, so its force condition can report a planted foot as lost (or
        hide nothing the position check would not). Position comes from the
        predicted body pose and the object state, which the model predicts
        accurately. The factual counters seeding each rollout are unchanged.
        """
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            raise ValueError("Soft termination requires a positive integer horizon")
        if not isinstance(contact_force, bool):
            raise ValueError("contact_force must be a bool")
        self.reward_model = reward_model
        self.horizon = horizon
        self.contact_force = contact_force
        self.terms = tuple(reward_model.termination_terms)
        self._regular_masks = {}
        for term in self.terms:
            if (not isinstance(term.min_steps, int) or isinstance(term.min_steps, bool)
                    or term.min_steps < 1):
                raise ValueError("Consecutive termination min_steps must be a positive integer")
            if not math.isfinite(float(term.threshold)):
                raise ValueError("Termination thresholds must be finite")

    def initialize(self, state):
        context = self.reward_model._decode_context(state)
        if self.terms:
            counts = torch.stack([
                self.reward_model._get(context, "termination." + term.name + "._cum_error_mixin__cum_steps")
                for term in self.terms
            ], -1).detach()
            if not counts.is_floating_point():
                counts = counts.to(torch.get_default_dtype())
        else:
            if isinstance(state, Mapping):
                name, template = next(iter(state.items()))
                event_ndim = len(self.reward_model.context_shapes.get(name, template.shape[1:]))
                batch_shape = template.shape[:-event_ndim] if event_ndim else template.shape
            else:
                template, batch_shape = state, state.shape[:-1]
            counts = template.new_zeros(*batch_shape, 0)
        cohorts = torch.cat((counts.new_zeros(*counts.shape, self.horizon),
                             counts.new_ones(*counts.shape, 1)), -1)
        if self.terms:
            self._regular_mask(counts.device)
        return SoftTerminationCarry(cohorts, counts, 0)

    def _regular_mask(self, device):
        # Construct with device kernels, never a CPU tensor copied during CUDA
        # capture. A cached helper reuses this constant across every rollout.
        if device not in self._regular_masks:
            ages = torch.arange(1, self.horizon + 1, device=device)
            self._regular_masks[device] = torch.stack([ages >= term.min_steps for term in self.terms], 0)
        return self._regular_masks[device]

    def _lost_contact_probability(self, term, state, next_state, next_reference):
        reward = self.reward_model
        distance = (reward._get(state, "command.contact_eef_pos_w")
                    - reward._get(state, "command.contact_target_pos_w")).norm(dim=-1)
        if term.pos_thres <= 0:
            position_ok = torch.zeros_like(distance)
        else:
            position_ok = torch.sigmoid((1 - distance / term.pos_thres) / self.softness)
        threshold = term.frc_thres
        if not self.contact_force:
            force_ok = torch.ones_like(distance)
        elif isinstance(threshold, torch.Tensor):
            force = reward._contact_force(state)
            threshold = threshold.to(force)
            magnitude = force.abs() if threshold.numel() > 1 else force.norm(dim=-1)
            safe_threshold = torch.where(threshold > 0, threshold, torch.ones_like(threshold))
            force_ok = torch.where(threshold > 0,
                                   torch.sigmoid((magnitude / safe_threshold - 1) / self.softness),
                                   torch.ones_like(magnitude))
            if threshold.numel() > 1:
                force_ok = _product(force_ok)
        elif threshold <= 0:
            force_ok = torch.ones_like(distance)
        else:
            force = reward._contact_force(state)
            force_ok = torch.sigmoid((force.norm(dim=-1) / threshold - 1) / self.softness)
        active = reward._reference(state, next_state, "ref_body_contact", next_reference).bool().to(distance.dtype)
        lost = active * (1 - position_ok * force_ok)
        # Original contact error is exactly 0 or 2. Preserve thresholds outside
        # that interval rather than silently substituting a different condition.
        if term.threshold <= 0:
            return torch.ones_like(distance[..., 0])
        if term.threshold > 2:
            return torch.zeros_like(distance[..., 0])
        return 1 - _product(1 - lost)

    def violation_probabilities(self, state, next_state, next_reference=None):
        """Continuous per-term threshold surrogate with the original timing."""
        reward = self.reward_model
        state, next_state, next_reference = (
            reward._decode_context(value) for value in (state, next_state, next_reference))
        if hasattr(reward, "termination_errors"):
            errors = reward.termination_errors(state, next_state, next_reference)
        else:
            errors = {name: values["error"] for name, values in
                      reward.termination_details(state, next_state, next_reference).items()}
        probabilities = []
        for term in self.terms:
            if term.kind == "cum_lost_contact_steps":
                probability = self._lost_contact_probability(term, state, next_state, next_reference)
            elif term.threshold <= 0:
                probability = torch.ones_like(errors[term.name])
            else:
                probability = torch.sigmoid((errors[term.name] / term.threshold - 1) / self.softness)
            probabilities.append(probability)
        if probabilities:
            return torch.stack(probabilities, -1)
        return self.initialize(state).initial_counts

    def step_probabilities(self, probabilities, carry):
        """Advance cohorts from supplied probabilities; useful for exact tests.

        Conditional cohort normalization avoids survival-mass underflow. If a
        term absorbs all its mass, later hazards are zero: the caller's survival
        weight has already reached zero. No epsilon clamp changes p=0 or p=1.
        """
        if carry.elapsed >= self.horizon:
            raise ValueError("Soft termination carry exceeded its initialized horizon")
        if probabilities.shape != carry.initial_counts.shape:
            raise ValueError("Violation probabilities must match the initial counter shape")
        q = carry.cohorts
        if not self.terms:
            return q.new_zeros(*q.shape[:-2], 1), SoftTerminationCarry(q, carry.initial_counts, carry.elapsed + 1)
        regular_eligible = self._regular_mask(q.device)
        initial_eligible = torch.stack([
            carry.initial_counts[..., index] + carry.elapsed + 1 >= term.min_steps
            for index, term in enumerate(self.terms)
        ], -1)
        p = probabilities.unsqueeze(-1)
        regular_success = q[..., :self.horizon] * p
        initial_success = q[..., -1] * probabilities
        absorbed = (regular_success * regular_eligible).sum(-1) + initial_success * initial_eligible
        prior_mass = q.sum(-1)
        term_hazard = (absorbed / torch.where(prior_mass > 0, prior_mass, torch.ones_like(prior_mass))).clamp(0, 1)
        remaining = torch.cat((
            ((1 - probabilities) * prior_mass).unsqueeze(-1),
            regular_success[..., :-1] * (~regular_eligible[..., :-1]),
            (initial_success * (~initial_eligible)).unsqueeze(-1),
        ), -1)
        remaining_mass = remaining.sum(-1, keepdim=True)
        conditional = remaining / torch.where(remaining_mass > 0, remaining_mass, torch.ones_like(remaining_mass))
        hazard = (1 - _product(1 - term_hazard, keepdim=True)).clamp(0, 1)
        return hazard, SoftTerminationCarry(conditional, carry.initial_counts, carry.elapsed + 1)

    def step(self, state, next_state, carry, next_reference=None):
        probabilities = self.violation_probabilities(state, next_state, next_reference)
        return self.step_probabilities(probabilities, carry)
