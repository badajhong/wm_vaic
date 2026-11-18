"""Analytic per-step reward ceilings for VAIC reward terms.

SuperMimic bounds its critic and its return-based reset sampler by
``v_max = max_step_reward / (1 - gamma)``, where ``max_step_reward`` is the
largest per-step reward the task can pay. VAIC terms are either bounded
kernels (at most 1, or ``gain`` for the contact kernel) or nonpositive
penalties. A term of unknown kind, or a weighting that could make a penalty
positive, has no analytic ceiling and reports ``inf``.
"""

from __future__ import annotations

import math

# exp(-error / sigma) kernels and the constant survival bonus: at most 1.
UNIT_KINDS = frozenset({
    "keypoint_pos_tracking_product", "keypoint_pos_tracking_local_product",
    "keypoint_ori_tracking_product", "keypoint_ori_tracking_local_product",
    "keypoint_lin_vel_tracking_product", "keypoint_ang_vel_tracking_product",
    "joint_pos_tracking_product", "joint_vel_tracking_product",
    "object_pos_tracking", "object_ori_tracking", "object_vel_tracking", "survival",
})
# Penalties whose raw value is never positive.
NONPOSITIVE_KINDS = frozenset({
    "action_rate_l2", "action_rate2_l2", "joint_vel_l2", "joint_pos_limits",
    "joint_torque_limits", "impact_force_l2", "feet_slip", "feet_air_time_skateboard",
})


def term_ceiling(kind, weight, gain=1.0):
    """Largest ``weight * value`` one term can pay per step (before dt scaling)."""
    weight = float(weight)
    if kind in NONPOSITIVE_KINDS:
        return 0. if weight >= 0 else math.inf
    if kind in UNIT_KINDS:
        return max(weight, 0.)
    if kind == "eef_contact_exp":
        return max(weight * float(gain), 0.)
    return math.inf


def group_ceiling(terms, multiplicative=False):
    """Ceiling of one reward group from ``(kind, weight, gain)`` triples.

    Additive groups sum the term ceilings. A multiplicative group is bounded only
    when every factor is a nonnegative bounded kernel; a product of penalties
    could be positive and is reported as unbounded.
    """
    terms = list(terms)
    if not multiplicative:
        return sum(term_ceiling(kind, weight, gain) for kind, weight, gain in terms)
    if not terms:
        return 1.
    product = 1.
    for kind, weight, gain in terms:
        if kind in NONPOSITIVE_KINDS or weight < 0:
            return math.inf
        product *= term_ceiling(kind, weight, gain)
    return product


def environment_step_ceiling(env):
    """Largest summed per-step reward the live environment's groups can pay."""
    scale = float(env.step_dt) if getattr(env, "mult_dt", True) else 1.
    total = 0.
    for group in env.reward_groups.values():
        terms = [(type(func).__name__, getattr(func, "weight", 1.), getattr(func, "gain", 1.))
                 for func in group.funcs.values() if getattr(func, "enabled", True)]
        total += group_ceiling(terms, bool(getattr(group, "multiplicative", False)))
    return total * scale
