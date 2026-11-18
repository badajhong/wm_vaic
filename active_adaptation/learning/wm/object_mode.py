"""Task-contract checks for the opt-in root-only object dynamics model.

These checks describe the existing VAIC observation/reward equations; they do
not change the real articulated asset or its randomization. Geometry safety
(the chosen contact body is the modeled root) is checked by the physical model.
This module deliberately imports neither Isaac nor the task implementations.
"""


# Fail closed if a future observation/reward adapter adds a new dependency on
# articulated object state. Robot joint observations/rewards remain supported.
_ROOT_ONLY_OBSERVATIONS = frozenset({
    "root_ang_vel_history", "projected_gravity_history", "joint_pos_history", "prev_actions",
    "ref_root_pos_future_b", "ref_root_ori_future_b", "diff_body_pos_future_local",
    "diff_body_ori_future_local", "diff_body_lin_vel_future_local", "diff_body_ang_vel_future_local",
    "root_linvel_b", "body_pos_b", "body_vel_b", "body_height", "applied_action", "applied_torque",
    "object_xy_b", "object_heading_b", "object_pos_b", "object_ori_b", "ref_contact_pos_b",
    "diff_object_pos_future", "diff_object_ori_future", "ref_object_contact_future", "diff_contact_pos_b",
    "ref_root_vel_future_b", "ref_contact_flag_future", "object_points",
    "ref_body_pos_future_local", "ref_joint_pos_future", "ref_motion_phase", "ref_joint_pos_action_policy",
})

_ROOT_ONLY_REWARDS = frozenset({
    "keypoint_pos_tracking_product", "keypoint_pos_tracking_local_product",
    "keypoint_ori_tracking_product", "keypoint_ori_tracking_local_product",
    "keypoint_lin_vel_tracking_product", "keypoint_ang_vel_tracking_product",
    "joint_pos_tracking_product", "joint_vel_tracking_product",
    "object_pos_tracking", "object_ori_tracking", "eef_contact_exp",
    "action_rate_l2", "action_rate2_l2", "joint_vel_l2", "joint_pos_limits",
    "joint_torque_limits", "survival", "impact_force_l2", "feet_slip", "feet_air_time_skateboard",
})

_ROOT_ONLY_TERMINATIONS = frozenset({
    "cum_body_pos_error", "cum_body_z_error", "cum_body_ori_error",
    "cum_body_pos_error_local", "cum_body_ori_error_local", "cum_joint_pos_error",
    "cum_object_pos_error", "cum_object_ori_error", "cum_lost_contact_steps",
})


def validate_root_only_task(adapter, reward):
    """Reject task equations that cannot consume root-only object predictions.

    Call after constructing the original observation and analytic reward
    adapters, and only when ``object_dynamics == 'root'``. The physical model
    separately checks the object kinematic root. No live state is read or changed.
    Disabled reward terms have already been filtered by ``VAICRewardModel``;
    its own constructor still rejects disabled terms with unsupported side effects.
    """
    command = adapter.command
    if getattr(command, "object2", None) is not None:
        raise ValueError("Root-only object dynamics supports one manipulated object; object2 is unsupported")

    selected_groups = set(adapter.codec.keys)
    for group_name, group in adapter.groups.items():
        if group_name not in selected_groups:
            continue
        for name, observation in group.funcs.items():
            kind = type(observation).__name__
            if kind not in _ROOT_ONLY_OBSERVATIONS:
                raise ValueError(
                    f"Root-only object dynamics does not support observation {group_name}.{name} ({kind}); "
                    "object joint/body dependencies must be audited before enabling this task")

    for terms, supported, label in (
        (reward.reward_terms, _ROOT_ONLY_REWARDS, "reward"),
        (reward.termination_terms, _ROOT_ONLY_TERMINATIONS, "termination"),
    ):
        for term in terms:
            if term.kind not in supported:
                raise ValueError(
                    f"Root-only object dynamics does not support {label} {term.name} ({term.kind}); "
                    "object joint/body dependencies must be audited before enabling this task")

