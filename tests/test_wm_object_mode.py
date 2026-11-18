"""Root-only task guards without simulator or GPU execution."""

from types import SimpleNamespace

import pytest

from active_adaptation.learning.wm.object_mode import validate_root_only_task


def _observation(kind):
    return type(kind, (), {})()


def _contract():
    adapter = SimpleNamespace(
        command=SimpleNamespace(object2=None),
        codec=SimpleNamespace(keys=("policy", "priv", "object_", "object_geo_")),
        groups={
            "policy": SimpleNamespace(funcs={"joint": _observation("joint_pos_history")}),
            "priv": SimpleNamespace(funcs={
                "object_difference": _observation("diff_object_pos_future"),
                "contact": _observation("diff_contact_pos_b"),
            }),
            "object_": SimpleNamespace(funcs={"pose": _observation("object_pos_b")}),
            "object_geo_": SimpleNamespace(funcs={"points": _observation("object_points")}),
            # A camera group excluded by the model codec is not an imagined input.
            "depth": SimpleNamespace(funcs={"camera": _observation("depth_camera")}),
        },
    )
    reward = SimpleNamespace(
        reward_terms=[SimpleNamespace(name=name, kind=name) for name in (
            "object_pos_tracking", "object_ori_tracking", "eef_contact_exp",
            "joint_pos_tracking_product", "joint_vel_l2", "impact_force_l2",
            "feet_slip", "feet_air_time_skateboard")],
        termination_terms=[SimpleNamespace(name=name, kind=name) for name in (
            "cum_object_pos_error", "cum_object_ori_error", "cum_lost_contact_steps", "cum_joint_pos_error")],
    )
    return adapter, reward


def test_root_only_task_accepts_root_contact_and_robot_joint_dependencies_without_mutation():
    adapter, reward = _contract()
    groups_before = {name: tuple(group.funcs.items()) for name, group in adapter.groups.items()}
    terms_before = (tuple(reward.reward_terms), tuple(reward.termination_terms))
    assert validate_root_only_task(adapter, reward) is None
    assert groups_before == {name: tuple(group.funcs.items()) for name, group in adapter.groups.items()}
    assert terms_before == (tuple(reward.reward_terms), tuple(reward.termination_terms))


@pytest.mark.parametrize("kind", ["object_joint_pos", "object_joint_vel", "diff_object_joint_pos_future",
                                  "object_joint_torque", "object_body_velocity", "unknown_future_observation"])
def test_root_only_task_rejects_wheel_state_and_unaudited_selected_observations(kind):
    adapter, reward = _contract()
    adapter.groups["priv"].funcs["wheel_dependency"] = _observation(kind)
    with pytest.raises(ValueError, match=r"observation priv.wheel_dependency"):
        validate_root_only_task(adapter, reward)


@pytest.mark.parametrize("attribute,kind,label", [
    ("reward_terms", "object_joint_pos_tracking", "reward"),
    ("reward_terms", "unknown_future_reward", "reward"),
    ("termination_terms", "cum_object_joint_pos_error", "termination"),
    ("termination_terms", "unknown_future_termination", "termination"),
])
def test_root_only_task_rejects_articulated_and_unaudited_reward_or_termination(attribute, kind, label):
    adapter, reward = _contract()
    getattr(reward, attribute).append(SimpleNamespace(name="wheel_dependency", kind=kind))
    with pytest.raises(ValueError, match=label + " wheel_dependency"):
        validate_root_only_task(adapter, reward)


def test_root_only_task_rejects_second_manipulated_object():
    adapter, reward = _contract()
    adapter.command.object2 = object()
    with pytest.raises(ValueError, match="object2"):
        validate_root_only_task(adapter, reward)
