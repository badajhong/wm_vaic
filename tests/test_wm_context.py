"""CPU checks for raw collection and hidden manipulation-state preservation."""

from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded
from torchrl.envs.transforms import VecNorm

from active_adaptation.learning.wm.context import RawWorldModelObservation


class _PhysicsView:
    def __init__(self, n, bodies):
        self.calls = 0
        self.masses = torch.full((n, bodies), 2.0)

    def get_masses(self):
        self.calls += 1
        return self.masses


def _asset(origins, bodies, joints):
    n = origins.shape[0]
    data = SimpleNamespace(
        root_link_pos_w=origins + torch.tensor([1.0, 2.0, 3.0]),
        root_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1),
        root_com_vel_w=torch.zeros(n, 6),
        body_link_pos_w=origins[:, None, :] + torch.ones(n, bodies, 3),
        body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, bodies, 1),
        body_com_vel_w=torch.zeros(n, bodies, 6),
        joint_pos=torch.zeros(n, joints),
        joint_vel=torch.zeros(n, joints),
        joint_stiffness=torch.ones(n, joints),
        joint_damping=torch.ones(n, joints),
        joint_armature=torch.ones(n, joints),
        joint_friction_coeff=torch.ones(n, joints),
    )
    return SimpleNamespace(data=data, root_physx_view=_PhysicsView(n, bodies),
                           _custom_friction=torch.full((n,), 0.1),
                           _custom_damping=torch.full((n,), 0.2),
                           custom_torques=torch.ones(n))


def _env():
    n = 2
    origins = torch.tensor([[0.0, 0.0, 0.0], [100.0, -100.0, 0.0]])
    robot, obj = _asset(origins, 3, 5), _asset(origins, 5, 4)
    command = SimpleNamespace(
        asset=robot, object=obj, object2=None, object_asset_name="skateboard",
        object_joint_idx_asset=[0, 1, 2, 3],
        eef_contact_forces_w=torch.ones(n, 2, 3), t=torch.arange(n),
        contact_target_pos_offset=torch.ones(n, 2, 3),
        contact_eef_pos_offset=torch.ones(n, 2, 3),
    )
    contact = SimpleNamespace(data=SimpleNamespace(
        net_forces_w=torch.ones(n, 3, 3), current_air_time=torch.zeros(n, 3),
        last_air_time=torch.ones(n, 3), current_contact_time=torch.zeros(n, 3),
    ))
    term = SimpleNamespace(error=torch.zeros(n),
                           _cum_error_mixin__cum_steps=torch.tensor([7, 8]),
                           _cum_error_mixin__exceeded=torch.tensor([False, True]))
    return SimpleNamespace(
        num_envs=n, device=torch.device("cpu"),
        scene=SimpleNamespace(env_origins=origins, sensors={"contact_forces": contact}),
        command_manager=command,
        action_manager=SimpleNamespace(action_buf=torch.zeros(n, 5, 3),
                                       applied_action=torch.zeros(n, 5), alpha=torch.ones(n, 1),
                                       delay=torch.zeros(n, 1, dtype=torch.long), offset=torch.zeros(n, 5)),
        episode_length_buf=torch.zeros(n, dtype=torch.long),
        termination_funcs={"cum_object_pos_error": term}, reward_groups={},
        observation_spec=Composite(policy=Unbounded((n, 3)), object_=Unbounded((n, 12)), shape=(n,)),
    )


class RawWorldModelObservationTests(unittest.TestCase):
    def test_context_contains_object_joints_hidden_state_and_relative_positions(self):
        env = _env()
        transform = RawWorldModelObservation(env, ["policy", "object_"])
        context = transform.snapshot_context()
        self.assertEqual(context.shape, (2, transform.context_dim))
        root = context[:, transform.context_slices["object.root_link_pos_w"]]
        torch.testing.assert_close(root, torch.tensor([[1., 2., 3.], [1., 2., 3.]]))
        self.assertIn("object.joint_vel", transform.context_slices)
        self.assertIn("action.action_buf", transform.context_slices)
        self.assertIn("contact.current_air_time", transform.context_slices)
        counter = context[:, transform.context_slices[
            "termination.cum_object_pos_error._cum_error_mixin__cum_steps"]]
        torch.testing.assert_close(counter, torch.tensor([[7.], [8.]]))
        env.command_manager.object.data.joint_vel.fill_(9)
        env.action_manager.delay.fill_(4)
        after = transform.snapshot_context()
        torch.testing.assert_close(after[:, transform.context_slices["object.joint_vel"]], torch.full((2, 4), 9.))
        torch.testing.assert_close(after[:, transform.context_slices["action.delay"]], torch.full((2, 1), 4.))
        self.assertFalse(torch.equal(context, after))

    def test_raw_snapshots_are_independent_of_vecnorm_and_simulator_mutation(self):
        env = _env()
        transform = RawWorldModelObservation(env, ["policy", "object_"])
        td = TensorDict({"policy": torch.tensor([[1., 2., 3.], [3., 4., 5.]]),
                         "object_": torch.zeros(2, 12)}, batch_size=[2])
        original = td["policy"].clone()
        transform(td)
        context_before = td["wm_observation", "wm_context"].clone()
        VecNorm(["policy"])(td)
        torch.testing.assert_close(td["wm_observation", "policy"], original)
        self.assertFalse(torch.equal(td["policy"], original))
        td["policy"].zero_()
        env.command_manager.object.data.joint_pos.add_(5)
        torch.testing.assert_close(td["wm_observation", "policy"], original)
        torch.testing.assert_close(td["wm_observation", "wm_context"], context_before)

    def test_terminal_snapshot_survives_reset_and_specs_match(self):
        env = _env()
        transform = RawWorldModelObservation(env, ["policy", "object_"])
        next_td = env.observation_spec.zero()
        env.command_manager.object.data.joint_pos.fill_(7)
        terminal = transform._step(None, next_td)
        env.command_manager.object.data.joint_pos.zero_()
        reset = transform._reset(None, env.observation_spec.zero())
        joint_slice = transform.context_slices["object.joint_pos"]
        torch.testing.assert_close(terminal["wm_observation", "wm_context"][:, joint_slice], torch.full((2, 4), 7.))
        torch.testing.assert_close(reset["wm_observation", "wm_context"][:, joint_slice], torch.zeros(2, 4))
        spec = transform.transform_observation_spec(env.observation_spec)
        self.assertEqual(spec["wm_observation", "wm_context"].shape, reset["wm_observation", "wm_context"].shape)
        self.assertNotIn("wm_observation", env.observation_spec.keys())
        self.assertTrue(spec.is_in(reset))

    def test_startup_physics_is_cached_without_per_step_queries(self):
        env = _env()
        transform = RawWorldModelObservation(env, ["policy"])
        physics = env.command_manager.object.root_physx_view
        self.assertEqual(physics.calls, 1)
        physics.masses.fill_(99)
        for _ in range(3):
            snapshot = transform.snapshot_context()
        self.assertEqual(physics.calls, 1)
        torch.testing.assert_close(snapshot[:, transform.context_slices["object.physics.mass"]], torch.full((2, 5), 2.))

    def test_missing_object_or_wheel_state_fails_at_construction(self):
        env = _env()
        del env.command_manager.object.data.joint_vel
        with self.assertRaisesRegex(ValueError, "object.joint_vel"):
            RawWorldModelObservation(env, ["policy"])
        env.command_manager.object = None
        with self.assertRaisesRegex(ValueError, "command_manager.object"):
            RawWorldModelObservation(env, ["policy"])

    def test_schema_changes_are_rejected(self):
        env = _env()
        transform = RawWorldModelObservation(env, ["policy"])
        env.command_manager.object.data.joint_vel = torch.zeros(2, 5)
        with self.assertRaisesRegex(RuntimeError, "schema changed.*object.joint_vel"):
            transform.snapshot_context()

    def test_static_slices_freeze_only_episode_constants_and_physics(self):
        env = _env()
        transform = RawWorldModelObservation(env, ["policy"])
        static = transform.static_context_slices
        for name in ("object.physics.mass", "robot.physics.mass", "action.alpha", "action.delay",
                     "action.offset", "object._custom_friction", "object._custom_damping",
                     "object.joint_stiffness", "robot.joint_damping", "robot.joint_armature",
                     "object.joint_friction_coeff", "command.contact_target_pos_offset",
                     "command.contact_eef_pos_offset"):
            self.assertEqual(static[name], transform.context_slices[name])
        for name in ("object.joint_pos", "object.joint_vel", "object.custom_torques",
                     "action.action_buf", "action.applied_action", "contact.net_forces_w",
                     "command.eef_contact_forces_w", "episode_length", "command.t"):
            self.assertNotIn(name, static)
        # The static label applies to imagination, not to subsequent real resets.
        old = transform.snapshot_context()
        env.command_manager.object._custom_friction.fill_(0.7)
        env.action_manager.alpha.fill_(0.85)
        new = transform.snapshot_context()
        self.assertFalse(torch.equal(old[:, static["object._custom_friction"]],
                                     new[:, static["object._custom_friction"]]))
        torch.testing.assert_close(new[:, static["action.alpha"]], torch.full((2, 1), 0.85))

    def test_force_history_does_not_duplicate_current_force(self):
        env = _env()
        env.scene.sensors["contact_forces"].data.net_forces_w_history = torch.ones(2, 3, 3, 3)
        transform = RawWorldModelObservation(env, ["policy"])
        self.assertIn("contact.net_forces_w_history", transform.context_slices)
        self.assertNotIn("contact.net_forces_w", transform.context_slices)

    def test_contact_substeps_capture_existing_data_and_reset_only_selected_envs(self):
        env = _env()
        sensor = env.scene.sensors["contact_forces"]
        sensor._data = sensor.data
        sensor._timestamp = torch.zeros(2)
        sensor._timestamp_last_update = torch.zeros(2)
        sensor.cfg = SimpleNamespace(history_length=3)
        env.decimation = 4
        env._post_step_callbacks, env._reset_callbacks = [], []
        transform = RawWorldModelObservation(env, ["policy"])
        self.assertEqual(len(env._post_step_callbacks), 1)
        callback = env._post_step_callbacks[0]
        # The callback must use the already updated data, without touching the
        # public property or changing any sensor state.
        sensor.data = None
        for substep in range(4):
            sensor._timestamp.add_(.005)
            sensor._timestamp_last_update.copy_(sensor._timestamp)
            sensor._data.net_forces_w.fill_(substep + 1)
            callback(substep)
            torch.testing.assert_close(sensor._data.net_forces_w, torch.full((2, 3, 3), substep + 1.))
        context = transform.snapshot_context()
        forces = context[:, transform.context_slices["contact.net_forces_w_substeps"]].reshape(2, 4, 3, 3)
        torch.testing.assert_close(forces, torch.arange(1., 5.).view(1, 4, 1, 1).expand(2, 4, 3, 3))
        for key in ("contact.timestamp", "contact.timestamp_last_update"):
            torch.testing.assert_close(context[:, transform.context_slices[key]], sensor._timestamp[:, None])
            self.assertNotIn(key, transform.exogenous_context_slices)
            self.assertNotIn(key, transform.static_context_slices)
        env._reset_callbacks[0](torch.tensor([1]))
        after = transform.snapshot_context()[:, transform.context_slices["contact.net_forces_w_substeps"]].reshape_as(forces)
        torch.testing.assert_close(after[0], forces[0])
        self.assertEqual(after[1].count_nonzero(), 0)

    def test_contact_capture_requires_sensor_updates_every_substep(self):
        env = _env()
        sensor = env.scene.sensors["contact_forces"]
        sensor._data = sensor.data
        sensor._timestamp = torch.zeros(2)
        sensor.cfg = SimpleNamespace(history_length=0)
        with self.assertRaisesRegex(ValueError, "per-substep"):
            RawWorldModelObservation(env, ["policy"])


if __name__ == "__main__":
    unittest.main()
