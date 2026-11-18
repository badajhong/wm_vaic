"""Physical observation reconstruction and counterfactual isolation on CPU."""

from collections import OrderedDict
from types import SimpleNamespace
import unittest

import torch
from torchrl.data import Composite, Unbounded

from active_adaptation.learning.wm.adapter import VAICWorldModelAdapter
from active_adaptation.learning.wm.context import RawWorldModelObservation
from active_adaptation.learning.wm.core import ObservationCodec
from test_wm_context import _env


def _observation(kind, **attributes):
    return type(kind, (), {})(**{}) if not attributes else _attributes(type(kind, (), {})(), attributes)


def _attributes(obj, attributes):
    vars(obj).update(attributes)
    return obj


def _fixture(*, references=False, noisy=False, contact_substeps=False):
    env = _env()
    env.cfg = SimpleNamespace(terrain="plane")
    command = env.command_manager
    command.tracking_body_indices_asset = [0, 1, 2]
    command.tracking_joint_indices_asset = [0, 1, 2, 3, 4]
    command.object_body_id_asset = 0
    command.contact_eef_body_indices_asset = [1, 2]
    command.robot_root_pos_w = command.asset.data.root_link_pos_w.clone()
    command.robot_root_quat_w = command.asset.data.root_link_quat_w.clone()
    command.contact_target_pos_w = env.scene.env_origins[:, None] + torch.ones(2, 2, 3)
    command.contact_eef_pos_w = command.contact_target_pos_w.clone()
    if references:
        command.ref_object_pos_future_w = env.scene.env_origins[:, None] + torch.full((2, 2, 3), 4.)
    joint_history = _observation("joint_pos_history", buffer=torch.zeros(2, 3, 5),
        joint_pos=torch.zeros(2, 2, 5), joint_ids=torch.arange(5),
        history_steps=[0, 1, 2], joint_mask=torch.ones(1, 1, 5), noise_std=0.)
    policy = OrderedDict(
        root_ang_vel_history=_observation("root_ang_vel_history", buffer=torch.zeros(2, 1, 3),
                                         history_steps=[0], noise_std=.1 if noisy else 0.),
        joint_pos_history=joint_history,
        prev_actions=_observation("prev_actions", steps=3, permute=False, flatten=True),
    )
    object_observations = OrderedDict(object_pos_b=_observation("object_pos_b"),
                                      object_ori_b=_observation("object_ori_b"))
    privileged_observations = object_observations.copy()
    if references:
        privileged_observations["diff_object_pos_future"] = _observation("diff_object_pos_future")
    env.observation_funcs = OrderedDict(policy=SimpleNamespace(funcs=policy),
        object_=SimpleNamespace(funcs=object_observations), priv=SimpleNamespace(funcs=privileged_observations))
    env.observation_spec = Composite(policy=Unbounded((2, 33)), object_=Unbounded((2, 12)),
                                     priv=Unbounded((2, 18 if references else 12)), shape=(2,))
    env._update_callbacks = []
    if contact_substeps:
        sensor = env.scene.sensors["contact_forces"]
        sensor._data = sensor.data
        sensor._timestamp = torch.full((2,), 0.02)
        sensor._timestamp_last_update = sensor._timestamp.clone()
        sensor.cfg = SimpleNamespace(history_length=3)
        env.decimation = 4
        env._post_step_callbacks = []
        env._reset_callbacks = []
    transform = RawWorldModelObservation(env, ["policy", "object_", "priv"])
    spec = transform.transform_observation_spec(env.observation_spec)["wm_observation"]
    codec = ObservationCodec(spec, list(spec.keys()))
    adapter = VAICWorldModelAdapter(env, transform, codec)

    def snapshot():
        return codec.pack(transform(env.observation_spec.zero())["wm_observation"])

    previous = snapshot()
    joint_history.joint_pos[:, 0] = 3.
    joint_history.joint_pos[:, 1] = 5.
    state = snapshot()
    return env, transform, codec, adapter, previous, state


class PhysicalObservationAdapterTests(unittest.TestCase):
    def test_context_unpack_handles_sequences_and_preserves_gradients(self):
        _, _, codec, adapter, _, state = _fixture()
        state = state[:, None].repeat(1, 3, 1).detach().requires_grad_()
        values = adapter._get_contexts(state)
        for name, value in values.items():
            torch.testing.assert_close(value, adapter.context(state, name))
        sum(value.sum() for value in values.values()).backward()
        expected = torch.zeros_like(state)
        expected[..., codec.slices["wm_context"]] = 1.
        torch.testing.assert_close(state.grad, expected)

    def test_context_replacement_preserves_schema_and_both_gradient_sources(self):
        _, _, codec, adapter, _, state = _fixture()
        state = state.detach().requires_grad_()
        replacement = torch.randn(2, 3, requires_grad=True)
        result = adapter.set_context(state, {"object.root_link_pos_w": replacement})
        section = adapter.context_slices["object.root_link_pos_w"]
        begin = codec.slices["wm_context"].start + section.start
        end = codec.slices["wm_context"].start + section.stop
        torch.testing.assert_close(result[:, :begin], state[:, :begin])
        torch.testing.assert_close(result[:, begin:end], replacement)
        torch.testing.assert_close(result[:, end:], state[:, end:])
        result.sum().backward()
        expected = torch.ones_like(state)
        expected[:, begin:end] = 0.
        torch.testing.assert_close(state.grad, expected)
        torch.testing.assert_close(replacement.grad, torch.ones_like(replacement))

    def test_histories_are_updated_from_predicted_physics_and_previous_history(self):
        env, transform, codec, adapter, previous, state = _fixture()
        action = torch.full((2, 5), 2.)
        result = adapter.rebuild(state, state, action, previous_state=previous)
        history = adapter.context(result, "obs_history.policy.joint_pos_history")
        torch.testing.assert_close(history[:, 0], torch.full((2, 5), 4.))
        torch.testing.assert_close(history[:, 1:], torch.zeros(2, 2, 5))
        actions = adapter.context(result, "action.action_buf")
        torch.testing.assert_close(actions[..., 0], action)
        torch.testing.assert_close(actions[..., 1:], torch.zeros(2, 5, 2))
        self.assertEqual(len(env._update_callbacks), 1)

    def test_context_only_training_path_keeps_all_context_updates(self):
        _, _, codec, adapter, previous, state = _fixture(references=True, noisy=True)
        action = torch.full((2, 5), 2.)
        full = adapter.rebuild(state, state, action, previous_state=previous)
        compact = adapter.rebuild(state, state, action, previous_state=previous,
                                  rebuild_observations=False)
        torch.testing.assert_close(compact[:, codec.slices["wm_context"]],
                                   full[:, codec.slices["wm_context"]])
        for key in codec.keys:
            if key != "wm_context":
                torch.testing.assert_close(compact[:, codec.slices[key]], state[:, codec.slices[key]])
        self.assertFalse(torch.equal(full[:, codec.slices["policy"]], compact[:, codec.slices["policy"]]))

    def test_next_template_physical_state_and_observations_cannot_leak(self):
        _, transform, codec, adapter, previous, state = _fixture()
        action = torch.ones(2, 5)
        malicious_template = state.clone()
        malicious_template = adapter.set_context(malicious_template, {
            "object.root_link_pos_w": torch.full((2, 3), 900.),
            "obs_history.policy.joint_pos_history": torch.full((2, 3, 5), 800.),
        })
        malicious_template[:, codec.slices["object_"]] = 700.
        expected = adapter.rebuild(state, state, action, previous_state=previous)
        actual = adapter.rebuild(state, malicious_template, action, previous_state=previous)
        torch.testing.assert_close(actual, expected)

    def test_substep_contact_state_stays_physical_and_differentiable(self):
        _, transform, _, adapter, previous, state = _fixture(contact_substeps=True)
        names = ("contact.timestamp", "contact.timestamp_last_update",
                 "contact.net_forces_w_substeps", "contact.current_contact_time",
                 "contact.current_air_time", "contact.last_air_time")
        for name in names:
            self.assertNotIn(name, transform.exogenous_context_slices)
            self.assertNotIn(name, transform.static_context_slices)
        state = state.detach().requires_grad_()
        malicious_template = adapter.set_context(state.detach(), {
            name: torch.full_like(adapter.context(state, name), 900.) for name in names
        })
        actual = adapter.rebuild(state, malicious_template, torch.zeros(2, 5),
                                 previous_state=previous)
        for name in names:
            torch.testing.assert_close(adapter.context(actual, name), adapter.context(state, name))
        adapter.context(actual, "contact.net_forces_w_substeps").sum().backward()
        torch.testing.assert_close(adapter.context(state.grad, "contact.net_forces_w_substeps"),
                                   torch.ones_like(adapter.context(state, "contact.net_forces_w_substeps")))

    def test_actor_and_critic_object_inputs_derive_from_same_differentiable_state(self):
        _, _, codec, adapter, previous, state = _fixture()
        state = state.clone().requires_grad_()
        result = adapter.rebuild(state, state.detach(), torch.ones(2, 5), previous_state=previous)
        obs = codec.unpack(result)
        torch.testing.assert_close(obs["object_"], obs["priv"])
        loss = obs["object_"][..., :3].sum() + obs["policy"].sum()
        loss.backward()
        context_start = codec.slices["wm_context"].start
        for field in ("object.root_link_pos_w", "robot.joint_pos_substeps"):
            selected = adapter.context_slices[field]
            self.assertGreater(state.grad[:, context_start+selected.start:context_start+selected.stop].abs().sum().item(), 0.)
        self.assertEqual(state.grad[:, codec.slices["object_"]].abs().sum().item(), 0.)
        self.assertEqual(state.grad[:, codec.slices["policy"]].abs().sum().item(), 0.)

    def test_future_reference_changes_tracking_error_without_changing_physical_state(self):
        _, transform, codec, adapter, previous, state = _fixture(references=True)
        field = "pre.ref_object_pos_future_w"
        self.assertIn(field, transform.exogenous_context_slices)
        template = adapter.set_context(state, {field: torch.full((2, 2, 3), 9.),
            "pre.robot_root_pos_w": torch.full((2, 3), 100.),
            "object.root_link_pos_w": torch.full((2, 3), 200.)})
        rebuilt = adapter.rebuild(state, template, torch.zeros(2, 5), previous_state=previous)
        actual = codec.unpack(rebuilt)["priv"][:, 12:].reshape(2, 2, 3)
        expected = 9. - adapter.context(state, "object.root_link_pos_w")[:, None]
        torch.testing.assert_close(actual, expected.expand_as(actual))
        torch.testing.assert_close(adapter.context(rebuilt, "pre.robot_root_pos_w"),
                                   adapter.context(previous, "command.robot_root_pos_w"))
        torch.testing.assert_close(adapter.context(rebuilt, "object.root_link_pos_w"),
                                   adapter.context(state, "object.root_link_pos_w"))

    def test_recorded_noise_is_used_with_predicted_sensor_value(self):
        _, transform, codec, adapter, previous, state = _fixture(noisy=True)
        noise_name = "obs_noise.policy.root_ang_vel_history"
        self.assertIn(noise_name, transform.exogenous_context_slices)
        velocity = torch.zeros(2, 6)
        velocity[:, 3:] = .75
        state = adapter.set_context(state, {"robot.root_com_vel_w": velocity})
        template = adapter.set_context(state, {noise_name: torch.full((2, 3), .125),
            "robot.root_com_vel_w": torch.full((2, 6), 900.),
            "obs_history.policy.root_ang_vel_history": torch.full((2, 1, 3), 800.)})
        rebuilt = adapter.rebuild(state, template, torch.zeros(2, 5), previous_state=previous)
        torch.testing.assert_close(codec.unpack(rebuilt)["policy"][:, :3], torch.full((2, 3), .875))

    def test_previous_physical_cache_controls_lagged_object_observations(self):
        _, _, codec, adapter, previous, state = _fixture()
        previous = adapter.set_context(previous, {"command.robot_root_pos_w": torch.full((2, 3), -2.)})
        state = adapter.set_context(state, {"robot.root_link_pos_w": torch.full((2, 3), 100.)})
        previous = previous.requires_grad_()
        rebuilt = adapter.rebuild(state, state.detach(), torch.zeros(2, 5), previous_state=previous)
        positions = codec.unpack(rebuilt)["object_"][:, :3]
        torch.testing.assert_close(positions, adapter.context(state, "object.root_link_pos_w") + 2.)
        positions.sum().backward()
        selected = adapter.context(previous.grad, "command.robot_root_pos_w")
        torch.testing.assert_close(selected, -torch.ones_like(selected))


if __name__ == "__main__":
    unittest.main()
