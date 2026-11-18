"""Independent reference time, motion boundaries and source-isolation on CPU."""

from types import SimpleNamespace

import torch
from torch import nn
from torchrl.data import Unbounded

from active_adaptation.learning.wm.adapter import VAICWorldModelAdapter
from active_adaptation.learning.wm.context import RawWorldModelObservation
from active_adaptation.learning.wm.core import ObservationCodec
from active_adaptation.learning.wm.grounded import GroundedWorldModel
from active_adaptation.learning.wm.imagination import VAICImaginationReference
from test_wm_adapter import _fixture, _observation


def _manual_command_update(env):
    """Direct mock of original RobotTracking/RobotObjectTracking update ordering."""
    cmd, data = env.command_manager, env.command_manager.dataset.data
    index = cmd.dataset.starts[cmd.motion_ids, None] + cmd.t[:, None] + cmd.future_steps
    index = torch.minimum(index, cmd.dataset.ends[cmd.motion_ids, None] - 1)
    origin = env.scene.env_origins
    for name in ("pos", "quat", "lin_vel", "ang_vel"):
        values = getattr(data, f"body_{name}_w")[index]
        robot = values[:, :, cmd.tracking_body_indices_motion]
        root = values[:, :, cmd.root_body_idx_motion]
        obj = values[:, :, cmd.object_body_id_motion]
        if name == "pos":
            robot, root, obj = robot + origin[:, None, None], root + origin[:, None], obj + origin[:, None]
        setattr(cmd, f"ref_body_{name}_future_w", robot)
        setattr(cmd, f"ref_body_{name}_w", robot[:, 0])
        setattr(cmd, f"ref_root_{name}_future_w", root)
        setattr(cmd, f"ref_root_{name}_w", root[:, 0])
        setattr(cmd, f"ref_object_{name}_future_w", obj)
        setattr(cmd, f"ref_object_{name}_w", obj[:, 0])
    for name in ("pos", "vel"):
        values = getattr(data, f"joint_{name}")[index]
        setattr(cmd, f"ref_joint_{name}_future_", values[..., cmd.tracking_joint_indices_motion])
        setattr(cmd, f"ref_joint_{name}", values[:, 0, cmd.tracking_joint_indices_motion])
        if name == "pos":
            cmd.current_joint_pos = values[:, 0]
    cmd.t += 1
    contact_index = cmd.dataset.starts[cmd.motion_ids, None] + cmd.t[:, None] + cmd.future_steps
    contact_index = torch.minimum(contact_index, cmd.dataset.ends[cmd.motion_ids, None] - 1)
    for name in ("body", "feet", "object"):
        contact = getattr(cmd, "_" + name + "_contact")[contact_index]
        setattr(cmd, f"ref_{name}_contact_future", contact)
        setattr(cmd, f"ref_{name}_contact", contact[:, 0])


def _manual_local_body(command):
    # Independent yaw rotation expression for the mock's pure-z quaternions.
    root = command.ref_root_pos_w.clone()
    root[:, 2] = 0
    delta = command.ref_body_pos_future_w - root[:, None, None]
    angle = 2 * torch.atan2(command.ref_root_quat_w[:, 3], command.ref_root_quat_w[:, 0])
    c, s = angle.cos()[:, None, None], angle.sin()[:, None, None]
    return torch.stack((c * delta[..., 0] + s * delta[..., 1],
                        -s * delta[..., 0] + c * delta[..., 1], delta[..., 2]), -1).flatten(1)


def _reference_fixture():
    env, _, _, _, _, _ = _fixture(noisy=True, references=True)
    env.max_episode_length = 1000
    command = env.command_manager
    command.future_steps = torch.tensor([1, 2, 4])
    command.tracking_body_indices_motion = [2, 0, 1]
    command.tracking_joint_indices_motion = [4, 1, 3]
    command.root_body_idx_motion, command.object_body_id_motion = 0, 3
    frames = torch.arange(100.)
    body = frames[:, None, None] + torch.arange(4.)[None, :, None] + torch.tensor([.1, .2, .3])
    angle = frames[:, None] * .013 + torch.arange(4.) * .03
    quat = torch.stack(((angle / 2).cos(), torch.zeros_like(angle), torch.zeros_like(angle),
                        (angle / 2).sin()), -1)
    command.dataset = SimpleNamespace(starts=torch.tensor([0, 50]), ends=torch.tensor([50, 100]),
        data=SimpleNamespace(body_pos_w=body, body_quat_w=quat, body_lin_vel_w=body * .01,
                             body_ang_vel_w=body * .02,
                             joint_pos=frames[:, None] * .1 + torch.arange(5.) * .03,
                             joint_vel=frames[:, None] * .01 + torch.arange(5.) * .02))
    for offset, name in enumerate(("body", "feet", "object")):
        setattr(command, "_" + name + "_contact", ((frames[:, None] + torch.arange(2.) + offset) % 3 == 0))
    command.motion_ids, command.motion_len = torch.tensor([0, 1]), torch.tensor([50, 50])
    command.t = torch.tensor([3, 4])
    _manual_command_update(env)
    env.observation_funcs["command"] = SimpleNamespace(funcs={
        "ref_body_pos_future_local": _observation("ref_body_pos_future_local"),
        "ref_joint_pos_future": _observation("ref_joint_pos_future"),
        "ref_motion_phase": _observation("ref_motion_phase"),
    })
    env.observation_funcs["ref_joint_pos_"] = SimpleNamespace(funcs={
        "ref_joint_pos_action_policy": _observation("ref_joint_pos_action_policy",
            action_indices_motion=[1, 3], default_joint_pos=torch.tensor([[.2, .4], [.2, .4]]),
            action_scaling=torch.tensor([.5, 2.])),
    })
    contact_obs = _observation("ref_contact_pos_b", noise_std=.01, episodic_noise_std=.02,
                               ref_contact_pos_b=torch.ones(2, 2, 3), yaw_only=True)
    env.observation_funcs["object_"].funcs["ref_contact_pos_b"] = contact_obs
    env.observation_spec["command"] = Unbounded((2, 37))
    env.observation_spec["ref_joint_pos_"] = Unbounded((2, 2))
    env.observation_spec["object_"] = Unbounded((2, 18))
    # The prior adapter fixture used two future frames.
    env.observation_spec["priv"] = Unbounded((2, 21))
    env._update_callbacks = []
    transform = RawWorldModelObservation(env, list(env.observation_spec.keys()))
    spec = transform.transform_observation_spec(env.observation_spec)["wm_observation"]
    codec = ObservationCodec(spec, list(spec.keys()))
    adapter = VAICWorldModelAdapter(env, transform, codec)
    provider = VAICImaginationReference(env, adapter)

    def snapshot(cached_local=None):
        td = env.observation_spec.zero()
        if cached_local is None:
            cached_local = _manual_local_body(command)
        td["command"] = torch.cat((cached_local, command.ref_joint_pos_future_.flatten(1),
                                    (command.t / command.motion_len)[:, None]), -1)
        td["ref_joint_pos_"] = (command.current_joint_pos[:, [1, 3]] - torch.tensor([.2, .4])) / torch.tensor([.5, 2.])
        return codec.pack(transform(td)["wm_observation"])

    return env, adapter, provider, transform, snapshot


def test_generated_references_match_original_pre_and_post_command_timing():
    env, adapter, provider, transform, snapshot = _reference_fixture()
    state = snapshot()
    generated, truncated = provider.step(state)
    # Same real transition callback order: cached observation and pre fields,
    # then the command update, then direct observations.
    cached_local = _manual_local_body(env.command_manager)
    transform._capture_pre_command()
    env.episode_length_buf += 1
    _manual_command_update(env)
    actual = snapshot(cached_local)
    for name in provider.reference_context_names:
        torch.testing.assert_close(adapter.context(generated, name), adapter.context(actual, name),
                                   atol=1e-5, rtol=1e-6, msg=name)
    for name in provider.reference_observation_groups:
        section = adapter.codec.slices[name]
        torch.testing.assert_close(generated[:, section], actual[:, section], atol=1e-5, rtol=1e-6, msg=name)
    assert not truncated.any()
    # Read-only provider has no persistent checkpoint state/schema additions.
    assert not provider.state_dict()


def test_imagined_reference_continues_past_factual_failure_and_clamps_own_motion_end():
    env, adapter, provider, _, snapshot = _reference_fixture()
    state = snapshot()
    original = state.clone()
    command_time, command_ids = env.command_manager.t.clone(), env.command_manager.motion_ids.clone()
    # Replay's next episode could have unrelated motion identity/time. It is
    # deliberately not an input to any of these 32 independent steps.
    env.command_manager.motion_ids.copy_(env.command_manager.motion_ids.flip(0))
    for step in range(32):
        state, boundary = provider.step(state)
        torch.testing.assert_close(adapter.context(state, "command.t"), command_time.float() + step + 1)
        torch.testing.assert_close(adapter.context(state, "command.motion_ids"), command_ids.float())
        assert not boundary.any()
    torch.testing.assert_close(env.command_manager.t, command_time)
    # Arbitrary changes to live data after construction cannot alter the source.
    env.command_manager.dataset.data.body_pos_w.fill_(-999.)
    one, _ = provider.step(original)
    assert adapter.context(one, "command.ref_body_pos_w").min() > 0
    near_end = adapter.set_context(original, {"command.t": torch.tensor([49., 48.]),
                                               "episode_length": torch.tensor([10., 999.])})
    target, boundary = provider.step(near_end)
    assert boundary.tolist() == [[True], [True]]  # Motion end / episode limit.
    # Future horizons never read the first frame of the following motion.
    torch.testing.assert_close(adapter.context(target, "command.ref_root_pos_w")[:, 0],
                               torch.tensor([49.1, 99.1]))


def test_noise_resampling_does_not_modify_reference_or_frozen_episode_noise():
    _, adapter, provider, _, snapshot = _reference_fixture()
    state = snapshot()
    frozen = "obs_noise.object_.ref_contact_pos_b"
    assert provider.frozen_noise_fields == (frozen,)
    torch.manual_seed(1)
    first, _ = provider.step(state, resample_noise=True)
    second, _ = provider.step(state, resample_noise=True)
    torch.testing.assert_close(adapter.context(first, frozen), adapter.context(state, frozen))
    assert not torch.equal(adapter.context(first, "obs_noise.policy.root_ang_vel_history"),
                           adapter.context(second, "obs_noise.policy.root_ang_vel_history"))
    for name in provider.reference_context_names:
        torch.testing.assert_close(adapter.context(first, name), adapter.context(second, name))


def test_grounded_imagination_keeps_predicted_termination_separate_from_reference_truncation():
    class Physics(nn.Module):
        pass
    class Reference(nn.Module):
        def step(self, state, *, resample_noise=False):
            assert resample_noise
            return state + 10, torch.tensor([[False], [True]])
    class Model(GroundedWorldModel):
        def forward(self, state, action, next_exogenous):
            return (state + action + next_exogenous * 0, action.square(),
                    torch.tensor([[1.], [0.]]), torch.ones(2, 1))
    model = Model(Physics(), nn.Identity(), SimpleNamespace(reward_terms=[]))
    model.set_imagination_reference(Reference())
    state = torch.zeros(2, 1, requires_grad=True)
    action = torch.ones(2, 1, requires_grad=True)
    prediction, reward, terminated, discount, truncated = model.imagine(state, action)
    assert terminated.tolist() == [[1.], [0.]]
    assert truncated.tolist() == [[False], [True]]
    (prediction.sum() + reward.sum()).backward()
    torch.testing.assert_close(action.grad, torch.full_like(action, 3.))
    torch.testing.assert_close(state.grad, torch.ones_like(state))
