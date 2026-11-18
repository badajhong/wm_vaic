"""Return-based RSI: SuperMimic ReturnSampler port and motion-command wiring.

The sampler tests mirror SuperMimic's tests/test_return_reset_sampler.py
(Apache-2.0), including its golden stream, so the port stays bit-faithful.
"""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from active_adaptation.utils.return_sampler import ReturnSampler, ReturnSamplerCfg

COMMAND = Path(__file__).resolve().parents[1] / "active_adaptation/envs/mdp/commands/hdmi/command.py"


def _sampler(num_envs=1, bin_count=4, frame_count=40, **cfg):
    return ReturnSampler(ReturnSamplerCfg(**cfg), bin_count, frame_count, num_envs, max_step_reward=1.0)


def _step(s, frame, reward, terminated=False, clip_end=False, timed_out=False):
    n = s._frames.shape[0]
    return s.record(torch.full((n,), frame), torch.full((n,), float(reward)), torch.full((n,), terminated),
                    torch.full((n,), clip_end), torch.full((n,), timed_out))


def test_unmeasured_bins_read_v_max_and_sampling_starts_uniform():
    s = _sampler(horizon=1)
    assert torch.equal(s.value, torch.full((4,), 100.0))
    assert torch.allclose(_step(s, 25, 1.0), torch.full((4,), 0.25))


def test_a_fall_stops_and_clip_end_bootstraps_from_mean_value():
    s = _sampler(horizon=1)
    _step(s, 5, 0.3, terminated=True)
    assert float(s.value[0]) == pytest.approx(0.3, abs=1e-5)
    s = _sampler(horizon=1)
    s.value[:] = torch.tensor([20.0, 40.0, 60.0, 80.0])
    _step(s, 39, 0.3, clip_end=True)
    assert float(s.value[3]) == pytest.approx(0.3 + 0.99 * 50.0, abs=1e-4)


def test_returns_discount_back_from_a_fall():
    s = _sampler(horizon=3)
    _step(s, 0, 1.0)
    _step(s, 1, 1.0)
    _step(s, 2, 1.0, terminated=True)
    expected = (1 + 0.99 * (1 + 0.99) + (1 + 0.99) + 1) / 3
    assert float(s.value[0]) == pytest.approx(expected, abs=1e-5)


def test_value_memory_and_focus_on_low_return():
    s = _sampler(horizon=1, memory=4)
    for reward in (1.0, 2.0, 3.0, 4.0):
        _step(s, 5, reward, terminated=True)
    assert float(s.value[0]) == pytest.approx(2.5, abs=1e-6)
    s = _sampler(horizon=1, explore=0.0, keep=0.0)
    s.value[1] = 70.0
    w = _step(s, 25, 1.0)
    assert float(w[1] / w[0]) == pytest.approx(math.exp(25 * 0.3), rel=1e-3)


def test_golden_stream_matches_supermimic():
    s = _sampler(num_envs=4, bin_count=16, frame_count=160)
    gen = torch.Generator().manual_seed(99)
    weights = None
    for _ in range(60):
        frames = torch.randint(0, 160, (4,), generator=gen)
        rewards = torch.rand(4, generator=gen)
        terminated = torch.rand(4, generator=gen) < 0.15
        clip_end = (torch.rand(4, generator=gen) < 0.05) & ~terminated
        timed_out = (torch.rand(4, generator=gen) < 0.05) & ~terminated & ~clip_end
        out = s.record(frames, rewards, terminated, clip_end, timed_out)
        if out is not None:
            weights = out
    golden = [0.03130187, 0.03134733, 0.03201599, 0.03608831, 0.04007206, 0.31132525, 0.03760373,
              0.03675647, 0.07033682, 0.0331141, 0.03324675, 0.05272575, 0.05815605, 0.03189738,
              0.0314665, 0.13254566]
    assert torch.allclose(weights, torch.tensor(golden), atol=1e-6)


def test_sampled_frames_follow_bins_and_stay_on_the_timeline():
    s = _sampler(bin_count=4, frame_count=42)
    probs = torch.tensor([0., 0., 1., 0.])
    frames = s.sample_frames(probs, 1000, generator=torch.Generator().manual_seed(0))
    assert ((frames >= 21) & (frames < 31)).all()
    frames = s.sample_frames(torch.full((4,), .25), 4000, generator=torch.Generator().manual_seed(1))
    assert frames.min() >= 0 and frames.max() <= 41 and len(frames.unique()) == 42


def test_config_rejects_unknown_options_and_unbounded_reward():
    with pytest.raises(ValueError, match="Unknown"):
        ReturnSamplerCfg.from_dict({"stepness": 3.})
    assert ReturnSamplerCfg.from_dict({"explore": .3}).explore == .3
    with pytest.raises(ValueError, match="max_step_reward"):
        ReturnSampler(ReturnSamplerCfg(), 4, 40, 1, max_step_reward=math.inf)


def _command_methods(*names):
    tree = ast.parse(COMMAND.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == "RobotTracking")
    functions = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
    namespace = {"torch": torch, "Dict": dict}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(COMMAND), "exec"), namespace)
    return {name: namespace[name] for name in names}


class _Env(SimpleNamespace):
    pass


def _command(reset_sampling="return", lengths=(30, 50), num_envs=3):
    methods = _command_methods("_setup_return_sampling", "_global_frames", "record_step",
                               "_sample_return_starts", "_sample_motions")
    cls = type("Command", (), {**methods, "finished": property(lambda self: (self.t >= self.motion_len).unsqueeze(1))})
    command = cls()
    lengths = torch.tensor(lengths)
    ends = torch.cumsum(lengths, 0)
    command.dataset = SimpleNamespace(lengths=lengths, starts=ends - lengths, ends=ends,
                                      num_motions=len(lengths))
    command.env = _Env(step_dt=.02, training=True, mult_dt=True,
                       reward_groups={"loco": SimpleNamespace(funcs={"s": type("survival", (), {"weight": 1., "enabled": True})()},
                                                               multiplicative=False)})
    command.device, command.num_envs = "cpu", num_envs
    command.reset_sampling = reset_sampling
    command.replay_motion = command.record_motion = False
    command.future_steps = torch.tensor([1, 2, 8])
    command.motion_ids = torch.tensor([0, 1, 1])
    command.motion_len = lengths[command.motion_ids]
    command.motion_starts = command.dataset.starts[command.motion_ids]
    command.motion_ends = command.dataset.ends[command.motion_ids]
    command.t = torch.tensor([5, 12, 50])
    command._setup_return_sampling({"horizon": 1})
    return command


def test_command_timeline_and_recorded_transition_frames():
    command = _command()
    assert command._timeline_frames == 80 and command._timeline_bins == 80 // 10 + 1
    # Motion 1 starts at global frame 30; frames clamp to each clip's last frame.
    assert command._global_frames(command.motion_ids, command.t - 1).tolist() == [4, 41, 79]
    reward = torch.full((3, 2), .01)
    command.record_step(reward, torch.tensor([[False], [True], [False]]), torch.tensor([[False], [False], [True]]))
    sampler = command._return_sampler
    # Survival weight 1 with dt .02 sets v_max = .02 / (1 - .99).
    assert sampler.v_max == pytest.approx(2.)
    assert command._reset_probs is not None and float(command._reset_probs.sum()) == pytest.approx(1.)
    # The fall (env 1, frame 41) is the only terminated sample; its bin value is its reward.
    assert float(sampler.value[sampler.bins(torch.tensor(41))]) == pytest.approx(.02)


def test_return_mode_draws_starts_from_the_distribution_inside_valid_clip_ranges():
    command = _command()
    probs = torch.zeros(command._timeline_bins)
    probs[4] = 1.  # 9 bins over 80 frames: bin 4 is global 35..43, motion 1 local 5..13
    command._reset_probs = probs
    command._return_sampler = ReturnSampler(ReturnSamplerCfg(), command._timeline_bins,
                                            command._timeline_frames, 3, 1.)
    env_ids = torch.tensor([0, 2])
    command._sample_motions(env_ids)
    assert command.motion_ids[env_ids].tolist() == [1, 1]
    assert ((command.t[env_ids] >= 5) & (command.t[env_ids] <= 13)).all()
    assert command.motion_len[env_ids].tolist() == [50, 50]
    assert command.motion_starts[env_ids].tolist() == [30, 30]
    # Starts never leave fewer than the last future step of reference motion.
    probs.zero_()
    probs[-1] = 1.
    command._sample_motions(env_ids)
    assert (command.t[env_ids] <= 50 - 8 - 1).all()


def test_uniform_mode_and_evaluation_never_record():
    command = _command(reset_sampling="uniform")
    command.record_step(torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.bool), torch.zeros(3, 1, dtype=torch.bool))
    assert command._return_sampler is None
    command = _command()
    command.env.training = False
    command.record_step(torch.ones(3, 1), torch.zeros(3, 1, dtype=torch.bool), torch.zeros(3, 1, dtype=torch.bool))
    assert command._return_sampler is None
