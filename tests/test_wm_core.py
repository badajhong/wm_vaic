"""CPU tests that do not import Isaac Sim or the training entrypoint."""

import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from torchrl.data import Composite, Unbounded


_SPEC = importlib.util.spec_from_file_location(
    "wm_core", Path(__file__).parents[1] / "active_adaptation/learning/wm/core.py"
)
wm_core = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(wm_core)
ObservationCodec = wm_core.ObservationCodec
RunningMoments = wm_core.RunningMoments


def make_codec():
    spec = Composite({
        "policy": Unbounded((2,)),
        "object_": Unbounded((2,)),
        "object_geo_": Unbounded((3,)),
        "command": Unbounded((2, 2)),
    })
    return ObservationCodec(spec, list(spec.keys()))


@pytest.mark.parametrize("batch_shape", [(), (3,), (3, 4)])
def test_codec_roundtrip_and_gradient_for_arbitrary_batches(batch_shape):
    codec = make_codec()
    values = {
        key: torch.randn(batch_shape + shape, requires_grad=True)
        for key, shape in codec.shapes.items()
    }
    td = TensorDict(values, batch_size=batch_shape)
    packed = codec.pack(td)
    recovered = codec.unpack(packed)
    assert packed.shape == batch_shape + (11,)
    for key, value in values.items():
        torch.testing.assert_close(value, recovered[key])
    recovered["object_"].sum().backward()
    torch.testing.assert_close(values["object_"].grad, torch.ones_like(values["object_"]))


def test_codec_removes_spec_batch_dimensions_and_retains_matrix_events():
    spec = Composite({"policy": Unbounded((8, 2, 3))}, shape=(8,))
    codec = ObservationCodec(spec, ["policy"])
    td = TensorDict({"policy": torch.randn(5, 4, 2, 3)}, batch_size=[5, 4])
    assert codec.pack(td).shape == (5, 4, 6)
    torch.testing.assert_close(codec.unpack(codec.pack(td))["policy"], td["policy"])


def test_moments_merge_without_reset_and_persist_in_checkpoint():
    stats = RunningMoments(3)
    first = torch.tensor([[1., 4., -2.], [3., 4., 0.]])
    second = torch.tensor([[5., 4., 2.], [7., 4., 4.], [9., 4., 6.]])
    stats.update(first)
    stats.update(second)
    all_samples = torch.cat((first, second))
    torch.testing.assert_close(stats.mean, all_samples.mean(0))
    torch.testing.assert_close(stats.var, all_samples.var(0, correction=0))
    assert stats.count == 5
    assert torch.isfinite(stats.normalize(all_samples)).all()
    restored = RunningMoments(3)
    restored.load_state_dict(stats.state_dict())
    torch.testing.assert_close(restored.normalize(all_samples), stats.normalize(all_samples))
    stats.update(torch.full((1, 3), float("nan")))
    assert stats.count == 5
