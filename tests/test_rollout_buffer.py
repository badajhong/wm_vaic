"""CPU rollout layout must preserve exact trajectories and source ownership."""

import pytest
import torch
from tensordict import TensorDict

from active_adaptation.utils.rollout import allocate_rollout_buffer


@pytest.mark.parametrize("contiguous_steps", [False, True])
@pytest.mark.parametrize("inference_mode", [False, True])
def test_per_step_collection_preserves_values_axes_and_ownership(contiguous_steps, inference_mode):
    n, steps = 5, 4
    with torch.inference_mode(inference_mode):
        source = TensorDict({
            "observation": torch.arange(n * 6).reshape(n, 2, 3).transpose(-1, -2).float(),
            "action": torch.arange(n * 2, dtype=torch.float64).reshape(n, 2),
            "counter": torch.arange(n, dtype=torch.int64),
            "empty": torch.empty(n, 0),
            "next": TensorDict({"done": torch.zeros(n, 1, dtype=torch.bool)}, [n]),
        }, [n])
        buffer = allocate_rollout_buffer(source, steps, device="cpu", contiguous_steps=contiguous_steps)
        for collection in range(2):
            expected = []
            for step in range(steps):
                source["observation"].add_(3)
                source["action"].mul_(-2)
                source["counter"].add_(1)
                source["next", "done"].fill_((collection + step) % 2 == 0)
                expected.append(source.clone())
                buffer[:, step] = source
            source.zero_()
            torch.testing.assert_close(buffer, torch.stack(expected, dim=1), rtol=0, atol=0)
        assert buffer.batch_size == (n, steps) and buffer.device == torch.device("cpu")
        for value in buffer.values(include_nested=True, leaves_only=True):
            assert value[:, 0].is_contiguous() if contiguous_steps else value.is_contiguous()
