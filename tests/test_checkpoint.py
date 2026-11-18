"""Atomic checkpoint replacement can be verified without importing Isaac Sim."""

from pathlib import Path

import pytest
import torch

from active_adaptation.utils.checkpoint import atomic_torch_save


def test_latest_checkpoint_stays_readable_until_successful_replacement(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint_latest.pt"
    torch.save({"step": 3}, path)
    original_save = torch.save
    def save_and_inspect(state, stream):
        assert torch.load(path, weights_only=True) == {"step": 3}
        assert tmp_path == Path(stream.name).parent
        original_save(state, stream)
        assert torch.load(path, weights_only=True) == {"step": 3}
    monkeypatch.setattr(torch, "save", save_and_inspect)
    expected = {"step": 4, "moment": torch.arange(5.)}
    atomic_torch_save(expected, path)
    actual = torch.load(path, weights_only=True)
    assert actual["step"] == 4
    torch.testing.assert_close(actual["moment"], expected["moment"])
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("failure", ["serialize", "flush", "replace"])
def test_failed_latest_save_preserves_previous_checkpoint_and_cleans_temp(tmp_path, monkeypatch, failure):
    import active_adaptation.utils.checkpoint as checkpoint
    path = tmp_path / "checkpoint_latest.pt"
    torch.save({"step": 3}, path)
    before = path.read_bytes()
    if failure == "serialize":
        def interrupted_save(_state, stream):
            stream.write(b"partial checkpoint")
            raise RuntimeError("injected serialization failure")
        monkeypatch.setattr(torch, "save", interrupted_save)
        expected_error = RuntimeError
    else:
        def interrupted(*_args):
            raise OSError("injected filesystem failure")
        monkeypatch.setattr(checkpoint.os, "fsync" if failure == "flush" else "replace", interrupted)
        expected_error = OSError
    with pytest.raises(expected_error, match="injected"):
        atomic_torch_save({"step": 4}, path)
    assert path.read_bytes() == before
    assert torch.load(path, weights_only=True) == {"step": 3}
    assert list(tmp_path.iterdir()) == [path]
