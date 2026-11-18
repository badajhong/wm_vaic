"""Checkpoint replacement that preserves the previous file on save failure."""

import os
from pathlib import Path
import tempfile

import torch


def atomic_torch_save(state, destination):
    """Serialize beside the destination, then atomically replace it.

    The destination remains untouched until serialization and file flushing
    succeed. A failed write or replacement removes the temporary file.
    """
    destination = Path(destination)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
                mode="w+b", dir=destination.parent,
                prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary_path = Path(stream.name)
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
