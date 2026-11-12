"""Helpers shared by the command-line tools."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA, then Apple's MPS, then the CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    return torch.device(name)
