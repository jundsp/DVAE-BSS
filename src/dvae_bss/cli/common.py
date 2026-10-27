"""Helpers shared by the command-line tools."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    """``"auto"`` picks CUDA when available, otherwise the CPU."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)
