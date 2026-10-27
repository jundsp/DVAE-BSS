"""Saving and loading models, including checkpoints written by the original code.

New checkpoints are a single ``.pt`` file holding the model configuration, the audio
front-end settings and the weights, so a model can be rebuilt without any other file.
Checkpoints from the dissertation code (a bare state dict plus a ``configuration.txt``)
can be loaded with :func:`load_legacy`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .audio import CompressedSTFT
from .model import DVAE

FORMAT_VERSION = 1


def save_checkpoint(
    path: str | Path, model: DVAE, stft: CompressedSTFT, extra: dict[str, Any] | None = None
) -> None:
    """Write ``model`` and its front-end settings to ``path``."""
    torch.save(
        {
            "format": FORMAT_VERSION,
            "model": model.config,
            "audio": audio_config(stft),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def audio_config(stft: CompressedSTFT) -> dict[str, Any]:
    """Front-end settings needed to reproduce the model's input."""
    return dict(n_fft=stft.n_fft, hop_length=stft.hop_length, compression=stft.compression)


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[DVAE, CompressedSTFT]:
    """Rebuild the model and front end saved by :func:`save_checkpoint`."""
    blob = torch.load(path, map_location=map_location, weights_only=True)
    if blob.get("format") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a dvae-bss checkpoint (use load_legacy for old ones)")
    model = DVAE(**blob["model"])
    model.load_state_dict(blob["state_dict"])
    return model, CompressedSTFT(**blob["audio"])


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop ``module.`` prefixes (DataParallel) and parameters the model no longer has."""
    cleaned = {}
    for key, value in state_dict.items():
        key = key.removeprefix("module.")
        if key != "likelihood.variance":
            cleaned[key] = value
    return cleaned


def config_from_legacy_args(args: dict[str, Any], n_freq: int = 257) -> dict[str, Any]:
    """Translate the ``configuration.txt`` written by the original ``train.py``.

    Args:
        args: Parsed JSON of ``configuration.txt``.
        n_freq: Number of frequency bins; the original code always used 257 (512-point FFT).

    Raises:
        ValueError: For model variants that were removed (pooled, complex-valued).
    """
    if not args.get("conv2d", True) or args.get("complex_version", False):
        raise ValueError("only the convolutional magnitude models (DCAE, DSVAE) are supported")
    dilated = bool(args.get("separator_version", False))
    config = dict(
        n_freq=n_freq,
        latent_dim=args["latent_dim"],
        rnn_dim=args["rnn_dim"],
        num_sources=args["num_sources"],
        encoder="dilated" if dilated else "strided",
        layer_channels=tuple(args["cnn_channels"]),
        kernel_size=(3, 3),
        stride=(2, 1) if dilated else (2, 2),
        padding=(1, 1),
        rnn_type=args.get("rnn_type", "lstm"),
        rnn_layers=args.get("num_layers", 1),
        inference_rnn=not args.get("inference_rnn_off", False),
        variance=args.get("variance", 0.1),
        activate_final_layer=args.get("activate_final_layer", False),
        autoencoder=args.get("autoencoder", False),
    )
    if dilated:
        config.update(
            separator_kernel_size=args["separator_kernel_size"],
            separator_layers=args["separator_layers"],
            separator_channels=args["separator_channels"],
        )
    return config


def load_legacy(
    weights: str | Path,
    configuration: str | Path | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[DVAE, CompressedSTFT]:
    """Load a checkpoint written by the dissertation code.

    Args:
        weights: The ``.pt`` state dict.
        configuration: The ``configuration.txt`` saved next to it; defaults to the file of
            that name in the same folder.
        map_location: Device to load the tensors on.
    """
    weights = Path(weights)
    configuration = Path(configuration) if configuration else weights.parent / "configuration.txt"
    args = json.loads(configuration.read_text())
    model = DVAE(**config_from_legacy_args(args))
    state_dict = torch.load(weights, map_location=map_location, weights_only=True)
    model.load_state_dict(clean_state_dict(state_dict))
    stft = CompressedSTFT(n_fft=512, hop_length=128, compression=args.get("compression", 1.0))
    return model, stft
