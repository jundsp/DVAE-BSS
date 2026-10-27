"""Spectrogram plots (requires the ``plots`` extra, i.e. matplotlib)."""

from __future__ import annotations

import numpy as np
import torch

from .audio import CompressedSTFT, mag2db


def spectrogram_db(
    wave: np.ndarray | torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Single-sided magnitude spectrogram in dB (a sinusoid of amplitude ``A`` peaks at ``A / 2``).

    Returns:
        ``(S, t, f)`` with ``S`` of shape ``(F, T)``, ``t`` in seconds and ``f`` in Hz.
    """
    wave = torch.as_tensor(np.asarray(wave), dtype=torch.float32).reshape(1, -1)
    stft = CompressedSTFT(n_fft, hop_length, compression=1.0, sample_rate=sample_rate)
    left, _, _ = stft.padding(wave.shape[-1])
    magnitude = mag2db(stft.magnitude(wave)[0] / stft.window.sum() * stft.normalizer)
    times = (np.arange(magnitude.shape[-1]) * hop_length + n_fft / 2 - left) / sample_rate
    freqs = np.linspace(0, sample_rate / 2, magnitude.shape[0])
    return magnitude.numpy(), times, freqs


def plot_spectrogram(  # noqa: ANN201
    ax, wave, sample_rate: int = 16000, dynamic_range: float = 80.0, vmax: float | None = None
):
    """Draw a spectrogram on ``ax``; colours span ``dynamic_range`` dB below ``vmax``.

    Args:
        ax: Matplotlib axes.
        wave: Waveform ``(L,)``.
        sample_rate: Sample rate in Hz.
        dynamic_range: Range of the colour scale (dB).
        vmax: Top of the colour scale in dB; default: the spectrogram's own peak. Pass the
            same value to several panels to make them comparable.
    """
    spec, times, freqs = spectrogram_db(wave, sample_rate)
    vmax = spec.max() if vmax is None else vmax
    image = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=(times[0], times[-1], freqs[0] / 1e3, freqs[-1] / 1e3),
        vmin=vmax - dynamic_range,
        vmax=vmax,
        interpolation="none",
    )
    ax.set_xlim(0, times[-1])
    return image


def plot_separation(mixture, sources, sample_rate: int = 16000):  # noqa: ANN001, ANN201
    """Figure with the mixture on top and one row per separated source.

    Args:
        mixture: Waveform ``(L,)``.
        sources: Waveforms ``(K, L)``.
        sample_rate: Sample rate in Hz.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = np.atleast_2d(np.asarray(sources))
    fig, axes = plt.subplots(
        1 + len(sources), 1, figsize=(6, 1.6 * (1 + len(sources))), sharex=True, sharey=True
    )
    plot_spectrogram(axes[0], mixture, sample_rate)
    axes[0].set_ylabel("mixture\nkHz")
    for k, source in enumerate(sources, start=1):
        plot_spectrogram(axes[k], source, sample_rate)
        axes[k].set_ylabel(f"source {k}\nkHz")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    return fig
