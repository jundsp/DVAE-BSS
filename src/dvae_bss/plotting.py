"""Spectrogram plots in the style of the dissertation (requires the ``plots`` extra).

The dissertation's figures used a LaTeX matplotlib style (Computer Modern, 10 pt) and the
``jet`` colour map for spectrograms. :func:`style` reproduces it with the Computer Modern fonts
bundled with matplotlib, so LaTeX is not needed unless ``usetex=True``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dvae_bss.audio import CompressedSTFT, mag2db

SPECTROGRAM_CMAP = "jet"


def style(usetex: bool = False) -> dict[str, Any]:
    """Matplotlib settings of the dissertation figures.

    Use with ``plt.rcParams.update(style())`` or ``matplotlib.rc_context(style())``.

    Args:
        usetex: Typeset text with LaTeX, as the dissertation did (needs a LaTeX install).
    """
    params: dict[str, Any] = {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "lines.linewidth": 1,
        "legend.fontsize": 10,
        "legend.frameon": False,
        "image.interpolation": "none",
        "image.aspect": "auto",
        "image.origin": "lower",
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
    }
    if usetex:
        params["text.usetex"] = True
    else:
        # cmr10 has no Unicode minus sign, so tick labels go through mathtext.
        params |= {
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": False,
        }
    return params


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
    ax,
    wave,
    sample_rate: int = 16000,
    dynamic_range: float = 100.0,
    vmax: float | None = None,
    cmap: str = SPECTROGRAM_CMAP,
):
    """Draw a spectrogram on ``ax``; colours span ``dynamic_range`` dB below ``vmax``.

    Args:
        ax: Matplotlib axes.
        wave: Waveform ``(L,)``.
        sample_rate: Sample rate in Hz.
        dynamic_range: Range of the colour scale (dB).
        vmax: Top of the colour scale in dB; default: the spectrogram's own peak. Pass the
            same value to several panels to make them comparable.
        cmap: Matplotlib colour map.
    """
    spec, times, freqs = spectrogram_db(wave, sample_rate)
    vmax = spec.max() if vmax is None else vmax
    image = ax.imshow(
        spec,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=(times[0], times[-1], freqs[0] / 1e3, freqs[-1] / 1e3),
        vmin=vmax - dynamic_range,
        vmax=vmax,
        interpolation="none",
    )
    ax.set_xlim(0, times[-1])
    return image


def plot_separation(mixture, sources, sample_rate: int = 16000):  # noqa: ANN001, ANN201
    """One row: the mixture, then each separated source, as in the dissertation's examples.

    Args:
        mixture: Waveform ``(L,)``.
        sources: Waveforms ``(K, L)``.
        sample_rate: Sample rate in Hz.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = np.atleast_2d(np.asarray(sources))
    with matplotlib.rc_context(style()):
        fig, axes = plt.subplots(
            1, 1 + len(sources), figsize=(1.75 * (1 + len(sources)), 2.1), sharex=True, sharey=True
        )
        # One colour scale for all panels, so that near-silent outputs look silent.
        top = spectrogram_db(mixture, sample_rate)[0].max()
        plot_spectrogram(axes[0], mixture, sample_rate, vmax=top)
        axes[0].set_title("mixture")
        axes[0].set_ylabel("frequency (kHz)")
        for k, source in enumerate(sources, start=1):
            plot_spectrogram(axes[k], source, sample_rate, vmax=top)
            axes[k].set_title(f"source {k}")
        for ax in axes:
            ax.set_xlabel("time (s)")
        fig.tight_layout()
    return fig
