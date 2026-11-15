"""Spectrogram plots in the style of the dissertation (requires the ``plots`` extra).

The dissertation's figures used a LaTeX matplotlib style (Computer Modern, 10 pt) and the
``jet`` colour map for spectrograms. :func:`style` reproduces it with the Computer Modern fonts
bundled with matplotlib, so LaTeX is not needed unless ``usetex=True``.
"""

from __future__ import annotations

import io
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from dvae_bss.audio import CompressedSTFT, mag2db

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage

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


def plot_spectrogram(
    ax: Axes,
    wave: np.ndarray | torch.Tensor,
    sample_rate: int = 16000,
    dynamic_range: float = 100.0,
    vmax: float | None = None,
    cmap: str = SPECTROGRAM_CMAP,
) -> AxesImage:
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


def plot_separation(
    mixture: np.ndarray | torch.Tensor,
    sources: np.ndarray | torch.Tensor,
    sample_rate: int = 16000,
) -> Figure:
    """One row: the mixture, then each separated source, as in the dissertation's examples.

    Args:
        mixture: Waveform ``(L,)``.
        sources: Waveforms ``(K, L)``.
        sample_rate: Sample rate in Hz.
    """
    import matplotlib

    plt = _pyplot()
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


def save_png(fig: Figure, path: str | Path, colors: int = 256) -> None:
    """Save a figure as a PNG with a palette of ``colors`` colours instead of full RGBA.

    Spectrograms are noisy and compress poorly; with the 256 colours of the colour map the
    figure looks the same and the file is about half the size.

    Args:
        fig: The figure.
        path: Output file.
        colors: Palette size; 256 is visually lossless for these figures.
    """
    from PIL import Image

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)
    with Image.open(buffer) as image:
        palette = image.convert("RGB").quantize(
            colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE
        )
    palette.save(path, format="PNG", optimize=True)


def _pyplot() -> ModuleType:
    """``matplotlib.pyplot`` with a non-interactive backend, safe on servers without a display."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


LABELS = {
    "val_sdr": "SI-SDR",
    "val_sir": "SIR",
    "gpu_reserved_gb": "reserved",
    "gpu_peak_gb": "tensors",
    "gpu_total_gb": "total",
}


def plot_training_curves(rows: Sequence[Mapping[str, float]], best_epoch: int = 0) -> Figure:
    """Learning curves from the rows of a run's ``log.csv``.

    Panels: the loss (negative ELBO), its two terms (reconstruction ``ell`` and ``kld``), the
    separation scores of the validation mixtures, the learning rate, and, for runs on a GPU,
    the peak memory against the device's total (the gap is the headroom). A falling ``ell``
    with a flat ``kld`` is the usual healthy picture; a validation loss that rises while the
    training loss falls is over-fitting.

    Args:
        rows: One mapping per epoch with ``epoch``, ``train_loss``, ``val_loss``, ... .
        best_epoch: Epoch to mark with a dashed line (the one saved as ``best.pt``).
    """
    import matplotlib

    plt = _pyplot()

    def column(name: str) -> np.ndarray:
        return np.array([row.get(name, math.nan) for row in rows], dtype=float)

    epochs = column("epoch")
    panels = [
        ("loss (negative ELBO)", ["train_loss", "val_loss"]),
        ("reconstruction term (ell)", ["train_ell", "val_ell"]),
        ("KL term (kld)", ["train_kld", "val_kld"]),
        # SAR is left out: it is often far above the others and would flatten them.
        ("validation separation (dB)", ["val_sdr", "val_sir"]),
        ("learning rate", ["lr"]),
    ]
    if np.isfinite(column("gpu_reserved_gb")).any():
        panels.append(("peak GPU memory (GB)", ["gpu_reserved_gb", "gpu_peak_gb", "gpu_total_gb"]))
    with matplotlib.rc_context(style()):
        fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 2.6), sharex=True)
        for ax, (title, names) in zip(axes, panels, strict=True):
            for name in names:
                values = column(name)
                if np.isfinite(values).any():
                    label = LABELS.get(name, name.split("_")[0])
                    if name == "gpu_total_gb":  # the limit, not a measurement
                        total = np.nanmax(values)
                        ax.axhline(total, color="0.3", linestyle=":", label=label)
                        ax.set_ylim(0, 1.05 * total)
                        continue
                    ax.plot(epochs, values, ".-" if len(epochs) < 30 else "-", label=label)
            if best_epoch:
                ax.axvline(best_epoch, color="0.5", linestyle="--", linewidth=0.8)
            ax.set_title(title)
            ax.set_xlabel("epoch")
            if len(names) > 1 and ax.get_legend_handles_labels()[0]:
                ax.legend()
        lr_axis = axes[[title for title, _ in panels].index("learning rate")]
        if np.isfinite(column("lr")).any() and (column("lr") > 0).all():
            lr_axis.set_yscale("log")
        fig.tight_layout()
    return fig


def plot_training_example(
    mixture: np.ndarray | torch.Tensor,
    references: np.ndarray | torch.Tensor,
    estimates: np.ndarray | torch.Tensor,
    raw_estimates: np.ndarray | torch.Tensor | None = None,
    reconstruction: np.ndarray | torch.Tensor | None = None,
    estimates_sum: np.ndarray | torch.Tensor | None = None,
    scores: Sequence[float] | None = None,
    raw_scores: Sequence[float] | None = None,
    sample_rate: int = 16000,
    title: str = "",
) -> Figure:
    """A validation mixture, its true sources, and what the model makes of it.

    Rows, all on one colour scale:

    1. The mixture and the streams it was made from.
    2. The model's raw output (if ``raw_estimates`` is given): the reconstruction, i.e. the
       sum of the decoded sources that the loss compares with the mixture, then each decoded
       source.
    3. The masked outputs, as returned by ``dvae-separate``: their sum (equal to the mixture
       by construction, a check of the mask and the inverse STFT), then each output.

    Outputs are in the same order in rows 2 and 3: first those matched to the true sources,
    then the unmatched ones, which should be near silent.

    Args:
        mixture: Waveform ``(L,)``.
        references: True sources ``(N, L)``.
        estimates: Masked outputs ``(K, L)``, the first ``N`` matched to ``references``.
        raw_estimates: Unmasked decoded sources ``(K, L)`` in the order of ``estimates``.
        reconstruction: The model's mixture estimate ``(L,)``.
        estimates_sum: Sum of the masked outputs ``(L,)``.
        scores: SI-SDR (dB) of each matched masked output, shown in its title.
        raw_scores: SI-SDR (dB) of each matched raw output, shown in its title.
        sample_rate: Sample rate in Hz.
        title: Figure title, e.g. the epoch.
    """
    import matplotlib

    plt = _pyplot()
    references = np.atleast_2d(np.asarray(references))
    estimates = np.atleast_2d(np.asarray(estimates))
    n_ref, n_out = len(references), len(estimates)
    n_cols = 1 + max(n_ref, n_out)
    rows = ["true", "masked"] if raw_estimates is None else ["true", "raw", "masked"]
    with matplotlib.rc_context(style()):
        fig, axes = plt.subplots(
            len(rows),
            n_cols,
            figsize=(1.75 * n_cols, 0.3 + 1.9 * len(rows)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        row = {name: axes[i] for i, name in enumerate(rows)}
        top = spectrogram_db(mixture, sample_rate)[0].max()

        def panel(ax: Axes, wave: np.ndarray | torch.Tensor, label: str) -> None:
            plot_spectrogram(ax, wave, sample_rate, vmax=top)
            ax.set_title(label)

        def output_label(k: int, prefix: str, values: Sequence[float] | None) -> str:
            if k >= n_ref:
                return f"{prefix}{k + 1} (unmatched)"
            dB = f" ({values[k]:.1f} dB)" if values is not None else ""
            return f"{prefix}{k + 1}{dB}"

        panel(row["true"][0], mixture, "mixture")
        for k, reference in enumerate(references):
            panel(row["true"][k + 1], reference, f"true source {k + 1}")
        if raw_estimates is not None:
            if reconstruction is not None:
                panel(row["raw"][0], reconstruction, "reconstruction")
            for k, estimate in enumerate(np.atleast_2d(np.asarray(raw_estimates))):
                panel(row["raw"][k + 1], estimate, output_label(k, "raw ", raw_scores))
        elif reconstruction is not None:
            panel(row["masked"][0], reconstruction, "reconstruction")
        if raw_estimates is not None and estimates_sum is not None:
            panel(row["masked"][0], estimates_sum, "sum of outputs")
        for k, estimate in enumerate(estimates):
            panel(row["masked"][k + 1], estimate, output_label(k, "output ", scores))

        for ax in axes.flat:
            if not ax.has_data():
                ax.set_axis_off()
        names = {"true": "data", "raw": "model output", "masked": "masked (separate)"}
        for name, axs in row.items():
            axs[0].set_ylabel(f"{names[name]}\nfrequency (kHz)")
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
    return fig
