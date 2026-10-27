r"""Spectrogram figures of separated examples: mixture, true sources and estimates.

Runs the model on chosen examples of a pre-mixed test set, matches the estimates to the true
sources and writes one figure per example, plus wav files of the mixture, sources and
estimates for listening.

    python experiments/figures/qualitative.py --model runs/k5/model.pt --data data/twonote \\
        --examples 0 12 --out figures/qualitative
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from dvae_bss import Separator, load_model
from dvae_bss.data import PremixedDataset
from dvae_bss.metrics import optimal_permute
from dvae_bss.plotting import plot_spectrogram, spectrogram_db


def active_span(wave: np.ndarray, sample_rate: int, threshold: float = 1e-4) -> slice:
    """Samples between the first and last point where the signal exceeds ``threshold``."""
    active = np.nonzero(np.abs(wave) > threshold)[0]
    if len(active) == 0:
        return slice(None)
    return slice(max(active[0] - sample_rate // 10, 0), active[-1] + sample_rate // 10)


def draw_example(mix: np.ndarray, targets: np.ndarray, estimates: np.ndarray, rate: int):
    """Figure with the mixture on top, true sources in the middle and estimates below."""
    span = active_span(mix, rate)
    n = len(targets)
    fig, axes = plt.subplots(3, n, figsize=(2.75 * n, 5), sharex=True, sharey=True, squeeze=False)
    top = spectrogram_db(mix[span], rate)[0].max()
    plot_spectrogram(axes[0, 0], mix[span], rate, vmax=top)
    axes[0, 0].set_title("mixture", y=0.97)
    for k in range(1, n):
        axes[0, k].remove()
    for k in range(n):
        plot_spectrogram(axes[1, k], targets[k][span], rate, vmax=top)
        plot_spectrogram(axes[2, k], estimates[k][span], rate, vmax=top)
        axes[1, k].set_title(f"source {k + 1}", y=0.97)
        axes[2, k].set_title(f"estimate {k + 1}", y=0.97)
        axes[2, k].set_xlabel("time (s)")
    for row in range(3):
        axes[row, 0].set_ylabel("frequency (kHz)")
    fig.tight_layout()
    return fig


def main(argv: list[str] | None = None) -> None:
    """Write figures and audio for the chosen examples."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("--legacy-config", type=Path)
    parser.add_argument("-d", "--data", type=Path, required=True, help="folder with mix.json")
    parser.add_argument("--examples", type=int, nargs="+", default=[0], help="indices in the set")
    parser.add_argument("-o", "--out", type=Path, default=Path("figures/qualitative"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    model, stft = load_model(args.model, args.legacy_config)
    separator = Separator(model, stft).to(args.device).eval()
    dataset = PremixedDataset(args.data, stft.sample_rate)
    for index in args.examples:
        item = dataset[index]
        with torch.no_grad():
            estimates = separator(item["mix"].to(args.device).unsqueeze(0)).cpu()
        estimates, _ = optimal_permute(estimates, item["stems"].unsqueeze(0))
        k = item["stems"].shape[0]
        folder = args.out / f"example_{index}"
        folder.mkdir(parents=True, exist_ok=True)
        arrays = dict(mixture=item["mix"], sources=item["stems"], estimates=estimates[0, :k])
        for name, wave in arrays.items():
            sf.write(
                folder / f"{name}.wav", wave.reshape(-1, wave.shape[-1]).T.numpy(), stft.sample_rate
            )
        figure = draw_example(
            item["mix"].numpy(), item["stems"].numpy(), estimates[0, :k].numpy(), stft.sample_rate
        )
        figure.savefig(folder / "spectrograms.pdf", bbox_inches="tight")
        plt.close(figure)
        print(folder)


if __name__ == "__main__":
    main()
