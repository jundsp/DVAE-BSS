"""Oracle baselines: the unprocessed mixture and ideal ratio / binary masks.

The ideal masks use the true source spectrograms, so they bound what any mask-based method
can reach. All three are scored like a model with ``dvae-evaluate`` and written to
``<out-dir>/{mix,irm,ibm}/``.

    python experiments/baselines/ideal_masks.py --data data/rwc_test --out-dir results/ideal
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from dvae_bss.audio import CompressedSTFT
from dvae_bss.data import PremixedDataset
from dvae_bss.evaluation import evaluate_methods, print_summary
from dvae_bss.metrics import ideal_masks


def build_estimators(stft: CompressedSTFT) -> dict[str, Any]:
    """Estimators ``batch -> (B, K, L)`` for the mixture, IRM and IBM baselines."""

    def masked(batch: dict[str, Any], binary: bool) -> torch.Tensor:
        stems = batch["stems"]
        n_examples, n_sources, n_samples = stems.shape
        spectra = stft.analysis(stems.reshape(-1, n_samples))
        spectra = spectra.reshape(n_examples, n_sources, *spectra.shape[-2:])
        ratio, hard = ideal_masks(spectra)
        chosen = hard if binary else ratio
        return stft.synthesis(chosen.reshape(-1, *chosen.shape[-2:]), n_samples).view(stems.shape)

    return {
        "mix": lambda batch: batch["mix"].unsqueeze(1).repeat(1, batch["stems"].shape[1], 1),
        "irm": lambda batch: masked(batch, binary=False),
        "ibm": lambda batch: masked(batch, binary=True),
    }


def main(argv: list[str] | None = None) -> None:
    """Evaluate the mixture, IRM and IBM baselines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--data", type=Path, required=True, help="folder with mix.json")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("results/ideal"))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)

    # Linear magnitudes (no compression): the masks are defined on the plain STFT.
    stft = CompressedSTFT(args.n_fft, args.hop_length, 1.0, args.sample_rate)
    dataset = PremixedDataset(args.data, args.sample_rate)
    summaries = evaluate_methods(
        build_estimators(stft), dataset, args.out_dir, args.batch_size, args.max_batches
    )
    for name, summary in summaries.items():
        print(f"\n{name}")
        print_summary(summary)


if __name__ == "__main__":
    main()
