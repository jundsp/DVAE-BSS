"""``dvae-evaluate``: SI-SDR, SIR and SAR of a model on a pre-mixed test set."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from ..checkpoint import load_model
from ..data import PremixedDataset
from ..evaluation import evaluate_methods, print_summary
from ..separator import Separator
from .common import resolve_device


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of ``dvae-evaluate``."""
    parser = argparse.ArgumentParser(
        prog="dvae-evaluate",
        description="Evaluate a model on a folder with mix.json and audio/ (see PremixedDataset). "
        "Writes results.json (one record per source) and summary.json.",
    )
    parser.add_argument("-m", "--model", type=Path, required=True)
    parser.add_argument("--legacy-config", type=Path, help="configuration.txt of an old checkpoint")
    parser.add_argument("-d", "--data", type=Path, required=True, help="test-set folder")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("evaluation"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, help="stop after this many batches")
    parser.add_argument("--save-audio", action="store_true", help="write the estimated sources")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-evaluate``."""
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, stft = load_model(args.model, args.legacy_config, map_location="cpu")
    separator = Separator(model, stft).to(device).eval()

    dataset = PremixedDataset(args.data, stft.sample_rate)
    print(f"evaluating {len(dataset)} mixtures")

    def save_audio(_: str, first: int, estimates: torch.Tensor) -> None:
        for i, example in enumerate(estimates):
            sf.write(
                args.out_dir / f"estimates_{first + i}.wav", example.T.numpy(), stft.sample_rate
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries = evaluate_methods(
        {"": lambda batch: separator(batch["mix"].to(device))},
        dataset,
        args.out_dir,
        args.batch_size,
        args.max_batches,
        on_batch=save_audio if args.save_audio else None,
    )
    print_summary(summaries[""])


if __name__ == "__main__":
    main()
