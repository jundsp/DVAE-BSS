"""Score a model on a pre-mixed test set with SI-SDR, SIR and SAR.

Writes ``results.json`` (one record per source, read by the scripts in ``experiments/``) and
``summary.json``. A test set is a folder with ``mix.json`` and ``audio/``; see
``PremixedDataset`` or ``experiments/data_prep/make_test_set.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from dvae_bss.checkpoint import load_model
from dvae_bss.cli.common import resolve_device
from dvae_bss.data import PremixedDataset
from dvae_bss.evaluation import evaluate_methods, print_summary
from dvae_bss.separator import Separator


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of ``dvae-evaluate``."""
    p = argparse.ArgumentParser(prog="dvae-evaluate", description=__doc__.split("\n")[0])
    p.add_argument("-m", "--model", type=Path, required=True)
    p.add_argument("--legacy-config", type=Path, help="configuration.txt of an old checkpoint")
    p.add_argument("-d", "--data", type=Path, required=True, help="folder with mix.json, audio/")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("evaluation"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-batches", type=int, help="stop early, for a quick check")
    p.add_argument("--save-audio", action="store_true", help="write the estimated sources")
    p.add_argument("--device", default="auto")
    return p


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
