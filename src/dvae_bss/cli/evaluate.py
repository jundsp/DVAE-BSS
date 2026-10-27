"""``dvae-evaluate``: SI-SDR, SIR and SAR of a model on a pre-mixed test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from ..checkpoint import load_model
from ..data import PremixedDataset
from ..metrics import bss_metrics
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


def _rows(
    batch: dict[str, Any], metrics: dict[str, torch.Tensor], first_index: int
) -> list[dict[str, Any]]:
    """One record per (example, target source), with the example's metadata copied over."""
    n_examples, n_sources = metrics["sdr"].shape
    skip = {"mix", "stems"}
    meta = {k: v for k, v in batch.items() if k not in skip}
    # Rank of each source by SI-SDR within its mixture (0 = worst).
    rank = metrics["sdr"].argsort(-1).argsort(-1)
    rows = []
    for i in range(n_examples):
        example = {k: (v[i].item() if torch.is_tensor(v) else v[i]) for k, v in meta.items()}
        for j in range(n_sources):
            row = {
                "example": first_index + i,
                "source": j,
                "sdr": metrics["sdr"][i, j].item(),
                "sir": metrics["sir"][i, j].item(),
                "sar": metrics["sar"][i, j].item(),
                "place": int(rank[i, j]),
                **example,
            }
            if f"instrument{j + 1}" in example:
                row["instrument"] = example[f"instrument{j + 1}"]
            if f"pitch{j + 1}" in example and "pitch1" in example and "pitch2" in example:
                row["pitch"] = example[f"pitch{j + 1}"]
                row["delta_pitch"] = example["pitch1"] - example["pitch2"]
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-evaluate``."""
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    model, stft = load_model(args.model, args.legacy_config, map_location="cpu")
    separator = Separator(model, stft).to(device).eval()

    dataset = PremixedDataset(args.data, stft.sample_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"evaluating {len(dataset)} mixtures")

    rows: list[dict[str, Any]] = []
    for batch_index, batch in enumerate(loader):
        if args.max_batches is not None and batch_index >= args.max_batches:
            break
        mix, stems = batch["mix"].to(device), batch["stems"].to(device)
        metrics = bss_metrics(separator(mix), stems)
        first = len(rows) // stems.shape[1]
        scores = {k: metrics[k].cpu() for k in ("sdr", "sir", "sar")}
        rows += _rows(batch, scores, first)
        if args.save_audio:
            for i, estimates in enumerate(metrics["estimates"].cpu()):
                target = args.out_dir / f"estimates_{first + i}.wav"
                sf.write(target, estimates.T.numpy(), stft.sample_rate)
        print(f"batch {batch_index + 1}/{len(loader)}")

    (args.out_dir / "results.json").write_text(
        json.dumps({str(i): row for i, row in enumerate(rows)}, indent=2)
    )
    scores = {name: np.array([row[name] for row in rows]) for name in ("sdr", "sir", "sar")}
    summary: dict[str, Any] = {
        name: {"mean": float(v.mean()), "std": float(v.std())} for name, v in scores.items()
    }
    summary["n_sources"] = len(rows)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for name in ("sdr", "sir", "sar"):
        print(f"{name.upper():4s} {summary[name]['mean']:6.2f} dB (std {summary[name]['std']:.2f})")


if __name__ == "__main__":
    main()
