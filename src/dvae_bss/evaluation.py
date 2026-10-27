"""Result records shared by ``dvae-evaluate`` and the baselines.

Every method writes the same ``results.json`` (one record per example and source) so that the
scripts in ``experiments/figures`` can compare any of them.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import bss_metrics

METRICS = ("sdr", "sir", "sar")


def result_rows(
    metadata: Mapping[str, Any], scores: Mapping[str, torch.Tensor], first_example: int = 0
) -> list[dict[str, Any]]:
    """One record per (example, reference source) of a batch.

    Args:
        metadata: Per-example metadata as collated by a ``DataLoader`` (tensors or lists of
            length ``B``); every entry is copied to the records of its example.
        scores: ``sdr``, ``sir`` and ``sar`` tensors of shape ``(B, K)`` in dB.
        first_example: Index of the batch's first example in the whole set.

    Returns:
        Records with ``example``, ``source``, the scores and ``place`` (rank of the source by
        SI-SDR within its mixture, 0 = worst). When the metadata has ``instrument1``,
        ``instrument2``, ... or ``pitch1``, ``pitch2`` the record also holds the instrument
        and pitch of its own source and the pitch difference of the pair.
    """
    n_examples, n_sources = scores["sdr"].shape
    rank = scores["sdr"].argsort(-1).argsort(-1)
    rows = []
    for i in range(n_examples):
        example = {
            key: (value[i].item() if torch.is_tensor(value) else value[i])
            for key, value in metadata.items()
        }
        for j in range(n_sources):
            row = {
                "example": first_example + i,
                "source": j,
                **{name: scores[name][i, j].item() for name in METRICS},
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


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and standard deviation of every metric over all records."""
    summary: dict[str, Any] = {}
    for name in METRICS:
        values = np.array([row[name] for row in rows])
        summary[name] = {"mean": float(values.mean()), "std": float(values.std())}
    summary["n_sources"] = len(rows)
    return summary


def write_results(out_dir: str | Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Write ``results.json`` (pandas ``orient="index"`` layout) and ``summary.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps({str(i): r for i, r in enumerate(rows)}, indent=2)
    )
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_summary(summary: Mapping[str, Any]) -> None:
    """Print mean (and standard deviation) of each metric."""
    for name in METRICS:
        print(f"{name.upper():4s} {summary[name]['mean']:6.2f} dB (std {summary[name]['std']:.2f})")


Estimator = Callable[[dict[str, Any]], torch.Tensor]
BatchCallback = Callable[[str, int, torch.Tensor], None]


def evaluate_methods(
    estimators: Mapping[str, Estimator],
    dataset: Dataset,
    out_dir: str | Path,
    batch_size: int = 8,
    max_batches: int | None = None,
    on_batch: BatchCallback | None = None,
) -> dict[str, dict[str, Any]]:
    """Score one or more separation methods on a pre-mixed dataset.

    Args:
        estimators: Method name -> function taking a collated batch (``mix`` ``(B, L)``,
            ``stems`` ``(B, K, L)``, metadata) and returning estimated sources ``(B, K_est, L)``.
            A method named ``""`` writes straight into ``out_dir``.
        dataset: Items must be dictionaries as produced by
            :class:`~dvae_bss.data.PremixedDataset`.
        out_dir: Results go to ``out_dir / name`` (see :func:`write_results`).
        batch_size: Examples per batch.
        max_batches: Stop after this many batches (default: the whole set).
        on_batch: Called as ``on_batch(name, first_example, matched_estimates)`` for each batch.

    Returns:
        The summary of every method.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in estimators}
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        metadata = {k: v for k, v in batch.items() if k not in ("mix", "stems")}
        for name, estimate in estimators.items():
            estimates = estimate(batch)
            stems = batch["stems"].to(estimates.device)
            metrics = bss_metrics(estimates, stems)
            first = len(rows[name]) // stems.shape[1]
            scores = {k: metrics[k].cpu() for k in METRICS}
            rows[name] += result_rows(metadata, scores, first)
            if on_batch is not None:
                on_batch(name, first, metrics["estimates"].cpu())
        print(f"batch {index + 1}/{len(loader)}", flush=True)
    return {
        name: write_results(Path(out_dir) / name, method_rows) for name, method_rows in rows.items()
    }
