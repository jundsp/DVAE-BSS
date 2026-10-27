r"""Mixture invariant training (MixIT, Wisdom et al. 2020) with the DVAE's networks.

``train``  trains a plain auto-encoder (the DVAE's encoder, temporal network and decoder
           without sampling or prior) on mixtures of two mixtures. The ``K`` decoded
           spectrograms are split into two equal groups; the loss is the smaller squared error
           over all splits when each group is matched to one of the two input mixtures.
``eval``   separates a pre-mixed test set. MixIT outputs have no fixed meaning, so the
           ``K`` estimates are grouped into the reference sources with the split that gives the
           best SI-SDR (an *oracle* grouping, as is usual for MixIT evaluation).

    python experiments/baselines/mixit.py train --train-data data/notes --out runs/mixit \\
        --num-sources 4
    python experiments/baselines/mixit.py eval --model runs/mixit/model.pt --data data/rwc_test
"""

from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dvae_bss import DVAE, CompressedSTFT, load_checkpoint, save_checkpoint
from dvae_bss.cli import train as dvae_train
from dvae_bss.cli.common import resolve_device
from dvae_bss.data import PremixedDataset, mix_batch
from dvae_bss.evaluation import evaluate_methods, print_summary
from dvae_bss.metrics import bss_metrics


def mixit_sources(model: DVAE, x: torch.Tensor) -> torch.Tensor:
    """Decode ``K`` source spectrograms ``(B, K, F, T)`` from the posterior means."""
    sources = model.decode(model.encode(x)[0])
    return sources.view(-1, model.num_sources, *sources.shape[-2:])


def equal_splits(n_sources: int) -> torch.Tensor:
    """All ways to give ``K / 2`` of ``K`` outputs to the first mixture: ``(C(K, K/2), K/2)``."""
    return torch.tensor(list(itertools.combinations(range(n_sources), n_sources // 2)))


def mixit_loss(sources: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """MixIT loss of ``sources`` ``(B, K, F, T)`` for the two input mixtures ``(B, 2, F, T)``."""
    n_sources = sources.shape[1]
    first = equal_splits(n_sources).to(sources.device)
    everything = torch.arange(n_sources, device=sources.device)
    second = torch.stack([everything[~torch.isin(everything, row)] for row in first])
    estimates = torch.stack([sources[:, first].sum(2), sources[:, second].sum(2)], dim=2)
    errors = (estimates - targets.unsqueeze(1)).pow(2).mean(dim=(2, 3, 4))  # (B, splits)
    return errors.min(dim=1).values.mean()


def train(args: argparse.Namespace) -> None:
    """Train the MixIT auto-encoder."""
    if args.num_sources % 2:
        raise SystemExit("--num-sources must be even")
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    stft = CompressedSTFT(args.n_fft, args.hop_length, args.compression, args.sample_rate)
    args.autoencoder = True
    model = dvae_train.build_model(args, stft).to(device)
    train_files, val_files = dvae_train.split_files(args)
    loaders = {
        name: DataLoader(
            dvae_train.make_dataset(files, args, seed=None if name == "train" else args.seed),
            batch_size=min(args.batch_size, len(files)),
            drop_last=True,
            num_workers=args.num_workers,
        )
        for name, files in (("train", train_files), ("val", val_files))
        if len(files) >= max(args.num_targets, 2)  # a mixture needs at least this many sounds
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "log.csv", "w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])
        for epoch in range(1, args.epochs + 1):
            started, losses = time.time(), {}
            for name, loader in loaders.items():
                model.train(name == "train")
                total = 0.0
                with torch.set_grad_enabled(name == "train"):
                    for streams in loader:
                        # Mix two streams into a mixture, then two mixtures into the input.
                        mixture, _ = mix_batch(streams, args.num_targets, random_count=False)
                        x, targets = mix_batch(stft.magnitude(mixture.to(device)), 2, False)
                        loss = mixit_loss(mixit_sources(model, x), targets)
                        if name == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            optimizer.step()
                        total += loss.item()
                losses[name] = total / max(len(loader), 1)
            val = losses.get("val", float("nan"))
            writer.writerow([epoch, losses["train"], val, round(time.time() - started, 1)])
            log.flush()
            print(f"epoch {epoch:3d}  train {losses['train']:.5f}  val {val:.5f}")
            save_checkpoint(
                args.out / "model.pt", model, stft, extra={"epoch": epoch, "method": "mixit"}
            )


@torch.no_grad()
def separate_oracle(
    model: DVAE, stft: CompressedSTFT, batch: dict[str, Any], device
) -> torch.Tensor:
    """Two source estimates per mixture, grouping the ``K`` outputs with the best oracle split."""
    mix, stems = batch["mix"].to(device), batch["stems"].to(device)
    mixture = stft.analysis(mix)
    magnitude = mixture.abs()
    sources = mixit_sources(model, magnitude)
    sources = sources * (magnitude / sources.sum(1).clamp_min(1e-8)).unsqueeze(1)  # ratio mask
    spectra = torch.polar(sources, mixture.angle().unsqueeze(1).expand_as(sources))
    n_examples, n_sources, n_freq, n_frames = spectra.shape
    waves = stft.synthesis(spectra.reshape(-1, n_freq, n_frames), mix.shape[-1])
    waves = waves.view(n_examples, n_sources, -1)

    everything = torch.arange(n_sources, device=device)
    best_score = torch.full((n_examples,), -float("inf"), device=device)
    best = torch.zeros(n_examples, 2, mix.shape[-1], device=device)
    for first in equal_splits(n_sources).to(device):
        second = everything[~torch.isin(everything, first)]
        grouped = torch.stack([waves[:, first].sum(1), waves[:, second].sum(1)], dim=1)
        score = bss_metrics(grouped, stems)["sdr"].mean(-1)
        better = score > best_score
        best_score = torch.where(better, score, best_score)
        best[better] = grouped[better]
    return best


def evaluate(args: argparse.Namespace) -> None:
    """Evaluate a trained MixIT model with oracle grouping."""
    device = resolve_device(args.device)
    model, stft = load_checkpoint(args.model)
    model.to(device).eval()
    dataset = PremixedDataset(args.data, stft.sample_rate)
    summaries = evaluate_methods(
        {"": lambda batch: separate_oracle(model, stft, batch, device)},
        dataset,
        args.out_dir,
        args.batch_size,
        args.max_batches,
    )
    print_summary(summaries[""])


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point with ``train`` and ``eval`` subcommands."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    # Training takes the same data and model options as dvae-train (the prior is not used).
    train_parser = commands.add_parser("train", parents=[dvae_train.build_parser(add_help=False)])
    train_parser.set_defaults(num_sources=4, run=train)
    eval_parser = commands.add_parser("eval", help="evaluate with oracle grouping")
    eval_parser.add_argument("-m", "--model", type=Path, required=True)
    eval_parser.add_argument("-d", "--data", type=Path, required=True)
    eval_parser.add_argument("-o", "--out-dir", type=Path, default=Path("results/mixit"))
    eval_parser.add_argument("--batch-size", type=int, default=8)
    eval_parser.add_argument("--max-batches", type=int)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.set_defaults(run=evaluate)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
