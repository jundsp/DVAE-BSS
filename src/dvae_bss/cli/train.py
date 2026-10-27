"""``dvae-train``: train a DVAE on mixtures generated on the fly from isolated sounds.

Single process (CPU or one GPU) by default. For several GPUs launch with ``torchrun``::

    torchrun --nproc_per_node=4 -m dvae_bss.cli.train --train-data data/notes --out runs/k5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ..audio import CompressedSTFT
from ..checkpoint import read_checkpoint, save_checkpoint
from ..data import NoteSequenceDataset, collect_files, mix_batch
from ..losses import ELBOLoss
from ..model import DVAE
from .common import resolve_device

LOG_FIELDS = [
    "epoch",
    "train_loss",
    "train_ell",
    "train_kld",
    "val_loss",
    "val_ell",
    "val_kld",
    "seconds",
]


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Command-line options of ``dvae-train`` (``add_help=False`` to reuse them as a parent)."""
    parser = argparse.ArgumentParser(
        prog="dvae-train",
        add_help=add_help,
        description="Train a DVAE for unsupervised blind source separation. The model only "
        "sees mixtures of randomly drawn sounds; no source labels are used.",
    )
    data = parser.add_argument_group("data")
    data.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="folder of isolated sounds (searched recursively) or a text file listing them",
    )
    data.add_argument(
        "--val-data",
        type=Path,
        help="validation sounds; default: hold out --val-fraction of the training files",
    )
    data.add_argument("--val-fraction", type=float, default=0.05)
    data.add_argument("--sample-rate", type=int, default=16000)
    data.add_argument("--duration", type=float, default=8.0, help="seconds per training stream")
    data.add_argument("--num-targets", type=int, default=2, help="streams added to form a mixture")
    data.add_argument("--max-sounds", type=int, default=20, help="max sounds per stream")
    data.add_argument("--max-silence", type=float, default=0.5, help="max gap between sounds (s)")
    data.add_argument("--trim-start", type=float, default=0.0, help="max random cut at start (s)")
    data.add_argument("--trim-end", type=float, default=0.0, help="max random cut at end (s)")
    data.add_argument(
        "--no-random-count",
        action="store_true",
        help="always add all --num-targets streams instead of a random number of them",
    )
    data.add_argument("--average-mix", action="store_true", help="average instead of sum streams")
    data.add_argument("--augment-levels", action="store_true", help="random gain of 0 to -30 dB")
    data.add_argument("--remove-silence", action="store_true", help="trim silence off every sound")

    model = parser.add_argument_group("model (ignored with --resume)")
    model.add_argument("--encoder", choices=["strided", "dilated"], default="strided")
    model.add_argument("--num-sources", type=int, default=5, help="number of sources K")
    model.add_argument("--latent-dim", type=int, default=32)
    model.add_argument("--rnn-dim", type=int, default=512)
    model.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    model.add_argument("--rnn-layers", type=int, default=1)
    model.add_argument("--layer-channels", type=int, nargs="+", default=[32, 64, 128, 128, 128])
    model.add_argument("--variance", type=float, default=0.1, help="fixed likelihood variance")
    model.add_argument("--no-inference-rnn", action="store_true")
    model.add_argument("--activate-final-layer", action="store_true")
    model.add_argument("--autoencoder", action="store_true", help="no sampling, no KL term")
    model.add_argument(
        "--laplace", action="store_true", help="Laplace instead of Gaussian likelihood"
    )
    model.add_argument("--separator-kernel-size", type=int, default=1)
    model.add_argument("--separator-layers", type=int, default=1)
    model.add_argument("--separator-channels", type=int, default=32)
    model.add_argument("--n-fft", type=int, default=512)
    model.add_argument("--hop-length", type=int, default=128)
    model.add_argument("--compression", type=float, default=0.5, help="magnitude exponent c")

    train = parser.add_argument_group("optimisation")
    train.add_argument("-o", "--out", type=Path, default=Path("runs/dvae"), help="output folder")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32, help="streams per step and process")
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=2e-5)
    train.add_argument("--save-interval", type=int, default=1, help="epochs between checkpoints")
    train.add_argument("--log-interval", type=int, default=50, help="steps between progress lines")
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--device", default="auto")
    train.add_argument("--resume", type=Path, help="resume.pt of an earlier run")
    return parser


def split_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Training and validation files, holding files out of the training set if needed."""
    files = collect_files(args.train_data)
    if args.val_data is not None:
        return files, collect_files(args.val_data)
    order = np.random.default_rng(args.seed).permutation(len(files))
    n_val = max(1, int(round(args.val_fraction * len(files))))
    if len(files) <= n_val:
        raise SystemExit("too few training files to hold out a validation set")
    return [files[i] for i in order[n_val:]], [files[i] for i in order[:n_val]]


def make_dataset(
    files: list[Path], args: argparse.Namespace, seed: int | None
) -> NoteSequenceDataset:
    """Dataset of random note streams built from the command-line options."""
    return NoteSequenceDataset(
        files,
        duration=args.duration,
        sample_rate=args.sample_rate,
        max_silence=args.max_silence,
        trim_start=args.trim_start,
        trim_end=args.trim_end,
        max_sounds=args.max_sounds,
        augment_levels=args.augment_levels,
        remove_silence=args.remove_silence,
        seed=seed,
    )


def build_model(args: argparse.Namespace, stft: CompressedSTFT) -> DVAE:
    """Create a new model from the command-line options."""
    return DVAE(
        n_freq=stft.n_freq,
        latent_dim=args.latent_dim,
        rnn_dim=args.rnn_dim,
        num_sources=args.num_sources,
        encoder=args.encoder,
        layer_channels=tuple(args.layer_channels),
        rnn_type=args.rnn_type,
        rnn_layers=args.rnn_layers,
        inference_rnn=not args.no_inference_rnn,
        variance=args.variance,
        activate_final_layer=args.activate_final_layer,
        autoencoder=args.autoencoder,
        separator_kernel_size=args.separator_kernel_size,
        separator_layers=args.separator_layers,
        separator_channels=args.separator_channels,
    )


def average_across_processes(values: torch.Tensor, world_size: int) -> torch.Tensor:
    """Mean of ``values`` over all distributed processes (no-op for one process)."""
    if world_size > 1:
        dist.all_reduce(values)
        values = values / world_size
    return values


def run_epoch(model, loader, loss_fn, stft, device, args, optimizer=None, rank=0, world_size=1):
    """One pass over ``loader``; trains if an optimizer is given. Returns mean (loss, ell, kld)."""
    training = optimizer is not None
    model.train(training)
    generator = None if training else torch.Generator().manual_seed(args.seed)
    totals, steps = torch.zeros(3, device=device), 0
    with torch.set_grad_enabled(training):
        for step, streams in enumerate(loader, start=1):
            mixture, _ = mix_batch(
                streams, args.num_targets, not args.no_random_count, args.average_mix, generator
            )
            x = stft.magnitude(mixture.to(device))
            ell, kld = loss_fn(x, model(x))
            loss = ell + kld
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            totals += torch.stack([loss.detach(), ell.detach(), kld.detach()])
            steps += 1
            if training and rank == 0 and step % args.log_interval == 0:
                if not torch.isfinite(loss):
                    raise FloatingPointError("loss is not finite; try a lower --lr")
                print(f"  step {step}/{len(loader)}  loss {loss.item():.5f}", flush=True)
    totals = average_across_processes(totals / max(steps, 1), world_size)
    if not torch.isfinite(totals).all():
        raise FloatingPointError("loss is not finite; try a lower --lr")
    return totals.tolist()


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-train``."""
    args = build_parser().parse_args(argv)

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    device = resolve_device(args.device)
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if device.type == "cuda":
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.manual_seed(args.seed)

    resume = None
    if args.resume is not None:
        model, stft, resume = read_checkpoint(args.resume)
        # The architecture and front end come from the checkpoint, not the command line.
        args.sample_rate = stft.sample_rate
    else:
        stft = CompressedSTFT(args.n_fft, args.hop_length, args.compression, args.sample_rate)
        model = build_model(args, stft)
    model = model.to(device)

    train_files, val_files = split_files(args)
    train_set = make_dataset(train_files, args, seed=None)
    val_set = make_dataset(val_files, args, seed=args.seed)
    batch_size = min(args.batch_size, len(train_set))
    if batch_size < args.num_targets:
        raise SystemExit(f"need at least {args.num_targets} training files")
    sampler = DistributedSampler(train_set, drop_last=True) if world_size > 1 else None
    loader_args = dict(batch_size=batch_size, num_workers=args.num_workers, drop_last=True)
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader_args)
    val_loader = DataLoader(val_set, **dict(loader_args, batch_size=min(batch_size, len(val_set))))
    if len(val_set) < args.num_targets:
        val_loader = None  # too few validation sounds to form a mixture

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0
    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])
        start_epoch = resume["epoch"]
    network = (
        DistributedDataParallel(model, device_ids=[device] if device.type == "cuda" else None)
        if world_size > 1
        else model
    )  # noqa: E501
    loss_fn = ELBOLoss(scale_invariant=True, laplace=args.laplace, autoencoder=model.autoencoder)

    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config = {"args": options, "model": model.config, "device": str(device)}
        (args.out / "config.json").write_text(json.dumps(config, indent=2))
        log_path = args.out / "log.csv"
        if start_epoch == 0 or not log_path.exists():
            log_path.write_text(",".join(LOG_FIELDS) + "\n")
        n_parameters = sum(p.numel() for p in model.parameters()) / 1e6
        print(
            f"{len(train_set)} training / {len(val_set)} validation sounds, "
            f"{n_parameters:.2f}M parameters, device {device}"
        )

    for epoch in range(start_epoch + 1, args.epochs + 1):
        started = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        train = run_epoch(
            network, train_loader, loss_fn, stft, device, args, optimizer, rank, world_size
        )
        val = (
            run_epoch(network, val_loader, loss_fn, stft, device, args, None, rank, world_size)
            if val_loader is not None
            else [float("nan")] * 3
        )
        if rank == 0:
            seconds = time.time() - started
            print(
                f"epoch {epoch:3d}  train {train[0]:.5f}  val {val[0]:.5f}  ({seconds:.0f}s)",
                flush=True,
            )
            with open(args.out / "log.csv", "a", newline="") as f:
                csv.writer(f).writerow([epoch, *train, *val, round(seconds, 1)])
            if epoch % args.save_interval == 0 or epoch == args.epochs:
                save_checkpoint(args.out / "model.pt", model, stft, extra={"epoch": epoch})
                save_checkpoint(
                    args.out / "resume.pt",
                    model,
                    stft,
                    extra={"epoch": epoch, "optimizer": optimizer.state_dict()},
                )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
