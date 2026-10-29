"""Train a DVAE on mixtures made on the fly from a folder of isolated sounds.

    dvae-train --train-data data/notes --out runs/k5

The defaults are the settings of the dissertation's RWC model (``pretrained/``). Each epoch
strings random sounds into streams, adds ``--num-targets`` streams to form each mixture, and
fits the model to the mixtures only. For several GPUs, launch the same command with
``torchrun --nproc_per_node=N -m dvae_bss.cli.train ...``.

The run folder gets ``config.json``, ``log.csv``, ``model.pt`` and ``resume.pt`` (pass it to
``--resume`` to continue). Options not exposed here (trimming, silences, separator network,
...) are arguments of ``NoteSequenceDataset`` and ``DVAE``: edit them below.
"""

from __future__ import annotations

import argparse
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

from dvae_bss.audio import CompressedSTFT
from dvae_bss.checkpoint import read_checkpoint, save_checkpoint
from dvae_bss.cli.common import resolve_device
from dvae_bss.data import NoteSequenceDataset, collect_files, mix_batch
from dvae_bss.losses import ELBOLoss
from dvae_bss.model import DVAE

SAMPLE_RATE = 16000
VAL_FRACTION = 0.05  # held out of --train-data when no --val-data is given
LOG_FIELDS = ["epoch", "train_loss", "train_ell", "train_kld", "val_loss", "val_ell", "val_kld"]


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    """Command-line options (``add_help=False`` to reuse them, as ``mixit.py`` does)."""
    p = argparse.ArgumentParser("dvae-train", description=__doc__.split("\n")[0], add_help=add_help)
    p.add_argument("--train-data", type=Path, required=True, help="folder of sounds or a file list")
    p.add_argument("--val-data", type=Path, help="default: 5%% of the training sounds")
    p.add_argument("-o", "--out", type=Path, default=Path("runs/dvae"), help="run folder")
    p.add_argument("--resume", type=Path, help="resume.pt of an earlier run")

    # What the dissertation's experiments varied.
    p.add_argument("--num-sources", type=int, default=5, help="model outputs K")
    p.add_argument("--num-targets", type=int, default=2, help="max streams per mixture")
    p.add_argument("--compression", type=float, default=0.5, help="magnitude exponent c")
    p.add_argument("--variance", type=float, default=0.1, help="likelihood variance")
    p.add_argument("--encoder", choices=["strided", "dilated"], default="strided")
    p.add_argument("--rnn-type", choices=["lstm", "gru"], default="lstm")
    p.add_argument("--laplace", action="store_true", help="Laplace instead of Normal likelihood")
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--rnn-dim", type=int, default=512)
    p.add_argument("--layer-channels", type=int, nargs="+", default=[32, 64, 128, 128, 128])

    p.add_argument("--duration", type=float, default=8.0, help="seconds per stream")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=20, help="streams per step and GPU")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=50, help="steps between progress lines")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    return p


def build_model(args: argparse.Namespace, stft: CompressedSTFT, **overrides) -> DVAE:
    """A new model from the options. Other ``DVAE`` arguments keep their defaults."""
    return DVAE(
        n_freq=stft.n_freq,
        num_sources=args.num_sources,
        latent_dim=args.latent_dim,
        rnn_dim=args.rnn_dim,
        layer_channels=tuple(args.layer_channels),
        encoder=args.encoder,
        rnn_type=args.rnn_type,
        variance=args.variance,
        **overrides,
    )


def split_files(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    """Training and validation sounds."""
    files = collect_files(args.train_data)
    if args.val_data is not None:
        return files, collect_files(args.val_data)
    order = np.random.default_rng(args.seed).permutation(len(files))
    n_val = max(1, round(VAL_FRACTION * len(files)))
    return [files[i] for i in order[n_val:]], [files[i] for i in order[:n_val]]


def setup_distributed(device: torch.device) -> tuple[int, int, torch.device]:
    """Rank, world size and device of this process. One process unless run by torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size == 1:
        return 0, 1, device
    if device.type == "cuda":
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return dist.get_rank(), world_size, device


def run_epoch(model, loader, loss_fn, stft, num_targets, optimizer=None, log_interval=0):
    """One pass over ``loader``, training if an optimizer is given. Returns mean loss, ELL, KLD."""
    training = optimizer is not None
    model.train(training)
    # Validation mixtures are the same every epoch.
    generator = None if training else torch.Generator().manual_seed(0)
    device = next(model.parameters()).device
    totals = torch.zeros(3, device=device)
    with torch.set_grad_enabled(training):
        for step, streams in enumerate(loader, start=1):
            # Add streams into mixtures. The sources are discarded: training is unsupervised.
            mixture, _ = mix_batch(streams, num_targets, generator=generator)
            x = stft.magnitude(mixture.to(device))  # compressed magnitudes (B, F, T)
            # Negative ELBO = reconstruction of the mixture (ell) + KL to the prior (kld).
            ell, kld = loss_fn(x, model(x))
            loss = ell + kld
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if log_interval and step % log_interval == 0:
                    print(f"  step {step}/{len(loader)}  loss {loss.item():.5f}", flush=True)
            totals += torch.stack([loss, ell, kld]).detach()
    totals /= max(len(loader), 1)
    if dist.is_initialized():
        dist.all_reduce(totals)
        totals /= dist.get_world_size()
    if not torch.isfinite(totals).all():
        raise FloatingPointError("loss is not finite; try a lower --lr")
    return totals.tolist()


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-train``."""
    args = build_parser().parse_args(argv)
    rank, world_size, device = setup_distributed(resolve_device(args.device))
    is_main = rank == 0
    torch.manual_seed(args.seed)

    # Model. With --resume the architecture comes from the checkpoint.
    if args.resume is not None:
        model, stft, resume = read_checkpoint(args.resume)
    else:
        stft = CompressedSTFT(compression=args.compression, sample_rate=SAMPLE_RATE)
        model = build_model(args, stft)
        resume = {"epoch": 0}

    # Data: each item is a stream of random sounds; run_epoch adds streams into mixtures.
    train_files, val_files = split_files(args)
    streams = dict(duration=args.duration, sample_rate=stft.sample_rate, augment_levels=True)
    train_set = NoteSequenceDataset(train_files, **streams)
    val_set = NoteSequenceDataset(val_files, **streams, seed=args.seed)
    if min(len(train_set), args.batch_size) < args.num_targets:
        raise SystemExit(f"need at least {args.num_targets} training sounds")

    sampler = DistributedSampler(train_set, drop_last=True) if world_size > 1 else None
    batch_size = min(args.batch_size, len(train_set))
    loader = dict(batch_size=batch_size, num_workers=args.num_workers, drop_last=True)
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader)
    val_loader = None  # needs at least num_targets sounds to form a mixture
    if len(val_set) >= args.num_targets:
        val_loader = DataLoader(val_set, **{**loader, "batch_size": min(batch_size, len(val_set))})

    # Training.
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-5)
    if "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
    network = model
    if world_size > 1:
        network = DistributedDataParallel(model, [device] if device.type == "cuda" else None)
    loss_fn = ELBOLoss(scale_invariant=True, laplace=args.laplace, autoencoder=model.autoencoder)

    log_path = args.out / "log.csv"
    if is_main:
        args.out.mkdir(parents=True, exist_ok=True)
        options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config = {"args": options, "model": model.config}
        (args.out / "config.json").write_text(json.dumps(config, indent=2))
        if resume["epoch"] == 0 or not log_path.exists():
            log_path.write_text(",".join([*LOG_FIELDS, "seconds"]) + "\n")
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{len(train_set)} training / {len(val_set)} validation sounds, "
              f"{n_params:.2f}M parameters, {device}")  # fmt: skip

    for epoch in range(resume["epoch"] + 1, args.epochs + 1):
        started = time.time()
        if sampler is not None:
            sampler.set_epoch(epoch)
        log_interval = args.log_interval if is_main else 0
        train = run_epoch(network, train_loader, loss_fn, stft, args.num_targets, optimizer,
                          log_interval)  # fmt: skip
        val = [float("nan")] * 3
        if val_loader is not None:
            val = run_epoch(network, val_loader, loss_fn, stft, args.num_targets)
        if is_main:
            seconds = time.time() - started
            print(f"epoch {epoch:3d}  train {train[0]:.5f}  val {val[0]:.5f}  ({seconds:.0f}s)",
                  flush=True)  # fmt: skip
            with log_path.open("a") as f:
                f.write(",".join(str(v) for v in [epoch, *train, *val, round(seconds, 1)]) + "\n")
            save_checkpoint(args.out / "model.pt", model, stft, {"epoch": epoch})
            save_checkpoint(args.out / "resume.pt", model, stft,
                            {"epoch": epoch, "optimizer": optimizer.state_dict()})  # fmt: skip

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
