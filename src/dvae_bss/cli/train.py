"""Train a DVAE on mixtures made on the fly from a folder of isolated sounds.

    dvae-train configs/rwc_k5.yaml                          # a new run
    dvae-train configs/rwc_k5.yaml out=runs/k5b lr=3e-4     # with overrides
    dvae-train --resume runs/k5                             # continue a run
    dvae-train --resume runs/k5 epochs=800                  # continue with changes
    dvae-train --print-config > my.yaml                     # every option, as a new config

Options come from a YAML config file (see ``configs/`` and :class:`~dvae_bss.config.TrainConfig`
for all of them and their defaults), then from ``key=value`` overrides. The options a run
used are saved as ``<out>/config.yaml``; ``--resume`` reads them from there, applies any new
overrides (except to the model, which the checkpoint fixes) and saves them back.

The defaults are the settings of the dissertation's RWC model (``pretrained/``). Each epoch
strings random sounds into streams, adds exactly ``num_targets`` streams to form each
mixture, and fits the model to the mixtures only. With ``random_count``, each stream after
the first is dropped half the time, so mixtures have between one and ``num_targets`` streams.
For several GPUs, launch the same command with
``torchrun --nproc_per_node=N -m dvae_bss.cli.train ...``.

The run folder gets, and updates after every epoch:

* ``config.yaml``: the options of the run, overrides included.
* ``train.log``: a copy of the console output.
* ``log.csv``: losses, validation SI-SDR/SIR/SAR, learning rate, timing and peak GPU
  memory per epoch (``gpu_total_gb - gpu_reserved_gb`` is the headroom).
* ``model.pt`` (last epoch), ``best.pt`` (best ``monitor`` value) and ``resume.pt`` (the
  optimiser and early-stopping state that ``--resume`` continues from).
* ``plots/curves.png``: learning curves; ``plots/examples/`` and ``samples/``: spectrograms
  and audio of a fixed validation mixture separated by the model, every ``plot_interval``
  epochs. Needs matplotlib (the ``plots`` extra).
* ``tensorboard/``: the same, per step, with ``tensorboard: true``.

Training is unsupervised, but the validation mixtures are made from known streams, so the
SI-SDR of the separated outputs shows how separation, not just reconstruction, progresses.

Options not exposed here (trimming, silences, separator network, ...) are arguments of
``NoteSequenceDataset`` and ``DVAE``: edit them below.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dvae_bss.audio import CompressedSTFT
from dvae_bss.checkpoint import read_checkpoint, save_checkpoint
from dvae_bss.cli.common import resolve_device
from dvae_bss.config import (
    MODEL_OPTIONS,
    ConfigError,
    TrainConfig,
    changed_options,
    config_to_yaml,
    load_config,
    parse_overrides,
    save_config,
    update_config,
)
from dvae_bss.data import NoteSequenceDataset, collect_files, mix_batch
from dvae_bss.losses import ELBOLoss, optimal_gain
from dvae_bss.metrics import (
    bss_metrics,
    compute_measures_batch,
    make_scale_invariant,
    optimal_permute,
)
from dvae_bss.model import DVAE
from dvae_bss.monitor import (
    Board,
    EarlyStopping,
    GpuMonitor,
    MetricTable,
    StepTimer,
    close_logging,
    format_duration,
    logger,
    model_size,
    rule,
    setup_logging,
)
from dvae_bss.separator import Separator

SAMPLE_RATE = 16000
VAL_FRACTION = 0.05  # held out of train_data when no val_data is given
LOG_FIELDS = [
    "epoch",
    "train_loss", "train_ell", "train_kld",
    "val_loss", "val_ell", "val_kld",
    "val_sdr", "val_sir", "val_sar",
    "lr", "grad_norm", "skipped_steps", "seconds",
    "gpu_peak_gb", "gpu_reserved_gb", "gpu_total_gb", "gpu_util",
]  # fmt: skip
SILENT = 1e-8  # energy below which a reference stream counts as absent (--random-count)


def build_parser() -> argparse.ArgumentParser:
    """The command line: a config file, ``key=value`` overrides and two flags."""
    p = argparse.ArgumentParser(
        "dvae-train",
        description=__doc__.split("\n")[0],
        epilog="Options and their defaults: dvae-train --print-config",
    )
    p.add_argument(
        "items",
        nargs="*",
        metavar="CONFIG | key=value",
        help="a YAML config file (for a new run), then option overrides such as epochs=10",
    )
    p.add_argument("--resume", type=Path, metavar="RUN", help="run folder to continue")
    p.add_argument("--print-config", action="store_true", help="print the options as YAML and exit")
    return p


def parse_command_line(argv: list[str] | None = None) -> tuple[TrainConfig, bool]:
    """The options of the run and whether it continues an earlier one.

    Raises:
        SystemExit: For a usage error, an invalid option, or after ``--print-config``.
    """
    parser = build_parser()
    cli = parser.parse_args(argv)
    files = [item for item in cli.items if "=" not in item]
    overrides = [item for item in cli.items if "=" in item]
    if len(files) > 1:
        parser.error(f"one config file at most, got {files}")
    try:
        if cli.resume is not None:
            if files:
                parser.error("--resume takes the options of the run; give overrides only")
            config = resumed_config(cli.resume, overrides)
        elif files or cli.print_config:
            config = load_config(files[0] if files else None, overrides)
        else:
            parser.error("give a config file (see configs/) or --resume RUN")
    except (ConfigError, OSError) as error:
        parser.error(str(error))
    if cli.print_config:
        print(config_to_yaml(config), end="")
        raise SystemExit(0)
    if config.train_data is None:
        parser.error("train_data is not set (in the config file or as train_data=...)")
    return config, cli.resume is not None


def resumed_config(run: Path, overrides: list[str]) -> TrainConfig:
    """The options saved in ``run`` (a run folder or its ``resume.pt``) with ``overrides``.

    Raises:
        ConfigError: If an override changes the model or the run folder.
    """
    if run.suffix in (".yaml", ".yml", ".json"):
        raise ConfigError(
            f"--resume takes a run folder (the out= of a run), not a config file: {run}"
        )
    if run.suffix == ".pt":
        run = run.parent
    if not (run / "resume.pt").exists():
        raise ConfigError(f"{run} has no resume.pt to continue from")
    # Runs made before config.yaml saved their options in config.json.
    saved = next((run / n for n in ("config.yaml", "config.json") if (run / n).exists()), None)
    config = load_config(saved)
    changes = parse_overrides(overrides)
    fixed = sorted({k.replace("-", "_") for k in changes} & MODEL_OPTIONS)
    if fixed:
        raise ConfigError(f"the checkpoint fixes {', '.join(fixed)}; start a new run instead")
    config = update_config(config, changes)
    if config.out.resolve() != run.resolve():
        if "out" in {k.replace("-", "_") for k in changes}:
            raise ConfigError("a resumed run stays in its folder; out cannot be changed")
        config = update_config(config, {"out": run})  # the folder was moved or renamed
    return config


def build_model(cfg: TrainConfig, stft: CompressedSTFT, **overrides: Any) -> DVAE:
    """A new model from the options. Other ``DVAE`` arguments keep their defaults."""
    return DVAE(
        n_freq=stft.n_freq,
        num_sources=cfg.num_sources,
        latent_dim=cfg.latent_dim,
        rnn_dim=cfg.rnn_dim,
        layer_channels=tuple(cfg.layer_channels),
        encoder=cfg.encoder,
        rnn_type=cfg.rnn_type,
        variance=cfg.variance,
        **overrides,
    )


def split_files(cfg: TrainConfig) -> tuple[list[Path], list[Path]]:
    """Training and validation sounds, limited to ``max_train_sounds``/``max_val_sounds``."""
    files = collect_files(cfg.train_data)
    if cfg.val_data is not None:
        train, val = files, collect_files(cfg.val_data)
    else:
        order = np.random.default_rng(cfg.seed).permutation(len(files))
        n_val = max(1, round(VAL_FRACTION * len(files)))
        train, val = [files[i] for i in order[n_val:]], [files[i] for i in order[:n_val]]
    return subset(train, cfg.max_train_sounds, cfg.seed), subset(val, cfg.max_val_sounds, cfg.seed)


def subset(files: list[Path], limit: int | None, seed: int) -> list[Path]:
    """At most ``limit`` of ``files``, picked at random but the same for a given seed."""
    if limit is None or len(files) <= limit:
        return files
    keep = np.random.default_rng(seed).choice(len(files), size=limit, replace=False)
    return [files[i] for i in sorted(keep)]


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


def all_finite(value: torch.Tensor) -> bool:
    """Whether ``value`` is finite on every process, so that all of them skip the same steps.

    A process that skipped a backward pass alone would leave the others waiting forever for
    its gradients.
    """
    finite = torch.isfinite(value.detach()).all().float()
    if dist.is_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


def mean_across_processes(values: torch.Tensor) -> torch.Tensor:
    """Average a tensor over the processes of a multi-GPU run (unchanged otherwise)."""
    if dist.is_initialized():
        dist.all_reduce(values)
        values = values / dist.get_world_size()
    return values


def max_across_processes(values: dict[str, float], device: torch.device) -> dict[str, float]:
    """The largest value of each entry over the processes of a multi-GPU run.

    Every process must pass the same keys. Unchanged in a single-process run.
    """
    if not dist.is_initialized() or not values:
        return values
    tensor = torch.tensor(list(values.values()), dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return dict(zip(values, tensor.tolist(), strict=True))


def broadcast_value(value: float, device: torch.device) -> float:
    """The value of process 0, on every process (unchanged in a single-process run)."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor([value], dtype=torch.float64, device=device)
    dist.broadcast(tensor, src=0)
    return float(tensor.item())


def previous_monitor(resume: dict[str, Any], out: Path) -> str | None:
    """The value a resumed run monitored, or ``None`` for a new run or if it is unknown.

    Older ``resume.pt`` files do not record it; ``best.pt`` holds it as its score's key.
    """
    if "monitor" in resume or "early_stopping" not in resume:
        return resume.get("monitor")
    if (out / "best.pt").exists():
        _, _, extra = read_checkpoint(out / "best.pt")
        keys = [key for key in extra if key != "epoch"]
        if len(keys) == 1:
            return keys[0]
    return None


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Learning rate of the first parameter group."""
    return float(optimizer.param_groups[0]["lr"])


def train_epoch(
    network: nn.Module,
    loader: DataLoader,
    loss_fn: ELBOLoss,
    stft: CompressedSTFT,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    epoch: int,
    board: Board,
    step: int,
    gpu: GpuMonitor,
) -> tuple[dict[str, float], int]:
    """One pass over the training streams.

    Args:
        network: The model, possibly wrapped in ``DistributedDataParallel``.
        loader: Batches of streams ``(B, L)``.
        loss_fn: Negative ELBO, split into reconstruction and KL terms.
        stft: Front end that turns mixtures into compressed magnitudes.
        optimizer: Updates the network's parameters.
        cfg: Options; uses ``num_targets``, ``random_count``, ``clip_grad_norm``,
            ``max_skipped`` and ``log_interval``.
        epoch: Epoch number, for the progress lines.
        board: TensorBoard writer (may be disabled).
        step: Number of updates before this epoch, the x axis of TensorBoard.
        gpu: Samples GPU memory and utilisation during each step.

    Returns:
        The epoch's means (``train_loss``, ``train_ell``, ``train_kld``, ``grad_norm``),
        ``skipped_steps``, and the updated step count.

    Raises:
        FloatingPointError: If more than ``cfg.max_skipped`` steps had a non-finite loss
            or gradient.
    """
    network.train()
    device = next(network.parameters()).device
    totals = torch.zeros(4, device=device)  # loss, ell, kld, gradient norm
    n_steps = n_skipped = 0
    timer = StepTimer()
    max_norm = cfg.clip_grad_norm or math.inf  # inf: only measure the gradient norm
    for batch, streams in enumerate(loader, start=1):
        timer.data_ready()
        # Add streams into mixtures. The sources are discarded: training is unsupervised.
        mixture, _ = mix_batch(streams, cfg.num_targets, cfg.random_count)
        x = stft.magnitude(mixture.to(device))  # compressed magnitudes (B, F, T)
        # Negative ELBO = reconstruction of the mixture (ell) + KL to the prior (kld).
        ell, kld = loss_fn(x, network(x))
        loss = ell + kld
        gpu.sample()  # activations are alive now: near the step's peak memory

        # Skip a step with a non-finite loss or gradient instead of corrupting the weights.
        optimizer.zero_grad(set_to_none=True)
        reason = "loss"
        if all_finite(loss):
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(network.parameters(), max_norm)
            reason = "gradient"
        if reason == "loss" or not torch.isfinite(grad_norm):
            n_skipped += 1
            logger.warning(f"epoch {epoch} step {batch}: non-finite {reason}, step skipped")
            if n_skipped > cfg.max_skipped:
                raise FloatingPointError(
                    f"{n_skipped} non-finite steps in epoch {epoch}; try a lower --lr "
                    "or --clip-grad-norm"
                )
            optimizer.zero_grad(set_to_none=True)
            timer.step_done()
            continue
        optimizer.step()
        timer.step_done()
        gpu.sample()

        values = torch.stack([loss, ell, kld, grad_norm]).detach()
        totals += values
        n_steps += 1
        step += 1
        if cfg.log_interval and batch % cfg.log_interval == 0:
            loss_, ell_, kld_, norm_ = values.tolist()
            mean = totals[0].item() / n_steps
            logger.info(
                f"epoch {epoch} [{batch:{len(str(len(loader)))}d}/{len(loader)}] "
                f"loss {loss_:.5f} (ell {ell_:.5f}, kld {kld_:.5f}) | mean {mean:.5f} | "
                f"grad {norm_:.3g} | lr {current_lr(optimizer):.2e} | "
                f"{timer.summary(len(loader) - batch)}"
                + (f" | {memory}" if (memory := gpu.summary()) else "")
            )
            step_values = dict(loss=loss_, ell=ell_, kld=kld_, grad_norm=norm_)
            board.scalars({**step_values, "lr": current_lr(optimizer)}, step, "step")

    totals = mean_across_processes(totals / max(n_steps, 1))
    loss_, ell_, kld_, norm_ = totals.tolist()
    stats = dict(train_loss=loss_, train_ell=ell_, train_kld=kld_, grad_norm=norm_)
    stats["skipped_steps"] = float(n_skipped)
    logger.info(f"epoch {epoch} training done: {timer.summary()} per step")
    return stats, step


@torch.no_grad()
def separation_example(
    model: DVAE, stft: CompressedSTFT, mixture: torch.Tensor, sources: torch.Tensor
) -> dict[str, Any]:
    """Separate the first mixture of a batch, for the example plots and audio.

    Returns:
        ``mixture`` ``(L,)`` and ``references`` ``(N, L)``; the model's raw output:
        ``reconstruction`` ``(L,)`` (the sum of the decoded sources, which the loss compares
        with the mixture) and ``raw_estimates`` ``(K, L)`` (each decoded source), both
        rescaled by the loss's gain and given the mixture phase; the masked outputs of
        ``dvae-separate``: ``estimates`` ``(K, L)`` and their sum ``estimates_sum`` ``(L,)``;
        and ``scores`` and ``raw_scores``, the SI-SDR of each matched masked and raw output.
        Raw and masked outputs are in the same order, the first ``N`` matched to the
        references (the matching is that of the masked outputs).
    """
    mixture, sources = mixture[:1], sources[:1]
    n_samples = mixture.shape[-1]
    spec = stft.analysis(mixture)
    magnitude, phase = spec.abs(), spec.angle()
    output = model(magnitude, stochastic=False)
    # The loss is scale invariant, so the output level is arbitrary: apply the gain the loss
    # uses, to show what is actually compared with the mixture.
    gain = optimal_gain(magnitude, output["recon_x"])
    recon_x = output["recon_x"] * gain  # (1, F, T)
    recon_s = output["recon_s"][0] * gain  # (K, F, T)
    reconstruction = stft.synthesis(torch.polar(recon_x, phase), n_samples)
    raw = stft.synthesis(torch.polar(recon_s, phase.expand_as(recon_s)), n_samples)

    masked = Separator(model, stft)(mixture)  # exactly what dvae-separate returns
    _, perm = optimal_permute(masked, sources)
    order = torch.as_tensor(perm[0], device=masked.device)
    result = bss_metrics(masked, sources)
    raw = raw[order]
    raw_sdr, _, _ = compute_measures_batch(
        raw[None, : sources.shape[1]], make_scale_invariant(raw[None], sources)
    )
    return dict(
        mixture=mixture[0].cpu(),
        references=sources[0].cpu(),
        reconstruction=reconstruction[0].cpu(),
        raw_estimates=raw.cpu(),
        estimates=masked[0, order].cpu(),
        estimates_sum=masked[0].sum(0).cpu(),
        scores=result["sdr"][0].tolist(),
        raw_scores=raw_sdr[0].tolist(),
    )


@torch.no_grad()
def validate(
    network: nn.Module,
    model: DVAE,
    loader: DataLoader,
    loss_fn: ELBOLoss,
    stft: CompressedSTFT,
    cfg: TrainConfig,
    score: bool,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    """Loss and, if ``score``, separation quality on the validation mixtures.

    The validation mixtures are the same every epoch, so the numbers are comparable from one
    epoch to the next. Their sources are known, so the outputs of the model (ratio-masked, as
    by ``dvae-separate``) are scored with permutation-invariant SI-SDR, SIR and SAR. Absent
    sources (``--random-count``) are left out of the scores.

    Args:
        network: The model, possibly wrapped in ``DistributedDataParallel``.
        model: The bare model, used for the separation scores.
        loader: Batches of validation streams.
        loss_fn: Negative ELBO.
        stft: Front end.
        cfg: Options; uses ``num_targets`` and ``random_count``.
        score: Compute the separation scores and the example (process 0 only, and only if
            the model has at least as many outputs as there are streams per mixture).

    Returns:
        The means (``val_loss``, ``val_ell``, ``val_kld``, and ``val_sdr``, ``val_sir``,
        ``val_sar`` if scored), and the separated first mixture (see
        :func:`separation_example`) if scored.
    """
    network.eval()
    device = next(network.parameters()).device
    generator = torch.Generator().manual_seed(0)  # the same mixtures every epoch
    totals = torch.zeros(3, device=device)
    scores = torch.zeros(3, dtype=torch.float64)
    n_scored = 0
    example = None
    for streams in loader:
        mixture, sources = mix_batch(
            streams, cfg.num_targets, cfg.random_count, generator=generator
        )
        mixture, sources = mixture.to(device), sources.to(device)
        x = stft.magnitude(mixture)
        ell, kld = loss_fn(x, network(x))
        totals += torch.stack([ell + kld, ell, kld])
        if score:
            result = bss_metrics(Separator(model, stft)(mixture), sources)
            active = sources.pow(2).sum(-1) > SILENT  # (B, N)
            for i, name in enumerate(("sdr", "sir", "sar")):
                scores[i] += result[name][active].sum().cpu().double()
            n_scored += int(active.sum())
            if example is None:
                example = separation_example(model, stft, mixture, sources)
    totals = mean_across_processes(totals / max(len(loader), 1))
    loss_, ell_, kld_ = totals.tolist()
    stats = dict(val_loss=loss_, val_ell=ell_, val_kld=kld_)
    if n_scored:
        sdr, sir, sar = (scores / n_scored).tolist()
        stats |= dict(val_sdr=sdr, val_sir=sir, val_sar=sar)
    return stats, example


class Plotter:
    """Writes the plots and audio of a run; a failure here is logged, never fatal.

    Args:
        out: Run folder; plots go to ``out/plots``, audio to ``out/samples``.
        sample_rate: Sample rate of the audio.
        board: TensorBoard writer, which also receives the figures and audio.
    """

    def __init__(self, out: Path, sample_rate: int, board: Board) -> None:
        self.folder = out / "plots"
        self.samples = out / "samples"
        self.sample_rate = sample_rate
        self.board = board
        self.enabled = True
        if importlib.util.find_spec("matplotlib") is None:
            logger.warning("matplotlib is not installed (pip install dvae-bss[plots]): no plots")
            self.enabled = False

    def curves(self, rows: list[dict[str, float]], best_epoch: int) -> None:
        """Redraw ``plots/curves.png`` from the rows of ``log.csv``."""
        if self.enabled:
            self._guard(self._curves, rows, best_epoch)

    def example(self, example: dict[str, Any], epoch: int) -> None:
        """Draw ``plots/examples/epoch_NNNN.png`` and write ``samples/*.wav``."""
        if self.enabled:
            self._guard(self._example, example, epoch)

    def _guard(self, draw: Callable[..., None], *args: object) -> None:
        try:
            draw(*args)
        except Exception as error:  # a plotting bug must not end a long training run
            logger.warning(f"plotting failed ({type(error).__name__}: {error}); continuing")

    def _curves(self, rows: list[dict[str, float]], best_epoch: int) -> None:
        import matplotlib.pyplot as plt

        from dvae_bss.plotting import plot_training_curves, save_png

        self.folder.mkdir(parents=True, exist_ok=True)
        fig = plot_training_curves(rows, best_epoch)
        save_png(fig, self.folder / "curves.png")
        plt.close(fig)

    def _example(self, example: dict[str, Any], epoch: int) -> None:
        import matplotlib.pyplot as plt

        from dvae_bss.plotting import plot_training_example, save_png

        folder = self.folder / "examples"
        folder.mkdir(parents=True, exist_ok=True)
        fig = plot_training_example(**example, sample_rate=self.sample_rate, title=f"epoch {epoch}")
        save_png(fig, folder / f"epoch_{epoch:04d}.png")
        self.board.figure("example", fig, epoch)
        plt.close(fig)

        # Audio of the latest example, overwritten each time: listen while training runs.
        self.samples.mkdir(parents=True, exist_ok=True)
        waves = {"mixture": example["mixture"], "reconstruction": example["reconstruction"]}
        waves |= {f"source{k + 1}": w for k, w in enumerate(example["references"])}
        waves |= {f"raw_output{k + 1}": w for k, w in enumerate(example["raw_estimates"])}
        waves |= {f"output{k + 1}": w for k, w in enumerate(example["estimates"])}
        for name, wave in waves.items():
            sf.write(self.samples / f"{name}.wav", wave.numpy(), self.sample_rate, "FLOAT")
            self.board.audio(f"example/{name}", wave, epoch, self.sample_rate)


def log_header(
    cfg: TrainConfig,
    model: DVAE,
    device: torch.device,
    world_size: int,
    n_sounds: tuple[int, int],
    steps_per_epoch: int,
) -> None:
    """Describe the run at the top of the log."""
    logger.info(rule("dvae-train"))
    logger.info(f"run folder: {cfg.out}")
    processes = f" x {world_size} processes" if world_size > 1 else ""
    logger.info(f"device: {device}{processes}")
    logger.info(
        f"data: {n_sounds[0]} training / {n_sounds[1]} validation sounds, "
        f"{steps_per_epoch} steps of {cfg.batch_size} streams per epoch, "
        f"mixtures of {'1 to ' if cfg.random_count else ''}{cfg.num_targets} streams"
    )
    logger.info(
        f"model: {model.num_sources} outputs, {cfg.encoder} encoder, {cfg.rnn_type}, "
        f"{model_size(model)}"
    )
    likelihood = "Laplace" if cfg.laplace else "Normal"
    logger.info(f"objective: {likelihood} likelihood (variance {cfg.variance}) + KL to the prior")
    stopping = f"patience {cfg.patience}" if cfg.patience else "off"
    schedule = (
        f"x{cfg.lr_factor} after {cfg.lr_patience} flat epochs" if cfg.lr_patience else "constant"
    )
    clip = cfg.clip_grad_norm or "off"
    logger.info(
        f"optimiser: AdamW, lr {cfg.lr:g} ({schedule}), gradient clipping {clip}, "
        f"early stopping {stopping} (on {cfg.monitor})"
    )
    logger.info(rule())


def log_epoch(
    epoch: int, cfg: TrainConfig, row: dict[str, float], status: str, gpu: GpuMonitor
) -> None:
    """Summary lines at the end of an epoch."""
    logger.info(
        rule(f"epoch {epoch}/{cfg.epochs} done in {format_duration(row['seconds'])}", char="-")
    )
    skipped = f", {row['skipped_steps']:.0f} steps skipped" if row["skipped_steps"] else ""
    logger.info(
        f"train  loss {row['train_loss']:.5f}  ell {row['train_ell']:.5f}  "
        f"kld {row['train_kld']:.5f}  grad {row['grad_norm']:.3g}{skipped}"
    )
    if math.isfinite(row.get("val_loss", math.nan)):
        scores = ""
        if math.isfinite(row.get("val_sdr", math.nan)):
            scores = (
                f"  SI-SDR {row['val_sdr']:.2f} dB  SIR {row['val_sir']:.2f} dB  "
                f"SAR {row['val_sar']:.2f} dB"
            )
        logger.info(
            f"val    loss {row['val_loss']:.5f}  ell {row['val_ell']:.5f}  "
            f"kld {row['val_kld']:.5f}{scores}"
        )
    logger.info(f"lr {row['lr']:.2e} | {status}")
    if memory := gpu.summary(row):
        logger.info(memory)


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-train``."""
    cfg, resuming = parse_command_line(argv)
    rank, world_size, device = setup_distributed(resolve_device(cfg.device))
    is_main = rank == 0
    torch.manual_seed(cfg.seed)
    if not resuming and (cfg.out / "resume.pt").exists():
        raise SystemExit(
            f"{cfg.out} already holds a run: continue it with --resume {cfg.out}, "
            "or choose another folder with out=..."
        )
    # Every process reads the earlier options, before process 0 replaces them below.
    saved = cfg.out / "config.yaml"
    previous = load_config(saved) if resuming and saved.exists() else None
    if is_main:
        cfg.out.mkdir(parents=True, exist_ok=True)
        header = f"Options of this run. Continue it with: dvae-train --resume {cfg.out}"
        save_config(cfg, saved, header)
    # Only process 0 talks and writes files; the others report warnings and errors only.
    setup_logging(cfg.out / "train.log" if is_main else None, verbose=is_main)
    if previous is not None:
        for name, (old, new) in changed_options(previous, cfg).items():
            logger.info(f"option changed on resume: {name} {old} -> {new}")
    board = Board(cfg.out / "tensorboard" if cfg.tensorboard and is_main else None)
    try:
        run(cfg, previous if resuming else None, resuming, rank, world_size, device, board)
    except Exception:
        logger.exception("training failed")
        raise
    finally:
        board.close()
        close_logging()
        if world_size > 1:
            dist.destroy_process_group()


def run(
    cfg: TrainConfig,
    previous_cfg: TrainConfig | None,
    resuming: bool,
    rank: int,
    world_size: int,
    device: torch.device,
    board: Board,
) -> None:
    """The training run of :func:`main`, once logging and processes are set up.

    ``previous_cfg`` holds the options a resumed run was started with, if known.
    """
    is_main = rank == 0

    # Model. A resumed run takes the architecture and its state from resume.pt.
    if resuming:
        try:
            model, stft, resume = read_checkpoint(cfg.out / "resume.pt")
        except (RuntimeError, EOFError) as error:  # cut short by a run stopped while saving
            raise SystemExit(
                f"{cfg.out / 'resume.pt'} is damaged ({error.__class__.__name__}): the run was "
                "probably stopped while saving it. Start a new run, or resume from a copy."
            ) from error
        logger.info(f"resuming {cfg.out} after epoch {resume['epoch']}")
    else:
        stft = CompressedSTFT(compression=cfg.compression, sample_rate=SAMPLE_RATE)
        model = build_model(cfg, stft)
        resume = {"epoch": 0}

    # Data: each item is a stream of random sounds; the epoch functions add them into mixtures.
    train_files, val_files = split_files(cfg)
    streams = dict(
        duration=cfg.duration,
        sample_rate=stft.sample_rate,
        max_sounds=cfg.max_sounds,
        max_silence=cfg.max_silence,
        augment_levels=True,
    )
    train_set = NoteSequenceDataset(train_files, **streams)
    val_set = NoteSequenceDataset(val_files, **streams, seed=cfg.seed)
    if min(len(train_set), cfg.batch_size) < cfg.num_targets:
        raise SystemExit(f"need at least {cfg.num_targets} training sounds")

    sampler = DistributedSampler(train_set, drop_last=True) if world_size > 1 else None
    batch_size = min(cfg.batch_size, len(train_set))
    # Keep the worker processes between epochs: starting and stopping them costs seconds per
    # pass (about 10 s on macOS). Safe here, as training items draw fresh randomness on every
    # access and validation items are seeded by their index.
    loader = dict(
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(train_set, shuffle=sampler is None, sampler=sampler, **loader)
    val_loader = None  # needs at least num_targets sounds to form a mixture
    if len(val_set) >= cfg.num_targets:
        val_loader = DataLoader(val_set, **{**loader, "batch_size": min(batch_size, len(val_set))})
    else:
        logger.warning(f"fewer than {cfg.num_targets} validation sounds: no validation")

    # Optimisation.
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=2e-5)
    network: nn.Module = model
    if world_size > 1:
        network = DistributedDataParallel(model, [device] if device.type == "cuda" else None)
    loss_fn = ELBOLoss(scale_invariant=True, laplace=cfg.laplace, autoencoder=model.autoencoder)

    # What decides the best epoch: SI-SDR needs a validation set and K >= num_targets.
    can_score = val_loader is not None and model.num_sources >= cfg.num_targets
    monitor = cfg.monitor
    if monitor == "val_sdr" and not can_score:
        logger.warning("cannot score separation here; monitoring val_loss instead")
        monitor = "val_loss"
    if monitor == "val_loss" and val_loader is None:
        monitor = "train_loss"
    mode = "max" if monitor == "val_sdr" else "min"
    stopper = EarlyStopping(cfg.patience, cfg.min_delta, mode)
    scheduler = None
    if cfg.lr_patience:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode,
            factor=cfg.lr_factor,
            patience=cfg.lr_patience,
            # Absolute, like early stopping: a relative threshold misbehaves for SI-SDR < 0.
            threshold=cfg.min_delta,
            threshold_mode="abs",
        )

    # Continue where an earlier run stopped (older resume.pt files hold only the optimiser).
    if "optimizer" in resume:
        optimizer.load_state_dict(resume["optimizer"])
    # Early stopping and the lr schedule only continue if they watch the same value as before:
    # a best SI-SDR (maximised) means nothing to a run now minimising the validation loss.
    previous = previous_monitor(resume, cfg.out)
    if previous not in (None, monitor):
        logger.info(f"monitor changed from {previous} to {monitor}: early stopping restarts")
    else:
        if "early_stopping" in resume:
            stopper.load_state_dict(resume["early_stopping"])
        if scheduler is not None and "scheduler" in resume:
            scheduler.load_state_dict(resume["scheduler"])
    # The restored optimiser carries the old lr; a new lr= given on resume replaces it. A plain
    # resume keeps the restored value, which the lr schedule may already have lowered.
    if previous_cfg is not None and previous_cfg.lr != cfg.lr:
        for group in optimizer.param_groups:
            group["lr"] = cfg.lr
        if scheduler is not None:
            scheduler._last_lr = [cfg.lr] * len(optimizer.param_groups)
        logger.info(f"lr set to {cfg.lr:.2e} on resume")
    step = int(resume.get("step", 0))

    table = plotter = None
    if is_main:
        board.text("config", f"```\n{config_to_yaml(cfg)}\n```")
        table = MetricTable(cfg.out / "log.csv", LOG_FIELDS, resume=resume["epoch"] > 0)
        plotter = Plotter(cfg.out, stft.sample_rate, board)
        log_header(
            cfg, model, device, world_size, (len(train_set), len(val_set)), len(train_loader)
        )

    gpu = GpuMonitor(device)
    for epoch in range(resume["epoch"] + 1, cfg.epochs + 1):
        started = time.time()
        gpu.reset()
        if sampler is not None:
            sampler.set_epoch(epoch)
        logger.info(rule(f"epoch {epoch}/{cfg.epochs}", char="-"))
        train_stats, step = train_epoch(
            network, train_loader, loss_fn, stft, optimizer, cfg, epoch, board, step, gpu
        )
        val_stats: dict[str, float] = {}
        example = None
        if val_loader is not None:
            logger.info("validating")
            val_stats, example = validate(
                network, model, val_loader, loss_fn, stft, cfg, score=is_main and can_score
            )
        row = {"epoch": epoch, **train_stats, **val_stats, "lr": current_lr(optimizer)}
        row["seconds"] = time.time() - started
        # Peaks over training and validation; the fullest GPU of a multi-GPU run.
        row.update(max_across_processes(gpu.stats(), device))

        # The monitored value comes from process 0 (only it computes SI-SDR), so that every
        # process takes the same LR and stopping decisions.
        value = broadcast_value(row.get(monitor, math.nan), device)
        improved = stopper.step(value, epoch)
        if scheduler is not None:
            scheduler.step(value)
            if current_lr(optimizer) < row["lr"]:
                logger.info(
                    f"{monitor} flat for {stopper.bad_epochs} epochs: "
                    f"lr -> {current_lr(optimizer):.2e}"
                )

        if is_main:
            assert table is not None and plotter is not None
            table.append(row)
            board.scalars({k: v for k, v in row.items() if k != "epoch"}, epoch, "epoch")
            save_checkpoint(cfg.out / "model.pt", model, stft, {"epoch": epoch})
            if improved:
                extra = {"epoch": epoch, monitor: value}
                save_checkpoint(cfg.out / "best.pt", model, stft, extra)
            state = {
                "epoch": epoch,
                "step": step,
                "optimizer": optimizer.state_dict(),
                "monitor": monitor,
                "early_stopping": stopper.state_dict(),
            }
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            save_checkpoint(cfg.out / "resume.pt", model, stft, state)

            plotter.curves(table.read(), stopper.best_epoch)
            if (
                example is not None
                and cfg.plot_interval
                and (epoch == 1 or epoch % cfg.plot_interval == 0)
            ):
                plotter.example(example, epoch)
            saved = " -> saved best.pt" if improved else ""
            log_epoch(epoch, cfg, row, f"{monitor} {stopper.status()}{saved}", gpu)

        if stopper.should_stop:
            logger.info(
                f"early stopping: {monitor} has not improved for {cfg.patience} epochs; "
                f"best was epoch {stopper.best_epoch} ({stopper.best:.5f}), saved as best.pt"
            )
            break
    logger.info(rule("done"))


if __name__ == "__main__":
    main()
