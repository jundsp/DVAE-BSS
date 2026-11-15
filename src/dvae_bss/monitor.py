"""Tools for watching a training run: a log file, a metric table, TensorBoard, early stopping.

``dvae-train`` uses them to leave a record in the run folder that can be read while
training is still going:

* ``train.log``: everything printed to the console, with timestamps.
* ``log.csv``: one row of losses, separation scores, learning rate and peak GPU memory
  per epoch (:class:`MetricTable`, :class:`GpuMonitor`).
* ``tensorboard/``: the same values per step and per epoch, plus figures and audio, when
  TensorBoard is installed and enabled (:class:`Board`).

:class:`EarlyStopping` and :class:`StepTimer` keep track of the best epoch and of where the
time goes (waiting for data vs. computing).
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from matplotlib.figure import Figure

logger = logging.getLogger("dvae_bss.train")

_FORMAT = "%(asctime)s %(levelname)-7s %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(log_file: Path | None = None, verbose: bool = True) -> logging.Logger:
    """Send training messages to the console and, if given, to ``log_file`` (appended).

    Args:
        log_file: File that receives a copy of every message.
        verbose: Show progress messages; ``False`` keeps only warnings and errors, as for the
            other processes of a multi-GPU run.

    Returns:
        The ``dvae_bss.train`` logger.
    """
    close_logging()
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.propagate = False
    formatter = logging.Formatter(_FORMAT, _DATE_FORMAT)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def close_logging() -> None:
    """Detach and close the handlers added by :func:`setup_logging`."""
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def rule(title: str = "", width: int = 79, char: str = "=") -> str:
    """A separator line for the log, with an optional title: ``==== title ====``."""
    if not title:
        return char * width
    side = max(width - len(title) - 2, 0)
    return f"{char * (side // 2)} {title} {char * (side - side // 2)}"


def format_duration(seconds: float) -> str:
    """``3725`` -> ``"1:02:05"``; ``65`` -> ``"1:05"``."""
    if not math.isfinite(seconds):
        return "?"
    minutes, secs = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class GpuMonitor:
    """Peak GPU memory (and, on CUDA with ``nvidia-ml-py``, utilisation) since :meth:`reset`.

    The headroom of a run is ``gpu_total_gb - gpu_reserved_gb``: what is left for a larger
    ``batch_size`` or ``duration``. Values, all in GB:

    * ``gpu_peak_gb``: peak memory held by tensors.
    * ``gpu_reserved_gb``: peak memory held by PyTorch, tensors plus its cache; the number
      that runs out first.
    * ``gpu_total_gb``: memory of the device (on Apple MPS, the recommended maximum).
    * ``gpu_util``: mean percentage of time the GPU was busy, sampled once per step (CUDA
      only, needs ``nvidia-ml-py``). Well below 100 points at the data loader.

    On the CPU every method does nothing and :meth:`stats` is empty. On MPS, which keeps no
    peak, the peaks are the largest values seen by :meth:`sample`.

    Args:
        device: The training device.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.kind = device.type if device.type in ("cuda", "mps") else None
        self._util_ok = self.kind == "cuda"
        self.reset()

    def reset(self) -> None:
        """Start a new measurement period (an epoch)."""
        self._peak = self._reserved = 0.0
        self._util: list[float] = []
        if self.kind == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def sample(self) -> None:
        """Take a reading; call once per step."""
        if self.kind == "mps":
            self._peak = max(self._peak, torch.mps.current_allocated_memory() / 1e9)
            self._reserved = max(self._reserved, torch.mps.driver_allocated_memory() / 1e9)
        elif self._util_ok:
            try:
                self._util.append(float(torch.cuda.utilization(self.device)))
            except Exception:  # no nvidia-ml-py, or no NVML on this machine
                self._util_ok = False

    def stats(self) -> dict[str, float]:
        """The values listed above, for ``log.csv``; empty on the CPU."""
        if self.kind is None:
            return {}
        self.sample()
        if self.kind == "cuda":
            values = {
                "gpu_peak_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
                "gpu_reserved_gb": torch.cuda.max_memory_reserved(self.device) / 1e9,
                "gpu_total_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9,
            }
        else:
            values = {
                "gpu_peak_gb": self._peak,
                "gpu_reserved_gb": self._reserved,
                "gpu_total_gb": torch.mps.recommended_max_memory() / 1e9,
            }
        if self._util:
            values["gpu_util"] = sum(self._util) / len(self._util)
        return values

    def summary(self, stats: Mapping[str, float] | None = None) -> str:
        """E.g. ``"GPU peak 3.1 GB, reserved 3.5 of 24.0 GB (85% free), busy 97%"``.

        Args:
            stats: Values from :meth:`stats` (e.g. the maximum over processes); default: now.
        """
        stats = self.stats() if stats is None else stats
        if not stats or not stats.get("gpu_total_gb"):
            return ""
        reserved, total = stats["gpu_reserved_gb"], stats["gpu_total_gb"]
        text = (
            f"GPU peak {stats['gpu_peak_gb']:.1f} GB, reserved {reserved:.1f} of {total:.1f} GB "
            f"({100 * (1 - reserved / total):.0f}% free)"
        )
        if "gpu_util" in stats and math.isfinite(stats["gpu_util"]):
            text += f", busy {stats['gpu_util']:.0f}%"
        return text


def model_size(model: torch.nn.Module) -> str:
    """Number of parameters, e.g. ``"4.21M parameters (4.21M trainable)"``."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"{total / 1e6:.2f}M parameters ({trainable / 1e6:.2f}M trainable)"


class StepTimer:
    """Average time per step spent waiting for the data loader and spent computing.

    Call :meth:`data_ready` when a batch arrives and :meth:`step_done` after the update.
    A slow data loader (``data`` close to ``compute``) calls for more ``--num-workers``.
    """

    def __init__(self) -> None:
        self.started = time.perf_counter()
        self._mark = self.started
        self.data = 0.0
        self.compute = 0.0
        self.steps = 0

    def data_ready(self) -> None:
        """A batch has arrived: the time since the last step was spent loading data."""
        now = time.perf_counter()
        self.data += now - self._mark
        self._mark = now

    def step_done(self) -> None:
        """The update is done: the time since the batch arrived was spent computing."""
        now = time.perf_counter()
        self.compute += now - self._mark
        self._mark = now
        self.steps += 1

    def summary(self, steps_left: int = 0) -> str:
        """``"data 12 ms, compute 85 ms per step, ETA 1:05"``."""
        n = max(self.steps, 1)
        per_step = (time.perf_counter() - self.started) / n
        eta = f", ETA {format_duration(per_step * steps_left)}" if steps_left else ""
        return f"data {1e3 * self.data / n:.0f} ms, compute {1e3 * self.compute / n:.0f} ms{eta}"


class EarlyStopping:
    """Track the best value of a metric and stop after ``patience`` epochs without progress.

    Args:
        patience: Epochs without improvement before stopping; ``0`` never stops (the best
            epoch is still tracked, to save ``best.pt``).
        min_delta: Smallest change that counts as an improvement.
        mode: ``"min"`` for losses, ``"max"`` for scores such as SI-SDR.
    """

    def __init__(self, patience: int = 0, min_delta: float = 0.0, mode: str = "min") -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', not {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = math.inf if mode == "min" else -math.inf
        self.best_epoch = 0
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record the value of ``epoch``; return whether it is the best so far.

        A non-finite value never counts as an improvement.
        """
        if self.mode == "min":
            improved = value < self.best - self.min_delta
        else:
            improved = value > self.best + self.min_delta
        # The first finite value is always the best so far, whatever min_delta is.
        improved = math.isfinite(value) and (improved or self.best_epoch == 0)
        if improved:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self) -> bool:
        """Whether ``patience`` epochs have passed without improvement."""
        return self.patience > 0 and self.bad_epochs >= self.patience

    def status(self) -> str:
        """``"best 0.01234 at epoch 12, 3/20 epochs without improvement"``."""
        limit = f"/{self.patience}" if self.patience > 0 else ""
        epochs = "epoch" if self.bad_epochs == 1 and not limit else "epochs"
        return (
            f"best {self.best:.5f} at epoch {self.best_epoch}, "
            f"{self.bad_epochs}{limit} {epochs} without improvement"
        )

    def state_dict(self) -> dict[str, float | int]:
        """What :meth:`load_state_dict` needs to continue after ``--resume``."""
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state: Mapping[str, float | int]) -> None:
        """Restore the state saved by :meth:`state_dict`."""
        self.best = float(state["best"])
        self.best_epoch = int(state["best_epoch"])
        self.bad_epochs = int(state["bad_epochs"])


class MetricTable:
    """Per-epoch values in a CSV file, appended to as training goes.

    Args:
        path: The CSV file.
        fields: Column names. Missing values are written as ``nan``.
        resume: Keep the rows already in ``path``. Rows written with other columns (by an
            older version) are rewritten under the new header.
    """

    def __init__(self, path: Path, fields: Sequence[str], resume: bool = False) -> None:
        self.path = path
        self.fields = list(fields)
        rows = []
        if resume and path.exists():
            with path.open(newline="") as f:
                rows = list(csv.DictReader(f))  # kept as written
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, self.fields, restval="nan", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

    def append(self, row: Mapping[str, Any]) -> None:
        """Add one row; floats are rounded to 6 significant digits."""
        values = {k: f"{v:.6g}" if isinstance(v, float) else v for k, v in row.items()}
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, self.fields, restval="nan", extrasaction="ignore").writerow(values)

    def read(self) -> list[dict[str, float]]:
        """All rows so far, as numbers."""
        with self.path.open(newline="") as f:
            return [{k: float(v) for k, v in row.items() if v} for row in csv.DictReader(f)]


class Board:
    """Optional TensorBoard writer; every method does nothing when it is disabled.

    Args:
        folder: Where to write the event files, or ``None`` to disable.
    """

    def __init__(self, folder: Path | None) -> None:
        self.writer = None
        if folder is None:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            logger.warning("TensorBoard is not installed (pip install tensorboard); skipping it")
            return
        self.writer = SummaryWriter(log_dir=str(folder))
        logger.info(f"TensorBoard: tensorboard --logdir {folder}")

    def scalars(self, values: Mapping[str, float], step: int, prefix: str = "") -> None:
        """Log each finite value as ``prefix/name``."""
        if self.writer is None:
            return
        for name, value in values.items():
            if math.isfinite(value):
                self.writer.add_scalar(f"{prefix}/{name}" if prefix else name, value, step)

    def figure(self, tag: str, figure: Figure, step: int) -> None:
        """Log a matplotlib figure as an image (the figure is left open)."""
        if self.writer is not None:
            self.writer.add_figure(tag, figure, step, close=False)

    def audio(self, tag: str, wave: torch.Tensor, step: int, sample_rate: int) -> None:
        """Log a waveform ``(L,)``, normalised to avoid clipping."""
        if self.writer is not None:
            wave = wave.detach().float().cpu()
            self.writer.add_audio(tag, wave / wave.abs().max().clamp_min(1e-8), step, sample_rate)

    def text(self, tag: str, text: str) -> None:
        """Log a block of text, such as the run's options."""
        if self.writer is not None:
            self.writer.add_text(tag, text)

    def close(self) -> None:
        """Flush and close the event file."""
        if self.writer is not None:
            self.writer.close()
