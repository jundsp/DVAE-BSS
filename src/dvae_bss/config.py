"""Training options: one typed dataclass, read from YAML files and ``key=value`` overrides.

A config file only needs the options that differ from the defaults below. The options a run
actually used, overrides included, are saved as ``config.yaml`` in its run folder, so that
``dvae-train --resume <run folder>`` needs nothing else.

    config = load_config("configs/rwc_k5.yaml", ["epochs=10", "lr=3e-4"])
    save_config(config, "runs/k5/config.yaml")
"""

from __future__ import annotations

import dataclasses
import types
import typing
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


class ConfigError(ValueError):
    """An unknown option or an invalid value in a config file or override."""


def option(default: Any, help: str, group: str) -> Any:  # noqa: A002 (as in argparse)
    """A dataclass field with the description and section printed in config files."""
    if isinstance(default, list):
        return field(default_factory=lambda: list(default), metadata={"help": help, "group": group})
    return field(default=default, metadata={"help": help, "group": group})


DATA, MODEL, OPTIM, MONITOR = (
    "Data and run folder",
    "Model and mixtures (what the dissertation's experiments varied)",
    "Optimisation",
    "Best model, early stopping, learning-rate schedule and monitoring",
)


@dataclass
class TrainConfig:
    """Every option of ``dvae-train``. The defaults are the dissertation's RWC model."""

    train_data: Path | None = option(None, "folder of isolated sounds, or a file list", DATA)
    val_data: Path | None = option(None, "null: hold out 5% of train_data", DATA)
    max_train_sounds: int | None = option(None, "null: all; else a fixed random subset", DATA)
    max_val_sounds: int | None = option(None, "null: all; else a fixed random subset", DATA)
    out: Path = option(Path("runs/dvae"), "run folder", DATA)

    num_sources: int = option(5, "model outputs K", MODEL)
    num_targets: int = option(2, "streams added into each mixture", MODEL)
    random_count: bool = option(False, "true: 1 to num_targets streams per mixture", MODEL)
    compression: float = option(0.5, "magnitude exponent c", MODEL)
    variance: float = option(0.1, "likelihood variance", MODEL)
    encoder: Literal["strided", "dilated"] = option("strided", "strided | dilated", MODEL)
    rnn_type: Literal["lstm", "gru"] = option("lstm", "lstm | gru", MODEL)
    laplace: bool = option(False, "Laplace instead of Normal likelihood", MODEL)
    latent_dim: int = option(256, "size of the latent state", MODEL)
    rnn_dim: int = option(512, "size of the recurrent networks", MODEL)
    layer_channels: list[int] = option([32, 64, 128, 128, 128], "encoder channels", MODEL)

    duration: float = option(8.0, "seconds per stream", OPTIM)
    max_sounds: int = option(20, "sounds per stream at most (1: one note per stream)", OPTIM)
    max_silence: float = option(0.5, "random silence before each sound, up to (seconds)", OPTIM)
    epochs: int = option(500, "last epoch", OPTIM)
    batch_size: int = option(20, "streams per step and GPU", OPTIM)
    lr: float = option(1e-4, "learning rate (AdamW)", OPTIM)
    num_workers: int = option(4, "data loading processes", OPTIM)
    seed: int = option(1, "random seed", OPTIM)
    device: str = option("auto", "auto | cuda | mps | cpu", OPTIM)
    clip_grad_norm: float | None = option(None, "e.g. 1.0 to clip the gradient norm", OPTIM)
    max_skipped: int = option(20, "non-finite steps skipped per epoch before giving up", OPTIM)

    monitor: Literal["val_loss", "val_sdr"] = option(
        "val_loss", "val_loss | val_sdr: picks best.pt, early stopping and the lr schedule", MONITOR
    )
    patience: int = option(0, "stop after this many epochs without progress (0: never)", MONITOR)
    min_delta: float = option(0.0, "smallest change that counts as progress", MONITOR)
    lr_patience: int = option(0, "lower the lr after this many flat epochs (0: never)", MONITOR)
    lr_factor: float = option(0.5, "factor applied to the lr", MONITOR)
    log_interval: int = option(10, "steps between progress lines", MONITOR)
    plot_interval: int = option(5, "epochs between example plots and audio (0: never)", MONITOR)
    tensorboard: bool = option(False, "also log to <out>/tensorboard", MONITOR)


# Options fixed by the checkpoint: a resumed run cannot change them.
MODEL_OPTIONS = frozenset(
    {"num_sources", "compression", "variance", "encoder", "rnn_type", "latent_dim", "rnn_dim",
     "layer_channels"}
)  # fmt: skip


def _hints() -> dict[str, Any]:
    return typing.get_type_hints(TrainConfig)


def _coerce(name: str, value: Any, hint: Any) -> Any:
    """``value`` converted to the type ``hint``, as the command line would convert text."""
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        (inner,) = [a for a in args if a is not type(None)]
        return _coerce(name, value, inner)
    if origin is Literal:
        if value not in args:
            raise ConfigError(f"{name} must be one of {list(args)}, not {value!r}")
        return value
    if origin is list:
        values = value if isinstance(value, list) else [value]
        return [_coerce(name, v, args[0]) for v in values]
    try:
        if hint is bool:
            if isinstance(value, bool):
                return value
            raise ValueError
        if hint is int:
            if isinstance(value, bool) or (isinstance(value, float) and not value.is_integer()):
                raise ValueError
            return int(value)
        if hint is float:
            if isinstance(value, bool):
                raise ValueError
            return float(value)  # YAML reads "1e-4" as text
        if hint is Path:
            return Path(str(value))
        return str(value)
    except (TypeError, ValueError):
        expected = getattr(hint, "__name__", str(hint))
        raise ConfigError(f"{name} must be {expected}, not {value!r}") from None


def update_config(config: TrainConfig, values: Mapping[str, Any]) -> TrainConfig:
    """A copy of ``config`` with ``values`` set; names may use ``-`` or ``_``.

    Raises:
        ConfigError: For an unknown name or an invalid value.
    """
    hints = _hints()
    changes = {}
    for key, value in values.items():
        name = str(key).replace("-", "_")
        if name not in hints:
            raise ConfigError(f"unknown option {key!r}")
        changes[name] = _coerce(name, value, hints[name])
    return dataclasses.replace(config, **changes)


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """``["epochs=10", "layer_channels=[4, 8]"]`` -> ``{"epochs": 10, "layer_channels": [4, 8]}``.

    Values are read as YAML, so ``true``, ``null`` and lists work as in a config file.
    """
    values = {}
    for item in items:
        key, sep, text = item.partition("=")
        if not sep or not key:
            raise ConfigError(f"expected key=value, got {item!r}")
        values[key.strip()] = yaml.safe_load(text) if text.strip() else None
    return values


def read_config_file(path: str | Path) -> dict[str, Any]:
    """The options in a YAML or JSON file.

    Also reads the ``config.json`` of runs made before ``config.yaml``, which held the options
    under ``"args"``.
    """
    path = Path(path)
    values = yaml.safe_load(path.read_text()) or {}  # YAML is a superset of JSON
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: expected option names and values")
    if isinstance(values.get("args"), dict):
        # Options of the old command line that are not options any more.
        values = {k: v for k, v in values["args"].items() if k not in ("resume", "config")}
    return values


def load_config(
    path: str | Path | None = None,
    overrides: Iterable[str] = (),
    defaults: Mapping[str, Any] | None = None,
) -> TrainConfig:
    """Built-in defaults, then ``defaults``, then the file at ``path``, then ``overrides``.

    Raises:
        ConfigError: For an unknown option or an invalid value, naming the file.
    """
    config = update_config(TrainConfig(), defaults or {})
    if path is not None:
        try:
            config = update_config(config, read_config_file(path))
        except ConfigError as error:
            raise ConfigError(f"{path}: {error}") from None
    return update_config(config, parse_overrides(overrides))


def config_values(config: TrainConfig) -> dict[str, Any]:
    """The options as plain YAML/JSON values (paths as text)."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in dataclasses.asdict(config).items()}


def config_to_yaml(config: TrainConfig, header: str = "") -> str:
    """The options as a commented YAML file, in sections, every option listed."""
    values = config_values(config)
    lines = [f"# {line}".rstrip() for line in header.splitlines()]
    group = None
    # One line per option; lists stay on one line too ("flow" style).
    texts = {
        k: f"{k}: {yaml.safe_dump(v, default_flow_style=True, width=1000).strip()}"
        if isinstance(v, list)
        else yaml.safe_dump({k: v}, width=1000).strip()
        for k, v in values.items()
    }
    width = max(len(text) for text in texts.values()) + 2
    for item in dataclasses.fields(TrainConfig):
        if item.metadata["group"] != group:
            group = item.metadata["group"]
            lines += ["", f"# {group}."]
        lines.append(f"{texts[item.name]:<{width}}# {item.metadata['help']}")
    return "\n".join(lines).lstrip("\n") + "\n"


def save_config(config: TrainConfig, path: str | Path, header: str = "") -> None:
    """Write :func:`config_to_yaml` to ``path``."""
    Path(path).write_text(config_to_yaml(config, header))


def changed_options(old: TrainConfig, new: TrainConfig) -> dict[str, tuple[Any, Any]]:
    """``{name: (old value, new value)}`` for every option that differs."""
    before, after = dataclasses.asdict(old), dataclasses.asdict(new)
    return {k: (before[k], after[k]) for k in before if before[k] != after[k]}
