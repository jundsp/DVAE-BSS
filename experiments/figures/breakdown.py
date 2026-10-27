"""Break the SI-SDR of one evaluation down by instrument, time offset and pitch difference.

Reads the ``results.json`` of ``dvae-evaluate``. Plots are drawn when the columns they need
exist (``instrument``, ``instrument1``/``instrument2``, ``delta_time``, ``delta_pitch``,
``delta_semitone``), so the same script serves the RWC, NSynth and two-note evaluations.

    python experiments/figures/breakdown.py results/nsynth/results.json --out figures/nsynth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dvae_bss.plotting import style

RENAME = {"sdr": "SI-SDR", "sir": "SIR", "sar": "SAR"}
# Display names of the two-note set's note types, as in the dissertation. Other names keep
# their spelling, with underscores as spaces (Computer Modern has no underscore glyph).
LABELS = {"forced": "sustain", "forced_vibrato": "vibrato"}


def save(fig, path: Path) -> None:
    """Save a figure tightly and close it."""
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def by_instrument(data: pd.DataFrame, out: Path) -> None:
    """Box plot of SI-SDR per source instrument, ordered by median."""
    order = data.groupby("instrument")["SI-SDR"].median().sort_values().index
    fig, ax = plt.subplots(figsize=(1.2 + 0.15 * len(order), 3))
    sns.boxplot(
        data=data, x="instrument", y="SI-SDR", order=order, hue="instrument", hue_order=order,
        palette="husl", legend=False, notch=True, ax=ax, showfliers=False, whis=0,
    )  # fmt: skip
    ax.tick_params(axis="x", rotation=90)
    ax.set_xlabel(None)
    save(fig, out / "boxplot_instrument_sisdr.pdf")


def by_pair(data: pd.DataFrame, out: Path) -> None:
    """Box plot of SI-SDR for every combination of the two instruments in the mixture."""
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    sns.boxplot(data=data, x="SI-SDR", y="instruments", ax=ax, showfliers=False, whis=0, notch=True)
    ax.set_ylabel(None)
    save(fig, out / "boxplot_instrument_pair_sisdr.pdf")


def instrument_pairs(data: pd.DataFrame, out: Path) -> None:
    """Mean SI-SDR for every pair of instruments in the mixture."""
    pair = np.sort(data[["instrument1", "instrument2"]].astype(str).to_numpy(), axis=1)
    pairs = pd.DataFrame(pair, columns=["first", "second"], index=data.index)
    pairs["SI-SDR"] = data["SI-SDR"]
    matrix = pairs.groupby(["first", "second"])["SI-SDR"].mean().reset_index()
    matrix = matrix.pivot(index="second", columns="first", values="SI-SDR")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(
        matrix,
        square=True,
        annot=True,
        fmt=".0f",
        cmap="plasma",
        ax=ax,
        cbar_kws={"label": "SI-SDR"},
    )
    ax.set(xlabel="", ylabel="")
    save(fig, out / "heatmap_instrument_pairs.pdf")


def line(
    data: pd.DataFrame, column: str, xlabel: str, filename: Path, hue: str | None = None
) -> None:
    """Mean SI-SDR against a mixture property."""
    fig, ax = plt.subplots(figsize=(4 if hue else 3, 2.5))
    sns.lineplot(data=data, x=column, y="SI-SDR", hue=hue, ax=ax)
    ax.set_xlabel(xlabel)
    if hue:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title=None)
    save(fig, filename)


def time_semitone_maps(data: pd.DataFrame, out: Path) -> None:
    """Mean SI-SDR, SIR and SAR on the grid of time offset and semitone difference."""
    mean = data.groupby(["delta_semitone", "delta_time"])[["SI-SDR", "SIR", "SAR"]].mean()
    for measure in mean.columns:
        grid = mean[measure].unstack()
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        times, semitones = grid.columns.to_numpy(float), grid.index.to_numpy()
        half_step = np.diff(times).min() / 2 if len(times) > 1 else 0.5
        image = ax.imshow(
            grid.to_numpy(),
            origin="lower",
            aspect="auto",
            cmap="plasma",
            interpolation="none",
            extent=(
                times[0] - half_step,
                times[-1] + half_step,
                semitones[0] - 0.5,
                semitones[-1] + 0.5,
            ),
        )
        ax.set_yticks(semitones)
        fig.colorbar(image).ax.set_ylabel(measure)
        ax.set(xlabel=r"$\Delta$ time (s)", ylabel=r"$\Delta$ semitone")
        save(fig, out / f"time_semitone_{measure}.pdf")


def main(argv: list[str] | None = None) -> None:
    """Draw every breakdown the results allow."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", type=Path, help="results.json of dvae-evaluate")
    parser.add_argument("-o", "--out", type=Path, default=Path("figures/breakdown"))
    parser.add_argument("--usetex", action="store_true", help="typeset text with LaTeX")
    args = parser.parse_args(argv)
    plt.rcParams.update(style(args.usetex))
    data = pd.read_json(args.results, orient="index").rename(columns=RENAME)
    for column in ("instrument", "instrument1", "instrument2"):
        if column in data:
            data[column] = [LABELS.get(str(n), str(n).replace("_", " ")) for n in data[column]]
    two_note = "delta_semitone" in data
    args.out.mkdir(parents=True, exist_ok=True)

    if "instrument" in data:
        by_instrument(data, args.out)
    if {"instrument1", "instrument2"} <= set(data):
        instrument_pairs(data, args.out)
        data["instruments"] = (
            data["instrument1"].astype(str) + " + " + data["instrument2"].astype(str)
        )
        by_pair(data, args.out)
    if "delta_time" in data:
        label = r"$\Delta$ time (s)" if two_note else r"$|\Delta|$ time (s)"
        hue = "instrument1" if two_note and "instrument1" in data else None
        line(data, "delta_time", label, args.out / "time_sdr.pdf", hue=hue)
    if "delta_pitch" in data:
        data["abs_delta_pitch"] = data["delta_pitch"].abs()
        line(data, "abs_delta_pitch", r"$|\Delta|$ pitch (semitones)", args.out / "pitch_sdr.pdf")
    if {"delta_semitone", "delta_time"} <= set(data):
        line(
            data,
            "delta_semitone",
            r"$\Delta$ semitone",
            args.out / "semitone_sdr.pdf",
            hue="instrument1",
        )
        time_semitone_maps(data, args.out)
    print(data[["SI-SDR", "SIR", "SAR"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
