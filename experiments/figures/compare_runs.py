r"""Compare models that differ in one setting (compression, number of sources, variance, ...).

Takes the ``results.json`` written by ``dvae-evaluate`` for each setting and draws SI-SDR, SIR
and SAR as box plots and line plots, overall and per estimated source (ranked by SI-SDR within
each mixture). This replaces the compression, sources and variance figures of the dissertation.

    python experiments/figures/compare_runs.py results/c1/results.json results/c05/results.json \\
        --labels 1 0.5 --xlabel "compression $c$" --out figures/compression
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dvae_bss.plotting import style

METRICS = {"sdr": "SI-SDR", "sir": "SIR", "sar": "SAR"}
BOX = dict(showfliers=False, whis=0)
BOX_OUTLINE = dict(
    showcaps=False,
    boxprops={"facecolor": "none", "edgecolor": "grey"},
    medianprops={"color": "blue"},
    whiskerprops={"color": "blue"},
    **BOX,
)


def load_runs(paths: list[Path], labels: list[str], exclude: list[str]) -> pd.DataFrame:
    """All results in one frame, with a ``setting`` column and ``estimate`` (1 = best)."""
    frames = []
    for path, label in zip(paths, labels, strict=True):
        frame = pd.read_json(path, orient="index")
        frame["setting"] = label
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True).rename(columns=METRICS)
    for column in ("instrument1", "instrument2"):
        if column in data:
            data = data[~data[column].isin(exclude)]
    n_sources = data.groupby(["setting", "example"])["source"].transform("count")
    data["estimate"] = n_sources - data["place"]
    return data


def facet(data: pd.DataFrame, draw, filename: Path, size=(8.5, 3), **kwargs) -> None:
    """One panel per metric, drawn with ``draw(data=..., y=metric, ax=...)``."""
    fig, axes = plt.subplots(1, 3, figsize=size)
    for ax, metric in zip(axes, METRICS.values(), strict=True):
        draw(data=data, y=metric, ax=ax, **kwargs)
    for ax in axes[:-1]:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    if axes[-1].get_legend() is not None:
        sns.move_legend(axes[-1], "upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Draw the comparison figures."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results", nargs="+", type=Path, help="results.json files, one per setting")
    parser.add_argument("--labels", nargs="+", help="setting of each file (default: folder name)")
    parser.add_argument("--xlabel", default="setting")
    parser.add_argument("-o", "--out", type=Path, default=Path("figures/comparison"))
    parser.add_argument(
        "--exclude-instrument", nargs="*", default=[], help="drop these instruments"
    )
    parser.add_argument("--usetex", action="store_true", help="typeset text with LaTeX")
    args = parser.parse_args(argv)
    plt.rcParams.update(style(args.usetex))
    labels = args.labels or [path.parent.name for path in args.results]
    if len(labels) != len(args.results):
        parser.error("--labels needs one entry per results file")

    data = load_runs(args.results, labels, args.exclude_instrument)
    args.out.mkdir(parents=True, exist_ok=True)

    def box(*, data, y, ax, **kwargs):
        sns.boxplot(data=data, x="setting", y=y, ax=ax, **kwargs)
        ax.set_xlabel(args.xlabel)

    def line(*, data, y, ax, **kwargs):
        sns.pointplot(data=data, x="setting", y=y, ax=ax, **kwargs)
        ax.set_xlabel(args.xlabel)

    facet(data, box, args.out / "boxplot_outline.pdf", (7, 2.5), **BOX_OUTLINE)
    facet(data, box, args.out / "boxplot.pdf", **BOX)
    facet(data, box, args.out / "boxplot_by_estimate.pdf", hue="estimate", **BOX)
    facet(data, line, args.out / "lineplot_median.pdf", estimator="median", hue="estimate")
    medians = data.groupby("setting")[list(METRICS.values())].median().round(2)
    medians.to_csv(args.out / "medians.csv")
    print(medians.to_string())


if __name__ == "__main__":
    main()
