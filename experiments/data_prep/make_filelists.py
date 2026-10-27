"""Write fixed train / validate / test file lists for a folder of sounds.

``dvae-train`` can hold out validation files itself; use this when the split should be fixed
and reused (for example to keep test sounds away from training). Each list has one path per
line, relative to ``--output`` (``--output`` defaults to the audio folder), and can be passed
to ``dvae-train --train-data`` and ``--val-data``.

    python experiments/data_prep/make_filelists.py --audio data/notes --fractions 0.9 0.05 0.05
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from dvae_bss.data import collect_files

SPLITS = ("train", "validate", "test")


def main(argv: list[str] | None = None) -> None:
    """Write ``train.txt``, ``validate.txt`` and ``test.txt``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--audio", type=Path, required=True, help="folder of audio files")
    parser.add_argument("-o", "--output", type=Path, help="where to write the lists")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=[0.9, 0.05, 0.05],
        metavar=("TRAIN", "VALIDATE", "TEST"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    output = args.output or args.audio
    output.mkdir(parents=True, exist_ok=True)

    files = collect_files(args.audio)
    order = np.random.default_rng(args.seed).permutation(len(files))
    fractions = np.array(args.fractions) / sum(args.fractions)
    edges = np.round(np.cumsum(fractions) * len(files)).astype(int)
    for name, indices in zip(SPLITS, np.split(order, edges[:-1]), strict=True):
        lines = [os.path.relpath(files[i], output) for i in sorted(indices)]
        (output / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        print(f"{name}: {len(lines)} files")


if __name__ == "__main__":
    main()
