"""Build a pre-mixed test set (``mix.json`` + ``audio/``) from isolated sounds.

Each example adds two randomly chosen sounds, the second delayed by a random time, and stores
the mixture, the two sources as a two-channel file, and metadata for the evaluation figures:
the ``instrument`` of each sound (name of its parent folder), ``delta_time`` and, if the file
names start with a MIDI pitch (like ``060_...``), the pitches. The result can be used with
``dvae-evaluate``. The work is done by ``dvae_bss.data.write_premixed_set``.

    python experiments/data_prep/make_test_set.py --audio data/test_notes --out data/test_set -n 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dvae_bss.data import collect_files, write_premixed_set


def main(argv: list[str] | None = None) -> None:
    """Write the test set."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--audio", type=Path, required=True, help="folder or list of isolated sounds"
    )
    parser.add_argument("-o", "--out", type=Path, required=True)
    parser.add_argument("-n", "--num-examples", type=int, default=100)
    parser.add_argument("--duration", type=float, default=4.0, help="length of each example (s)")
    parser.add_argument("--max-delay", type=float, default=1.0, help="max onset difference (s)")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    files = collect_files(args.audio)
    if len(files) < 2:
        raise SystemExit("need at least two sounds")
    write_premixed_set(files, args.out, args.num_examples, args.duration, args.max_delay,
                       args.sample_rate, args.seed)  # fmt: skip
    print(f"wrote {args.num_examples} examples to {args.out}")


if __name__ == "__main__":
    main()
