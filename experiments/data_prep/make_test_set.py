"""Build a pre-mixed test set (``mix.json`` + ``audio/``) from isolated sounds.

Each example adds two randomly chosen sounds, the second delayed by a random time, and stores
the mixture, the two sources as a two-channel file, and metadata for the evaluation figures:
the ``instrument`` of each sound (name of its parent folder), ``delta_time`` and, if the file
names start with a MIDI pitch (like ``060_...``), the pitches. The result can be used with
``dvae-evaluate``.

    python experiments/data_prep/make_test_set.py --audio data/test_notes --out data/test_set -n 500
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from dvae_bss.data import collect_files, read_audio


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
    rng = np.random.default_rng(args.seed)
    n_samples = int(args.duration * args.sample_rate)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)

    records = {}
    for i in range(args.num_examples):
        first, second = rng.choice(len(files), size=2, replace=False)
        delay = round(float(rng.uniform(0, args.max_delay)), 4)
        stems = torch.zeros(2, n_samples)
        for k, (index, onset) in enumerate(((first, 0), (second, int(delay * args.sample_rate)))):
            wave = read_audio(files[index], args.sample_rate)
            wave = wave / wave.abs().max().clamp_min(1e-8)
            wave = wave[: n_samples - onset]
            stems[k, onset : onset + len(wave)] = wave
        sf.write(
            args.out / "audio" / f"mix_{i}.wav",
            stems.sum(0).numpy(),
            args.sample_rate,
            subtype="FLOAT",
        )
        sf.write(
            args.out / "audio" / f"stems_{i}.wav",
            stems.T.numpy(),
            args.sample_rate,
            subtype="FLOAT",
        )
        record = dict(filename=f"mix_{i}.wav", stemname=f"stems_{i}.wav", delta_time=delay)
        for k, index in enumerate((first, second), start=1):
            record[f"instrument{k}"] = files[index].parent.name
            match = re.match(r"(\d{1,3})_", files[index].name)
            if match:
                record[f"pitch{k}"] = int(match.group(1))
        records[str(i)] = record
    (args.out / "mix.json").write_text(json.dumps(records, indent=2))
    print(f"wrote {args.num_examples} examples to {args.out}")


if __name__ == "__main__":
    main()
