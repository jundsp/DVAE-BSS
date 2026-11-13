r"""Two-note mixtures of real instrument recordings from NSynth, for the demo page and README.

Each mixture adds two acoustic notes from the NSynth test split, each scaled to unit peak and
starting at its own onset (equal onsets for notes that start together). The pair is then
scaled so the mixture peaks at 0.9 (SI-SDR does not change). Writes a pre-mixed set
(``mix.json``, ``audio/``) for ``dvae-evaluate`` and ``experiments/figures/make_demo_page.py``:

    python experiments/data_prep/make_nsynth_examples.py \
        --nsynth ~/data/nsynth-test --out data/NSynthExamples

``--nsynth`` is the unpacked ``nsynth-test.jsonwav.tar.gz`` (``examples.json``, ``audio/``)
from https://magenta.tensorflow.org/datasets/nsynth. NSynth is by Engel et al. (2017),
Google Magenta, licensed CC BY 4.0. Only the test split is used, so the mixtures are out of
sample for the pretrained model, which was trained and validated on RWC only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import soundfile as sf
import torch

from dvae_bss.data import read_audio

# (first note, second note, onset of the first, onset of the second in seconds); NSynth names.
PAIRS = [
    ("reed_acoustic_011-065-075", "guitar_acoustic_014-058-050", 0.0, 0.5),
    ("brass_acoustic_046-059-075", "guitar_acoustic_021-067-127", 0.0, 0.3),
    ("string_acoustic_056-064-100", "keyboard_acoustic_004-059-127", 0.4, 0.4),
    ("flute_acoustic_002-072-127", "mallet_acoustic_062-063-100", 0.0, 0.4),
    ("brass_acoustic_046-059-075", "brass_acoustic_046-067-075", 0.0, 0.0),
]
SAMPLE_RATE = 16000
DURATION = 4.0
PEAK = 0.9


def main(argv: list[str] | None = None) -> None:
    """Write the pre-mixed set."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--nsynth", type=Path, required=True, help="unpacked NSynth test split")
    parser.add_argument("-o", "--out", type=Path, default=Path("data/NSynthExamples"))
    args = parser.parse_args(argv)

    meta = json.loads((args.nsynth / "examples.json").read_text())
    n_samples = int(DURATION * SAMPLE_RATE)
    (args.out / "audio").mkdir(parents=True, exist_ok=True)
    records = {}
    for i, (first, second, start1, start2) in enumerate(PAIRS):
        stems = torch.zeros(2, n_samples)
        for k, (name, start) in enumerate(((first, start1), (second, start2))):
            onset = int(start * SAMPLE_RATE)
            wave = read_audio(args.nsynth / "audio" / f"{name}.wav", SAMPLE_RATE)
            wave = wave / wave.abs().max().clamp_min(1e-8)
            wave = wave[: n_samples - onset]
            stems[k, onset : onset + len(wave)] = wave
        stems = stems * PEAK / stems.sum(0).abs().max()
        sf.write(args.out / "audio" / f"mix_{i}.wav", stems.sum(0).numpy(), SAMPLE_RATE,
                 subtype="FLOAT")  # fmt: skip
        sf.write(args.out / "audio" / f"stems_{i}.wav", stems.T.numpy(), SAMPLE_RATE,
                 subtype="FLOAT")  # fmt: skip
        pitch1, pitch2 = meta[first]["pitch"], meta[second]["pitch"]
        records[str(i)] = dict(
            filename=f"mix_{i}.wav",
            stemname=f"stems_{i}.wav",
            onset1=start1,
            delta_time=start2 - start1,
            instrument1=meta[first]["instrument_family_str"],
            instrument2=meta[second]["instrument_family_str"],
            pitch1=pitch1,
            pitch2=pitch2,
            delta_semitone=pitch2 - pitch1,
            note1=first,
            note2=second,
        )
        print(f"{i}: {first} + {second}")
    (args.out / "mix.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
