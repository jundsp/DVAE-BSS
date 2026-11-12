"""The TwoNote test set: two synthetic notes with controlled pitch interval and onset offset.

This is the test set behind the TwoNote results in Chapter 6 of the dissertation (appendix,
"Two-note test dataset"). Each note is a sum of harmonics of one of three production types:

* ``free``: a freely decaying, slightly inharmonic note, like a plucked string. It starts
  at full level and its harmonics decay exponentially, the higher ones faster.
* ``forced``: a sustained note with a short attack and release.
* ``forced_vibrato``: the sustained note with a vibrato of ±50 cents at 4 Hz.

Every mixture adds a reference note (C4, starting at ``LEAD_IN``) and a second note that is
``delta_semitone`` semitones higher (0 to 12) and starts ``delta_time`` seconds later
(0.005 to 1.005 s), for six pairs of production types. The full grid is
6 × 13 × 17 = 1326 mixtures, numbered with the onset offset varying fastest, then the
interval, then the pair, so items keep the numbers they had in the dissertation
(the examples on the demo page are items 222, 357, 365 and 619).

The original generator was lost. This one was rebuilt from the appendix and fitted to the
four TwoNote items kept in ``docs/audio``: their spectrograms match within about 1 dB, and
the free and sustained notes match sample for sample (over 20 dB SI-SDR). The vibrato note
matches in spectrum but not in phase. Two details differ from the appendix: the vibrato rate
is 4 Hz, not 5 Hz, and each note is scaled to the same RMS, not the same peak.

    dvae-make-twonote --out data/TwoNote
    dvae-evaluate -m pretrained/dvae_rwc_k5.pt -d data/TwoNote -o results/twonote
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
NOTE_TYPES = ("free", "forced", "forced_vibrato")

#: Pairs of (reference, second note) production types, in the order of the dataset.
PAIRS: tuple[tuple[str, str], ...] = (
    ("free", "free"),
    ("forced", "free"),
    ("forced_vibrato", "free"),
    ("forced", "forced"),
    ("forced", "forced_vibrato"),
    ("forced_vibrato", "forced_vibrato"),
)
INTERVALS = tuple(range(13))  # semitones above the reference
DELAYS = tuple(round(0.005 + 0.0625 * k, 4) for k in range(17))  # seconds

REFERENCE_HZ = 440.0 * 2.0 ** (-9 / 12)  # C4, 261.63 Hz
NOTE_DURATION = 2.0  # seconds
LEAD_IN = 0.1  # silence before the reference note (s)
FILE_DURATION = 3.2  # seconds
STEM_RMS = 0.0424  # every stem is scaled to this RMS over the whole file

ROLLOFF = {"free": 0.3, "forced": 0.4, "forced_vibrato": 0.4}  # harmonic m: (-1)^m e^(-r(m-1))
INHARMONICITY = 5e-5  # stiffness B of the free note: f_m = m f0 sqrt(1 + B m²)
DECAY = (3.0, 0.15)  # free note: harmonic m decays as e^(-(a + b m) t)
ATTACK = 0.05  # sustained notes: linear attack (s)
RELEASE = 0.05  # sustained notes: release (s)
VIBRATO_CENTS = 50.0
VIBRATO_HZ = 4.0  # the appendix says 5 Hz; the surviving items have 4 Hz


def synth_note(
    kind: str,
    f0: float,
    duration: float = NOTE_DURATION,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """One note of a production type, before level scaling.

    Args:
        kind: One of :data:`NOTE_TYPES`.
        f0: Fundamental frequency (Hz).
        duration: Length of the note (s).
        sample_rate: Sample rate (Hz).

    Returns:
        ``float64`` waveform of ``duration * sample_rate`` samples.
    """
    if kind not in NOTE_TYPES:
        raise ValueError(f"unknown note type {kind!r}; choose from {NOTE_TYPES}")
    n = int(round(duration * sample_rate))
    t = np.arange(n) / sample_rate
    nyquist = 0.5 * sample_rate

    m = np.arange(1, int(nyquist // f0) + 2, dtype=float)
    ratios = m * np.sqrt(1.0 + INHARMONICITY * m**2) if kind == "free" else m
    keep = ratios * f0 * 2.0 ** (VIBRATO_CENTS / 1200.0) < nyquist
    m, ratios = m[keep], ratios[keep]
    amps = (-1.0) ** m * np.exp(-ROLLOFF[kind] * (m - 1))

    if kind == "forced_vibrato":
        cents = -VIBRATO_CENTS * np.sin(2 * np.pi * VIBRATO_HZ * t)
        phase = 2 * np.pi * np.cumsum(f0 * 2.0 ** (cents / 1200.0)) / sample_rate
    elif kind == "forced":
        phase = 2 * np.pi * f0 * (t + 1.0 / sample_rate)
    else:
        phase = 2 * np.pi * f0 * t

    if kind == "free":
        env = np.exp(-(DECAY[0] + DECAY[1] * m)[:, None] * t[None])
    else:
        ramp = np.ones(n)
        n_attack = int(ATTACK * sample_rate)
        ramp[:n_attack] = np.arange(n_attack) / n_attack
        n_release = int(RELEASE * sample_rate)
        ramp[n - n_release :] *= np.linspace(1.0, 0.0, n_release) ** 2
        env = ramp[None]
    return (amps[:, None] * env * np.sin(ratios[:, None] * phase[None])).sum(0)


def make_example(
    pair: tuple[str, str],
    interval: int,
    delay: float,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """The two stems ``(2, samples)`` of one mixture; the mixture is their sum."""
    n = int(round(FILE_DURATION * sample_rate))
    stems = np.zeros((2, n))
    for k, (kind, semitones, onset) in enumerate(
        ((pair[0], 0, LEAD_IN), (pair[1], interval, LEAD_IN + delay))
    ):
        note = synth_note(kind, REFERENCE_HZ * 2.0 ** (semitones / 12.0), sample_rate=sample_rate)
        start = int(round(onset * sample_rate))
        note = note[: n - start]
        stems[k, start : start + len(note)] = note
        stems[k] *= STEM_RMS / np.sqrt(np.mean(stems[k] ** 2))
    return stems


def grid(
    pairs: Sequence[tuple[str, str]] = PAIRS,
    intervals: Sequence[int] = INTERVALS,
    delays: Sequence[float] = DELAYS,
) -> list[dict[str, Any]]:
    """Metadata of every mixture, in dataset order (offset fastest, then interval, then pair)."""
    return [
        dict(instrument1=a, instrument2=b, delta_semitone=int(i), delta_time=float(d))
        for (a, b), i, d in itertools.product(pairs, intervals, delays)
    ]


def write_twonote(
    out: str | Path,
    pairs: Sequence[tuple[str, str]] = PAIRS,
    intervals: Sequence[int] = INTERVALS,
    delays: Sequence[float] = DELAYS,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Write the set as a :class:`~dvae_bss.data.PremixedDataset` (``mix.json`` + ``audio/``)."""
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    records = {}
    for i, record in enumerate(grid(pairs, intervals, delays)):
        stems = make_example(
            (record["instrument1"], record["instrument2"]),
            record["delta_semitone"],
            record["delta_time"],
            sample_rate,
        )
        sf.write(out / "audio" / f"mix_{i}.wav", stems.sum(0), sample_rate, subtype="FLOAT")
        sf.write(out / "audio" / f"stems_{i}.wav", stems.T, sample_rate, subtype="FLOAT")
        records[str(i)] = dict(filename=f"mix_{i}.wav", stemname=f"stems_{i}.wav", **record)
    (out / "mix.json").write_text(json.dumps(records, indent=2))
    return out


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of ``dvae-make-twonote``."""
    p = argparse.ArgumentParser(prog="dvae-make-twonote", description=__doc__.split("\n")[0])
    p.add_argument("-o", "--out", type=Path, default=Path("data/TwoNote"))
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-make-twonote``."""
    args = build_parser().parse_args(argv)
    write_twonote(args.out)
    print(f"wrote {len(grid())} mixtures to {args.out}")


if __name__ == "__main__":
    main()
