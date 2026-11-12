"""Synthetic instrument notes, for training and testing without a licensed dataset.

This dataset is new in this repository and is not part of the dissertation, which used RWC
for training and RWC, NSynth and TwoNote for testing. It lets anyone train and evaluate the
model without downloading a licensed dataset. Nothing is stored in the repository: the
command below regenerates the same notes from the seed.

Each note is a sum of sinusoidal partials shaped like one of six instrument families
(:data:`FAMILIES`): plucked strings, bowed strings, reeds, brass, bells and organs. Pitch,
duration, spectral slope, a random formant envelope, vibrato, inharmonicity and decay are drawn
at random, so every note differs. The layout matches what the other tools expect: one folder of
notes per split, organised by family and named ``<midi>_<family>_<index>.wav``, and a pre-mixed
test set (``mix.json`` + ``audio/``) for ``dvae-evaluate``.

    dvae-make-synthetic --out data/synthetic
    dvae-train --train-data data/synthetic/train --val-data data/synthetic/val --out runs/syn
    dvae-evaluate -m runs/syn/model.pt -d data/synthetic/test -o results/syn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

from dvae_bss.data import collect_files, write_premixed_set

SAMPLE_RATE = 16000
MAX_PARTIALS = 40

#: Family name -> MIDI pitch range (inclusive).
FAMILIES: dict[str, tuple[int, int]] = {
    "pluck": (40, 84),
    "bowed": (43, 96),
    "reed": (50, 91),
    "brass": (40, 82),
    "bell": (60, 96),
    "organ": (36, 96),
}

# Frequency ratios of a church bell's partials (hum, prime, tierce, quint, nominal, ...).
BELL_RATIOS = np.array([0.5, 1.0, 1.183, 1.506, 2.0, 2.514, 2.662, 3.011, 4.166, 5.433, 6.796])
ORGAN_RATIOS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0])


def midi_to_hz(midi: float) -> float:
    """Frequency of a MIDI pitch, A4 = 440 Hz."""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def _envelope(n: int, sr: int, attack: float, release: float, decay: float = 0.0) -> np.ndarray:
    """Attack ramp, exponential decay at ``decay`` (1/s), and a raised-cosine release."""
    t = np.arange(n) / sr
    env = np.exp(-decay * t)
    n_attack = min(max(1, int(attack * sr)), n)
    env[:n_attack] *= np.sin(0.5 * np.pi * np.arange(n_attack) / n_attack) ** 2
    n_release = min(max(1, int(release * sr)), n - n_attack)
    env[n - n_release :] *= np.cos(0.5 * np.pi * np.arange(n_release) / n_release) ** 2
    return env


def _formant_gain(freqs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A smooth random spectral envelope: one or two resonances on a log-frequency axis."""
    log_f = np.log2(np.maximum(freqs, 20.0))
    gain = np.ones_like(freqs)
    for _ in range(int(rng.integers(1, 3))):
        centre = rng.uniform(np.log2(300.0), np.log2(4000.0))
        width = rng.uniform(0.4, 1.2)
        gain += rng.uniform(0.5, 3.0) * np.exp(-0.5 * ((log_f - centre) / width) ** 2)
    return gain


def synth_note(
    family: str, midi: float, rng: np.random.Generator, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """One note of an instrument family, peak-normalised to 0.9.

    Args:
        family: One of :data:`FAMILIES`.
        midi: MIDI pitch; fractional values give microtonal pitches.
        rng: Random generator for every other property of the note.
        sample_rate: Sample rate (Hz).

    Returns:
        ``float32`` waveform.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {sorted(FAMILIES)}")
    f0 = midi_to_hz(midi) * 2.0 ** (rng.normal(0.0, 5.0) / 1200.0)  # ~5 cents of detuning
    sustained = family in ("bowed", "reed", "brass", "organ")
    duration = rng.uniform(0.4, 2.0) if sustained else rng.uniform(0.6, 2.5)
    n = int(duration * sample_rate)
    t = np.arange(n) / sample_rate
    nyquist = 0.45 * sample_rate

    # Partial frequency ratios, amplitudes (P,) and per-partial decay rates (1/s).
    h = np.arange(1, MAX_PARTIALS + 1, dtype=float)
    decay = np.zeros(MAX_PARTIALS)
    if family == "pluck":
        stiffness = rng.uniform(0.0, 3e-4)
        ratios = h * np.sqrt(1.0 + stiffness * h**2)
        position = rng.uniform(0.08, 0.3)  # where the string is plucked
        amps = np.abs(np.sin(np.pi * h * position)) / h ** rng.uniform(0.8, 1.5)
        decay = rng.uniform(1.0, 3.0) * (1.0 + rng.uniform(0.1, 0.4) * h)
    elif family == "bowed":
        ratios = h
        amps = 1.0 / h ** rng.uniform(0.9, 1.3)
    elif family == "reed":
        ratios = h
        amps = np.where(h % 2 == 1, 1.0, rng.uniform(0.05, 0.3)) / h ** rng.uniform(0.8, 1.4)
    elif family == "brass":
        ratios = h
        amps = 1.0 / h ** rng.uniform(0.6, 1.0)
    elif family == "bell":
        ratios = BELL_RATIOS * (1.0 + rng.normal(0.0, 0.01, len(BELL_RATIOS)))
        amps = rng.uniform(0.3, 1.0, len(ratios)) / np.sqrt(ratios)
        decay = rng.uniform(0.8, 2.0) * ratios
    else:  # organ
        ratios = ORGAN_RATIOS
        amps = rng.uniform(0.0, 1.0, len(ratios)) * (rng.random(len(ratios)) < 0.7)
        amps[0] = max(amps[0], 0.5)
    decay = decay[: len(ratios)]
    keep = ratios * f0 < nyquist
    ratios, amps, decay = ratios[keep], amps[keep], decay[keep]
    amps = amps * _formant_gain(ratios * f0, rng)

    # Instantaneous frequency of the fundamental: vibrato for bowed strings and reeds.
    pitch = np.full(n, f0)
    if family in ("bowed", "reed") and rng.random() < 0.8:
        rate, depth = rng.uniform(4.5, 6.5), rng.uniform(5.0, 30.0)  # Hz, cents
        onset = np.clip(t / rng.uniform(0.2, 0.5), 0.0, 1.0)  # vibrato fades in
        pitch = f0 * 2.0 ** (onset * depth * np.sin(2 * np.pi * rate * t) / 1200.0)
    phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
    start = rng.uniform(0, 2 * np.pi, len(ratios))[:, None]
    partials = np.sin(ratios[:, None] * phase[None] + start)  # (P, n)

    # Amplitude envelopes, one per partial.
    if family in ("pluck", "bell"):
        env = np.exp(-decay[:, None] * t[None])
        env *= _envelope(n, sample_rate, attack=0.003, release=0.05)[None]
    else:
        attack = {"bowed": (0.06, 0.2), "reed": (0.02, 0.08), "brass": (0.03, 0.1),
                  "organ": (0.005, 0.02)}[family]  # fmt: skip
        main = _envelope(n, sample_rate, rng.uniform(*attack), rng.uniform(0.05, 0.2),
                         decay=rng.uniform(0.0, 0.6))  # fmt: skip
        env = np.broadcast_to(main, (len(ratios), n))
        if family == "brass":  # brighter when louder: upper partials enter later
            env = main[None] ** (1.0 + 0.15 * (np.arange(len(ratios))[:, None]))
    wave = (amps[:, None] * env * partials).sum(0)

    if family == "pluck":  # a short noise burst on the attack
        burst = rng.normal(0.0, 1.0, min(n, int(0.01 * sample_rate)))
        wave[: len(burst)] += 0.1 * np.abs(wave).max() * burst * np.linspace(1, 0, len(burst))
    if family in ("bowed", "reed"):  # bow or breath noise following the envelope
        wave += rng.uniform(0.002, 0.01) * np.abs(wave).max() * rng.normal(0, 1, n) * env[0]

    return (0.9 * wave / max(np.abs(wave).max(), 1e-9)).astype(np.float32)


def write_notes(
    folder: str | Path, num_notes: int, seed: int = 0, sample_rate: int = SAMPLE_RATE
) -> Path:
    """Write ``num_notes`` random notes to ``folder/<family>/<midi>_<family>_<index>.wav``.

    Families are used in turn and pitches are uniform in each family's range.
    """
    folder = Path(folder)
    rng = np.random.default_rng(seed)
    families = list(FAMILIES)
    for i in range(num_notes):
        family = families[i % len(families)]
        midi = int(rng.integers(FAMILIES[family][0], FAMILIES[family][1] + 1))
        path = folder / family / f"{midi:03d}_{family}_{i:05d}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, synth_note(family, midi, rng, sample_rate), sample_rate, subtype="PCM_16")
    return folder


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of ``dvae-make-synthetic``."""
    p = argparse.ArgumentParser(prog="dvae-make-synthetic", description=__doc__.split("\n")[0])
    p.add_argument("-o", "--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--train", type=int, default=3000, help="training notes")
    p.add_argument("--val", type=int, default=300, help="validation notes")
    p.add_argument("--test", type=int, default=300, help="notes for the test mixtures")
    p.add_argument("--test-mixtures", type=int, default=500, help="two-note test mixtures")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-make-synthetic``: train/, val/, test_notes/ and test/."""
    args = build_parser().parse_args(argv)
    splits = {"train": args.train, "val": args.val, "test_notes": args.test}
    for offset, (name, count) in enumerate(splits.items()):
        write_notes(args.out / name, count, seed=args.seed * 10 + offset)
        print(f"wrote {count} notes to {args.out / name}")
    write_premixed_set(collect_files(args.out / "test_notes"), args.out / "test",
                       args.test_mixtures, seed=args.seed)  # fmt: skip
    print(f"wrote {args.test_mixtures} mixtures to {args.out / 'test'}")


if __name__ == "__main__":
    main()
