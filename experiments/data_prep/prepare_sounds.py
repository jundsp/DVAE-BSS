"""Convert a folder of recordings to mono 16 kHz wav files, optionally trimmed or split.

Training wants isolated sounds (one note or event per file). This script resamples and
downmixes everything under ``--input`` into ``--output`` (same folder structure), can trim
silence off the ends, and can cut long recordings into separate sounds wherever the signal
stays quiet.

    python experiments/data_prep/prepare_sounds.py --input raw --output data/notes --trim-db 70
    python experiments/data_prep/prepare_sounds.py --input raw --output data/notes --split
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

from dvae_bss.data import AUDIO_SUFFIXES, read_audio, trim_silence


def split_recording(
    wave: np.ndarray, sample_rate: int, min_duration: float = 0.1, threshold_db: float = -100.0,
    smoothing: int = 500,
) -> list[np.ndarray]:  # fmt: skip
    """Cut a recording into the segments where its smoothed envelope exceeds ``threshold_db``.

    Args:
        wave: Mono waveform.
        sample_rate: Sample rate in Hz.
        min_duration: Discard segments shorter than this (seconds).
        threshold_db: Envelope level (dB relative to the peak) below which the signal is quiet.
        smoothing: Length of the moving average that forms the envelope (samples).

    Returns:
        The audible segments in order.
    """
    peak = np.abs(wave).max()
    if peak == 0:
        return []
    envelope = lfilter(np.ones(smoothing) / smoothing, 1, np.abs(wave) / peak)
    quiet = (20 * np.log10(np.maximum(envelope, 1e-12)) < threshold_db).astype(int)
    quiet[0] = quiet[-1] = 1
    change = np.diff(quiet)
    starts, stops = np.nonzero(change == -1)[0], np.nonzero(change == 1)[0]
    return [
        wave[a:b] for a, b in zip(starts, stops, strict=False) if b - a > min_duration * sample_rate
    ]


def main(argv: list[str] | None = None) -> None:
    """Convert and write the files."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument(
        "--trim-db", type=float, help="trim ends quieter than this many dB below peak"
    )
    parser.add_argument(
        "--split", action="store_true", help="split recordings into separate sounds"
    )
    parser.add_argument("--split-db", type=float, default=-100.0)
    parser.add_argument("--min-duration", type=float, default=0.1, help="shortest sound kept (s)")
    args = parser.parse_args(argv)

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    if not files:
        raise SystemExit(f"no audio files under {args.input}")
    for n, path in enumerate(files, start=1):
        wave = read_audio(path, args.sample_rate)
        if args.trim_db is not None:
            wave = trim_silence(wave, args.trim_db)
        target = args.output / path.relative_to(args.input).with_suffix("")
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.split:
            pieces = split_recording(
                wave.numpy(), args.sample_rate, args.min_duration, args.split_db
            )
        else:
            pieces = [wave.numpy()]
        for i, piece in enumerate(pieces, start=1):
            name = (
                target.with_name(f"{target.name}_{i}.wav")
                if args.split
                else target.with_suffix(".wav")
            )
            sf.write(name, np.clip(piece, -1, 1), args.sample_rate)
        print(f"[{n}/{len(files)}] {path.name}: {len(pieces)} file(s)")


if __name__ == "__main__":
    main()
