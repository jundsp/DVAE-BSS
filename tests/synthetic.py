"""Tiny synthetic data for the tests: decaying harmonic notes and pre-mixed pairs."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

CONFIGS = Path(__file__).parents[1] / "configs"

TINY_MODEL = ["layer_channels=[4, 8, 8]", "num_sources=2", "latent_dim=4", "rnn_dim=8"]


def note(pitch_hz: float, seconds: float = 0.5, rng=None) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    wave = sum(np.sin(2 * np.pi * pitch_hz * h * t) / h for h in (1, 2, 3))
    return (wave * np.exp(-4 * t)).astype(np.float32) * 0.3


def write_notes(folder: Path, n: int = 12) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        sf.write(folder / f"note{i:02d}.wav", note(200 * 2 ** (i / 12)), SAMPLE_RATE)
    return folder


def write_premixed(folder: Path, n: int = 4) -> Path:
    """Folder with mix.json and audio/ holding ``n`` two-note mixtures."""
    audio = folder / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    records = {}
    for i in range(n):
        low, high = 220 * 2 ** (i / 12), 330 * 2 ** (i / 12)
        stems = np.stack([note(low), note(high)], axis=-1)
        sf.write(audio / f"mix{i}.wav", stems.sum(-1), SAMPLE_RATE)
        sf.write(audio / f"stems{i}.wav", stems, SAMPLE_RATE)
        records[str(i)] = dict(
            filename=f"mix{i}.wav", stemname=f"stems{i}.wav", instrument1="a", instrument2="b",
            pitch1=low, pitch2=high, delta_time=0.0,
        )  # fmt: skip
    (folder / "mix.json").write_text(json.dumps(records))
    return folder


# dvae-train arguments for a tiny, fast run on the synthetic notes: the shipped config, then
# overrides. Add train_data=..., out=... and more overrides after them.
TRAIN = [
    str(CONFIGS / "rwc_k5.yaml"), "duration=2", "batch_size=4", "num_workers=0", "device=cpu",
    "log_interval=1", *TINY_MODEL,
]  # fmt: skip
