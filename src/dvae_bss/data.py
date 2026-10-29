"""Audio I/O and datasets.

Training needs only a pool of isolated sounds (for example single instrument notes).
:class:`NoteSequenceDataset` strings random sounds into long, sparse streams and
:func:`mix_batch` adds several streams together, so the model is trained on mixtures
without ever seeing the sources separately. Evaluation uses pre-mixed sets that do come with
their sources (:class:`PremixedDataset`).
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from dvae_bss.audio import db2mag

AUDIO_SUFFIXES = (".wav", ".flac", ".ogg")


def read_audio(path: str | Path, sample_rate: int | None = 16000) -> torch.Tensor:
    """Read an audio file as a mono float32 waveform.

    Args:
        path: Any format supported by libsndfile (wav, flac, ogg).
        sample_rate: Resample to this rate; ``None`` keeps the file's rate.

    Returns:
        Waveform of shape ``(L,)``, channels averaged.
    """
    data, file_rate = sf.read(path, dtype="float32", always_2d=True)
    wave = data.mean(axis=1)
    if sample_rate is not None and file_rate != sample_rate:
        divisor = math.gcd(sample_rate, file_rate)
        wave = resample_poly(wave, sample_rate // divisor, file_rate // divisor).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(wave))


def collect_files(source: str | Path) -> list[Path]:
    """List audio files from a folder (searched recursively) or from a file list.

    A file list is a text file with one path per line, relative to the list's folder. Extra
    tab-separated columns are ignored.
    """
    source = Path(source).expanduser()
    if source.is_dir():
        files = sorted(p for p in source.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    elif source.is_file():
        lines = [line.split("\t")[0].strip() for line in source.read_text().splitlines()]
        files = [source.parent / line for line in lines if line]
    else:
        raise FileNotFoundError(source)
    if not files:
        raise ValueError(f"no audio files found in {source}")
    return files


def trim_silence(wave: torch.Tensor, top_db: float = 70.0, frame: int = 512) -> torch.Tensor:
    """Remove leading and trailing frames more than ``top_db`` below the loudest frame."""
    if len(wave) < frame:
        return wave
    rms = wave[: len(wave) // frame * frame].view(-1, frame).pow(2).mean(-1).sqrt()
    audible = (rms > rms.max() * db2mag(torch.tensor(-top_db))).nonzero()
    if len(audible) == 0:
        return wave
    return wave[int(audible[0]) * frame : (int(audible[-1]) + 1) * frame]


class NoteSequenceDataset(Dataset):
    """Random streams of isolated sounds separated by random silences.

    Item ``n`` starts with file ``n`` and keeps appending randomly chosen files until
    ``duration`` seconds are filled. Silences, trimming and the choice of files are random on
    every access, so epochs differ.

    Args:
        files: Audio files, each holding one isolated sound.
        duration: Length of each stream (seconds).
        sample_rate: Sample rate of the streams.
        max_silence: Silence before each sound is uniform in ``[0, max_silence]`` seconds.
        trim_start: Up to this many seconds are cut randomly from the start of each sound.
        trim_end: Likewise for the end of each sound.
        max_sounds: Upper bound on the number of sounds in a stream.
        normalize: Scale every sound to unit peak.
        augment_levels: Apply a random gain in ``[-30, 0]`` dB to each sound.
        remove_silence: Trim digital silence off both ends of every sound.
        seed: If given, item ``n`` is deterministic (used for validation); otherwise random.
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        duration: float = 8.0,
        sample_rate: int = 16000,
        max_silence: float = 0.5,
        trim_start: float = 0.0,
        trim_end: float = 0.0,
        max_sounds: int = 20,
        normalize: bool = True,
        augment_levels: bool = False,
        remove_silence: bool = False,
        seed: int | None = None,
    ) -> None:
        self.files = [Path(f) for f in files]
        self.sample_rate = sample_rate
        self.n_samples = int(duration * sample_rate)
        self.max_silence = int(max(0.0, max_silence) * sample_rate) + 1
        self.max_trim_start = int(max(0.0, trim_start) * sample_rate) + 1
        self.max_trim_end = int(max(0.0, trim_end) * sample_rate) + 1
        self.max_sounds = max_sounds
        self.normalize = normalize
        self.augment_levels = augment_levels
        self.remove_silence = remove_silence
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, index: int, rng: np.random.Generator) -> torch.Tensor:
        wave = read_audio(self.files[index], self.sample_rate)
        if self.normalize and wave.abs().max() > 1e-3:
            wave = wave / wave.abs().max()
        if self.remove_silence:
            wave = trim_silence(wave)
        if self.augment_levels:
            wave = wave * db2mag(torch.tensor(-30.0 * rng.random(), dtype=torch.float32))
        return wave

    def __getitem__(self, index: int) -> torch.Tensor:
        rng = np.random.default_rng(None if self.seed is None else [self.seed, index])
        stream = torch.zeros(self.n_samples)
        # Write sounds one after another into a silent buffer, each after a random gap, until
        # the buffer is full. A stream holds sounds of one source in sequence, never overlapping.
        position = int(rng.integers(self.max_silence))
        file_index = index
        for _ in range(self.max_sounds):
            if position >= self.n_samples:
                break
            wave = self._load(file_index, rng)
            length = min(len(wave), self.n_samples - position)
            start = min(int(rng.integers(self.max_trim_start)), length // 2)
            length -= start
            end = min(int(rng.integers(self.max_trim_end)), length // 2)
            length -= end
            stream[position : position + length] = wave[start : start + length]
            position += length + int(rng.integers(self.max_silence))
            file_index = int(rng.integers(len(self.files)))
        return stream


def mix_batch(
    streams: torch.Tensor,
    num_sources: int = 2,
    random_count: bool = True,
    average: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create mixtures by adding differently shifted copies of a batch of streams.

    Source ``k`` of example ``i`` is stream ``i - k`` of the batch, so every stream is used
    in several mixtures and the batch must hold at least ``num_sources`` streams.

    Args:
        streams: ``(B, L)`` streams.
        num_sources: Streams per mixture.
        random_count: Randomly drop each source after the first, so mixtures contain between
            one and ``num_sources`` active sources.
        average: Divide by ``num_sources`` instead of summing.
        generator: Random generator for the dropping.

    Returns:
        Mixtures ``(B, L)`` and their sources ``(B, num_sources, L)``.
    """
    batch = streams.shape[0]
    if batch < num_sources:
        raise ValueError(f"need a batch of at least {num_sources} streams, got {batch}")
    # Rolling the batch pairs every stream with different neighbours, so a batch of B streams
    # gives B mixtures without loading any extra audio.
    sources = torch.stack([streams.roll(k, dims=0) for k in range(num_sources)], dim=1)
    if random_count and num_sources > 1:
        # Some mixtures then hold fewer sources, so the model also learns to leave outputs
        # silent.
        keep = torch.randint(0, 2, (batch, num_sources - 1, 1), generator=generator)
        sources = torch.cat([sources[:, :1], sources[:, 1:] * keep], dim=1)
    if average:
        sources = sources / num_sources
    return sources.sum(1), sources


class PremixedDataset(Dataset):
    """Evaluation set of mixtures with known sources.

    The folder contains ``mix.json`` and an ``audio/`` folder. ``mix.json`` maps an index to
    a record with the mixture file name (``filename``), the file holding the stacked sources
    (``stemname``; one channel per source) and any metadata, for example ``instrument1``,
    ``instrument2``, ``pitch1``, ``pitch2``, ``delta_time``. Items are dictionaries with
    ``mix`` ``(L,)``, ``stems`` ``(K, L)`` and the metadata.
    """

    def __init__(self, root: str | Path, sample_rate: int | None = 16000) -> None:
        self.root = Path(root)
        self.sample_rate = sample_rate
        records = json.loads((self.root / "mix.json").read_text())
        self.records: list[dict[str, Any]] = (
            list(records.values()) if isinstance(records, dict) else list(records)
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        audio = self.root / "audio"
        mix = read_audio(audio / record.pop("filename"), self.sample_rate)
        stem_data, stem_rate = sf.read(
            audio / record.pop("stemname"), dtype="float32", always_2d=True
        )
        stems = torch.from_numpy(stem_data.T.copy())
        if self.sample_rate is not None and stem_rate != self.sample_rate:
            divisor = math.gcd(self.sample_rate, stem_rate)
            resampled = resample_poly(
                stems.numpy(), self.sample_rate // divisor, stem_rate // divisor, axis=-1
            )
            stems = torch.from_numpy(resampled.astype(np.float32))
        return {"mix": mix, "stems": stems, **record}
