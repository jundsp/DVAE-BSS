import numpy as np
import pytest
import soundfile as sf
import torch

from dvae_bss.data import (
    NoteSequenceDataset,
    PremixedDataset,
    collect_files,
    mix_batch,
    read_audio,
    trim_silence,
)
from synthetic import SAMPLE_RATE, note, write_notes, write_premixed


def test_collect_files_from_folder_and_list(tmp_path):
    folder = write_notes(tmp_path / "notes", 3)
    assert len(collect_files(folder)) == 3
    listing = tmp_path / "list.txt"
    listing.write_text("notes/note00.wav\tmetadata\nnotes/note02.wav\n\n")
    assert [p.name for p in collect_files(listing)] == ["note00.wav", "note02.wav"]
    with pytest.raises(FileNotFoundError):
        collect_files(tmp_path / "missing")


def test_read_audio_resamples_and_downmixes(tmp_path):
    stereo = np.stack([note(440), note(440)], axis=-1)
    sf.write(tmp_path / "a.wav", stereo, 32000)
    wave = read_audio(tmp_path / "a.wav", SAMPLE_RATE)
    assert wave.ndim == 1 and abs(len(wave) - len(stereo) // 2) <= 1
    assert len(read_audio(tmp_path / "a.wav", None)) == len(stereo)


def test_sequence_dataset(tmp_path):
    files = collect_files(write_notes(tmp_path, 6))
    random_set = NoteSequenceDataset(files, duration=2.0)
    stream = random_set[0]
    assert (
        stream.shape == (2 * SAMPLE_RATE,) and stream.abs().max() <= 1 and stream.abs().max() > 0.1
    )
    assert not torch.equal(random_set[0], random_set[0])
    fixed = NoteSequenceDataset(files, duration=2.0, seed=3)
    assert torch.equal(fixed[1], fixed[1]) and not torch.equal(fixed[1], fixed[2])


def test_sequence_dataset_levels_and_trimming(tmp_path):
    files = collect_files(write_notes(tmp_path, 4))
    quiet = NoteSequenceDataset(files, duration=1.0, augment_levels=True, remove_silence=True,
                                trim_start=0.1, trim_end=0.1, max_silence=0.0)[0]  # fmt: skip
    assert torch.isfinite(quiet).all() and quiet.abs().max() <= 1


def test_trim_silence():
    wave = torch.cat([torch.zeros(4096), torch.ones(2048), torch.zeros(4096)])
    trimmed = trim_silence(wave)
    assert 2048 <= len(trimmed) <= 2048 + 1024 and trimmed.abs().max() == 1


def test_mix_batch():
    streams = torch.arange(1, 5.0).view(4, 1).repeat(1, 6)
    mix, sources = mix_batch(streams, num_sources=2, random_count=False)
    assert mix.shape == (4, 6) and sources.shape == (4, 2, 6)
    assert torch.equal(mix[:, 0], torch.tensor([5.0, 3.0, 5.0, 7.0]))  # x_i + x_(i-1)
    assert torch.equal(mix_batch(streams, 2, False, average=True)[0], mix / 2)
    generator = torch.Generator().manual_seed(0)
    _, dropped = mix_batch(streams.repeat(4, 1), 3, True, generator=generator)
    assert (dropped[:, 1:].abs().sum(-1) == 0).any()  # some sources are switched off
    assert torch.equal(dropped[:, 0], streams.repeat(4, 1))
    with pytest.raises(ValueError):
        mix_batch(streams[:1], num_sources=2)


def test_premixed_dataset(tmp_path):
    folder = write_premixed(tmp_path, 3)
    dataset = PremixedDataset(folder)
    item = dataset[1]
    assert len(dataset) == 3 and item["stems"].shape[0] == 2
    torch.testing.assert_close(item["mix"], item["stems"].sum(0), atol=1e-4, rtol=0)
    assert item["instrument1"] == "a" and "filename" not in item
