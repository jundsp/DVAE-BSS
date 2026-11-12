from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dvae_bss.data import PremixedDataset
from dvae_bss.datagen import twonote

AUDIO = Path(__file__).resolve().parents[1] / "docs" / "audio"

# Items of the original TwoNote set kept for the demo page, with their metadata.
DEMO = {
    222: ("forced", "free", 0, 0.0675),
    357: ("forced", "free", 8, 0.005),
    365: ("forced", "free", 8, 0.505),
    619: ("forced_vibrato", "free", 10, 0.4425),
}


def log_spectrogram(wave: np.ndarray) -> np.ndarray:
    frames = np.lib.stride_tricks.sliding_window_view(wave, 512)[::128] * np.hanning(512)
    return 20 * np.log10(np.abs(np.fft.rfft(frames)) + 1e-6)


def test_grid_has_dissertation_size_and_order():
    grid = twonote.grid()
    assert len(grid) == 6 * 13 * 17
    for index, (a, b, interval, delay) in DEMO.items():
        record = grid[index]
        assert (record["instrument1"], record["instrument2"]) == (a, b)
        assert record["delta_semitone"] == interval and record["delta_time"] == delay


@pytest.mark.parametrize("kind", twonote.NOTE_TYPES)
def test_note_is_finite_and_pitched(kind):
    wave = twonote.synth_note(kind, 440.0)
    assert len(wave) == 2 * 16000 and np.isfinite(wave).all()
    spectrum = np.abs(np.fft.rfft(wave[1600:9600] * np.hanning(8000)))
    assert abs(np.argmax(spectrum) * 2.0 - 440.0) < 10.0
    with pytest.raises(ValueError):
        twonote.synth_note("kazoo", 440.0)


@pytest.mark.skipif(not AUDIO.exists(), reason="demo audio missing")
@pytest.mark.parametrize("index", list(DEMO))
def test_matches_original_items(index):
    record = twonote.grid()[index]
    stems = twonote.make_example(
        (record["instrument1"], record["instrument2"]),
        record["delta_semitone"],
        record["delta_time"],
    )
    for k in range(2):
        original, _ = sf.read(AUDIO / f"ex{index}_source{k + 1}.wav")
        ours = stems[k, : len(original)]
        ref, est = log_spectrogram(original), log_spectrogram(ours)
        loud = ref > ref.max() - 60
        assert np.mean(np.abs(est - ref)[loud]) < 1.5
        assert np.isclose(np.std(ours), np.std(original), rtol=0.01)


def test_writes_premixed_set(tmp_path):
    twonote.write_twonote(tmp_path, pairs=twonote.PAIRS[:2], intervals=(0, 7), delays=(0.005, 0.5))
    data = PremixedDataset(tmp_path)
    item = data[3]
    assert len(data) == 8 and item["stems"].shape == (2, int(3.2 * 16000))
    assert np.allclose(item["stems"].sum(0), item["mix"], atol=1e-6)
    assert item["delta_semitone"] == 7 and item["delta_time"] == 0.5
