import numpy as np
import pytest

from dvae_bss.data import PremixedDataset, collect_files
from dvae_bss.datagen import synthetic


@pytest.mark.parametrize("family", list(synthetic.FAMILIES))
def test_note_is_finite_normalised_and_pitched(family):
    low, high = synthetic.FAMILIES[family]
    for midi in (low, high):
        wave = synthetic.synth_note(family, midi, np.random.default_rng(midi))
        assert wave.dtype == np.float32 and 0.3 * 16000 < len(wave) < 2.6 * 16000
        assert np.isfinite(wave).all() and np.isclose(np.abs(wave).max(), 0.9, atol=1e-3)
    # The strongest bin of a sustained note lies on a partial of its pitch.
    if family in ("bowed", "reed", "brass", "organ"):
        spectrum = np.abs(np.fft.rfft(wave[:8000] * np.hanning(8000)))
        peak = np.argmax(spectrum) * 2.0  # bin spacing 16000 / 8000 Hz
        f0 = synthetic.midi_to_hz(high)
        assert abs(peak / f0 - round(peak / f0)) < 0.05


def test_notes_are_reproducible():
    a = synthetic.synth_note("pluck", 60, np.random.default_rng(1))
    b = synthetic.synth_note("pluck", 60, np.random.default_rng(1))
    assert np.array_equal(a, b)
    with pytest.raises(ValueError):
        synthetic.synth_note("kazoo", 60, np.random.default_rng(1))


def test_cli_writes_splits_and_test_set(tmp_path):
    synthetic.main(["-o", str(tmp_path), "--train", "12", "--val", "6", "--test", "6",
                    "--test-mixtures", "3"])  # fmt: skip
    train = collect_files(tmp_path / "train")
    assert len(train) == 12 and {f.parent.name for f in train} == set(synthetic.FAMILIES)
    assert all(f.name.split("_")[1] == f.parent.name for f in train)
    assert len(collect_files(tmp_path / "val")) == 6
    test = PremixedDataset(tmp_path / "test")
    item = test[0]
    assert len(test) == 3 and item["stems"].shape == (2, 4 * 16000)
    assert np.allclose(item["stems"].sum(0), item["mix"], atol=1e-6)
    assert item["instrument1"] in synthetic.FAMILIES and isinstance(item["pitch1"], int)
