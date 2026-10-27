"""The scripts in experiments/ run end to end on tiny synthetic data."""

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from dvae_bss.cli import evaluate
from synthetic import SAMPLE_RATE, TRAIN, note, write_notes, write_premixed

EXPERIMENTS = Path(__file__).parents[1] / "experiments"


def script(relative: str):
    """Import an experiment script by path."""
    path = EXPERIMENTS / relative
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def test_set(tmp_path_factory):
    return write_premixed(tmp_path_factory.mktemp("premixed"), 4)


@pytest.fixture(scope="module")
def results(run, test_set, tmp_path_factory):
    """results.json of dvae-evaluate with instrument and pitch metadata."""
    out = tmp_path_factory.mktemp("eval")
    evaluate.main(["-m", str(run / "run" / "model.pt"), "-d", str(test_set), "-o", str(out),
                   "--device", "cpu"])  # fmt: skip
    return out / "results.json"


def means(folder: Path) -> dict:
    return json.loads((folder / "summary.json").read_text())


# ----------------------------------------------------------------------------- baselines
def test_ideal_masks_beat_the_mixture(test_set, tmp_path):
    script("baselines/ideal_masks.py").main(["-d", str(test_set), "-o", str(tmp_path)])
    mix, irm, ibm = (means(tmp_path / name)["sdr"]["mean"] for name in ("mix", "irm", "ibm"))
    assert abs(mix) < 3  # two equally loud sources: the mixture is no better than 0 dB
    assert irm > mix + 5 and ibm > mix + 5
    assert len(json.loads((tmp_path / "irm" / "results.json").read_text())) == 8


def test_ica_baseline_and_missing_nmf(test_set, tmp_path):
    module = script("baselines/ica_nmf.py")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module.main(["-d", str(test_set), "-o", str(tmp_path), "--methods", "ica"])
    assert len(json.loads((tmp_path / "ica" / "results.json").read_text())) == 8
    if importlib.util.find_spec("NMFtoolbox") is None:
        with pytest.raises(SystemExit, match="NMF Toolbox"):
            module.NMFDeconvolution(10, 10)


def test_mixit_loss_picks_the_best_split():
    mixit = script("baselines/mixit.py")
    a, b = torch.rand(2, 1, 4, 5), torch.rand(2, 1, 4, 5)
    targets = torch.cat([a + b, b], dim=1)  # mixture 1 = a + b, mixture 2 = b
    sources = torch.cat([a, b, b / 2, b / 2], dim=1)  # K = 4 outputs
    assert mixit.mixit_loss(sources, targets).item() == pytest.approx(0, abs=1e-12)
    assert mixit.mixit_loss(sources * 2, targets).item() > 0.01
    assert mixit.equal_splits(4).shape == (6, 2)


def test_mixit_train_and_eval(tmp_path, test_set):
    mixit = script("baselines/mixit.py")
    notes = write_notes(tmp_path / "notes")
    model_dir = tmp_path / "mixit"
    mixit.main(
        ["train", "--train-data", str(notes), "--out", str(model_dir), "--epochs", "1", *TRAIN]
    )
    assert (model_dir / "model.pt").exists() and (model_dir / "log.csv").exists()
    mixit.main(["eval", "-m", str(model_dir / "model.pt"), "-d", str(test_set),
                "-o", str(tmp_path / "ev"), "--device", "cpu"])  # fmt: skip
    assert len(json.loads((tmp_path / "ev" / "results.json").read_text())) == 8


# ------------------------------------------------------------------------------- figures
def test_compare_runs(results, tmp_path):
    script("figures/compare_runs.py").main(
        [
            str(results),
            str(results),
            "--labels",
            "a",
            "b",
            "--xlabel",
            "setting",
            "-o",
            str(tmp_path),
        ]
    )
    for name in (
        "boxplot.pdf",
        "boxplot_outline.pdf",
        "boxplot_by_estimate.pdf",
        "lineplot_median.pdf",
        "medians.csv",
    ):
        assert (tmp_path / name).stat().st_size > 0


def test_breakdown(results, tmp_path):
    script("figures/breakdown.py").main([str(results), "-o", str(tmp_path)])
    for name in (
        "boxplot_instrument_sisdr.pdf",
        "heatmap_instrument_pairs.pdf",
        "time_sdr.pdf",
        "pitch_sdr.pdf",
    ):
        assert (tmp_path / name).stat().st_size > 0


def test_qualitative(run, test_set, tmp_path):
    script("figures/qualitative.py").main(
        [
            "-m",
            str(run / "run" / "model.pt"),
            "-d",
            str(test_set),
            "--examples",
            "1",
            "-o",
            str(tmp_path),
        ]
    )
    folder = tmp_path / "example_1"
    assert (folder / "spectrograms.pdf").exists()
    estimates, rate = sf.read(folder / "estimates.wav")
    assert estimates.shape[1] == 2 and rate == SAMPLE_RATE


def test_model_internals(run, tmp_path):
    audio = tmp_path / "mix.wav"
    sf.write(audio, note(262, 1.0) + note(330, 1.0), SAMPLE_RATE)
    script("figures/model_internals.py").main(
        [
            "-m",
            str(run / "run" / "model.pt"),
            "--audio",
            str(audio),
            "-o",
            str(tmp_path / "out"),
            "--samples",
            "2",
        ]  # fmt: skip
    )
    names = {p.name for p in (tmp_path / "out").iterdir()}
    assert {
        "sources.pdf",
        "latent_states.pdf",
        "prior_sample_2.pdf",
        "encoder_features_3.png",
    } <= names
    assert "decoder_features_3.png" in names


# --------------------------------------------------------------------------- data preparation
def test_make_test_set_roundtrip(tmp_path):
    sounds = tmp_path / "sounds"
    for instrument in ("flute", "oboe"):
        write_notes(sounds / instrument, 3)
    script("data_prep/make_test_set.py").main(
        ["--audio", str(sounds), "-o", str(tmp_path / "set"), "-n", "5", "--duration", "1"]
    )
    from dvae_bss.data import PremixedDataset

    dataset = PremixedDataset(tmp_path / "set")
    item = dataset[2]
    assert len(dataset) == 5 and item["stems"].shape == (2, SAMPLE_RATE)
    torch.testing.assert_close(item["mix"], item["stems"].sum(0), atol=1e-5, rtol=0)
    assert item["instrument1"] in {"flute", "oboe"} and 0 <= item["delta_time"] <= 1


def test_prepare_sounds_splits_a_recording(tmp_path):
    gap = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)
    recording = np.concatenate([gap, note(220), gap, note(330), gap])
    (tmp_path / "raw").mkdir()
    sf.write(tmp_path / "raw" / "scale.wav", recording, 44100)  # resampled while converting
    module = script("data_prep/prepare_sounds.py")
    module.main(
        ["-i", str(tmp_path / "raw"), "-o", str(tmp_path / "out"), "--split", "--split-db", "-60"]
    )
    pieces = sorted((tmp_path / "out").glob("scale_*.wav"))
    assert len(pieces) == 2
    assert sf.info(pieces[0]).samplerate == 16000
    module.main(["-i", str(tmp_path / "raw"), "-o", str(tmp_path / "plain"), "--trim-db", "60"])
    assert 0 < sf.info(tmp_path / "plain" / "scale.wav").frames < len(recording) * 16000 // 44100


def test_make_filelists(tmp_path):
    write_notes(tmp_path / "notes", 20)
    script("data_prep/make_filelists.py").main(["--audio", str(tmp_path / "notes"), "--seed", "1"])
    from dvae_bss.data import collect_files

    counts = {n: len(collect_files(tmp_path / "notes" / f"{n}.txt")) for n in ("train", "validate")}
    assert counts == {"train": 18, "validate": 1}
    everything = [
        p
        for n in ("train", "validate", "test")
        for p in (tmp_path / "notes" / f"{n}.txt").read_text().split()
    ]
    assert len(everything) == len(set(everything)) == 20


def test_make_demo_page(run, test_set, tmp_path):
    script("figures/make_demo_page.py").main(
        ["-m", str(run / "run" / "model.pt"), "-d", str(test_set), "--examples", "0", "2",
         "--notes", "", "a remark", "-o", str(tmp_path)]
    )  # fmt: skip
    page = (tmp_path / "index.html").read_text()
    assert page.count("<section>") == 2 and "a remark" in page
    for name in ("ex0_mixture", "ex2_source2", "ex2_estimate1"):
        assert f'src="audio/{name}.wav"' in page and (tmp_path / "audio" / f"{name}.wav").exists()
        assert (tmp_path / "img" / f"{name}.png").exists()
