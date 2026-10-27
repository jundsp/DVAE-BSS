import csv
import json
import os
import subprocess
import sys

import pytest
import soundfile as sf
import torch

from dvae_bss import load_checkpoint
from dvae_bss.cli import evaluate, separate, train
from synthetic import SAMPLE_RATE, TRAIN, note, write_premixed


def test_train_writes_outputs(run):
    out = run / "run"
    assert {p.name for p in out.iterdir()} == {"config.json", "log.csv", "model.pt", "resume.pt"}
    rows = list(csv.DictReader((out / "log.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert all(float(r["train_loss"]) == float(r["train_loss"]) for r in rows)  # not NaN
    model, stft = load_checkpoint(out / "model.pt")
    assert model.num_sources == 2 and stft.sample_rate == SAMPLE_RATE
    assert json.loads((out / "config.json").read_text())["model"]["latent_dim"] == 4


def test_resume_continues_the_log(run, tmp_path):
    out = tmp_path / "resumed"
    for name in ("log.csv", "model.pt", "resume.pt", "config.json"):
        out.mkdir(exist_ok=True)
        (out / name).write_bytes((run / "run" / name).read_bytes())
    train.main(["--train-data", str(run / "notes"), "--out", str(out), "--epochs", "3",
                "--resume", str(out / "resume.pt"), *TRAIN])  # fmt: skip
    assert [r["epoch"] for r in csv.DictReader((out / "log.csv").open())] == ["1", "2", "3"]


def test_separate(run, tmp_path):
    mixture = note(262) + note(330)
    sf.write(tmp_path / "mix.wav", mixture, SAMPLE_RATE)
    separate.main([str(tmp_path / "mix.wav"), "-m", str(run / "run" / "model.pt"),
                   "-o", str(tmp_path / "out"), "--device", "cpu", "--plot"])  # fmt: skip
    outputs = sorted((tmp_path / "out").glob("mix_source*.wav"))
    assert len(outputs) == 2 and (tmp_path / "out" / "mix.png").exists()
    estimate, rate = sf.read(outputs[0])
    assert rate == SAMPLE_RATE and len(estimate) == len(mixture)


def test_separate_without_model_explains(tmp_path, monkeypatch):
    monkeypatch.setattr(separate, "PRETRAINED", tmp_path / "missing.pt")
    with pytest.raises(SystemExit):
        separate.main([str(tmp_path / "x.wav")])


def test_evaluate(run, tmp_path):
    data = write_premixed(tmp_path / "set", 4)
    evaluate.main(["-m", str(run / "run" / "model.pt"), "-d", str(data), "-o", str(tmp_path / "ev"),
                   "--batch-size", "2", "--device", "cpu", "--save-audio"])  # fmt: skip
    results = json.loads((tmp_path / "ev" / "results.json").read_text())
    assert len(results) == 8  # 4 mixtures x 2 sources
    first = results["0"]
    assert {
        "sdr",
        "sir",
        "sar",
        "place",
        "instrument",
        "pitch",
        "delta_pitch",
        "delta_time",
    } <= first.keys()
    assert first["instrument"] == "a" and results["1"]["instrument"] == "b"
    assert {results["0"]["place"], results["1"]["place"]} == {0, 1}
    summary = json.loads((tmp_path / "ev" / "summary.json").read_text())
    assert summary["n_sources"] == 8 and set(summary) >= {"sdr", "sir", "sar"}
    assert len(list((tmp_path / "ev").glob("estimates_*.wav"))) == 4


def test_evaluate_max_batches(run, tmp_path):
    data = write_premixed(tmp_path / "set", 4)
    evaluate.main(["-m", str(run / "run" / "model.pt"), "-d", str(data), "-o", str(tmp_path / "ev"),
                   "--batch-size", "2", "--max-batches", "1", "--device", "cpu"])  # fmt: skip
    assert len(json.loads((tmp_path / "ev" / "results.json").read_text())) == 4


def test_train_distributed_on_cpu(run, tmp_path):
    """Two gloo processes via torchrun, as in the Slurm example."""
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
               "-m", "dvae_bss.cli.train", "--train-data", str(run / "notes"),
               "--out", str(tmp_path / "ddp"), "--epochs", "1", *TRAIN]  # fmt: skip
    result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert (tmp_path / "ddp" / "model.pt").exists()
    assert len(list(csv.DictReader((tmp_path / "ddp" / "log.csv").open()))) == 1


def test_cli_help_runs():
    for module in ("train", "separate", "evaluate"):
        result = subprocess.run([sys.executable, "-m", f"dvae_bss.cli.{module}", "--help"],
                                capture_output=True, text=True)  # fmt: skip
        assert result.returncode == 0 and "usage" in result.stdout.lower()


def test_loaded_model_is_deterministic(run):
    model, _ = load_checkpoint(run / "run" / "model.pt")
    x = torch.rand(1, 257, 257)
    model.eval()
    torch.testing.assert_close(
        model(x, stochastic=False)["recon_x"], model(x, stochastic=False)["recon_x"]
    )
