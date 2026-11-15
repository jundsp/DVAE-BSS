import csv
import json
import os
import subprocess
import sys

import pytest
import soundfile as sf
import torch

from dvae_bss import load_checkpoint
from dvae_bss.checkpoint import read_checkpoint
from dvae_bss.cli import evaluate, separate, train
from dvae_bss.config import config_values, load_config
from synthetic import CONFIGS, SAMPLE_RATE, TRAIN, note, write_premixed


def test_train_writes_outputs(run):
    out = run / "run"
    files = {"config.yaml", "train.log", "log.csv", "model.pt", "best.pt", "resume.pt", "plots"}
    assert {p.name for p in out.iterdir()} == files
    assert (out / "plots" / "curves.png").exists()
    assert "epoch 2/2 done" in (out / "train.log").read_text()
    rows = list(csv.DictReader((out / "log.csv").open()))
    assert [r["epoch"] for r in rows] == ["1", "2"]
    assert all(float(r["train_loss"]) == float(r["train_loss"]) for r in rows)  # not NaN
    model, stft = load_checkpoint(out / "model.pt")
    assert model.num_sources == 2 and stft.sample_rate == SAMPLE_RATE
    saved = load_config(out / "config.yaml")  # the options used, overrides included
    assert saved.latent_dim == 4 and saved.epochs == 2 and saved.out == out


def test_mixtures_have_exactly_num_targets_streams_by_default():
    # The pretrained model was trained on mixtures of exactly two streams.
    assert load_config().random_count is False
    assert load_config(CONFIGS / "rwc_k5.yaml").random_count is False
    assert load_config(None, ["random_count=true"]).random_count is True


def copy_run(run, out):
    out.mkdir()
    for name in ("config.yaml", "log.csv", "model.pt", "resume.pt"):
        (out / name).write_bytes((run / "run" / name).read_bytes())
    return out


def test_resume_takes_the_run_folder_and_saves_overrides(run, tmp_path):
    out = copy_run(run, tmp_path / "resumed")
    train.main(["--resume", str(out), "epochs=3"])
    assert [r["epoch"] for r in csv.DictReader((out / "log.csv").open())] == ["1", "2", "3"]
    assert load_config(out / "config.yaml").epochs == 3
    assert "option changed on resume: epochs 2 -> 3" in (out / "train.log").read_text()


def test_resume_from_a_run_with_the_old_config_json(run, tmp_path):
    out = copy_run(run, tmp_path / "old")
    options = config_values(load_config(out / "config.yaml"))
    (out / "config.yaml").unlink()
    (out / "config.json").write_text(json.dumps({"args": {**options, "resume": None}}))
    config, resuming = train.parse_command_line(["--resume", str(out / "resume.pt")])
    assert resuming and config.latent_dim == 4 and config.out == out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["--resume", "{out}", "latent_dim=8"], "checkpoint fixes latent_dim"),
        (["--resume", "{out}", "out=elsewhere"], "out cannot be changed"),
        (["--resume", "{out}", "{config}"], "overrides only"),
        (["{config}", "{config}"], "one config file"),
        ([], "give a config file"),
        (["{config}", "epoch=3"], "unknown option 'epoch'"),
        (["{config}", "encoder=conv"], "encoder must be one of"),
        (["{config}", "lr=fast"], "lr must be float"),
        (["{config}", "epochs"], "one config file"),  # no "=": a second config file
    ],
)
def test_command_line_errors(run, tmp_path, capsys, argv, message):
    out = copy_run(run, tmp_path / "run")
    config = CONFIGS / "rwc_k5.yaml"
    argv = [a.format(out=out, config=config) for a in argv]
    with pytest.raises(SystemExit):
        train.parse_command_line(argv)
    assert message in capsys.readouterr().err


def test_a_new_run_does_not_overwrite_an_old_one(run, tmp_path):
    out = copy_run(run, tmp_path / "taken")
    with pytest.raises(SystemExit, match="--resume"):
        train.main([*TRAIN, f"train_data={run / 'notes'}", f"out={out}"])


def test_config_file_overrides_and_print_config(tmp_path, capsys):
    config = tmp_path / "options.yaml"
    config.write_text("train-data: notes\nepochs: 7\nlr: 1e-3\nlayer_channels: [4, 8]\n")
    options, resuming = train.parse_command_line([str(config), "epochs=3"])
    assert not resuming and options.train_data.name == "notes"
    assert options.epochs == 3  # the override wins
    assert options.lr == 1e-3 and options.layer_channels == [4, 8]  # "1e-3" is text to YAML
    with pytest.raises(SystemExit):
        train.parse_command_line([str(config), "--print-config"])
    printed = capsys.readouterr().out
    assert "epochs: 7" in printed and "layer_channels: [4, 8]" in printed
    # What --print-config writes is a config file that gives the same options back.
    (tmp_path / "printed.yaml").write_text(printed)
    assert load_config(tmp_path / "printed.yaml") == load_config(config)


def test_shipped_configs_parse():
    for path in CONFIGS.glob("*.yaml"):
        assert train.parse_command_line([str(path)])[0].train_data is not None


def test_validation_scores_plots_audio_and_tensorboard(run, tmp_path):
    pytest.importorskip("tensorboard")
    out = tmp_path / "monitored"
    notes = run / "notes"
    train.main([*TRAIN, f"train_data={notes}", f"val_data={notes}", f"out={out}", "epochs=1",
                "tensorboard=true"])  # fmt: skip
    (row,) = csv.DictReader((out / "log.csv").open())
    assert all(float(row[k]) == float(row[k]) for k in ("val_loss", "val_sdr", "lr"))
    assert (out / "plots" / "examples" / "epoch_0001.png").exists()
    names = {p.name for p in (out / "samples").iterdir()}
    assert {"mixture.wav", "source1.wav", "output1.wav", "reconstruction.wav"} <= names
    assert list((out / "tensorboard").glob("events.out.tfevents.*"))


def test_early_stopping(run, tmp_path):
    # With a huge min_delta nothing after epoch 1 counts as progress.
    out = tmp_path / "stopped"
    train.main([*TRAIN, f"train_data={run / 'notes'}", f"out={out}", "epochs=5", "patience=2",
                "min_delta=1e9"])  # fmt: skip
    assert [r["epoch"] for r in csv.DictReader((out / "log.csv").open())] == ["1", "2", "3"]
    assert load_checkpoint(out / "best.pt")[0].num_sources == 2


def test_resume_with_another_monitor_restarts_early_stopping(run, tmp_path):
    out = tmp_path / "switched"
    notes = run / "notes"
    data = [f"train_data={notes}", f"val_data={notes}", f"out={out}"]
    train.main([*TRAIN, *data, "epochs=1", "monitor=val_sdr", "lr_patience=3"])
    train.main(["--resume", str(out), "epochs=2", "monitor=val_loss"])
    assert "monitor changed from val_sdr to val_loss" in (out / "train.log").read_text()
    state = torch.load(out / "resume.pt", weights_only=True)["extra"]
    assert state["monitor"] == "val_loss" and state["scheduler"]["mode"] == "min"
    assert set(read_checkpoint(out / "best.pt")[2]) == {"epoch", "val_loss"}


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
               "-m", "dvae_bss.cli.train", *TRAIN, f"train_data={run / 'notes'}",
               f"out={tmp_path / 'ddp'}", "epochs=1"]  # fmt: skip
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


def test_sound_limits_pick_a_fixed_subset(run):
    notes = run / "notes"
    config = load_config(None, [f"train_data={notes}", f"val_data={notes}",
                                "max_train_sounds=5", "max_val_sounds=3"])  # fmt: skip
    train_files, val_files = train.split_files(config)
    assert len(train_files) == 5 and len(val_files) == 3
    assert train.split_files(config) == (train_files, val_files)  # same seed, same subset
