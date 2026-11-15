import math

import pytest
import torch

from dvae_bss.monitor import EarlyStopping, GpuMonitor, MetricTable, format_duration


def test_early_stopping_counts_epochs_without_progress():
    stopper = EarlyStopping(patience=2, min_delta=0.1)
    assert stopper.step(1.0, 1)
    assert not stopper.step(0.95, 2)  # within min_delta
    assert not stopper.should_stop
    assert not stopper.step(math.nan, 3)  # never an improvement
    assert stopper.should_stop and stopper.best_epoch == 1


def test_early_stopping_max_mode_and_state():
    stopper = EarlyStopping(patience=0, mode="max")
    for epoch, value in enumerate([-3.0, -1.0, -2.0], start=1):
        stopper.step(value, epoch)
    assert stopper.best == -1.0 and stopper.bad_epochs == 1
    assert not stopper.should_stop  # patience 0 never stops
    restored = EarlyStopping(mode="max")
    restored.load_state_dict(stopper.state_dict())
    assert restored.state_dict() == stopper.state_dict()
    with pytest.raises(ValueError):
        EarlyStopping(mode="up")


def test_metric_table_keeps_rows_on_resume_and_adds_columns(tmp_path):
    path = tmp_path / "log.csv"
    table = MetricTable(path, ["epoch", "loss"])
    table.append({"epoch": 1, "loss": 0.5})
    table = MetricTable(path, ["epoch", "loss", "sdr"], resume=True)
    table.append({"epoch": 2, "loss": 0.25, "sdr": 3.0})
    rows = table.read()
    assert [r["epoch"] for r in rows] == [1.0, 2.0]
    assert math.isnan(rows[0]["sdr"]) and rows[1]["sdr"] == 3.0
    assert path.read_text().splitlines()[1] == "1,0.5,nan"


def test_format_duration():
    assert format_duration(65) == "1:05"
    assert format_duration(3725) == "1:02:05"
    assert format_duration(math.inf) == "?"


def test_gpu_monitor_is_silent_on_cpu():
    gpu = GpuMonitor(torch.device("cpu"))
    gpu.sample()
    assert gpu.stats() == {} and gpu.summary() == ""


def test_gpu_monitor_summary_shows_headroom():
    stats = {"gpu_peak_gb": 3.0, "gpu_reserved_gb": 6.0, "gpu_total_gb": 24.0, "gpu_util": 97.0}
    text = GpuMonitor(torch.device("cpu")).summary(stats)
    assert text == "GPU peak 3.0 GB, reserved 6.0 of 24.0 GB (75% free), busy 97%"


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Apple MPS")
def test_gpu_monitor_tracks_mps_peak():
    gpu = GpuMonitor(torch.device("mps"))
    x = torch.ones(64, 1024, 1024, device="mps")  # 256 MB
    gpu.sample()
    del x
    stats = gpu.stats()
    assert stats["gpu_peak_gb"] >= 0.25 and stats["gpu_total_gb"] > stats["gpu_peak_gb"]


def test_training_curves_add_a_memory_panel_for_gpu_runs():
    pytest.importorskip("matplotlib")
    from dvae_bss.plotting import plot_training_curves

    rows = [{"epoch": 1.0, "train_loss": -1.0, "lr": 1e-4}]
    assert len(plot_training_curves(rows).axes) == 5
    gpu = {"gpu_peak_gb": 2.0, "gpu_reserved_gb": 3.0, "gpu_total_gb": 16.0}
    fig = plot_training_curves([{**rows[0], **gpu}])
    assert len(fig.axes) == 6 and fig.axes[4].get_yscale() == "log"
