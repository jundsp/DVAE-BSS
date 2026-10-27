import json

import pytest
import torch

from conftest import new_config
from dvae_bss import DVAE, CompressedSTFT, load_checkpoint, load_legacy, save_checkpoint
from dvae_bss.checkpoint import config_from_legacy_args

LEGACY_ARGS = dict(
    latent_dim=8, rnn_dim=16, num_sources=2, num_layers=1, cnn_channels=[4, 8, 8],
    rnn_type="lstm", variance=0.1, activate_final_layer=True, compression=0.5,
    conv2d=True, complex_version=False, separator_version=False, inference_rnn_off=False,
    autoencoder=False,
)  # fmt: skip


def test_roundtrip(tmp_path):
    model = DVAE(n_freq=65, latent_dim=4, rnn_dim=8, num_sources=2, layer_channels=(4, 8, 8))
    stft = CompressedSTFT(n_fft=128, hop_length=32, compression=0.7)
    save_checkpoint(tmp_path / "m.pt", model, stft, extra={"epoch": 3})
    loaded, loaded_stft = load_checkpoint(tmp_path / "m.pt")
    assert loaded.config == model.config
    assert (loaded_stft.n_fft, loaded_stft.hop_length, loaded_stft.compression) == (128, 32, 0.7)
    for (name, a), (_, b) in zip(
        model.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_legacy_config_matches_original_architecture(golden):
    """The legacy loader must choose the architecture the original scripts built."""
    expected = new_config(golden["strided"], "strided")
    got = config_from_legacy_args(LEGACY_ARGS, n_freq=65)
    for key in ("layer_channels", "kernel_size", "stride", "padding"):
        assert tuple(got[key]) == tuple(expected[key]), key
    for key in ("rnn_type", "encoder", "inference_rnn", "num_sources", "latent_dim"):
        assert got[key] == expected[key], key
    assert config_from_legacy_args(
        dict(
            LEGACY_ARGS,
            separator_version=True,
            separator_kernel_size=3,
            separator_layers=1,
            separator_channels=8,
        )
    )["stride"] == (2, 1)


def test_load_legacy_files(tmp_path):
    args = dict(LEGACY_ARGS)
    model = DVAE(**config_from_legacy_args(args))
    state = {"module." + k: v for k, v in model.state_dict().items()}
    state["module.likelihood.variance"] = torch.tensor(1.0)  # parameter of the old model
    torch.save(state, tmp_path / "svae.pt")
    (tmp_path / "configuration.txt").write_text(json.dumps(args))
    loaded, stft = load_legacy(tmp_path / "svae.pt")
    assert stft.compression == 0.5 and (stft.n_fft, stft.hop_length) == (512, 128)
    for (_, a), (_, b) in zip(model.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("override", [{"conv2d": False}, {"complex_version": True}])
def test_legacy_rejects_removed_variants(override):
    with pytest.raises(ValueError):
        config_from_legacy_args(dict(LEGACY_ARGS, **override))
