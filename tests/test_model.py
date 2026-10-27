"""Behaviour checks that do not need the golden fixtures (all on CPU, tiny models)."""

import pytest
import torch

from dvae_bss import DVAE, CompressedSTFT, ELBOLoss, Separator
from dvae_bss.losses import optimal_gain
from dvae_bss.metrics import bss_metrics, make_scale_invariant

TINY = dict(n_freq=65, latent_dim=4, rnn_dim=8, num_sources=3, layer_channels=(4, 8, 8))


@pytest.fixture(params=["strided", "dilated"])
def encoder(request):
    return request.param


@pytest.mark.parametrize(
    "overrides",
    [{}, {"inference_rnn": False}, {"autoencoder": True}, {"rnn_type": "gru", "rnn_layers": 2}],
)
def test_train_step_is_finite_and_updates_weights(encoder, overrides):
    model = DVAE(encoder=encoder, **TINY, **overrides).train()
    x = torch.rand(2, 65, 33) ** 2
    before = {k: v.clone() for k, v in model.named_parameters()}
    ell, kld = ELBOLoss(autoencoder=overrides.get("autoencoder", False))(x, model(x))
    (ell + kld).backward()
    torch.optim.SGD(model.parameters(), lr=0.1).step()
    assert torch.isfinite(ell + kld)
    assert any(not torch.equal(v, before[k]) for k, v in model.named_parameters())


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_dilated_separator_depth(layers):
    """The original code crashed for ``separator_layers > 1``."""
    model = DVAE(encoder="dilated", separator_layers=layers, separator_kernel_size=3, **TINY).eval()
    out = model(torch.rand(2, 65, 33), stochastic=False)
    assert out["recon_s"].shape == (2, 3, 65, 33)
    assert out["recon_x"].shape == (2, 65, 33)


def test_output_shapes_and_source_sum(encoder):
    model = DVAE(encoder=encoder, **TINY).eval()
    out = model(torch.rand(2, 65, 33), stochastic=False)
    assert out["mu_q"].shape[0] == 2 * 3 and out["mu_q"].shape[-1] == 4
    torch.testing.assert_close(out["recon_x"], out["recon_s"].sum(1))
    assert out["recon_s"].min() >= 0 and out["recon_s"].max() <= 1  # sigmoid output


def test_sample_from_prior(encoder):
    model = DVAE(encoder=encoder, **TINY).eval()
    sources, z = model.sample(n_steps=9, n_samples=2)
    assert z.shape == (2 * 3, 9, 4)
    assert sources.shape[:2] == (2, 3) and torch.isfinite(sources).all()


def test_unknown_encoder_rejected():
    with pytest.raises(ValueError):
        DVAE(encoder="nope", **TINY)


def test_optimal_gain_is_detached_and_least_squares():
    target = torch.rand(3, 5, 7)
    estimate = (2.5 * target).requires_grad_()
    gain = optimal_gain(target, estimate)
    assert not gain.requires_grad
    torch.testing.assert_close(gain.flatten(), torch.full((3,), 1 / 2.5), atol=1e-5, rtol=1e-5)


def test_silent_estimate_does_not_produce_nan():
    gain = optimal_gain(torch.rand(2, 4, 4), torch.zeros(2, 4, 4))
    assert torch.isfinite(gain).all()


def test_separator_shapes_and_mask():
    stft = CompressedSTFT(n_fft=128, hop_length=32, compression=0.5)
    model = DVAE(encoder="strided", **TINY).eval()
    wave = torch.randn(2, 3000) * 0.1
    sources = Separator(model, stft)(wave)
    assert sources.shape == (2, 3, 3000) and torch.isfinite(sources).all()


@pytest.mark.parametrize("n_samples", [1000, 2048, 5001])
def test_stft_roundtrip_any_length(n_samples):
    stft = CompressedSTFT(n_fft=128, hop_length=32, compression=0.5)
    wave = torch.randn(2, n_samples)
    _, _, n_frames = stft.padding(n_samples)
    spec = stft.analysis(wave)
    assert spec.shape[-1] == n_frames and (n_frames - 1) % 64 == 0
    torch.testing.assert_close(stft.synthesis(spec, n_samples), wave, atol=1e-5, rtol=0)


def test_metrics_perfect_swapped_and_silent():
    g = torch.Generator().manual_seed(0)
    targets = torch.randn(2, 2, 4000, generator=g)
    noisy = targets + 0.01 * torch.randn(2, 2, 4000, generator=g)
    swapped = noisy.flip(1)
    ordered = bss_metrics(noisy, targets)
    shuffled = bss_metrics(swapped, targets)
    assert ordered["sdr"].min() > 30
    torch.testing.assert_close(ordered["sdr"], shuffled["sdr"])
    silent = torch.zeros(1, 2, 100)
    assert torch.isfinite(make_scale_invariant(silent, silent)).all()
