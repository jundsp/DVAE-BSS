"""The refactored package must reproduce the dissertation code (golden fixtures).

Fixtures come from ``tests/make_fixtures.py`` run on the ``dissertation-2024`` tag.
"""

import numpy as np
import pytest
import torch

from dvae_bss import DVAE, CompressedSTFT, ELBOLoss, Separator
from dvae_bss.checkpoint import clean_state_dict
from dvae_bss.metrics import (
    bss_metrics,
    compute_measures_batch,
    ideal_masks,
    make_scale_invariant,
    optimal_permute,
)

TOL = dict(atol=1e-5, rtol=1e-4)


def build(entry) -> DVAE:
    model = DVAE(**entry["new_config"])
    model.load_state_dict(clean_state_dict(entry["state_dict"]), strict=True)
    return model


def assert_outputs_match(output: dict, reference: dict) -> None:
    assert output.keys() == reference.keys()
    for key, value in reference.items():
        try:
            torch.testing.assert_close(output[key], value, **TOL)
        except AssertionError as error:
            raise AssertionError(f"{key}: {error}") from error


def test_eval_forward(entry):
    model = build(entry).eval()
    with torch.no_grad():
        assert_outputs_match(model(entry["x"], stochastic=False), entry["forward_eval"])


@pytest.mark.parametrize("stochastic", [True, False])
def test_train_forward(entry, stochastic):
    model = build(entry).train()
    torch.manual_seed(123)  # same noise draws as the reference
    output = model(entry["x"], stochastic=stochastic)
    assert_outputs_match(output, entry[f"forward_train_stochastic={stochastic}"])


@pytest.mark.parametrize("laplace", [False, True])
def test_loss(entry, laplace):
    model = build(entry).train()
    torch.manual_seed(7)
    ell, kld = ELBOLoss(scale_invariant=True, laplace=laplace)(entry["x"], model(entry["x"]))
    expected = entry["loss_laplace" if laplace else "loss_normal"]
    assert (ell.item(), kld.item()) == pytest.approx(expected, rel=1e-4, abs=1e-5)


def test_training_trajectory(entry):
    model = build(entry).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    loss_fn = ELBOLoss(scale_invariant=True)
    torch.manual_seed(11)
    trajectory = []
    for _ in range(20):
        ell, kld = loss_fn(entry["x"], model(entry["x"]))
        optimizer.zero_grad()
        (ell + kld).backward()
        optimizer.step()
        trajectory.append((ell.item(), kld.item()))
    np.testing.assert_allclose(trajectory, entry["train_trajectory"], rtol=1e-3, atol=1e-5)


def stft_of(golden) -> CompressedSTFT:
    return CompressedSTFT(
        **{k: v for k, v in golden["constants"].items() if k != "hop"},
        hop_length=golden["constants"]["hop"],
    )


def test_stft_matches_separator_preprocess(golden):
    wave, stft = golden["wave"], stft_of(golden)
    spec = stft.analysis(wave)
    torch.testing.assert_close(
        spec, golden["strided"]["separator_preprocess"], atol=1e-5, rtol=1e-4
    )
    roundtrip = stft.synthesis(spec, wave.shape[-1])
    torch.testing.assert_close(
        roundtrip, golden["strided"]["separator_roundtrip"], atol=1e-5, rtol=1e-4
    )
    torch.testing.assert_close(roundtrip, wave, atol=1e-5, rtol=0)


def test_stft_matches_torchaudio_on_interior_frames(golden):
    wave, stft = golden["wave"], stft_of(golden)
    left, _, _ = stft.padding(wave.shape[-1])
    assert left % stft.hop_length == 0  # frames of both transforms are then aligned
    shift = left // stft.hop_length - 2
    ours = stft.magnitude(wave)[..., shift + 2 : shift + 32]
    reference = golden["audio_pipeline"][..., 2:32]  # edge frames differ: reflect vs zero padding
    torch.testing.assert_close(ours, reference, atol=1e-5, rtol=1e-4)


def test_separator(entry, golden):
    model = build(entry).eval()
    separator = Separator(model, stft_of(golden), mask=True)
    torch.testing.assert_close(
        separator(golden["wave"]), entry["separator_out"], atol=1e-5, rtol=1e-4
    )


def test_metrics(golden):
    ref = golden["metrics"]
    permuted, perm = optimal_permute(ref["estimates"], ref["targets"])
    np.testing.assert_array_equal(perm, ref["perm"])
    torch.testing.assert_close(permuted, ref["permuted"])
    invariant = make_scale_invariant(permuted, ref["targets"])
    torch.testing.assert_close(invariant, ref["invariant"], atol=1e-5, rtol=1e-4)
    sdr, sir, sar = compute_measures_batch(permuted[:, :2], invariant)
    for got, name in ((sdr, "sdr"), (sir, "sir"), (sar, "sar")):
        torch.testing.assert_close(got, ref[name], atol=1e-3, rtol=1e-4)
    assert bss_metrics(ref["estimates"], ref["targets"])["sdr"].shape == (3, 2)


def test_ideal_masks(golden):
    ref = golden["ideal_masks"]
    irm, ibm = ideal_masks(ref["stft_sources"])
    torch.testing.assert_close(irm, ref["irm"])
    torch.testing.assert_close(ibm, ref["ibm"])
