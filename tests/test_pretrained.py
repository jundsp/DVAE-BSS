"""The bundled model separates the bundled example (skipped if either file is missing)."""

from pathlib import Path

import pytest

from dvae_bss import Separator, load_model
from dvae_bss.data import read_audio

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "pretrained" / "dvae_rwc_k5.pt"
EXAMPLE = ROOT / "examples" / "mixture.wav"

pytestmark = pytest.mark.skipif(
    not (MODEL.exists() and EXAMPLE.exists()), reason="pretrained model or example missing"
)


def test_pretrained_separates_example():
    model, stft = load_model(MODEL)
    assert model.config["num_sources"] == 5 and stft.sample_rate == 16000
    mixture = read_audio(EXAMPLE)
    sources = Separator(model.eval(), stft)(mixture[None])[0]
    energy = sources.pow(2).sum(-1)
    share = (energy / energy.sum()).sort(descending=True).values
    # two notes: two outputs carry the signal, the other three stay near silent
    assert share[:2].min() > 0.2 and share[2:].sum() < 1e-3
    assert sources.shape == (5, len(mixture))
