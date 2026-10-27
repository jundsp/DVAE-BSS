from pathlib import Path

import pytest
import torch

FIXTURE = Path(__file__).parent / "fixtures" / "golden.pt"

OLD_TO_NEW = dict(
    dimx="n_freq", dimz="latent_dim", dimh="rnn_dim", num_layers="rnn_layers",
)  # fmt: skip
DROPPED = {"dimg", "scale_invariant", "dilation"}


def new_config(entry: dict, encoder: str) -> dict:
    """Map a constructor call of the original ``DCAE``/``DSVAE`` to ``DVAE`` arguments."""
    config = {OLD_TO_NEW.get(k, k): v for k, v in entry["config"].items() if k not in DROPPED}
    return dict(config, encoder=encoder)


@pytest.fixture(scope="session")
def golden() -> dict:
    return torch.load(FIXTURE, weights_only=False)


@pytest.fixture(params=["strided", "dilated"])
def entry(request, golden):
    """Golden data of one encoder type, with ``name`` and a ready ``DVAE`` config."""
    data = dict(golden[request.param], name=request.param)
    data["new_config"] = new_config(data, request.param)
    return data
