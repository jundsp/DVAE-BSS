"""Regenerate ``tests/fixtures/golden.pt`` from the original dissertation code.

The fixtures are reference outputs produced by the code at the git tag
``dissertation-2024``; ``tests/test_equivalence.py`` checks that the refactored package
reproduces them. Everything is tiny (CPU, float32, 65 frequency bins) so the file stays
well under 1 MB.

Usage:
    git worktree add --detach /path/to/orig dissertation-2024
    python tests/make_fixtures.py --orig /path/to/orig
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Small configurations: 3 conv layers (time frames must be 8k+1), 65 frequency bins.
DIMX, FRAMES, BATCH = 65, 33, 2
N_FFT, HOP = 128, 32
COMPRESSION = 0.5

STRIDED = dict(
    dimx=DIMX, dimz=8, dimh=16, dimg=16, num_sources=2, num_layers=1, scale_invariant=True,
    layer_channels=(4, 8, 8), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
    variance=0.1, activate_final_layer=True, rnn_type="lstm", inference_rnn=True,
    autoencoder=False,
)  # fmt: skip
DILATED = dict(
    dimx=DIMX, dimz=8, dimh=16, dimg=16, num_sources=2, num_layers=1, scale_invariant=True,
    layer_channels=(4, 8, 8), kernel_size=(3, 3), stride=(2, 1), padding=(1, 1),
    variance=0.1, activate_final_layer=True, rnn_type="gru", inference_rnn=True,
    autoencoder=False, separator_layers=1, separator_kernel_size=3, separator_channels=8,
)  # fmt: skip


def tensors(output: dict) -> dict:
    return {k: v.detach().clone() for k, v in output.items() if torch.is_tensor(v)}


def capture_model(old, name: str, cls, config: dict, x: torch.Tensor, wave: torch.Tensor) -> dict:
    torch.manual_seed(0)
    model = cls(**config)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    entry = {"cls": name, "config": config, "state_dict": initial, "x": x}

    model.train()
    for stochastic in (True, False):
        torch.manual_seed(123)
        entry[f"forward_train_stochastic={stochastic}"] = tensors(model(x, stochastic=stochastic))

    # Losses use the stochastic train-mode forward pass with a fixed seed.
    for label, laplace in (("normal", False), ("laplace", True)):
        torch.manual_seed(7)
        output = model(x)
        ell, kld = old.LossFunction(scale_invariant=True, laplace=laplace)(x, output)
        entry[f"loss_{label}"] = (ell.item(), kld.item())

    # Optimisation trajectory from the same initial weights.
    model.load_state_dict(initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-5)
    lossfun = old.LossFunction(scale_invariant=True)
    torch.manual_seed(11)
    trajectory = []
    for _ in range(20):
        ell, kld = lossfun(x, model(x))
        optimizer.zero_grad()
        (ell + kld).backward()
        optimizer.step()
        trajectory.append((ell.item(), kld.item()))
    entry["train_trajectory"] = trajectory

    model.load_state_dict(initial)
    model.eval()
    with torch.no_grad():
        entry["forward_eval"] = tensors(model(x, stochastic=False))
        separator = old.Separator(
            model, compression=COMPRESSION, n_fft=N_FFT, hop_length=HOP, mask=True
        )
        entry["separator_out"] = separator(wave)
        spec = separator.preprocess(wave)
        entry["separator_preprocess"] = spec
        entry["separator_roundtrip"] = separator.postprocess(spec)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--orig", type=Path, required=True, help="worktree of dissertation-2024")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent / "fixtures" / "golden.pt"
    )
    args = parser.parse_args()

    np.float = float  # the original code predates its removal from NumPy
    sys.path.insert(0, str(args.orig))
    import dataset as old_dataset
    import model as old
    from utils import utils as old_utils

    g = torch.Generator().manual_seed(0)
    x = torch.rand(BATCH, DIMX, FRAMES, generator=g) ** 2
    wave = torch.randn(BATCH, 2048, generator=g) * 0.1

    golden = {
        "torch": torch.__version__,
        "constants": dict(n_fft=N_FFT, hop=HOP, compression=COMPRESSION),
        "wave": wave,
        "strided": capture_model(old, "DCAE", old.DCAE, STRIDED, x, wave),
        "dilated": capture_model(old, "DSVAE", old.DSVAE, DILATED, x, wave),
    }

    pipeline = old_dataset.AudioPipeline(
        n_fft=N_FFT, hop_length=HOP, compression=COMPRESSION, n_time=FRAMES
    )
    golden["audio_pipeline"] = pipeline(wave)

    estimates = torch.randn(3, 3, 1000, generator=g)
    targets = torch.randn(3, 2, 1000, generator=g)
    estimates[:, :2] += 0.5 * targets
    permuted, perm = old_utils.optimal_permute(estimates, targets)
    invariant = old_utils.make_scale_invariant(estimates=permuted, targets=targets)
    sdr, sir, sar = old_utils.compute_measures_batch(estimates=permuted, targets=invariant)
    golden["metrics"] = dict(
        estimates=estimates, targets=targets, permuted=permuted, perm=perm,
        invariant=invariant, sdr=sdr, sir=sir, sar=sar,
    )  # fmt: skip

    stft_sources = torch.randn(2, 2, 17, 9, dtype=torch.complex64, generator=g)
    irm, ibm = old_utils.ideal_masks(stft_sources)
    golden["ideal_masks"] = dict(stft_sources=stft_sources, irm=irm, ibm=ibm)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(golden, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
