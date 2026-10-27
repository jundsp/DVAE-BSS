"""``dvae-separate``: split audio files into sources with a trained model."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from ..checkpoint import load_model
from ..data import read_audio
from ..separator import Separator
from .common import resolve_device

# Default model location relative to a source checkout of the repository.
PRETRAINED = Path(__file__).resolve().parents[3] / "pretrained" / "dvae_rwc_k5.pt"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of ``dvae-separate``."""
    parser = argparse.ArgumentParser(
        prog="dvae-separate",
        description="Separate mixtures into sources. Writes one wav file per source.",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="mixture files (wav, flac, ogg)")
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=PRETRAINED if PRETRAINED.exists() else None,
        help="trained model; the pretrained one if the repository includes it",
    )
    parser.add_argument(
        "--legacy-config",
        type=Path,
        help="configuration.txt of a checkpoint from the dissertation code "
        "(default: the file next to the checkpoint)",
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=Path("separated"), help="output folder"
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', ...")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="do not scale the mixture to unit peak before separation",
    )
    parser.add_argument(
        "--plot", action="store_true", help="also save a spectrogram figure (needs matplotlib)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point of ``dvae-separate``."""
    args = build_parser().parse_args(argv)
    if args.model is None:
        raise SystemExit("no model given: pass --model (no pretrained model found)")
    device = resolve_device(args.device)
    model, stft = load_model(args.model, args.legacy_config, map_location="cpu")
    separator = Separator(model, stft).to(device).eval()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.audio:
        mixture = read_audio(path, stft.sample_rate)
        peak = mixture.abs().max().item()
        scale = 1.0 if args.no_normalize or peak < 1e-6 else 1.0 / peak
        with torch.no_grad():
            sources = separator((mixture * scale).to(device).unsqueeze(0))[0].cpu() / scale
        for k, source in enumerate(sources, start=1):
            target = args.out_dir / f"{path.stem}_source{k}.wav"
            sf.write(target, source.clamp(-1, 1).numpy(), stft.sample_rate, subtype="PCM_24")
            print(target)
        if args.plot:
            from ..plotting import plot_separation

            figure = plot_separation(mixture.numpy(), sources.numpy(), stft.sample_rate)
            figure.savefig(args.out_dir / f"{path.stem}.png", dpi=150)
            print(args.out_dir / f"{path.stem}.png")


if __name__ == "__main__":
    main()
