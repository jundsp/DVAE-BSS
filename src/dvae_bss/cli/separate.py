"""Separate audio files into sources with a trained model, one wav per source."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from dvae_bss.checkpoint import load_model
from dvae_bss.cli.common import resolve_device
from dvae_bss.data import read_audio
from dvae_bss.separator import Separator

# Default model location relative to a source checkout of the repository.
PRETRAINED = Path(__file__).resolve().parents[3] / "pretrained" / "dvae_rwc_k5.pt"


def build_parser() -> argparse.ArgumentParser:
    """Command-line options of ``dvae-separate``."""
    p = argparse.ArgumentParser(prog="dvae-separate", description=__doc__.split("\n")[0])
    p.add_argument("audio", nargs="+", type=Path, help="mixture files (wav, flac, ogg)")
    default_model = PRETRAINED if PRETRAINED.exists() else None
    p.add_argument("-m", "--model", type=Path, default=default_model, help="default: pretrained/")
    p.add_argument("--legacy-config", type=Path, help="configuration.txt of an old checkpoint")
    p.add_argument("-o", "--out-dir", type=Path, default=Path("separated"))
    p.add_argument("--device", default="auto")
    p.add_argument("--no-normalize", action="store_true", help="do not scale input to unit peak")
    p.add_argument("--plot", action="store_true", help="also save a spectrogram figure")
    return p


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
            from dvae_bss.plotting import plot_separation

            figure = plot_separation(mixture.numpy(), sources.numpy(), stft.sample_rate)
            figure.savefig(args.out_dir / f"{path.stem}.png", dpi=150)
            print(args.out_dir / f"{path.stem}.png")


if __name__ == "__main__":
    main()
