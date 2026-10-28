r"""Static web page of separation examples: audio players and spectrograms.

Separates chosen examples of a pre-mixed test set and writes a self-contained ``index.html``
with the mixture, the true sources and the matched estimates of each example, plus the
``audio/`` and ``img/`` files it links to. The repository's ``docs/`` page (GitHub Pages) is
built with

    python experiments/figures/make_demo_page.py -d data/TwoNote --examples 365 619 357 222 \\
        --notes "" "" "" "Typical failure: notes at the same pitch are not separated." -o docs
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from dvae_bss import Separator, load_model
from dvae_bss.data import PremixedDataset
from dvae_bss.metrics import bss_metrics
from dvae_bss.plotting import plot_spectrogram, spectrogram_db, style

PRETRAINED = Path(__file__).resolve().parents[2] / "pretrained" / "dvae_rwc_k5.pt"

# Note types of the TwoNote set; other names are shown as they are.
NOTE_TYPES = {
    "forced": "sustained note",
    "free": "decaying note",
    "forced_vibrato": "sustained note with vibrato",
}


def describe(record: dict[str, Any]) -> str:
    """One-line description from the test set's metadata (TwoNote and make_test_set keys)."""
    parts = []
    names = [record.get(f"instrument{k}") for k in (1, 2)]
    if all(names):
        parts.append(" + ".join(NOTE_TYPES.get(str(n), str(n)) for n in names))
    if "delta_semitone" in record:
        parts.append(f"{record['delta_semitone']} semitones apart")
    if "delta_time" in record:
        parts.append(f"second note starts {float(record['delta_time']):.2f} s later")
    return ", ".join(parts)


def save_spectrogram(path: Path, wave: np.ndarray, rate: int, vmax: float) -> None:
    """Small spectrogram image without axes, on a fixed colour scale."""
    fig, ax = plt.subplots(figsize=(3.2, 1.6), dpi=100)
    plot_spectrogram(ax, wave, rate, vmax=vmax)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path)
    plt.close(fig)


def clip(name: str, label: str, sdr: float | None = None) -> str:
    """HTML for one spectrogram with its audio player."""
    score = f'<span class="score">{sdr:.1f} dB</span>' if sdr is not None else ""
    return (
        f"<figure><figcaption>{html.escape(label)}{score}</figcaption>"
        f'<img src="img/{name}.png" alt="spectrogram: {html.escape(label)}" width="320" '
        f'height="160" loading="lazy">'
        f'<audio controls preload="none" src="audio/{name}.wav"></audio></figure>'
    )


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DVAE-BSS examples</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --card: #f4f5f7; --line: #dcdfe4;
  --accent: #0b6bcb; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg: #111214; --fg: #e8e8ea; --muted: #a0a4ab; --card: #1b1d21; --line: #2c3036;
    --accent: #6cb4ff; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.55 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }}
main {{ max-width: 1080px; margin: 0 auto; padding: 32px 16px 64px; }}
h1 {{ font-size: 1.8rem; margin: 0 0 4px; }}
h2 {{ font-size: 1.15rem; margin: 0; }}
p {{ margin: 8px 0; }}
a {{ color: var(--accent); }}
.lead, .meta, .note {{ color: var(--muted); }}
.lead {{ max-width: 760px; }}
.diagram {{ display: block; width: 100%; max-width: 900px; margin-top: 16px; background: #fff;
  border-radius: 8px; }}
section {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px; margin-top: 24px; }}
.row {{ display: grid; grid-template-columns: 120px 1fr; gap: 12px; align-items: start;
  margin-top: 14px; }}
.row > span {{ font-weight: 600; padding-top: 4px; }}
.clips {{ display: flex; flex-wrap: wrap; gap: 12px; }}
figure {{ margin: 0; width: 320px; max-width: 100%; }}
figcaption {{ font-size: .9rem; display: flex; justify-content: space-between; }}
.score {{ color: var(--muted); font-variant-numeric: tabular-nums; }}
img {{ display: block; width: 100%; height: auto; border-radius: 4px; background: #000; }}
audio {{ width: 100%; height: 36px; margin-top: 4px; }}
footer {{ margin-top: 32px; font-size: .9rem; color: var(--muted); }}
@media (max-width: 640px) {{ .row {{ grid-template-columns: 1fr; gap: 4px; }} }}
</style>
</head>
<body>
<main>
<h1>DVAE-BSS: separation examples</h1>
<p class="lead">A dynamical variational auto-encoder trained only on mixtures of instrument
notes (RWC), with no isolated sources, separates two-note mixtures from a synthetic test set it
never saw. It has {num_sources} outputs. The two closest to the true notes are shown with their
SI-SDR; the remaining outputs carry the share of the output energy given under each example.
Spectrograms share one colour scale per example.</p>
<p class="lead">Code, pretrained model and paper references:
<a href="https://github.com/jundsp/DVAE-BSS">github.com/jundsp/DVAE-BSS</a>. From Chapter 6 of
J. Neri, <em>Bayesian Hierarchical Modeling and Inference of Mixed Audio Sources</em>, Ph.D.
dissertation, McGill University, 2024.</p>
{diagram}
{sections}
<footer>Generated by <code>experiments/figures/make_demo_page.py</code>. Audio: 16 kHz
mono.</footer>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> None:
    """Write ``index.html``, ``audio/`` and ``img/`` for the chosen examples."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "-m", "--model", type=Path, default=PRETRAINED, help="default: the pretrained model"
    )
    parser.add_argument("-d", "--data", type=Path, required=True, help="folder with mix.json")
    parser.add_argument("--examples", type=int, nargs="+", default=[0], help="indices in the set")
    parser.add_argument("--notes", nargs="*", default=[], help="one remark per example (optional)")
    parser.add_argument("--diagram", help="image in the output folder to show at the top")
    parser.add_argument("-o", "--out", type=Path, default=Path("docs"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    plt.rcParams.update(style())

    model, stft = load_model(args.model)
    separator = Separator(model, stft).to(args.device).eval()
    dataset = PremixedDataset(args.data, stft.sample_rate)
    rate = stft.sample_rate
    for sub in ("audio", "img"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    sections = []
    for number, index in enumerate(args.examples, start=1):
        item = dataset[index]
        mix, stems = item["mix"], item["stems"]
        with torch.no_grad():
            outputs = separator(mix.to(args.device)[None]).cpu()
        metrics = bss_metrics(outputs, stems[None])
        matched = metrics["estimates"][0]
        k = stems.shape[0]
        energy = matched.pow(2).sum(-1)
        rest = float(energy[k:].sum() / energy.sum())

        vmax = spectrogram_db(mix.numpy(), rate)[0].max()
        rows = {"mixture": [(f"ex{index}_mixture", "", mix, None)]}
        rows["true notes"] = [
            (f"ex{index}_source{j + 1}", f"note {j + 1}", stems[j], None) for j in range(k)
        ]
        rows["separated"] = [
            (f"ex{index}_estimate{j + 1}", f"output for note {j + 1}", matched[j],
             float(metrics["sdr"][0, j]))
            for j in range(k)
        ]  # fmt: skip
        html_rows = []
        for title, clips in rows.items():
            for name, _, wave, _ in clips:
                wave = wave.numpy()
                sf.write(args.out / "audio" / f"{name}.wav", wave, rate, subtype="PCM_16")
                save_spectrogram(args.out / "img" / f"{name}.png", wave, rate, vmax)
            cells = "".join(clip(name, label, sdr) for name, label, _, sdr in clips)
            html_rows.append(f'<div class="row"><span>{title}</span><div class="clips">{cells}'
                             "</div></div>")  # fmt: skip
        record = dataset.records[index]
        remark = args.notes[number - 1] if number <= len(args.notes) else ""
        remark = f" {html.escape(remark)}" if remark else ""
        sections.append(
            f'<section><h2>Example {number}</h2><p class="meta">{html.escape(describe(record))}'
            f" (test-set item {index}).{remark}</p>{''.join(html_rows)}"
            f'<p class="note">Other {outputs.shape[1] - k} outputs: {100 * rest:.2f} % of the '
            "output energy.</p></section>"
        )
        print(f"example {index}: SI-SDR {metrics['sdr'][0].numpy().round(1)}, rest {rest:.4f}")

    diagram = (
        f'<img class="diagram" src="{html.escape(args.diagram)}" alt="Concept of the DVAE: '
        "an encoder maps the mixture to latent source trajectories, a decoder maps each to a "
        'source spectrogram, and the sources add up to the mixture.">'
        if args.diagram
        else ""
    )
    page = PAGE.format(num_sources=outputs.shape[1], diagram=diagram, sections="\n".join(sections))
    (args.out / "index.html").write_text(page)
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
