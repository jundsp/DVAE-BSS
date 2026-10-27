"""Classical baselines on the mixture spectrogram: FastICA and NMF (with deconvolution).

Each mixture's magnitude spectrogram is factorised into two components, which are given the
mixture phase and inverted. Scored like a model; results go to ``<out-dir>/{ica,nmf}/``.

ICA needs scikit-learn (``pip install -e .[experiments]``). NMF needs the NMF Toolbox from
AudioLabs Erlangen, which is not on PyPI: clone https://github.com/groupmm/NMFtoolbox and add
its folder to ``PYTHONPATH``.

    python experiments/baselines/ica_nmf.py --data data/rwc_test --methods ica nmf
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dvae_bss.audio import CompressedSTFT
from dvae_bss.data import PremixedDataset
from dvae_bss.evaluation import evaluate_methods, print_summary


def ica_components(magnitude: np.ndarray, n_components: int = 2, seed: int = 0) -> np.ndarray:
    """Rank-one components ``(n_components, F, T)`` from FastICA over the time axis."""
    from sklearn.decomposition import FastICA

    ica = FastICA(n_components=n_components, random_state=seed)
    spectra = ica.fit_transform(magnitude)  # (F, n): one spectrum per component
    activations = ica.mixing_.T  # (n, T): its activation over time
    return np.einsum("nt,fn->nft", activations, spectra)


class NMFDeconvolution:
    """NMF with deconvolution and a Wiener-style split, from the NMF Toolbox.

    Args:
        n_bins: Frequency bins of the spectrogram.
        n_frames: Time frames of the spectrogram.
        n_components: Number of components (sources).
        n_templates: Time frames of each template.
        n_iter: Update iterations.
        sample_rate: Sample rate; with ``n_fft`` it sets the bin spacing for the templates.
        n_fft: FFT size of the spectrogram.
    """

    def __init__(
        self,
        n_bins: int,
        n_frames: int,
        n_components: int = 2,
        n_templates: int = 8,
        n_iter: int = 30,
        sample_rate: int = 16000,
        n_fft: int = 512,
    ) -> None:
        try:
            from NMFtoolbox.alphaWienerFilter import alphaWienerFilter
            from NMFtoolbox.initActivations import initActivations
            from NMFtoolbox.initTemplates import initTemplates
            from NMFtoolbox.NMFD import NMFD
        except ImportError as error:
            raise SystemExit(
                "NMF needs the NMF Toolbox (https://github.com/groupmm/NMFtoolbox) on PYTHONPATH"
            ) from error
        self._nmfd, self._wiener = NMFD, alphaWienerFilter
        self.parameters = dict(
            numComp=n_components,
            numFrames=n_frames,
            numIter=n_iter,
            numTemplateFrames=n_templates,
            initW=initTemplates(
                dict(
                    deltaF=sample_rate / n_fft,
                    numComp=n_components,
                    numBins=n_bins,
                    numTemplateFrames=n_templates,
                ),
                "random",
            ),
            initH=initActivations(dict(numComp=n_components, numFrames=n_frames), "uniform"),
        )

    def __call__(self, magnitude: np.ndarray) -> np.ndarray:
        """Component spectrograms ``(n_components, F, T)`` of a magnitude spectrogram."""
        _, _, approximation, _, _ = self._nmfd(magnitude, self.parameters)
        components, _ = self._wiener(magnitude, approximation, 1.0)
        return np.asarray(components)


def build_estimators(stft: CompressedSTFT, methods: list[str], seed: int) -> dict[str, Any]:
    """Estimators ``batch -> (B, 2, L)`` for the requested methods."""

    def estimator(factorise):
        def run(batch: dict[str, Any]) -> torch.Tensor:
            mix = batch["mix"]
            spectra = stft.analysis(mix)
            outputs = []
            for spectrum in spectra:
                components = torch.from_numpy(
                    np.ascontiguousarray(factorise(spectrum.abs().numpy()))
                ).float()
                phase = torch.exp(1j * spectrum.angle())
                outputs.append(stft.synthesis(components * phase, mix.shape[-1]))
            return torch.stack(outputs)

        return run

    estimators = {}
    if "ica" in methods:
        estimators["ica"] = estimator(lambda magnitude: ica_components(magnitude, seed=seed))
    if "nmf" in methods:
        np.random.seed(seed)
        cache: dict[tuple[int, int], NMFDeconvolution] = {}

        def nmf(magnitude: np.ndarray) -> np.ndarray:
            if magnitude.shape not in cache:
                cache[magnitude.shape] = NMFDeconvolution(
                    *magnitude.shape, sample_rate=stft.sample_rate, n_fft=stft.n_fft
                )
            return cache[magnitude.shape](magnitude)

        estimators["nmf"] = estimator(nmf)
    return estimators


def main(argv: list[str] | None = None) -> None:
    """Evaluate the ICA and/or NMF baselines."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-d", "--data", type=Path, required=True, help="folder with mix.json")
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("results/classical"))
    parser.add_argument("--methods", nargs="+", choices=["ica", "nmf"], default=["ica", "nmf"])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    stft = CompressedSTFT(args.n_fft, args.hop_length, 1.0, args.sample_rate)
    dataset = PremixedDataset(args.data, args.sample_rate)
    estimators = build_estimators(stft, args.methods, args.seed)
    summaries = evaluate_methods(
        estimators, dataset, args.out_dir, args.batch_size, args.max_batches
    )
    for name, summary in summaries.items():
        print(f"\n{name}")
        print_summary(summary)


if __name__ == "__main__":
    main()
