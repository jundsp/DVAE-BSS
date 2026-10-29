"""Waveform-in, waveform-out separation with a trained :class:`~dvae_bss.model.DVAE`."""

from __future__ import annotations

import torch
from torch import nn

from dvae_bss.audio import CompressedSTFT
from dvae_bss.model import DVAE


class Separator(nn.Module):
    """Separate mixture waveforms into ``num_sources`` waveforms.

    The mixture is analysed with :class:`~dvae_bss.audio.CompressedSTFT`, the model
    estimates compressed source magnitudes from the posterior means, and each source is
    synthesised with the mixture phase.

    Args:
        model: Trained network.
        stft: Front end the model was trained with.
        mask: Rescale the source estimates so that they sum to the mixture magnitude
            (a ratio mask) before synthesis.
    """

    def __init__(self, model: DVAE, stft: CompressedSTFT, mask: bool = True) -> None:
        super().__init__()
        self.model = model
        self.stft = stft
        self.mask = mask

    @torch.no_grad()
    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """Separate waveforms.

        Args:
            wave: Mixtures, shape ``(B, L)``.

        Returns:
            Source waveforms, shape ``(B, K, L)``; sources are in arbitrary order.
        """
        n_samples = wave.shape[-1]
        mixture = self.stft.analysis(wave)  # compressed complex STFT (B, F, T)
        magnitude = mixture.abs()
        # Posterior means, no sampling: a deterministic estimate of every source.
        output = self.model(magnitude, stochastic=False)
        sources = output["recon_s"]  # (B, K, F, T)
        if self.mask:
            # Ratio mask: source k gets the fraction S_k / sum_j S_j of the mixture in every
            # time-frequency bin, so the estimates add up exactly to the mixture.
            sources = sources * (magnitude / output["recon_x"]).unsqueeze(1)
        # The model estimates magnitudes only; every source reuses the mixture's phase.
        spectra = torch.polar(sources, mixture.angle().unsqueeze(1).expand_as(sources))
        batch, k, n_freq, n_frames = spectra.shape
        out = self.stft.synthesis(spectra.reshape(-1, n_freq, n_frames), n_samples)
        return out.view(batch, k, n_samples)
