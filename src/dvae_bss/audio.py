"""Spectrogram front end shared by training, separation and evaluation.

The network operates on magnitude-compressed STFT magnitudes, ``|X| ** c`` with ``c`` in
``(0, 1]``. :class:`CompressedSTFT` computes them from a waveform and converts masked
estimates back to waveforms, so the same transform is used in every stage.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

# Strided encoders need ``frames = k * FRAME_MULTIPLE + 1`` so that decoding restores the length.
FRAME_MULTIPLE = 64


def db2mag(db: torch.Tensor) -> torch.Tensor:
    """Convert decibels to a linear amplitude ratio."""
    return torch.pow(10.0, db / 20.0)


def mag2db(mag: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Convert linear amplitudes to decibels; values below ``eps`` are clipped."""
    return 20.0 * torch.log10(mag.abs().clamp_min(eps))


class CompressedSTFT(nn.Module):
    """STFT with a Hann window, power-law magnitude compression and its inverse.

    Signals are zero-padded so that the number of frames is a multiple of
    :data:`FRAME_MULTIPLE` plus one, which the strided convolutional encoder needs. The
    analysis window is scaled so that the transform is unitary-like (``normalized=True`` in
    torchaudio).

    Args:
        n_fft: FFT size, also the window length (samples).
        hop_length: Hop between frames (samples). Must be at most ``n_fft // 4`` for the
            overlap-add gain in :meth:`synthesis` to be exact.
        compression: Exponent ``c`` applied to the STFT magnitude.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 128, compression: float = 0.5) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.compression = compression
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.normalizer = float(self.window.pow(2).sum().sqrt())
        # With a Hann^2 window the overlapped windows sum to (3/8) * n_fft / hop_length.
        self.overlap_gain = 8.0 * hop_length / (3.0 * n_fft)

    @property
    def n_freq(self) -> int:
        """Number of frequency bins, ``n_fft // 2 + 1``."""
        return self.n_fft // 2 + 1

    def padding(self, n_samples: int) -> tuple[int, int, int]:
        """Zero padding ``(left, right)`` and the resulting number of frames."""
        n_frames = math.ceil(
            (self.n_fft + n_samples - 2 * self.hop_length - 1) / self.hop_length + 1
        )
        n_frames = math.ceil(n_frames / FRAME_MULTIPLE) * FRAME_MULTIPLE + 1
        total = (n_frames - 1) * self.hop_length + self.n_fft - n_samples
        left = total // 2
        return left, total - left, n_frames

    def stft(self, wave: torch.Tensor) -> torch.Tensor:
        """Complex STFT of ``(B, L)`` waveforms; shape ``(B, F, T)``."""
        left, right, _ = self.padding(wave.shape[-1])
        spec = torch.stft(
            F.pad(wave, (left, right)),
            self.n_fft,
            self.hop_length,
            window=self.window.to(wave.device),
            center=False,
            return_complex=True,
        )
        return spec / self.normalizer

    def analysis(self, wave: torch.Tensor) -> torch.Tensor:
        """Compressed complex STFT ``|X|**c * exp(i angle(X))`` of ``(B, L)`` waveforms."""
        spec = self.stft(wave)
        return torch.polar(spec.abs().pow(self.compression), spec.angle())

    def magnitude(self, wave: torch.Tensor) -> torch.Tensor:
        """Compressed magnitudes ``|X|**c`` of ``(B, L)`` waveforms; shape ``(B, F, T)``."""
        return self.stft(wave).abs().pow(self.compression)

    def synthesis(self, spec: torch.Tensor, n_samples: int) -> torch.Tensor:
        """Invert :meth:`analysis`.

        Args:
            spec: Compressed complex STFT, shape ``(B, F, T)``.
            n_samples: Length of the original waveform, used to remove the padding.

        Returns:
            Waveforms of shape ``(B, n_samples)``.
        """
        left, right, n_frames = self.padding(n_samples)
        spec = torch.polar(spec.abs().pow(1.0 / self.compression), spec.angle())
        window = self.window.to(spec.device)
        frames = (
            torch.fft.irfft(spec, n=self.n_fft, dim=-2) * window.view(1, -1, 1) * self.normalizer
        )
        length = (n_frames - 1) * self.hop_length + self.n_fft
        wave = F.fold(
            frames,
            output_size=(1, length),
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop_length),
        ).view(-1, length)
        return wave[:, left : length - right] * self.overlap_gain
