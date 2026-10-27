"""Dynamical variational auto-encoder (DVAE) for unsupervised blind sound separation.

The model sees only mixture spectrograms. A feature encoder and a temporal network split
each mixture into ``K`` latent source trajectories, a decoder turns every trajectory into a
source spectrogram, and the sources are summed to reconstruct the mixture. The latent
trajectories follow a learned nonlinear dynamical prior (an RNN), which lets the model
process signals of any duration.

Tensor shape letters used throughout the package:

    B  batch size             K  number of sources
    F  frequency bins         T  spectrogram frames (``k * 64 + 1`` for strided models)
    D  latent dimension       T' latent time steps (T / 2**layers for strided models)

Symbols of the dissertation (Ch. 6) map to code as: ``z`` = latent state (``mu_q``/``logvar_q``
is its approximate posterior, ``mu_p``/``logvar_p`` its dynamical prior), ``z0`` the initial
state, and ``smu``/``smu0`` the learned per-source prior means.

Sources are folded into the batch axis inside the network: latent tensors have shape
``(B * K, T', D)``, ordered batch-major (all sources of example 0, then example 1, ...).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from .layers import (
    Pair,
    StridedDecoder,
    StridedDilatedDecoder,
    StridedDilatedEncoder,
    StridedEncoder,
    TConvSeparator,
    TConvTranspose,
    conv_output_size,
)

EncoderType = Literal["strided", "dilated"]


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Sample ``N(mu, exp(logvar))`` with the reparameterization trick."""
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def product_of_gaussians(
    mean_1: torch.Tensor, logvar_1: torch.Tensor, mean_2: torch.Tensor, logvar_2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and log-variance of the (normalised) product of two Gaussian densities."""
    precision_1, precision_2 = torch.exp(-logvar_1), torch.exp(-logvar_2)
    precision = precision_1 + precision_2
    return (mean_1 * precision_1 + mean_2 * precision_2) / precision, -precision.log()


def _mlp(in_features: int, hidden: int, out_features: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden), nn.LeakyReLU(), nn.Linear(hidden, out_features)
    )


def _rnn(
    kind: str, input_size: int, hidden_size: int, layers: int, bidirectional: bool
) -> nn.RNNBase:
    cls = {"lstm": nn.LSTM, "gru": nn.GRU}[kind]
    return cls(
        input_size, hidden_size, num_layers=layers, bidirectional=bidirectional, batch_first=True
    )


class DVAE(nn.Module):
    """Dynamical VAE that separates a mixture spectrogram into ``num_sources`` sources.

    Args:
        n_freq: Number of input frequency bins ``F``.
        latent_dim: Dimension ``D`` of every source's latent state.
        rnn_dim: Hidden size of the inference and prior RNNs.
        num_sources: Number of sources ``K`` the model can represent.
        encoder: ``"strided"`` downsamples time and frequency; ``"dilated"`` downsamples
            frequency only and keeps one latent state per spectrogram frame.
        layer_channels: Channels of the 2-D convolutional layers (one layer per entry).
        kernel_size: 2-D convolution kernel (frequency, time).
        stride: 2-D convolution stride. Defaults to ``(2, 2)`` (strided) or ``(2, 1)`` (dilated).
        padding: 2-D convolution padding.
        rnn_type: ``"lstm"`` or ``"gru"``.
        rnn_layers: Number of stacked RNN layers.
        inference_rnn: Use a bidirectional RNN over the encoder features. If false, the
            per-step features feed the posterior directly.
        variance: Fixed observation variance of the likelihood.
        activate_final_layer: Apply LeakyReLU after the last transposed convolution.
        autoencoder: Deterministic latent states (no sampling), for ablations.
        separator_kernel_size: Temporal kernel of the layer between convolutions and RNN.
        separator_layers: Depth of that temporal network (``"dilated"`` encoder only).
        separator_channels: Hidden width of that temporal network (``"dilated"`` only).

    Note:
        Submodule and parameter names (``Encoder``, ``fc_encoder``, ``gru_generate``, ...)
        are kept from the original code so that its checkpoints load unchanged. ``gru_*``
        are LSTMs unless ``rnn_type="gru"``.
    """

    def __init__(
        self,
        n_freq: int = 257,
        latent_dim: int = 32,
        rnn_dim: int = 512,
        num_sources: int = 5,
        encoder: EncoderType = "strided",
        layer_channels: tuple[int, ...] = (32, 64, 128, 128, 128),
        kernel_size: Pair = (3, 3),
        stride: Pair | None = None,
        padding: Pair = (1, 1),
        rnn_type: Literal["lstm", "gru"] = "lstm",
        rnn_layers: int = 1,
        inference_rnn: bool = True,
        variance: float = 0.1,
        activate_final_layer: bool = True,
        autoencoder: bool = False,
        separator_kernel_size: int = 1,
        separator_layers: int = 1,
        separator_channels: int = 32,
    ) -> None:
        super().__init__()
        if encoder not in ("strided", "dilated"):
            raise ValueError(f"unknown encoder type {encoder!r}")
        stride = stride or ((2, 2) if encoder == "strided" else (2, 1))
        layer_channels = tuple(layer_channels)
        self.config: dict[str, Any] = dict(
            n_freq=n_freq, latent_dim=latent_dim, rnn_dim=rnn_dim, num_sources=num_sources,
            encoder=encoder, layer_channels=list(layer_channels), kernel_size=list(kernel_size),
            stride=list(stride), padding=list(padding), rnn_type=rnn_type, rnn_layers=rnn_layers,
            inference_rnn=inference_rnn, variance=variance,
            activate_final_layer=activate_final_layer, autoencoder=autoencoder,
            separator_kernel_size=separator_kernel_size,
            separator_layers=separator_layers, separator_channels=separator_channels,
        )  # fmt: skip

        self.n_freq = n_freq
        self.latent_dim = latent_dim
        self.rnn_dim = rnn_dim
        self.num_sources = num_sources
        self.rnn_layers = rnn_layers
        self.rnn_type = rnn_type
        self.inference_rnn = inference_rnn
        self.autoencoder = autoencoder

        # Without the inference RNN the posterior heads read the raw per-step features,
        # which are then sized to hold a (mean, log-variance) pair.
        feature_dim = rnn_dim if inference_rnn else latent_dim * 2
        hidden = (latent_dim + feature_dim) // 2
        self.feature_dim = feature_dim

        conv_channels = layer_channels[-1]
        freq_features = n_freq
        for _ in layer_channels:
            freq_features = conv_output_size(freq_features, kernel_size[0], stride[0], padding[0])
        encoded_features = freq_features * conv_channels
        self.conv_channels = conv_channels

        if inference_rnn:
            self.gru_infer = _rnn(
                rnn_type, feature_dim, feature_dim, rnn_layers, bidirectional=True
            )
        self.gru_generate = _rnn(rnn_type, latent_dim, rnn_dim, rnn_layers, bidirectional=False)

        self.mu_q = _mlp(feature_dim, hidden, latent_dim)
        self.logvar_q = _mlp(feature_dim, hidden, latent_dim)
        self.mu0_q = _mlp(feature_dim, hidden, latent_dim)
        self.logvar0_q = _mlp(feature_dim, hidden, latent_dim)
        self.mu_p = _mlp(rnn_dim, hidden, latent_dim)
        self.logvar_p = _mlp(rnn_dim, hidden, latent_dim)
        self.transition_z = nn.Sequential(nn.Linear(latent_dim, rnn_dim), nn.LeakyReLU())

        conv = dict(kernel_size=kernel_size, stride=stride, padding=padding)
        if encoder == "strided":
            self.Encoder = StridedEncoder(1, layer_channels, **conv)
            self.Decoder = StridedDecoder(
                conv_channels, 1, layer_channels, activate_final_layer=activate_final_layer, **conv
            )
            self.fc_encoder = nn.Sequential(
                nn.Conv1d(
                    encoded_features, feature_dim * num_sources, separator_kernel_size,
                    padding=separator_kernel_size // 2,
                ),
                nn.LeakyReLU(),
            )  # fmt: skip
            self.fc_decoder = nn.Sequential(
                nn.ConvTranspose1d(latent_dim, encoded_features, kernel_size=1),
                nn.LeakyReLU(),
                nn.BatchNorm1d(encoded_features),
            )
        else:
            self.Encoder = StridedDilatedEncoder(1, layer_channels, **conv)
            self.Decoder = StridedDilatedDecoder(
                conv_channels, 1, layer_channels, activate_final_layer=activate_final_layer, **conv
            )
            temporal = dict(
                kernel_size=separator_kernel_size,
                layers=separator_layers,
                hidden_channels=separator_channels,
                first_layer=len(layer_channels),
            )
            self.fc_encoder = TConvSeparator(
                encoded_features, feature_dim * num_sources, **temporal
            )
            self.fc_decoder = TConvTranspose(latent_dim, encoded_features, **temporal)

        self.activation = nn.Sigmoid()
        # Per-source prior over the latent states: the initial-state prior is N(smu0, 1), and
        # the dynamical prior is multiplied by N(smu, 1) to anchor each source's latent code.
        self.register_parameter(
            "variance", nn.Parameter(torch.tensor([variance]), requires_grad=False)
        )
        self.register_parameter("smu", nn.Parameter(torch.zeros(1, num_sources, latent_dim)))
        self.register_parameter("smu0", nn.Parameter(torch.zeros(1, num_sources, latent_dim)))
        self.register_buffer("slogvar", torch.zeros(1, num_sources, latent_dim))
        self.register_buffer("slogvar0", torch.zeros(1, num_sources, latent_dim))

    # ------------------------------------------------------------------ inference
    def feature_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional features per source: ``(B, F, T)`` -> ``(B * K, T', feature_dim)``."""
        features = self.Encoder(x.unsqueeze(1))
        features = features.reshape(x.shape[0], -1, features.shape[-1])
        features = self.fc_encoder(features)
        return features.reshape(-1, self.feature_dim, features.shape[-1]).transpose(-2, -1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Approximate posterior of the latent states.

        Args:
            x: Mixture magnitudes, shape ``(B, F, T)``.

        Returns:
            ``(mu, logvar, mu0, logvar0)``: posterior of the state sequence, each
            ``(B * K, T', D)``, and of the initial state, each ``(B * K, 1, D)``.
        """
        g = self.feature_encoder(x)
        if self.inference_rnn:
            g, _ = self.gru_infer(g)
            # Average the forward and backward passes.
            g = 0.5 * (g[..., : self.feature_dim] + g[..., self.feature_dim :])
        first = g[:, :1]
        return self.mu_q(g), self.logvar_q(g), self.mu0_q(first), self.logvar0_q(first)

    # ----------------------------------------------------------------- generation
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states ``(B * K, T', D)`` into source magnitudes ``(B * K, 1, F, T)``."""
        z = self.fc_decoder(z.transpose(-2, -1))
        z = z.reshape(z.shape[0], self.conv_channels, -1, z.shape[-1])
        return self.activation(self.Decoder(z))

    def _per_source(self, parameter: torch.Tensor, batch: int) -> torch.Tensor:
        """Tile a ``(1, K, D)`` per-source tensor to ``(batch * K, 1, D)``."""
        return parameter.repeat(batch, 1, 1).view(-1, 1, self.latent_dim)

    def _combine(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return 0.5 * (self.transition_z(z) + h)

    def _prior_stats(self, zh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = zh.shape[0] // self.num_sources
        return product_of_gaussians(
            self.mu_p(zh),
            self.logvar_p(zh),
            self._per_source(self.smu, batch),
            self._per_source(self.slogvar, batch),
        )

    def prior(self, z: torch.Tensor, z0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dynamical prior ``p(z_t | z_<t)`` evaluated along a state sequence.

        Args:
            z: States, shape ``(B * K, T', D)``.
            z0: Initial states, shape ``(B * K, 1, D)``.
        """
        previous = torch.cat([z0, z[:, :-1]], dim=1)
        h, _ = self.gru_generate(previous)
        return self._prior_stats(self._combine(previous, h))

    @torch.no_grad()
    def sample(self, n_steps: int = 10, n_samples: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate from the prior.

        Returns:
            Source magnitudes ``(n_samples, K, F, T)`` and latent states
            ``(n_samples * K, n_steps, D)``.
        """
        batch = n_samples * self.num_sources
        z_t = reparameterize(
            self._per_source(self.smu0, n_samples), self._per_source(self.slogvar0, n_samples)
        )
        state, states = None, []
        for _ in range(n_steps):
            h, state = self.gru_generate(z_t, state)
            mu, logvar = self._prior_stats(self._combine(z_t, h))
            z_t = reparameterize(mu, logvar)
            states.append(z_t)
        z = torch.cat(states, dim=1)
        sources = self.decode(z)
        return sources.view(n_samples, self.num_sources, *sources.shape[-2:]), z.view(
            batch, n_steps, -1
        )

    # -------------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor, stochastic: bool = True) -> dict[str, torch.Tensor]:
        """Encode a mixture, sample latent states and reconstruct the sources.

        Args:
            x: Mixture magnitudes ``(B, F, T)``.
            stochastic: Sample states from the posterior. If false (or for an auto-encoder),
                use the posterior means, as at separation time.

        Returns:
            Dictionary with ``recon_x`` ``(B, F, T)`` (sum of the sources), ``recon_s``
            ``(B, K, F, T)``, the posterior and prior statistics, ``z``, and the
            hyper-parameters the loss needs (``smu0``, ``slogvar0``, ``variance``).
        """
        mu_q, logvar_q, mu0_q, logvar0_q = self.encode(x)
        if stochastic and not self.autoencoder:
            z = reparameterize(mu_q, logvar_q)
            z0 = reparameterize(mu0_q, logvar0_q)
        else:
            z, z0 = mu_q, mu0_q
        mu_p, logvar_p = self.prior(z, z0)

        sources = self.decode(z)
        recon_s = sources.view(-1, self.num_sources, *sources.shape[-2:])
        batch = x.shape[0]
        return {
            "recon_x": recon_s.sum(1),
            "recon_s": recon_s,
            "mu_q": mu_q,
            "logvar_q": logvar_q,
            "mu_p": mu_p,
            "logvar_p": logvar_p,
            "mu0_q": mu0_q,
            "logvar0_q": logvar0_q,
            "smu0": self._per_source(self.smu0, batch),
            "slogvar0": self._per_source(self.slogvar0, batch),
            "variance": self.variance,
            "z": z,
        }
