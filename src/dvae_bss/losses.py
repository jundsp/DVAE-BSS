"""Variational objective: likelihoods, KL divergence and the combined negative ELBO.

The model is trained on the *mixture* only. The decoded sources are summed to a mixture
estimate, whose likelihood is evaluated against the observed mixture spectrogram.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def kl_divergence(
    mean: torch.Tensor,
    logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """Sum of ``KL(N(mean, exp(logvar)) || N(prior_mean, exp(prior_logvar)))`` over all entries."""
    return -0.5 * torch.sum(
        1 + logvar - prior_logvar - (logvar.exp() + (mean - prior_mean).pow(2)) / prior_logvar.exp()
    )


def optimal_gain(target: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    """Per-example least-squares gain ``a`` minimising ``|target - a * estimate|``.

    Args:
        target: Observed spectrograms, shape ``(B, ...)``.
        estimate: Model output with the same shape.

    Returns:
        Gains of shape ``(B, 1, ..., 1)``. The gain is detached: it rescales the estimate
        but is not optimised through, so the loss cannot be reduced by shrinking the output.
    """
    # Least squares: a = <target, estimate> / <estimate, estimate>, one gain per example.
    estimate = estimate.detach()
    energy = (estimate * estimate).flatten(1).sum(1)
    # Guard against an all-zero or non-finite estimate (e.g. early in training).
    degenerate = (energy.abs() < 1e-9) | ~torch.isfinite(energy)
    energy = torch.where(degenerate, torch.ones_like(energy), energy)
    gain = (target * estimate).flatten(1).sum(1) / (energy + 1e-6)
    return gain.view(-1, *([1] * (target.ndim - 1)))


class NormalLikelihood(nn.Module):
    """Negative log-likelihood (up to a constant) of a Gaussian with fixed variance."""

    def __init__(self, scale_invariant: bool = True) -> None:
        super().__init__()
        self.scale_invariant = scale_invariant

    def forward(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of ``x`` given mean estimate ``mean`` and variance ``var``."""
        # Scale invariance: compare the mixture with the best-scaled estimate, so the model
        # is not penalised for the overall level of its output.
        gain = optimal_gain(x, mean) if self.scale_invariant else 1.0
        eps = x - mean * gain
        return 0.5 * torch.sum(var.log() + eps.pow(2) / var)


class LaplaceLikelihood(nn.Module):
    """Negative log-likelihood of a Laplace distribution with variance ``var``."""

    def __init__(self, scale_invariant: bool = True) -> None:
        super().__init__()
        self.scale_invariant = scale_invariant

    def forward(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood of ``x`` given mean estimate ``mean`` and variance ``var``."""
        scale = torch.sqrt(0.5 * var)
        gain = optimal_gain(x, mean) if self.scale_invariant else 1.0
        l1 = (mean * gain - x).abs()
        return torch.sum(math.log(2.0) + scale.log() + l1 / scale)


class ELBOLoss(nn.Module):
    """Negative evidence lower bound, split into likelihood and KL terms.

    Args:
        scale_invariant: Rescale the mixture estimate to the observation before comparing.
        laplace: Use a Laplace instead of a Gaussian likelihood.
        autoencoder: Drop the KL term, which turns the VAE into a plain auto-encoder.
        averaged: Normalise by the number of spectrogram entries instead of by batch size.
    """

    def __init__(
        self,
        scale_invariant: bool = True,
        laplace: bool = False,
        autoencoder: bool = False,
        averaged: bool = True,
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.averaged = averaged
        cls = LaplaceLikelihood if laplace else NormalLikelihood
        self.likelihood = cls(scale_invariant=scale_invariant)

    def forward(
        self, x: torch.Tensor, output: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(likelihood term, KL term)``; their sum is the loss to minimise.

        Args:
            x: Mixture spectrogram ``(B, F, T)``.
            output: Dictionary returned by :meth:`dvae_bss.model.DVAE.forward`.
        """
        norm = float(x.numel() if self.averaged else x.size(0))
        # Reconstruction term: how well the *sum* of the decoded sources explains the mixture.
        # Nothing compares individual sources with anything, as none are observed.
        ell = self.likelihood(x, output["recon_x"], output["variance"])
        # Regularisation term: the posterior trajectories must stay close to the dynamical
        # prior. The KL is taken step by step, between q(z_t | X) and p(z_t | z_<t) evaluated
        # at the sampled previous states, plus the KL of the initial state against N(smu0, 1).
        kld = kl_divergence(
            output["mu_q"], output["logvar_q"], output["mu_p"], output["logvar_p"]
        ) + kl_divergence(output["mu0_q"], output["logvar0_q"], output["smu0"], output["slogvar0"])
        if self.autoencoder:
            kld = kld * 0
        return ell / norm, kld / norm
