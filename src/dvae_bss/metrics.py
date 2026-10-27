"""Source-separation metrics: SI-SDR, SIR and SAR with permutation-invariant matching.

Estimates are first matched to the references with the Hungarian algorithm, then each
reference is rescaled to the estimate's gain (scale invariance), then BSS-eval style
projections give the signal-to-distortion/interference/artifact ratios.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

EPS = 1e-12


def optimal_permute(
    estimates: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, np.ndarray]:
    """Reorder estimates to best match the targets (minimum summed L1 distance).

    Args:
        estimates: ``(B, K_est, L)`` estimated waveforms.
        targets: ``(B, K_ref, L)`` reference waveforms, ``K_est >= K_ref``.

    Returns:
        The reordered estimates, shape ``(B, K_est, L)``, with the matched estimates first
        and the unmatched ones after them, and the permutation indices ``(B, K_est)``.
    """
    n, k_est, k_ref = estimates.shape[0], estimates.shape[1], targets.shape[1]
    cost = torch.cdist(estimates, targets, p=1).detach().cpu().numpy()
    perm = np.zeros((n, k_est), dtype=int)
    for i in range(n):
        matched = linear_sum_assignment(cost[i].T)[1]
        unmatched = sorted(set(range(k_est)) - set(matched))
        perm[i, :k_ref] = matched
        perm[i, k_ref:] = unmatched
    index = torch.from_numpy(perm).to(estimates.device)
    out = torch.gather(estimates, 1, index.unsqueeze(-1).expand(-1, -1, estimates.shape[-1]))
    return out, perm


def make_scale_invariant(estimates: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Scale each target by the gain that best fits the matching estimate."""
    estimates = estimates[:, : targets.shape[1]]
    energy = (targets * targets).sum(-1)
    gain = (estimates * targets).sum(-1) / (energy + EPS)
    return targets * gain.unsqueeze(-1)


def compute_measures_batch(
    estimates: torch.Tensor, targets: torch.Tensor, salt: float = EPS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SDR, SIR and SAR (dB) of each estimate against all targets.

    Args:
        estimates: ``(B, K, L)`` estimates, already matched to the targets.
        targets: ``(B, K, L)`` references, scaled by :func:`make_scale_invariant`.
        salt: Small constant that keeps the projections and logarithms finite.

    Returns:
        Three ``(B, K)`` tensors: SDR (equal to SI-SDR for scaled targets), SIR, SAR.
    """
    k = targets.shape[1]
    gram = targets @ targets.transpose(-1, -2) + torch.eye(k, device=targets.device) * salt
    measures = [_measures_for_source(estimates[:, j], targets, gram, j, salt) for j in range(k)]
    sdr, sir, sar = (torch.stack(m, dim=1) for m in zip(*measures, strict=True))
    return sdr, sir, sar


def _measures_for_source(
    estimate: torch.Tensor, targets: torch.Tensor, gram: torch.Tensor, j: int, salt: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = targets[:, j]
    residual = estimate - target
    # Project the residual on all targets; the projection is the interference.
    cross = (residual.unsqueeze(1) * targets).sum(-1)
    coefficients = torch.linalg.solve(gram, cross)
    interference = (targets * coefficients.unsqueeze(-1)).sum(1)
    artifacts = residual - interference

    def energy(x: torch.Tensor) -> torch.Tensor:
        return (x * x).sum(-1) + salt

    e_target = energy(target)
    return (
        10 * (e_target.log10() - energy(residual).log10()),
        10 * (e_target.log10() - energy(interference).log10()),
        10 * (e_target.log10() - energy(artifacts).log10()),
    )


def bss_metrics(estimates: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
    """Permutation- and scale-invariant SI-SDR, SIR and SAR of a batch.

    Args:
        estimates: ``(B, K_est, L)`` estimated sources (``K_est >= K_ref``).
        targets: ``(B, K_ref, L)`` reference sources.

    Returns:
        ``{"sdr", "sir", "sar", "estimates"}`` where the metrics are ``(B, K_ref)`` in dB and
        ``estimates`` holds the matched estimates.
    """
    matched, _ = optimal_permute(estimates, targets)
    scaled = make_scale_invariant(matched, targets)
    sdr, sir, sar = compute_measures_batch(matched[:, : targets.shape[1]], scaled)
    return {"sdr": sdr, "sir": sir, "sar": sar, "estimates": matched}


def ideal_masks(
    stft_sources: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Oracle ratio and binary mask estimates of every source from its true STFT.

    Args:
        stft_sources: Complex STFTs ``(B, K, F, T)`` of the true sources.
        alpha: Exponent applied to the magnitudes before forming the mask.

    Returns:
        Ratio-masked and binary-masked complex STFTs of the mixture, ``(B, K, F, T)``.
    """
    mixture = stft_sources.sum(1, keepdim=True)
    magnitudes = stft_sources.abs() ** alpha
    mask = magnitudes / (magnitudes.sum(1, keepdim=True) + np.finfo(np.float64).eps)
    phase = torch.exp(1j * mixture.angle())
    base = mixture.abs() ** alpha
    return base * mask * phase, base * mask.round() * phase
