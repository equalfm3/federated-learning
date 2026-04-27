"""Noise mechanisms for differential privacy.

Implements Gaussian and Laplace noise calibrated to privacy parameters.
These mechanisms are the building blocks for DP-SGD and other
privacy-preserving algorithms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch


class NoiseType(Enum):
    """Supported noise distribution types."""

    GAUSSIAN = "gaussian"
    LAPLACE = "laplace"


@dataclass
class NoiseConfig:
    """Configuration for noise generation.

    Attributes:
        noise_type: Distribution type (Gaussian or Laplace).
        noise_multiplier: Scale factor relative to sensitivity.
        sensitivity: L2 (Gaussian) or L1 (Laplace) sensitivity of the query.
        seed: Optional random seed for reproducibility.
    """

    noise_type: NoiseType = NoiseType.GAUSSIAN
    noise_multiplier: float = 1.0
    sensitivity: float = 1.0
    seed: Optional[int] = None


def calibrate_gaussian_sigma(
    epsilon: float,
    delta: float,
    sensitivity: float,
) -> float:
    """Calibrate Gaussian noise standard deviation for (epsilon, delta)-DP.

    Uses the analytic Gaussian mechanism formula:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Args:
        epsilon: Privacy budget parameter.
        delta: Failure probability parameter.
        sensitivity: L2 sensitivity of the query.

    Returns:
        Standard deviation sigma for the Gaussian noise.

    Raises:
        ValueError: If epsilon or delta are non-positive.
    """
    if epsilon <= 0 or delta <= 0:
        raise ValueError(f"epsilon ({epsilon}) and delta ({delta}) must be positive")
    return sensitivity * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


def calibrate_laplace_scale(
    epsilon: float,
    sensitivity: float,
) -> float:
    """Calibrate Laplace noise scale for epsilon-DP.

    The Laplace mechanism adds noise with scale b = sensitivity / epsilon.

    Args:
        epsilon: Privacy budget parameter.
        sensitivity: L1 sensitivity of the query.

    Returns:
        Scale parameter b for the Laplace distribution.

    Raises:
        ValueError: If epsilon is non-positive.
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon ({epsilon}) must be positive")
    return sensitivity / epsilon


def add_noise(
    tensor: torch.Tensor,
    config: NoiseConfig,
) -> torch.Tensor:
    """Add calibrated noise to a tensor.

    Args:
        tensor: Input tensor to perturb.
        config: Noise configuration specifying type and scale.

    Returns:
        Noisy tensor with same shape as input.
    """
    generator = None
    if config.seed is not None:
        generator = torch.Generator()
        generator.manual_seed(config.seed)

    scale = config.noise_multiplier * config.sensitivity

    if config.noise_type == NoiseType.GAUSSIAN:
        noise = torch.normal(
            mean=0.0,
            std=scale,
            size=tensor.shape,
            generator=generator,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    elif config.noise_type == NoiseType.LAPLACE:
        # Laplace via inverse CDF: L(0, b) = -b * sign(U-0.5) * ln(1 - 2|U-0.5|)
        u = torch.zeros_like(tensor).uniform_(-0.5, 0.5, generator=generator)
        noise = -scale * u.sign() * torch.log1p(-2.0 * u.abs())
    else:
        raise ValueError(f"Unknown noise type: {config.noise_type}")

    return tensor + noise


def clip_tensor(
    tensor: torch.Tensor,
    max_norm: float,
) -> torch.Tensor:
    """Clip a tensor to have at most the given L2 norm.

    This is the per-sample gradient clipping step in DP-SGD.

    Args:
        tensor: Input tensor to clip.
        max_norm: Maximum L2 norm.

    Returns:
        Clipped tensor with L2 norm <= max_norm.
    """
    norm = torch.norm(tensor.float(), p=2)
    clip_factor = max_norm / (norm + 1e-8)
    if clip_factor < 1.0:
        return tensor * clip_factor
    return tensor.clone()


def estimate_noise_impact(
    model_dim: int,
    noise_multiplier: float,
    sensitivity: float,
    batch_size: int,
) -> float:
    """Estimate the expected L2 norm of noise added per aggregation step.

    Useful for understanding the signal-to-noise ratio in DP-SGD.

    Args:
        model_dim: Total number of model parameters.
        noise_multiplier: Noise multiplier (sigma / sensitivity).
        sensitivity: Clipping norm (L2 sensitivity).
        batch_size: Number of samples in the batch.

    Returns:
        Expected L2 norm of the noise vector.
    """
    sigma = noise_multiplier * sensitivity / batch_size
    # E[||N(0, sigma^2 I_d)||_2] ≈ sigma * sqrt(d)
    return sigma * math.sqrt(model_dim)


if __name__ == "__main__":
    print("=== Noise Mechanisms Demo ===\n")

    # Gaussian calibration
    eps, delta, sens = 1.0, 1e-5, 1.0
    sigma = calibrate_gaussian_sigma(eps, delta, sens)
    print(f"Gaussian mechanism: eps={eps}, delta={delta}, sensitivity={sens}")
    print(f"  Calibrated sigma: {sigma:.4f}\n")

    # Laplace calibration
    scale = calibrate_laplace_scale(eps, sens)
    print(f"Laplace mechanism: eps={eps}, sensitivity={sens}")
    print(f"  Calibrated scale: {scale:.4f}\n")

    # Add noise to a tensor
    x = torch.ones(5)
    for noise_type in [NoiseType.GAUSSIAN, NoiseType.LAPLACE]:
        config = NoiseConfig(noise_type=noise_type, noise_multiplier=sigma, seed=42)
        noisy = add_noise(x, config)
        print(f"{noise_type.value} noise on ones(5): {noisy.tolist()}")

    # Gradient clipping
    print(f"\nGradient clipping:")
    grad = torch.randn(100)
    original_norm = torch.norm(grad).item()
    clipped = clip_tensor(grad, max_norm=1.0)
    clipped_norm = torch.norm(clipped).item()
    print(f"  Original norm: {original_norm:.4f}")
    print(f"  Clipped norm:  {clipped_norm:.4f}")

    # Noise impact estimation
    impact = estimate_noise_impact(
        model_dim=100_000, noise_multiplier=1.0, sensitivity=1.0, batch_size=64
    )
    print(f"\nExpected noise L2 norm (100K params, batch=64): {impact:.2f}")
