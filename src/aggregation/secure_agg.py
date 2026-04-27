"""Secure aggregation simulation for federated learning.

Simulates the Bonawitz et al. (2017) secure aggregation protocol where
clients mask their updates with pairwise random masks that cancel out
during summation. The server only sees the aggregate — never individual
client updates. This implementation simulates the cryptographic protocol
without actual key exchange for educational purposes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class SecureAggConfig:
    """Configuration for secure aggregation.

    Attributes:
        num_clients: Total number of participating clients.
        threshold: Minimum clients needed to reconstruct (t-of-n).
        mask_seed_base: Base seed for deterministic mask generation.
        quantize_bits: Bits for quantization (0 = no quantization).
    """

    num_clients: int = 10
    threshold: int = 7
    mask_seed_base: int = 42
    quantize_bits: int = 0


@dataclass
class MaskedUpdate:
    """A client's masked model update.

    Attributes:
        client_id: Client identifier.
        masked_weights: Weights with pairwise masks applied.
        num_samples: Number of training samples.
    """

    client_id: int
    masked_weights: Dict[str, torch.Tensor]
    num_samples: int


def _generate_pairwise_seed(client_a: int, client_b: int, base_seed: int) -> int:
    """Generate a deterministic seed for a pair of clients.

    The seed is symmetric: seed(a, b) == seed(b, a), which ensures
    masks cancel during aggregation.

    Args:
        client_a: First client ID.
        client_b: Second client ID.
        base_seed: Base seed for the round.

    Returns:
        Integer seed for the pair.
    """
    pair = tuple(sorted([client_a, client_b]))
    h = hashlib.sha256(f"{pair[0]}:{pair[1]}:{base_seed}".encode())
    return int.from_bytes(h.digest()[:8], "big")


def generate_mask(
    shape: torch.Size,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate a pseudorandom mask tensor from a seed.

    Args:
        shape: Shape of the mask tensor.
        seed: Random seed for reproducibility.
        dtype: Data type of the mask.

    Returns:
        Random mask tensor with zero mean.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    return torch.randn(shape, generator=gen, dtype=dtype)


def mask_client_update(
    client_id: int,
    weights: Dict[str, torch.Tensor],
    all_client_ids: List[int],
    config: SecureAggConfig,
    num_samples: int = 1,
    total_samples: int = 1,
) -> Dict[str, torch.Tensor]:
    """Apply pairwise masks to a client's weighted model contribution.

    Masks are applied to the weighted contribution (n_k/N * w_k) so that
    they cancel when summed across all clients.

    Args:
        client_id: This client's ID.
        weights: Model weights to mask.
        all_client_ids: IDs of all participating clients.
        config: Secure aggregation configuration.
        num_samples: This client's sample count.
        total_samples: Total samples across all clients.

    Returns:
        Masked weight dictionary.
    """
    weight_factor = num_samples / total_samples if total_samples > 0 else 1.0
    masked = {k: v.clone().float() * weight_factor for k, v in weights.items()}

    for other_id in all_client_ids:
        if other_id == client_id:
            continue

        seed = _generate_pairwise_seed(client_id, other_id, config.mask_seed_base)

        for key in masked:
            mask = generate_mask(masked[key].shape, seed)
            if client_id < other_id:
                masked[key] = masked[key] + mask
            else:
                masked[key] = masked[key] - mask

    return masked


def secure_aggregate(
    masked_updates: List[MaskedUpdate],
    config: SecureAggConfig,
) -> Dict[str, torch.Tensor]:
    """Aggregate masked client updates.

    Sums all masked updates. Because pairwise masks cancel, the result
    is the sum of the original (unmasked) weights. The server then
    divides by total samples for the weighted average.

    Args:
        masked_updates: List of masked client updates.
        config: Secure aggregation configuration.

    Returns:
        Aggregated (unmasked) weight dictionary.

    Raises:
        ValueError: If fewer than threshold clients participate.
    """
    if len(masked_updates) < config.threshold:
        raise ValueError(
            f"Need at least {config.threshold} clients, got {len(masked_updates)}"
        )

    total_samples = sum(u.num_samples for u in masked_updates)

    # Sum all masked weights (masks cancel, pre-weighted contributions remain)
    result: Dict[str, torch.Tensor] = {}
    for key in masked_updates[0].masked_weights:
        total = torch.zeros_like(
            masked_updates[0].masked_weights[key], dtype=torch.float32
        )
        for update in masked_updates:
            total += update.masked_weights[key].float()
        result[key] = total

    return result


def verify_mask_cancellation(
    original_weights: List[Dict[str, torch.Tensor]],
    sample_counts: List[int],
    aggregated: Dict[str, torch.Tensor],
    tolerance: float = 1e-4,
) -> bool:
    """Verify that masks properly cancelled during aggregation.

    Computes the expected weighted average from original (unmasked) weights
    and compares to the secure aggregation result.

    Args:
        original_weights: List of original client weights.
        sample_counts: Number of samples per client.
        aggregated: Result from secure_aggregate.
        tolerance: Maximum allowed L2 error.

    Returns:
        True if the aggregation matches the expected result.
    """
    total = sum(sample_counts)
    for key in aggregated:
        expected = torch.zeros_like(aggregated[key])
        for w, n in zip(original_weights, sample_counts):
            expected += (n / total) * w[key].float()
        diff = torch.norm(aggregated[key] - expected).item()
        if diff > tolerance:
            return False
    return True


def quantize_weights(
    weights: Dict[str, torch.Tensor],
    bits: int = 8,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Tuple[float, float]]]:
    """Quantize weights to reduce communication cost.

    Maps floating-point weights to integer range [0, 2^bits - 1].

    Args:
        weights: Model weights to quantize.
        bits: Number of bits for quantization.

    Returns:
        Tuple of (quantized weights, scale parameters for dequantization).
    """
    max_val = (1 << bits) - 1
    quantized = {}
    scales = {}

    for key, tensor in weights.items():
        t_min = tensor.min().item()
        t_max = tensor.max().item()
        t_range = t_max - t_min if t_max != t_min else 1.0

        q = ((tensor - t_min) / t_range * max_val).round().clamp(0, max_val)
        quantized[key] = q
        scales[key] = (t_min, t_range)

    return quantized, scales


def dequantize_weights(
    quantized: Dict[str, torch.Tensor],
    scales: Dict[str, Tuple[float, float]],
    bits: int = 8,
) -> Dict[str, torch.Tensor]:
    """Dequantize weights back to floating point.

    Args:
        quantized: Quantized weight tensors.
        scales: Scale parameters from quantization.
        bits: Number of bits used in quantization.

    Returns:
        Dequantized floating-point weights.
    """
    max_val = (1 << bits) - 1
    result = {}
    for key, q in quantized.items():
        t_min, t_range = scales[key]
        result[key] = q / max_val * t_range + t_min
    return result


if __name__ == "__main__":
    print("=== Secure Aggregation Demo ===\n")

    num_clients = 5
    config = SecureAggConfig(num_clients=num_clients, threshold=3)
    client_ids = list(range(num_clients))
    param_keys, shapes = ["fc1.weight", "fc1.bias"], [(32, 16), (32,)]

    original_weights, sample_counts, masked_updates = [], [], []
    total_samples = sum(100 + cid * 50 for cid in client_ids)
    for cid in client_ids:
        weights = {k: torch.randn(s) for k, s in zip(param_keys, shapes)}
        n_samples = 100 + cid * 50
        original_weights.append(weights)
        sample_counts.append(n_samples)
        masked_w = mask_client_update(cid, weights, client_ids, config, n_samples, total_samples)
        masked_updates.append(MaskedUpdate(cid, masked_w, n_samples))
        print(f"Client {cid}: {n_samples} samples, masked update ready")

    aggregated = secure_aggregate(masked_updates, config)
    correct = verify_mask_cancellation(original_weights, sample_counts, aggregated)
    print(f"\nMask cancellation verified: {correct}")

    for bits in [8, 16]:
        q, scales = quantize_weights(original_weights[0], bits=bits)
        dq = dequantize_weights(q, scales, bits=bits)
        err = sum(torch.norm(original_weights[0][k] - dq[k]).item() for k in param_keys)
        print(f"  {bits}-bit quantization error: {err:.6f}")
