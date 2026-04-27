"""Federated Averaging (FedAvg) aggregation strategy.

Implements the FedAvg algorithm from McMahan et al. (2017). The server
collects model updates from selected clients, computes a weighted average
of their parameters, and broadcasts the new global model. Weighting is
proportional to each client's dataset size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class ClientUpdate:
    """A model update from a single client.

    Attributes:
        client_id: Identifier for the client.
        weights: Model state dict after local training.
        num_samples: Number of training samples used.
        metrics: Optional training metrics (loss, accuracy, etc.).
    """

    client_id: int
    weights: Dict[str, torch.Tensor]
    num_samples: int
    metrics: Optional[Dict[str, float]] = None


@dataclass
class AggregationResult:
    """Result of aggregating client updates.

    Attributes:
        global_weights: New global model weights.
        num_clients: Number of clients that contributed.
        total_samples: Total samples across all clients.
        client_contributions: Weight given to each client.
    """

    global_weights: Dict[str, torch.Tensor]
    num_clients: int
    total_samples: int
    client_contributions: Dict[int, float]


def fedavg_aggregate(
    updates: List[ClientUpdate],
    global_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> AggregationResult:
    """Aggregate client updates using Federated Averaging.

    Computes a weighted average of client model weights, where each
    client's contribution is proportional to its dataset size.

    w_global = sum(n_k * w_k) / sum(n_k)

    Args:
        updates: List of client updates with weights and sample counts.
        global_weights: Previous global weights (unused in standard FedAvg,
            included for API consistency with FedProx).

    Returns:
        AggregationResult with the new global model weights.

    Raises:
        ValueError: If no updates are provided.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty update list")

    total_samples = sum(u.num_samples for u in updates)
    contributions = {
        u.client_id: u.num_samples / total_samples for u in updates
    }

    # Weighted average of all parameters
    new_weights: Dict[str, torch.Tensor] = {}
    for key in updates[0].weights:
        weighted_sum = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)
        for update in updates:
            weight = update.num_samples / total_samples
            weighted_sum += weight * update.weights[key].float()
        new_weights[key] = weighted_sum

    return AggregationResult(
        global_weights=new_weights,
        num_clients=len(updates),
        total_samples=total_samples,
        client_contributions=contributions,
    )


def fedavg_aggregate_deltas(
    global_weights: Dict[str, torch.Tensor],
    updates: List[ClientUpdate],
    learning_rate: float = 1.0,
) -> AggregationResult:
    """Aggregate client weight deltas and apply to global model.

    Instead of averaging absolute weights, this computes the weighted
    average of (client_weights - global_weights) and applies it as
    an update. This is equivalent to standard FedAvg but allows
    controlling the server-side learning rate.

    Args:
        global_weights: Current global model weights.
        updates: List of client updates.
        learning_rate: Server-side learning rate (1.0 = standard FedAvg).

    Returns:
        AggregationResult with updated global weights.
    """
    if not updates:
        raise ValueError("Cannot aggregate empty update list")

    total_samples = sum(u.num_samples for u in updates)
    contributions = {
        u.client_id: u.num_samples / total_samples for u in updates
    }

    new_weights: Dict[str, torch.Tensor] = {}
    for key in global_weights:
        delta = torch.zeros_like(global_weights[key], dtype=torch.float32)
        for update in updates:
            weight = update.num_samples / total_samples
            client_delta = update.weights[key].float() - global_weights[key].float()
            delta += weight * client_delta
        new_weights[key] = global_weights[key].float() + learning_rate * delta

    return AggregationResult(
        global_weights=new_weights,
        num_clients=len(updates),
        total_samples=total_samples,
        client_contributions=contributions,
    )


def compute_update_norm(
    global_weights: Dict[str, torch.Tensor],
    client_weights: Dict[str, torch.Tensor],
) -> float:
    """Compute L2 norm of the update (client - global).

    Args:
        global_weights: Global model weights.
        client_weights: Client model weights after local training.

    Returns:
        L2 norm of the weight difference.
    """
    total = 0.0
    for key in global_weights:
        diff = client_weights[key].float() - global_weights[key].float()
        total += torch.sum(diff ** 2).item()
    return total ** 0.5


if __name__ == "__main__":
    print("=== FedAvg Aggregation Demo ===\n")

    # Simulate 3 clients with different dataset sizes
    torch.manual_seed(42)
    param_keys = ["layer1.weight", "layer1.bias", "layer2.weight", "layer2.bias"]
    shapes = [(64, 784), (64,), (10, 64), (10,)]

    # Create "global" model
    global_w = {k: torch.randn(s) for k, s in zip(param_keys, shapes)}

    # Simulate client updates (slightly perturbed from global)
    updates = []
    for i in range(3):
        client_w = {k: v + 0.1 * torch.randn_like(v) for k, v in global_w.items()}
        n_samples = [500, 300, 200][i]
        updates.append(ClientUpdate(
            client_id=i,
            weights=client_w,
            num_samples=n_samples,
        ))
        norm = compute_update_norm(global_w, client_w)
        print(f"Client {i}: {n_samples} samples, update norm = {norm:.4f}")

    # Standard FedAvg
    result = fedavg_aggregate(updates)
    print(f"\nFedAvg result:")
    print(f"  Clients: {result.num_clients}, Total samples: {result.total_samples}")
    print(f"  Contributions: {result.client_contributions}")

    # Delta-based FedAvg with server LR
    result_delta = fedavg_aggregate_deltas(global_w, updates, learning_rate=0.5)
    drift = compute_update_norm(global_w, result_delta.global_weights)
    print(f"\nDelta FedAvg (server_lr=0.5): global drift = {drift:.4f}")
