"""FedProx aggregation strategy for heterogeneous federated learning.

Extends FedAvg with a proximal term that penalizes client models for
drifting too far from the global model. This stabilizes training when
clients have heterogeneous (non-IID) data distributions. From Li et al.
(2020), "Federated Optimization in Heterogeneous Networks."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .fedavg import AggregationResult, ClientUpdate, fedavg_aggregate


@dataclass
class FedProxConfig:
    """Configuration for FedProx.

    Attributes:
        mu: Proximal term coefficient. Higher values keep clients
            closer to the global model. mu=0 reduces to FedAvg.
        local_epochs: Number of local training epochs per round.
        learning_rate: Client-side learning rate.
    """

    mu: float = 0.01
    local_epochs: int = 5
    learning_rate: float = 0.01


def compute_proximal_loss(
    model: nn.Module,
    global_weights: Dict[str, torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """Compute the proximal regularization term.

    The proximal term is: (mu / 2) * ||w - w_global||^2

    This penalizes the local model for deviating from the global model,
    which helps with convergence under non-IID data.

    Args:
        model: Current local model.
        global_weights: Global model weights to regularize toward.
        mu: Proximal coefficient.

    Returns:
        Scalar proximal loss tensor.
    """
    prox_loss = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if name in global_weights:
            global_param = global_weights[name].to(param.device)
            prox_loss = prox_loss + torch.sum((param - global_param) ** 2)
    return (mu / 2.0) * prox_loss


def fedprox_local_step(
    model: nn.Module,
    data_batch: torch.Tensor,
    targets: torch.Tensor,
    global_weights: Dict[str, torch.Tensor],
    config: FedProxConfig,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    """Perform one FedProx local training step.

    Adds the proximal term to the standard cross-entropy loss:
        L_total = L_task + (mu/2) * ||w - w_global||^2

    Args:
        model: Local model to train.
        data_batch: Input data tensor.
        targets: Target labels.
        global_weights: Global model weights for proximal term.
        config: FedProx configuration.
        loss_fn: Task loss function (e.g., CrossEntropyLoss).
        optimizer: PyTorch optimizer.

    Returns:
        Dictionary with task_loss, prox_loss, and total_loss.
    """
    model.train()
    optimizer.zero_grad()

    logits = model(data_batch)
    task_loss = loss_fn(logits, targets)
    prox_loss = compute_proximal_loss(model, global_weights, config.mu)
    total_loss = task_loss + prox_loss

    total_loss.backward()
    optimizer.step()

    return {
        "task_loss": task_loss.item(),
        "prox_loss": prox_loss.item(),
        "total_loss": total_loss.item(),
    }


def fedprox_aggregate(
    updates: List[ClientUpdate],
    global_weights: Optional[Dict[str, torch.Tensor]] = None,
) -> AggregationResult:
    """Aggregate FedProx client updates.

    The aggregation step is identical to FedAvg — the proximal term
    only affects local training. The difference is that client updates
    will be closer to the global model due to the regularization.

    Args:
        updates: List of client updates.
        global_weights: Previous global weights (for reference).

    Returns:
        AggregationResult with new global weights.
    """
    return fedavg_aggregate(updates, global_weights)


def compute_client_drift(
    updates: List[ClientUpdate],
    global_weights: Dict[str, torch.Tensor],
) -> Dict[int, float]:
    """Measure how far each client drifted from the global model.

    Useful for monitoring the effect of the proximal term.

    Args:
        updates: Client updates after local training.
        global_weights: Global model weights before the round.

    Returns:
        Dictionary mapping client_id to L2 drift distance.
    """
    drifts = {}
    for update in updates:
        total = 0.0
        for key in global_weights:
            diff = update.weights[key].float() - global_weights[key].float()
            total += torch.sum(diff ** 2).item()
        drifts[update.client_id] = total ** 0.5
    return drifts


if __name__ == "__main__":
    print("=== FedProx Demo ===\n")

    # Create a simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    global_weights = {n: p.data.clone() for n, p in model.state_dict().items()}

    config = FedProxConfig(mu=0.01, learning_rate=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Simulate local training steps
    print(f"FedProx config: mu={config.mu}, lr={config.learning_rate}")
    for step in range(5):
        batch = torch.randn(16, 1, 28, 28)
        targets = torch.randint(0, 10, (16,))
        metrics = fedprox_local_step(
            model, batch, targets, global_weights, config, loss_fn, optimizer
        )
        print(f"  Step {step}: task={metrics['task_loss']:.4f}, "
              f"prox={metrics['prox_loss']:.6f}, total={metrics['total_loss']:.4f}")

    # Compare drift with different mu values
    print("\nDrift comparison (5 local steps):")
    for mu in [0.0, 0.01, 0.1, 1.0]:
        test_model = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 10),
        )
        test_model.load_state_dict(
            {k: v.clone() for k, v in global_weights.items()}
        )
        test_opt = torch.optim.SGD(test_model.parameters(), lr=0.01)
        test_config = FedProxConfig(mu=mu)

        for _ in range(5):
            b = torch.randn(16, 1, 28, 28)
            t = torch.randint(0, 10, (16,))
            fedprox_local_step(test_model, b, t, global_weights, test_config, loss_fn, test_opt)

        drift = 0.0
        for name, param in test_model.state_dict().items():
            drift += torch.sum((param.float() - global_weights[name].float()) ** 2).item()
        print(f"  mu={mu:.2f}: drift = {drift ** 0.5:.4f}")
