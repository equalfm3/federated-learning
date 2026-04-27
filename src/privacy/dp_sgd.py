"""Differentially Private Stochastic Gradient Descent (DP-SGD).

Implements the DP-SGD optimizer from Abadi et al. (2016) which provides
formal differential privacy guarantees during training. The key steps are:
1. Per-sample gradient computation
2. Gradient clipping to bound sensitivity
3. Noise addition calibrated to the clipping norm
4. Standard SGD update with the noisy gradient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .noise import NoiseConfig, NoiseType, add_noise, clip_tensor


@dataclass
class DPSGDConfig:
    """Configuration for DP-SGD optimizer.

    Attributes:
        learning_rate: SGD learning rate.
        max_grad_norm: Per-sample gradient clipping bound (L2).
        noise_multiplier: Noise scale relative to clipping norm.
        batch_size: Training batch size.
        noise_type: Type of noise to add (Gaussian for standard DP-SGD).
    """

    learning_rate: float = 0.01
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    batch_size: int = 32
    noise_type: NoiseType = NoiseType.GAUSSIAN


class DPSGD:
    """DP-SGD optimizer with per-sample gradient clipping and noise.

    This optimizer wraps a model and provides differentially private
    gradient updates. It clips each sample's gradient independently,
    averages the clipped gradients, and adds calibrated noise.

    Args:
        model: PyTorch model to optimize.
        config: DP-SGD configuration.
    """

    def __init__(self, model: nn.Module, config: DPSGDConfig) -> None:
        self.model = model
        self.config = config
        self._step_count: int = 0

    @property
    def step_count(self) -> int:
        """Number of optimization steps performed."""
        return self._step_count

    def compute_per_sample_gradients(
        self,
        loss_per_sample: torch.Tensor,
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute gradients for each sample independently.

        Uses backward on each per-sample loss to get individual gradients.
        This is the key difference from standard SGD where gradients are
        averaged before clipping.

        Args:
            loss_per_sample: Tensor of shape (batch_size,) with per-sample losses.

        Returns:
            List of gradient dictionaries, one per sample.
        """
        per_sample_grads: List[Dict[str, torch.Tensor]] = []

        for i in range(loss_per_sample.shape[0]):
            self.model.zero_grad()
            loss_per_sample[i].backward(retain_graph=True)

            sample_grad = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    sample_grad[name] = param.grad.clone()
            per_sample_grads.append(sample_grad)

        return per_sample_grads

    def clip_gradients(
        self,
        per_sample_grads: List[Dict[str, torch.Tensor]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Clip each sample's gradient to the maximum norm.

        Computes the global L2 norm across all parameters for each sample,
        then scales down if the norm exceeds max_grad_norm.

        Args:
            per_sample_grads: List of per-sample gradient dictionaries.

        Returns:
            List of clipped gradient dictionaries.
        """
        clipped = []
        for grad_dict in per_sample_grads:
            # Compute global norm across all parameters
            total_norm_sq = sum(
                torch.sum(g.float() ** 2).item() for g in grad_dict.values()
            )
            total_norm = total_norm_sq ** 0.5
            clip_factor = min(1.0, self.config.max_grad_norm / (total_norm + 1e-8))

            clipped_dict = {
                name: grad * clip_factor for name, grad in grad_dict.items()
            }
            clipped.append(clipped_dict)
        return clipped

    def aggregate_and_noise(
        self,
        clipped_grads: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Average clipped gradients and add calibrated noise.

        Args:
            clipped_grads: List of clipped per-sample gradient dictionaries.

        Returns:
            Noisy averaged gradient dictionary.
        """
        n = len(clipped_grads)
        if n == 0:
            return {}

        # Average the clipped gradients
        avg_grad: Dict[str, torch.Tensor] = {}
        for name in clipped_grads[0]:
            stacked = torch.stack([g[name] for g in clipped_grads])
            avg_grad[name] = stacked.mean(dim=0)

        # Add noise scaled to sensitivity / batch_size
        noise_config = NoiseConfig(
            noise_type=self.config.noise_type,
            noise_multiplier=self.config.noise_multiplier * self.config.max_grad_norm / n,
            sensitivity=1.0,
        )

        noisy_grad = {}
        for name, grad in avg_grad.items():
            noisy_grad[name] = add_noise(grad, noise_config)

        return noisy_grad

    def step(
        self,
        loss_per_sample: torch.Tensor,
    ) -> Dict[str, float]:
        """Perform one DP-SGD optimization step.

        Computes per-sample gradients, clips them, adds noise,
        and updates model parameters.

        Args:
            loss_per_sample: Per-sample loss tensor of shape (batch_size,).

        Returns:
            Dictionary with step metrics (grad_norm, noise_norm, etc.).
        """
        # 1. Per-sample gradients
        per_sample_grads = self.compute_per_sample_gradients(loss_per_sample)

        # 2. Clip
        clipped_grads = self.clip_gradients(per_sample_grads)

        # 3. Aggregate + noise
        noisy_grad = self.aggregate_and_noise(clipped_grads)

        # 4. SGD update
        grad_norm = 0.0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in noisy_grad:
                    param.sub_(self.config.learning_rate * noisy_grad[name])
                    grad_norm += torch.sum(noisy_grad[name] ** 2).item()

        self._step_count += 1
        return {
            "step": self._step_count,
            "grad_norm": grad_norm ** 0.5,
            "num_samples": len(per_sample_grads),
        }


def compute_dp_sgd_privacy(
    sample_rate: float,
    noise_multiplier: float,
    num_steps: int,
    delta: float = 1e-5,
) -> float:
    """Quick utility to compute epsilon for DP-SGD parameters.

    Args:
        sample_rate: batch_size / dataset_size.
        noise_multiplier: Noise multiplier sigma.
        num_steps: Number of training steps.
        delta: Target delta.

    Returns:
        Epsilon value.
    """
    from .accountant import PrivacyAccountant, AccountantConfig

    config = AccountantConfig(target_delta=delta)
    accountant = PrivacyAccountant(config)
    return accountant.get_epsilon_for_steps(noise_multiplier, sample_rate, num_steps)


if __name__ == "__main__":
    print("=== DP-SGD Demo ===\n")

    # Create a simple model
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    config = DPSGDConfig(
        learning_rate=0.01,
        max_grad_norm=1.0,
        noise_multiplier=1.1,
        batch_size=8,
    )
    optimizer = DPSGD(model, config)

    print(f"Config: lr={config.learning_rate}, clip={config.max_grad_norm}, "
          f"noise={config.noise_multiplier}")

    # Simulate a training step
    batch = torch.randn(8, 1, 28, 28)
    targets = torch.randint(0, 10, (8,))

    logits = model(batch)
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    loss_per_sample = loss_fn(logits, targets)

    metrics = optimizer.step(loss_per_sample)
    print(f"\nStep {metrics['step']}:")
    print(f"  Gradient norm: {metrics['grad_norm']:.6f}")
    print(f"  Samples: {metrics['num_samples']}")

    # Privacy cost
    eps = compute_dp_sgd_privacy(
        sample_rate=8 / 60000,
        noise_multiplier=1.1,
        num_steps=1000,
    )
    print(f"\nPrivacy cost (1000 steps, MNIST): epsilon = {eps:.4f}")
