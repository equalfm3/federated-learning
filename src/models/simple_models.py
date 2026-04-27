"""Simple neural network models for federated learning experiments.

Provides lightweight CNN and MLP architectures suitable for MNIST/CIFAR
experiments in federated settings. Models are kept small to make
multi-client simulation feasible on a single machine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    """Configuration for model architecture.

    Attributes:
        model_type: One of 'cnn', 'mlp'.
        input_channels: Number of input channels (1 for MNIST, 3 for CIFAR).
        input_size: Spatial dimension of input images.
        num_classes: Number of output classes.
        hidden_dim: Hidden layer dimension for MLP.
    """

    model_type: str = "cnn"
    input_channels: int = 1
    input_size: int = 28
    num_classes: int = 10
    hidden_dim: int = 200


class FedCNN(nn.Module):
    """Lightweight CNN for federated experiments.

    Two convolutional layers followed by two fully connected layers.
    Designed for 28x28 (MNIST) or 32x32 (CIFAR) inputs.

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv2d(config.input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Compute flattened size after two pool layers
        reduced = config.input_size // 4
        self.flat_size = 64 * reduced * reduced

        self.fc1 = nn.Linear(self.flat_size, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CNN.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class FedMLP(nn.Module):
    """Simple MLP for federated experiments.

    Two hidden layers with ReLU activation. Flattens input images
    before the first linear layer.

    Args:
        config: Model configuration dataclass.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.input_channels * config.input_size * config.input_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP.

        Args:
            x: Input tensor of shape (batch, channels, height, width).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        return self.net(x)


def create_model(config: ModelConfig) -> nn.Module:
    """Factory function to create a model from config.

    Args:
        config: Model configuration specifying architecture.

    Returns:
        Instantiated PyTorch model.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if config.model_type == "cnn":
        return FedCNN(config)
    elif config.model_type == "mlp":
        return FedMLP(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_weights(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract model weights as a dictionary of detached tensors.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary mapping parameter names to weight tensors.
    """
    return {name: param.data.clone() for name, param in model.state_dict().items()}


def set_model_weights(model: nn.Module, weights: Dict[str, torch.Tensor]) -> None:
    """Load weights into a model.

    Args:
        model: PyTorch model to update.
        weights: Dictionary mapping parameter names to weight tensors.
    """
    model.load_state_dict(weights)


def compute_weight_divergence(
    weights_a: Dict[str, torch.Tensor],
    weights_b: Dict[str, torch.Tensor],
) -> float:
    """Compute L2 divergence between two sets of model weights.

    Useful for measuring how far a client has drifted from the global model.

    Args:
        weights_a: First set of weights.
        weights_b: Second set of weights.

    Returns:
        L2 norm of the weight difference.
    """
    total = 0.0
    for key in weights_a:
        diff = weights_a[key].float() - weights_b[key].float()
        total += torch.sum(diff ** 2).item()
    return math.sqrt(total)


if __name__ == "__main__":
    print("=== Federated Learning Models Demo ===\n")

    for model_type in ["cnn", "mlp"]:
        config = ModelConfig(model_type=model_type, input_channels=1, input_size=28)
        model = create_model(config)
        n_params = count_parameters(model)
        print(f"{model_type.upper()}: {n_params:,} trainable parameters")

        dummy = torch.randn(4, 1, 28, 28)
        out = model(dummy)
        print(f"  Input: {dummy.shape} -> Output: {out.shape}")

        weights = get_model_weights(model)
        print(f"  Weight tensors: {len(weights)}")

    # Demonstrate weight divergence
    config = ModelConfig(model_type="mlp")
    model_a = create_model(config)
    model_b = create_model(config)
    div = compute_weight_divergence(
        get_model_weights(model_a), get_model_weights(model_b)
    )
    print(f"\nWeight divergence between two random MLPs: {div:.4f}")
