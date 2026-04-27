"""Federated learning client for local training.

Each client holds a local dataset partition and trains a copy of the
global model. Supports standard SGD, DP-SGD, and FedProx local training.
After local training, the client returns its updated weights to the server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .aggregation.fedavg import ClientUpdate
from .aggregation.fedprox import FedProxConfig, fedprox_local_step
from .privacy.dp_sgd import DPSGD, DPSGDConfig


@dataclass
class ClientConfig:
    """Configuration for a federated client.

    Attributes:
        client_id: Unique client identifier.
        local_epochs: Number of local training epochs per round.
        batch_size: Local training batch size.
        learning_rate: Local SGD learning rate.
        use_dp: Whether to use DP-SGD for local training.
        dp_config: DP-SGD configuration (if use_dp is True).
        fedprox_mu: FedProx proximal term coefficient (0 = standard FedAvg).
    """

    client_id: int = 0
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    use_dp: bool = False
    dp_config: Optional[DPSGDConfig] = None
    fedprox_mu: float = 0.0


class FederatedClient:
    """A single federated learning client.

    Manages local training on a private dataset partition and
    communicates model updates to the server.

    Args:
        config: Client configuration.
        model: Local copy of the global model.
        dataset: Client's local dataset partition.
    """

    def __init__(
        self,
        config: ClientConfig,
        model: nn.Module,
        dataset: Dataset,
    ) -> None:
        self.config = config
        self.model = model
        self.dataset = dataset
        self._round_metrics: List[Dict[str, float]] = []

    @property
    def client_id(self) -> int:
        """Client identifier."""
        return self.config.client_id

    @property
    def num_samples(self) -> int:
        """Number of local training samples."""
        return len(self.dataset)

    def receive_global_model(self, global_weights: Dict[str, torch.Tensor]) -> None:
        """Update local model with global weights from the server.

        Args:
            global_weights: Global model state dict.
        """
        self.model.load_state_dict(
            {k: v.clone() for k, v in global_weights.items()}
        )

    def train_local(self) -> ClientUpdate:
        """Perform local training and return the model update.

        Trains for config.local_epochs on the local dataset using
        either standard SGD, DP-SGD, or FedProx depending on config.

        Returns:
            ClientUpdate with trained weights and metrics.
        """
        global_weights = {
            n: p.data.clone() for n, p in self.model.state_dict().items()
        }

        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False,
        )

        if self.config.use_dp and self.config.dp_config is not None:
            metrics = self._train_dp(loader)
        elif self.config.fedprox_mu > 0:
            metrics = self._train_fedprox(loader, global_weights)
        else:
            metrics = self._train_standard(loader)

        self._round_metrics.append(metrics)

        return ClientUpdate(
            client_id=self.config.client_id,
            weights={n: p.data.clone() for n, p in self.model.state_dict().items()},
            num_samples=self.num_samples,
            metrics=metrics,
        )

    def _train_standard(self, loader: DataLoader) -> Dict[str, float]:
        """Standard SGD local training.

        Args:
            loader: DataLoader for local dataset.

        Returns:
            Training metrics dictionary.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.learning_rate
        )
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for epoch in range(self.config.local_epochs):
            for data, targets in loader:
                optimizer.zero_grad()
                logits = self.model(data)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(targets)
                preds = logits.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += len(targets)

        return {
            "loss": total_loss / max(total_samples, 1),
            "accuracy": total_correct / max(total_samples, 1),
            "epochs": self.config.local_epochs,
        }

    def _train_dp(self, loader: DataLoader) -> Dict[str, float]:
        """DP-SGD local training with per-sample gradient clipping and noise.

        Args:
            loader: DataLoader for local dataset.

        Returns:
            Training metrics dictionary.
        """
        dp_config = self.config.dp_config or DPSGDConfig()
        dp_optimizer = DPSGD(self.model, dp_config)
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for epoch in range(self.config.local_epochs):
            for data, targets in loader:
                logits = self.model(data)
                loss_per_sample = loss_fn(logits, targets)
                dp_optimizer.step(loss_per_sample)

                total_loss += loss_per_sample.sum().item()
                total_samples += len(targets)

        return {
            "loss": total_loss / max(total_samples, 1),
            "dp_steps": dp_optimizer.step_count,
            "epochs": self.config.local_epochs,
        }

    def _train_fedprox(
        self,
        loader: DataLoader,
        global_weights: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """FedProx local training with proximal regularization.

        Args:
            loader: DataLoader for local dataset.
            global_weights: Global model weights for proximal term.

        Returns:
            Training metrics dictionary.
        """
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.config.learning_rate
        )
        loss_fn = nn.CrossEntropyLoss()
        prox_config = FedProxConfig(mu=self.config.fedprox_mu)
        self.model.train()

        total_loss = 0.0
        total_prox = 0.0
        total_samples = 0

        for epoch in range(self.config.local_epochs):
            for data, targets in loader:
                metrics = fedprox_local_step(
                    self.model, data, targets, global_weights,
                    prox_config, loss_fn, optimizer,
                )
                total_loss += metrics["task_loss"] * len(targets)
                total_prox += metrics["prox_loss"] * len(targets)
                total_samples += len(targets)

        return {
            "loss": total_loss / max(total_samples, 1),
            "prox_loss": total_prox / max(total_samples, 1),
            "epochs": self.config.local_epochs,
        }

    def get_history(self) -> List[Dict[str, float]]:
        """Get training metrics history across rounds.

        Returns:
            List of per-round metric dictionaries.
        """
        return list(self._round_metrics)


if __name__ == "__main__":
    from .models.simple_models import ModelConfig, create_model

    print("=== Federated Client Demo ===\n")

    # Create synthetic dataset
    class SyntheticDataset(Dataset):
        def __init__(self, n: int = 200) -> None:
            self.data = torch.randn(n, 1, 28, 28)
            self.targets = torch.randint(0, 10, (n,))

        def __len__(self) -> int:
            return len(self.targets)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            return self.data[idx], self.targets[idx]

    model_config = ModelConfig(model_type="mlp", input_size=28)
    model = create_model(model_config)
    global_weights = {n: p.data.clone() for n, p in model.state_dict().items()}

    # Standard client
    client = FederatedClient(
        config=ClientConfig(client_id=0, local_epochs=2, batch_size=32),
        model=create_model(model_config),
        dataset=SyntheticDataset(200),
    )
    client.receive_global_model(global_weights)
    update = client.train_local()
    print(f"Client {update.client_id}: loss={update.metrics['loss']:.4f}, "
          f"acc={update.metrics['accuracy']:.4f}, samples={update.num_samples}")

    # FedProx client
    prox_client = FederatedClient(
        config=ClientConfig(client_id=1, local_epochs=2, fedprox_mu=0.1),
        model=create_model(model_config),
        dataset=SyntheticDataset(150),
    )
    prox_client.receive_global_model(global_weights)
    prox_update = prox_client.train_local()
    print(f"Client {prox_update.client_id} (FedProx): "
          f"loss={prox_update.metrics['loss']:.4f}, "
          f"prox={prox_update.metrics['prox_loss']:.6f}")
