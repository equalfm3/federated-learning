"""Federated learning server for orchestrating distributed training.

Coordinates FedAvg/FedProx training with optional secure aggregation and DP-SGD.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .aggregation.fedavg import (
    AggregationResult,
    ClientUpdate,
    fedavg_aggregate,
    fedavg_aggregate_deltas,
)
from .aggregation.secure_agg import (
    MaskedUpdate,
    SecureAggConfig,
    mask_client_update,
    secure_aggregate,
)
from .client import ClientConfig, FederatedClient
from .data.partitioner import (
    PartitionResult,
    compute_heterogeneity,
    partition_dirichlet,
    partition_iid,
)
from .models.simple_models import ModelConfig, create_model, get_model_weights
from .privacy.accountant import AccountantConfig, PrivacyAccountant


@dataclass
class ServerConfig:
    """Configuration for the federated server."""

    num_rounds: int = 50
    num_clients: int = 10
    clients_per_round: int = 0
    model_config: ModelConfig = field(default_factory=ModelConfig)
    partition_strategy: str = "iid"
    dirichlet_alpha: float = 0.5
    use_secure_agg: bool = False
    use_dp: bool = False
    server_lr: float = 1.0
    seed: int = 42


@dataclass
class RoundResult:
    """Result of a single federated training round."""

    round_num: int
    num_clients: int
    avg_loss: float
    avg_accuracy: Optional[float]
    aggregation: AggregationResult


class FederatedServer:
    """Orchestrates federated learning across simulated clients.

    Args:
        config: Server configuration.
    """

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.global_model = create_model(config.model_config)
        self.clients: List[FederatedClient] = []
        self.round_history: List[RoundResult] = []
        self.privacy_accountant: Optional[PrivacyAccountant] = None

        if config.use_dp:
            self.privacy_accountant = PrivacyAccountant(
                AccountantConfig(target_delta=1e-5)
            )

        random.seed(config.seed)
        torch.manual_seed(config.seed)

    def setup_clients(
        self, dataset: Dataset, client_config_fn: Optional[callable] = None,
    ) -> PartitionResult:
        """Partition data and create client instances."""
        # Partition data
        if self.config.partition_strategy == "dirichlet":
            partition = partition_dirichlet(
                dataset,
                self.config.num_clients,
                alpha=self.config.dirichlet_alpha,
                seed=self.config.seed,
            )
        else:
            partition = partition_iid(
                dataset, self.config.num_clients, seed=self.config.seed
            )

        # Create clients
        self.clients = []
        for cid in range(self.config.num_clients):
            if client_config_fn:
                client_cfg = client_config_fn(cid)
            else:
                client_cfg = ClientConfig(
                    client_id=cid,
                    use_dp=self.config.use_dp,
                )

            indices = partition.client_indices[cid]
            client_dataset = torch.utils.data.Subset(dataset, indices)

            client_model = create_model(self.config.model_config)
            client = FederatedClient(client_cfg, client_model, client_dataset)
            self.clients.append(client)

        return partition

    def select_clients(self) -> List[FederatedClient]:
        """Select clients for the current round."""
        k = self.config.clients_per_round
        if k <= 0 or k >= len(self.clients):
            return list(self.clients)
        return random.sample(self.clients, k)

    def run_round(self, round_num: int) -> RoundResult:
        """Execute one federated training round."""
        global_weights = get_model_weights(self.global_model)
        selected = self.select_clients()

        # Broadcast and train
        updates: List[ClientUpdate] = []
        for client in selected:
            client.receive_global_model(global_weights)
            update = client.train_local()
            updates.append(update)

        # Aggregate
        if self.config.use_secure_agg:
            agg_result = self._secure_aggregate(updates, global_weights)
        elif self.config.server_lr != 1.0:
            agg_result = fedavg_aggregate_deltas(
                global_weights, updates, self.config.server_lr
            )
        else:
            agg_result = fedavg_aggregate(updates)

        # Update global model
        self.global_model.load_state_dict(agg_result.global_weights)

        # Track privacy
        if self.privacy_accountant and self.config.use_dp:
            sample_rate = self.config.clients_per_round / self.config.num_clients
            if sample_rate <= 0:
                sample_rate = 1.0
            self.privacy_accountant.accumulate(1.0, sample_rate, num_steps=1)

        # Compute metrics
        losses = [u.metrics.get("loss", 0) for u in updates if u.metrics]
        accs = [u.metrics.get("accuracy", 0) for u in updates if u.metrics and "accuracy" in u.metrics]

        result = RoundResult(
            round_num=round_num,
            num_clients=len(selected),
            avg_loss=sum(losses) / max(len(losses), 1),
            avg_accuracy=sum(accs) / len(accs) if accs else None,
            aggregation=agg_result,
        )
        self.round_history.append(result)
        return result

    def _secure_aggregate(
        self, updates: List[ClientUpdate], global_weights: Dict[str, torch.Tensor],
    ) -> AggregationResult:
        """Perform secure aggregation of client updates."""
        client_ids = [u.client_id for u in updates]
        config = SecureAggConfig(
            num_clients=len(updates),
            threshold=max(1, len(updates) // 2),
        )

        masked_updates = []
        total_samples = sum(u.num_samples for u in updates)
        for update in updates:
            masked_w = mask_client_update(
                update.client_id, update.weights, client_ids, config,
                update.num_samples, total_samples,
            )
            masked_updates.append(MaskedUpdate(
                client_id=update.client_id,
                masked_weights=masked_w,
                num_samples=update.num_samples,
            ))

        agg_weights = secure_aggregate(masked_updates, config)
        total_samples = sum(u.num_samples for u in updates)

        return AggregationResult(
            global_weights=agg_weights,
            num_clients=len(updates),
            total_samples=total_samples,
            client_contributions={
                u.client_id: u.num_samples / total_samples for u in updates
            },
        )

    def train(self, verbose: bool = True) -> List[RoundResult]:
        """Run the full federated training loop."""
        for r in range(self.config.num_rounds):
            result = self.run_round(r)
            if verbose and (r % max(1, self.config.num_rounds // 10) == 0 or r == self.config.num_rounds - 1):
                msg = f"Round {r:3d}: loss={result.avg_loss:.4f}"
                if result.avg_accuracy is not None:
                    msg += f", acc={result.avg_accuracy:.4f}"
                if self.privacy_accountant:
                    spent = self.privacy_accountant.get_privacy_spent()
                    msg += f", eps={spent.epsilon:.4f}"
                print(msg)

        return self.round_history

    def evaluate(self, test_dataset: Dataset, batch_size: int = 128) -> Dict[str, float]:
        """Evaluate the global model on a test dataset."""
        loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )
        self.global_model.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for data, targets in loader:
                logits = self.global_model(data)
                loss = loss_fn(logits, targets)
                total_loss += loss.item() * len(targets)
                preds = logits.argmax(dim=1)
                total_correct += (preds == targets).sum().item()
                total_samples += len(targets)

        return {
            "test_loss": total_loss / max(total_samples, 1),
            "test_accuracy": total_correct / max(total_samples, 1),
        }


if __name__ == "__main__":
    from torch.utils.data import Dataset as TorchDataset

    print("=== Federated Server Demo ===\n")

    class SyntheticDataset(TorchDataset):
        def __init__(self, n: int = 1000) -> None:
            self.data = torch.randn(n, 1, 28, 28)
            self.targets = torch.randint(0, 10, (n,)).tolist()
        def __len__(self) -> int:
            return len(self.targets)
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            return self.data[idx], self.targets[idx]

    config = ServerConfig(
        num_rounds=10, num_clients=5, clients_per_round=3,
        model_config=ModelConfig(model_type="mlp", input_size=28),
        partition_strategy="dirichlet", dirichlet_alpha=0.5,
    )
    server = FederatedServer(config)
    train_data, test_data = SyntheticDataset(1000), SyntheticDataset(200)
    partition = server.setup_clients(train_data)
    print(f"Partition: {partition.strategy}, heterogeneity={compute_heterogeneity(partition):.4f}")
    results = server.train(verbose=True)
    test_metrics = server.evaluate(test_data)
    print(f"\nTest: loss={test_metrics['test_loss']:.4f}, acc={test_metrics['test_accuracy']:.4f}")
