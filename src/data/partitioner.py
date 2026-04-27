"""Data partitioning strategies for federated learning.

Supports IID (uniform random) and non-IID (Dirichlet allocation)
partitioning of datasets across simulated clients. Non-IID partitioning
creates heterogeneous label distributions that reflect real-world
federated scenarios where each device sees different data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset


@dataclass
class PartitionStats:
    """Statistics for a single client's data partition.

    Attributes:
        client_id: Identifier for this client.
        num_samples: Total samples assigned.
        label_counts: Mapping from label to count.
        label_distribution: Normalized distribution over labels.
    """

    client_id: int
    num_samples: int
    label_counts: Dict[int, int]
    label_distribution: Dict[int, float]


@dataclass
class PartitionResult:
    """Result of partitioning a dataset across clients.

    Attributes:
        client_indices: Mapping from client_id to list of dataset indices.
        stats: Per-client partition statistics.
        strategy: Name of the partitioning strategy used.
    """

    client_indices: Dict[int, List[int]]
    stats: List[PartitionStats]
    strategy: str


def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Extract labels from a PyTorch dataset.

    Args:
        dataset: Dataset with (data, label) items.

    Returns:
        Numpy array of integer labels.
    """
    if hasattr(dataset, "targets"):
        return np.array(dataset.targets)
    elif hasattr(dataset, "labels"):
        return np.array(dataset.labels)
    else:
        # Fallback: iterate (slow but universal)
        labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            labels.append(int(label))
        return np.array(labels)


def partition_iid(
    dataset: Dataset,
    num_clients: int,
    seed: int = 42,
) -> PartitionResult:
    """Partition dataset uniformly at random across clients (IID).

    Each client receives approximately the same number of samples,
    with label distributions matching the global distribution.

    Args:
        dataset: Source dataset to partition.
        num_clients: Number of federated clients.
        seed: Random seed for reproducibility.

    Returns:
        PartitionResult with client index assignments.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    indices = rng.permutation(n).tolist()
    labels = _extract_labels(dataset)

    splits = np.array_split(indices, num_clients)
    client_indices = {i: split.tolist() for i, split in enumerate(splits)}

    stats = _compute_stats(client_indices, labels)
    return PartitionResult(client_indices=client_indices, stats=stats, strategy="iid")


def partition_dirichlet(
    dataset: Dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_samples: int = 10,
) -> PartitionResult:
    """Partition dataset using Dirichlet allocation (non-IID).

    Lower alpha values create more heterogeneous partitions. At alpha -> 0,
    each client gets samples from only one class. At alpha -> inf, the
    partition approaches IID.

    Args:
        dataset: Source dataset to partition.
        num_clients: Number of federated clients.
        alpha: Dirichlet concentration parameter. Lower = more heterogeneous.
        seed: Random seed for reproducibility.
        min_samples: Minimum samples per client (redistributed if needed).

    Returns:
        PartitionResult with non-IID client index assignments.
    """
    rng = np.random.default_rng(seed)
    labels = _extract_labels(dataset)
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        class_indices.setdefault(int(label), []).append(idx)

    # Draw Dirichlet proportions for each class
    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

    for cls in range(num_classes):
        cls_idx = np.array(class_indices.get(cls, []))
        if len(cls_idx) == 0:
            continue

        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Scale proportions to actual counts
        proportions = proportions / proportions.sum()
        counts = (proportions * len(cls_idx)).astype(int)

        # Distribute remainder
        remainder = len(cls_idx) - counts.sum()
        for j in range(remainder):
            counts[j % num_clients] += 1

        start = 0
        for client_id in range(num_clients):
            end = start + counts[client_id]
            client_indices[client_id].extend(cls_idx[start:end].tolist())
            start = end

    # Ensure minimum samples per client by redistributing from largest
    for client_id in range(num_clients):
        if len(client_indices[client_id]) < min_samples:
            # Find client with most samples
            largest = max(client_indices, key=lambda k: len(client_indices[k]))
            needed = min_samples - len(client_indices[client_id])
            donated = client_indices[largest][-needed:]
            client_indices[largest] = client_indices[largest][:-needed]
            client_indices[client_id].extend(donated)

    stats = _compute_stats(client_indices, labels)
    return PartitionResult(
        client_indices=client_indices,
        stats=stats,
        strategy=f"dirichlet(alpha={alpha})",
    )


def _compute_stats(
    client_indices: Dict[int, List[int]],
    labels: np.ndarray,
) -> List[PartitionStats]:
    """Compute partition statistics for each client.

    Args:
        client_indices: Mapping from client_id to dataset indices.
        labels: Full label array for the dataset.

    Returns:
        List of PartitionStats, one per client.
    """
    stats = []
    for client_id, indices in client_indices.items():
        client_labels = labels[indices]
        unique, counts = np.unique(client_labels, return_counts=True)
        label_counts = {int(u): int(c) for u, c in zip(unique, counts)}
        total = sum(label_counts.values())
        label_dist = {k: v / total for k, v in label_counts.items()} if total > 0 else {}
        stats.append(PartitionStats(
            client_id=client_id,
            num_samples=len(indices),
            label_counts=label_counts,
            label_distribution=label_dist,
        ))
    return stats


def get_client_dataset(
    dataset: Dataset,
    partition: PartitionResult,
    client_id: int,
) -> Subset:
    """Get a PyTorch Subset for a specific client.

    Args:
        dataset: Original full dataset.
        partition: Partition result from IID or Dirichlet partitioning.
        client_id: Client identifier.

    Returns:
        Subset of the dataset for the specified client.

    Raises:
        KeyError: If client_id is not in the partition.
    """
    if client_id not in partition.client_indices:
        raise KeyError(f"Client {client_id} not found in partition")
    return Subset(dataset, partition.client_indices[client_id])


def compute_heterogeneity(partition: PartitionResult) -> float:
    """Measure label distribution heterogeneity across clients.

    Computes the average Jensen-Shannon divergence between each client's
    label distribution and the global distribution. Higher values indicate
    more heterogeneous (non-IID) partitions.

    Args:
        partition: Partition result to analyze.

    Returns:
        Average JS divergence (0 = perfectly IID, 1 = maximally non-IID).
    """
    if not partition.stats:
        return 0.0

    # Compute global distribution
    all_labels: Dict[int, int] = {}
    for s in partition.stats:
        for label, count in s.label_counts.items():
            all_labels[label] = all_labels.get(label, 0) + count

    total = sum(all_labels.values())
    if total == 0:
        return 0.0

    all_classes = sorted(all_labels.keys())
    global_dist = np.array([all_labels[c] / total for c in all_classes])

    js_divs = []
    for s in partition.stats:
        client_dist = np.array([s.label_distribution.get(c, 0.0) for c in all_classes])
        m = 0.5 * (global_dist + client_dist)
        # KL divergence with epsilon for numerical stability
        eps = 1e-10
        kl_pm = np.sum(global_dist * np.log((global_dist + eps) / (m + eps)))
        kl_qm = np.sum(client_dist * np.log((client_dist + eps) / (m + eps)))
        js = 0.5 * (kl_pm + kl_qm)
        js_divs.append(js)

    return float(np.mean(js_divs))


if __name__ == "__main__":
    print("=== Data Partitioner Demo ===\n")

    class SyntheticDataset(Dataset):
        def __init__(self, n: int = 1000, num_classes: int = 10) -> None:
            self.data = torch.randn(n, 1, 28, 28)
            self.targets = torch.randint(0, num_classes, (n,)).tolist()
        def __len__(self) -> int:
            return len(self.targets)
        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
            return self.data[idx], self.targets[idx]

    dataset = SyntheticDataset(n=2000, num_classes=10)

    iid_result = partition_iid(dataset, 5)
    print(f"IID: heterogeneity={compute_heterogeneity(iid_result):.4f}")
    for s in iid_result.stats:
        print(f"  Client {s.client_id}: {s.num_samples} samples")

    for alpha in [0.1, 0.5, 5.0]:
        noniid = partition_dirichlet(dataset, 5, alpha=alpha)
        print(f"\nDirichlet(alpha={alpha}): heterogeneity={compute_heterogeneity(noniid):.4f}")
        for s in noniid.stats:
            top = max(s.label_distribution, key=s.label_distribution.get) if s.label_distribution else -1
            print(f"  Client {s.client_id}: {s.num_samples} samples, top class={top}")
