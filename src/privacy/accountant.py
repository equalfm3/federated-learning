"""Privacy budget accounting for differential privacy.

Tracks cumulative privacy loss across multiple rounds of DP-SGD training
using Rényi Differential Privacy (RDP) composition and conversion to
(epsilon, delta)-DP. Implements the moments accountant approach from
Abadi et al. (2016).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PrivacySpent:
    """Record of privacy budget consumed.

    Attributes:
        epsilon: Total epsilon spent.
        delta: Delta parameter.
        num_steps: Number of composition steps.
        noise_multiplier: Noise multiplier used.
        sample_rate: Subsampling rate (batch_size / dataset_size).
    """

    epsilon: float
    delta: float
    num_steps: int
    noise_multiplier: float
    sample_rate: float


@dataclass
class AccountantConfig:
    """Configuration for the privacy accountant.

    Attributes:
        target_delta: Target delta for (epsilon, delta)-DP.
        target_epsilon: Optional epsilon budget cap.
        max_rdp_orders: RDP orders to evaluate for tight conversion.
    """

    target_delta: float = 1e-5
    target_epsilon: Optional[float] = None
    max_rdp_orders: List[float] = field(default_factory=lambda: [
        1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0,
        16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    ])


def _compute_rdp_single_step(
    order: float,
    noise_multiplier: float,
    sample_rate: float,
) -> float:
    """Compute RDP guarantee for a single step of subsampled Gaussian mechanism.

    Uses the analytic bound from Mironov (2017) for the subsampled
    Gaussian mechanism.

    Args:
        order: Rényi divergence order (alpha > 1).
        noise_multiplier: Ratio of noise std to sensitivity.
        sample_rate: Probability of including each sample.

    Returns:
        RDP epsilon for this single step at the given order.
    """
    if noise_multiplier == 0:
        return float("inf")

    if sample_rate == 0:
        return 0.0

    if sample_rate == 1.0:
        # No subsampling: standard Gaussian RDP
        return order / (2.0 * noise_multiplier ** 2)

    # Subsampled Gaussian mechanism bound
    # Simplified analytic bound for practical use
    if order <= 1:
        return 0.0

    log_term = math.log1p(-sample_rate)
    rdp_no_subsample = order / (2.0 * noise_multiplier ** 2)

    # Use the log-sum-exp trick for numerical stability
    log_a = math.log(sample_rate) + (order - 1) * rdp_no_subsample
    log_b = (order - 1) * log_term

    # Bound: (1/(alpha-1)) * log((1-q)^(alpha-1) + q * exp((alpha-1) * rdp))
    if log_a > log_b:
        result = log_a + math.log1p(math.exp(log_b - log_a))
    else:
        result = log_b + math.log1p(math.exp(log_a - log_b))

    return result / (order - 1)


def _rdp_to_dp(
    rdp_epsilon: float,
    order: float,
    delta: float,
) -> float:
    """Convert RDP guarantee to (epsilon, delta)-DP.

    Uses the conversion formula:
        epsilon = rdp_epsilon - log(delta) / (order - 1)

    Args:
        rdp_epsilon: RDP epsilon at the given order.
        order: Rényi divergence order.
        delta: Target delta.

    Returns:
        Epsilon for (epsilon, delta)-DP.
    """
    if order <= 1:
        return float("inf")
    return rdp_epsilon - math.log(delta) / (order - 1)


class PrivacyAccountant:
    """Tracks cumulative privacy loss using RDP composition.

    The accountant accumulates RDP guarantees across training steps
    and converts to (epsilon, delta)-DP using the optimal order.

    Args:
        config: Accountant configuration.
    """

    def __init__(self, config: Optional[AccountantConfig] = None) -> None:
        self.config = config or AccountantConfig()
        self._rdp_epsilons: List[float] = [0.0] * len(self.config.max_rdp_orders)
        self._steps: int = 0
        self._history: List[PrivacySpent] = []

    @property
    def steps(self) -> int:
        """Number of composition steps recorded."""
        return self._steps

    def accumulate(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int = 1,
    ) -> None:
        """Record privacy cost of training steps.

        Args:
            noise_multiplier: Noise multiplier (sigma / sensitivity).
            sample_rate: Subsampling rate (batch_size / dataset_size).
            num_steps: Number of identical steps to record.
        """
        for i, order in enumerate(self.config.max_rdp_orders):
            rdp = _compute_rdp_single_step(order, noise_multiplier, sample_rate)
            self._rdp_epsilons[i] += rdp * num_steps
        self._steps += num_steps

    def get_privacy_spent(self) -> PrivacySpent:
        """Compute current (epsilon, delta)-DP guarantee.

        Evaluates all RDP orders and returns the tightest bound.

        Returns:
            PrivacySpent with the best epsilon at the configured delta.
        """
        best_epsilon = float("inf")
        for i, order in enumerate(self.config.max_rdp_orders):
            eps = _rdp_to_dp(self._rdp_epsilons[i], order, self.config.target_delta)
            best_epsilon = min(best_epsilon, eps)

        return PrivacySpent(
            epsilon=best_epsilon,
            delta=self.config.target_delta,
            num_steps=self._steps,
            noise_multiplier=0.0,  # Aggregate
            sample_rate=0.0,
        )

    def is_budget_exceeded(self) -> bool:
        """Check if the privacy budget has been exceeded.

        Returns:
            True if target_epsilon is set and current epsilon exceeds it.
        """
        if self.config.target_epsilon is None:
            return False
        spent = self.get_privacy_spent()
        return spent.epsilon > self.config.target_epsilon

    def get_epsilon_for_steps(
        self,
        noise_multiplier: float,
        sample_rate: float,
        num_steps: int,
        delta: Optional[float] = None,
    ) -> float:
        """Compute epsilon for a hypothetical number of steps (without recording).

        Useful for planning how many rounds of training are possible
        within a given privacy budget.

        Args:
            noise_multiplier: Noise multiplier.
            sample_rate: Subsampling rate.
            num_steps: Number of steps to evaluate.
            delta: Override delta (uses config default if None).

        Returns:
            Epsilon value for the given parameters.
        """
        target_delta = delta or self.config.target_delta
        best_epsilon = float("inf")

        for order in self.config.max_rdp_orders:
            rdp = _compute_rdp_single_step(order, noise_multiplier, sample_rate)
            total_rdp = rdp * num_steps
            eps = _rdp_to_dp(total_rdp, order, target_delta)
            best_epsilon = min(best_epsilon, eps)

        return best_epsilon

    def max_steps_for_budget(
        self,
        noise_multiplier: float,
        sample_rate: float,
        target_epsilon: float,
        delta: Optional[float] = None,
    ) -> int:
        """Find maximum training steps within a privacy budget.

        Uses binary search to find the largest number of steps
        that keeps epsilon below the target.

        Args:
            noise_multiplier: Noise multiplier.
            sample_rate: Subsampling rate.
            target_epsilon: Maximum allowed epsilon.
            delta: Override delta.

        Returns:
            Maximum number of steps.
        """
        lo, hi = 1, 1_000_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            eps = self.get_epsilon_for_steps(noise_multiplier, sample_rate, mid, delta)
            if eps <= target_epsilon:
                lo = mid
            else:
                hi = mid - 1
        return lo


if __name__ == "__main__":
    print("=== Privacy Accountant Demo ===\n")

    config = AccountantConfig(target_delta=1e-5, target_epsilon=10.0)
    accountant = PrivacyAccountant(config)

    noise_mult = 1.0
    sample_rate = 0.01  # batch_size / dataset_size

    print(f"Noise multiplier: {noise_mult}, Sample rate: {sample_rate}")
    print(f"Target: epsilon={config.target_epsilon}, delta={config.target_delta}\n")

    # Simulate training rounds
    for step in [10, 50, 100, 500, 1000]:
        eps = accountant.get_epsilon_for_steps(noise_mult, sample_rate, step)
        print(f"  After {step:5d} steps: epsilon = {eps:.4f}")

    # Track actual training
    print("\nSimulating 100 training steps:")
    for i in range(100):
        accountant.accumulate(noise_mult, sample_rate, num_steps=1)
        if (i + 1) % 25 == 0:
            spent = accountant.get_privacy_spent()
            print(f"  Step {i+1:3d}: epsilon = {spent.epsilon:.4f}")

    # Budget planning
    max_steps = accountant.max_steps_for_budget(
        noise_mult, sample_rate, target_epsilon=10.0
    )
    print(f"\nMax steps for epsilon=10.0: {max_steps}")
    print(f"Budget exceeded: {accountant.is_budget_exceeded()}")
