import numpy as np
import torch
from typing import Tuple


class StratifiedZoneSampler:
    """
    Stratified zone sampling for controlled imbalance ratio distribution.

    Implements Option C from the proposal: instead of a single Beta distribution,
    we partition the sampling space into zones with specific target proportions.

    Zone A: r ∈ [1, 5)       → 60% of datasets (balanced to moderate)
    Zone B: r ∈ [5, 10)      → 10% of datasets (moderate to severe)
    Zone C: r ∈ [10, 100]    → 30% of datasets (severe imbalance)
    """

    def __init__(
        self,
        zone_a_ratio: Tuple[float, float] = (1.0, 5.0),
        zone_b_ratio: Tuple[float, float] = (5.0, 10.0),
        zone_c_ratio: Tuple[float, float] = (10.0, 100.0),
        zone_proportions: Tuple[float, float, float] = (0.60, 0.10, 0.30),
        power_law_exponent: float = 1.5,
    ):
        self.zone_a_ratio = zone_a_ratio
        self.zone_b_ratio = zone_b_ratio
        self.zone_c_ratio = zone_c_ratio
        self.zone_proportions = zone_proportions
        self.power_law_exponent = power_law_exponent

    def sample_imbalance_ratio(self) -> float:
        """Sample a single imbalance ratio r = majority/minority."""
        u = np.random.random()

        if u < self.zone_proportions[0]:
            return self._sample_zone_a()
        elif u < self.zone_proportions[0] + self.zone_proportions[1]:
            return self._sample_zone_b()
        else:
            return self._sample_zone_c()

    def _sample_zone_a(self) -> float:
        """Zone A: r ∈ [1, 5), uniform sampling."""
        r_min, r_max = self.zone_a_ratio
        return r_min + np.random.random() * (r_max - r_min)

    def _sample_zone_b(self) -> float:
        """Zone B: r ∈ [5, 10), uniform sampling."""
        r_min, r_max = self.zone_b_ratio
        return r_min + np.random.random() * (r_max - r_min)

    def _sample_zone_c(self) -> float:
        """Zone C: r ∈ [10, 100], power-law sampling for heavy tail."""
        r_min, r_max = self.zone_c_ratio
        u = np.random.random()
        r = r_min * (1 - u * (1 - r_max ** (-self.power_law_exponent))) ** (
            -1 / self.power_law_exponent
        )
        return min(max(r, r_min), r_max)

    def sample_minority_proportion(self) -> float:
        """Sample minority class proportion π from stratified zones."""
        r = self.sample_imbalance_ratio()
        return 1.0 / (r + 1.0)

    def sample_batch(self, batch_size: int) -> np.ndarray:
        """Sample a batch of minority proportions."""
        return np.array([self.sample_minority_proportion() for _ in range(batch_size)])


def verify_zone_properties(sampler: StratifiedZoneSampler, n_samples: int = 100000):
    """Verify that the sampler produces the expected distribution of imbalance ratios."""
    ratios = np.array([sampler.sample_imbalance_ratio() for _ in range(n_samples)])

    zone_a_count = np.sum(ratios < 5.0)
    zone_b_count = np.sum((ratios >= 5.0) & (ratios < 10.0))
    zone_c_count = np.sum(ratios >= 10.0)

    print(f"Zone distribution (n={n_samples}):")
    print(f"  Zone A (r < 5:1):   {zone_a_count/n_samples*100:.1f}% (target: 60%)")
    print(f"  Zone B (5:1 ≤ r < 10:1): {zone_b_count/n_samples*100:.1f}% (target: 10%)")
    print(f"  Zone C (r ≥ 10:1):  {zone_c_count/n_samples*100:.1f}% (target: 30%)")
    print(f"  Ratio range: [{ratios.min():.2f}, {ratios.max():.2f}]")
    print(f"  Mean ratio: {ratios.mean():.2f}")
    print(f"  Median ratio: {np.median(ratios):.2f}")


if __name__ == "__main__":
    sampler = StratifiedZoneSampler()
    verify_zone_properties(sampler)
