from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class BernoulliBandit:
    means: NDArray[np.float64]
    rng: np.random.Generator

    @classmethod
    def from_means(
        cls,
        means: list[float] | NDArray[np.float64],
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "BernoulliBandit":
        mu = np.asarray(means, dtype=np.float64)
        if mu.ndim != 1 or mu.size == 0:
            raise ValueError("means must be a non-empty 1D array")
        if np.any((mu < 0.0) | (mu > 1.0)):
            raise ValueError("means must be in [0, 1]")

        generator = rng if rng is not None else np.random.default_rng(seed)
        return cls(means=mu, rng=generator)

    @property
    def n_arms(self) -> int:
        return int(self.means.size)

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.means))

    def pull(self, arm: int) -> int:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} out of range")
        reward = self.rng.binomial(n=1, p=float(self.means[arm]))
        return int(reward)
