from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.envs.weapon_damage import WeaponDamageBandit


ObjectiveName = str


@dataclass(slots=True)
class ObjectiveBandit:
    base_env: BernoulliBandit | WeaponDamageBandit
    objective: ObjectiveName
    threshold: float | None = None

    def __post_init__(self) -> None:
        if self.objective not in {"dps", "oneshot"}:
            raise ValueError("objective must be dps or oneshot")
        if self.objective == "oneshot" and self.threshold is None:
            raise ValueError("threshold is required for oneshot objective")

    @property
    def n_arms(self) -> int:
        return int(self.base_env.n_arms)

    @property
    def reward_min(self) -> float:
        if self.objective == "oneshot":
            return 0.0
        return float(getattr(self.base_env, "reward_min", 0.0))

    @property
    def reward_max(self) -> float:
        if self.objective == "oneshot":
            return 1.0
        return float(getattr(self.base_env, "reward_max", 1.0))

    @property
    def reward_range(self) -> float:
        return float(self.reward_max - self.reward_min)

    def pull(self, arm: int) -> float:
        raw = float(self.base_env.pull(arm))
        if self.objective == "dps":
            return raw

        if self.threshold is None:
            raise RuntimeError("threshold is not set")
        return 1.0 if raw >= self.threshold else 0.0



def bernoulli_objective_values(
    means: NDArray[np.float64],
    objective: ObjectiveName,
    threshold: float | None,
) -> NDArray[np.float64]:
    if objective == "dps":
        return np.asarray(means, dtype=np.float64)

    if threshold is None:
        raise ValueError("threshold is required for oneshot objective")

    if threshold <= 0.0:
        return np.ones_like(means, dtype=np.float64)
    if threshold > 1.0:
        return np.zeros_like(means, dtype=np.float64)
    return np.asarray(means, dtype=np.float64)



def weapon_objective_values(
    env: WeaponDamageBandit,
    objective: ObjectiveName,
    threshold: float | None,
) -> NDArray[np.float64]:
    if objective == "dps":
        return env.expected_damages.astype(np.float64)

    if threshold is None:
        raise ValueError("threshold is required for oneshot objective")
    return env.oneshot_probabilities(threshold).astype(np.float64)
