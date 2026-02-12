from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class HistoryRecord:
    round_id: int
    total_pulls: int
    selected_arms: tuple[int, ...]
    active_arms: tuple[int, ...]
    counts: NDArray[np.int_]
    means: NDArray[np.float64]
    lcbs: NDArray[np.float64]
    ucbs: NDArray[np.float64]
    a_hat: int | None = None
    challenger: int | None = None


RoundCallback = Callable[[HistoryRecord], None]


class BanditLike(Protocol):
    @property
    def n_arms(self) -> int:
        ...

    @property
    def reward_min(self) -> float:
        ...

    @property
    def reward_max(self) -> float:
        ...

    @property
    def reward_range(self) -> float:
        ...

    def pull(self, arm: int) -> float | int:
        ...


@dataclass(slots=True)
class BAIResult:
    recommend_arm: int
    total_pulls: int
    pulls_per_arm: NDArray[np.int_]
    history: list[HistoryRecord]
