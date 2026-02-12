from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult


@dataclass(slots=True)
class TrialAggregate:
    num_trials: int
    mean_total_pulls: float
    misidentification_rate: float
    mean_pulls_per_arm: NDArray[np.float64]


def misidentification_rate(results: Sequence[BAIResult], true_best_arm: int) -> float:
    if not results:
        raise ValueError("results must be non-empty")
    return float(np.mean([res.recommend_arm != true_best_arm for res in results]))


def summarize_trials(results: Sequence[BAIResult], true_best_arm: int) -> TrialAggregate:
    if not results:
        raise ValueError("results must be non-empty")

    total_pulls = np.asarray([res.total_pulls for res in results], dtype=np.float64)
    pulls_matrix = np.stack([res.pulls_per_arm.astype(np.float64) for res in results], axis=0)
    return TrialAggregate(
        num_trials=len(results),
        mean_total_pulls=float(np.mean(total_pulls)),
        misidentification_rate=misidentification_rate(results, true_best_arm),
        mean_pulls_per_arm=np.mean(pulls_matrix, axis=0),
    )
