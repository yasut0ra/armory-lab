from __future__ import annotations

import math


def delta_i_t(total_delta: float, n_arms: int, t: int) -> float:
    """Allocate confidence budget so sum_{i,t} delta_{i,t} <= total_delta."""
    if total_delta <= 0.0 or total_delta >= 1.0:
        raise ValueError("total_delta must be in (0, 1)")
    if n_arms <= 0:
        raise ValueError("n_arms must be positive")
    if t <= 0:
        raise ValueError("t must be positive")
    return total_delta / (2.0 * float(n_arms) * float(t * t))


def hoeffding_radius(n: int, local_delta: float, reward_range: float = 1.0) -> float:
    if n <= 0:
        return math.inf
    if local_delta <= 0.0 or local_delta >= 1.0:
        raise ValueError("local_delta must be in (0, 1)")
    if reward_range <= 0.0:
        raise ValueError("reward_range must be positive")

    return reward_range * math.sqrt(math.log(1.0 / local_delta) / (2.0 * float(n)))
