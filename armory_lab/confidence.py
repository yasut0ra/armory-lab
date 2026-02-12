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


def binary_kl(p: float, q: float, eps: float = 1e-12) -> float:
    p_clip = min(max(p, eps), 1.0 - eps)
    q_clip = min(max(q, eps), 1.0 - eps)
    return p_clip * math.log(p_clip / q_clip) + (1.0 - p_clip) * math.log((1.0 - p_clip) / (1.0 - q_clip))


def kl_ucb(mean: float, n: int, local_delta: float, eps: float = 1e-9) -> float:
    if n <= 0:
        return 1.0
    if local_delta <= 0.0 or local_delta >= 1.0:
        raise ValueError("local_delta must be in (0, 1)")

    target = math.log(1.0 / local_delta) / float(n)
    if mean >= 1.0 - eps:
        return 1.0

    lo = mean
    hi = 1.0 - eps
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if binary_kl(mean, mid, eps=eps) > target:
            hi = mid
        else:
            lo = mid
    return min(max(lo, 0.0), 1.0)


def kl_lcb(mean: float, n: int, local_delta: float, eps: float = 1e-9) -> float:
    if n <= 0:
        return 0.0
    if local_delta <= 0.0 or local_delta >= 1.0:
        raise ValueError("local_delta must be in (0, 1)")

    target = math.log(1.0 / local_delta) / float(n)
    if mean <= eps:
        return 0.0

    lo = eps
    hi = mean
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if binary_kl(mean, mid, eps=eps) > target:
            lo = mid
        else:
            hi = mid
    return min(max(hi, 0.0), 1.0)
