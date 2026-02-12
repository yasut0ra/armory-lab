from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult, HistoryRecord, RoundCallback
from armory_lab.confidence import delta_i_t, hoeffding_radius
from armory_lab.envs.bernoulli import BernoulliBandit


@dataclass(slots=True)
class TrackAndStop:
    delta: float
    max_pulls: int = 1_000_000
    forced_exploration_scale: float = 0.5
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.delta <= 0.0 or self.delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.max_pulls <= 0:
            raise ValueError("max_pulls must be positive")
        if self.forced_exploration_scale <= 0.0:
            raise ValueError("forced_exploration_scale must be positive")
        if self.eps <= 0.0:
            raise ValueError("eps must be positive")

    def _update_bounds(
        self,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
        lcbs: NDArray[np.float64],
        ucbs: NDArray[np.float64],
        n_arms: int,
        bound_step: int,
    ) -> None:
        local_delta = delta_i_t(self.delta, n_arms, bound_step)
        for arm in range(n_arms):
            n = int(counts[arm])
            if n == 0:
                lcbs[arm] = 0.0
                ucbs[arm] = 1.0
                continue
            radius = hoeffding_radius(n, local_delta)
            lcbs[arm] = max(0.0, float(means[arm] - radius))
            ucbs[arm] = min(1.0, float(means[arm] + radius))

    def _beta(self, total_pulls: int, n_arms: int) -> float:
        t = max(1, total_pulls)
        return math.log((2.0 * float(max(1, n_arms - 1)) * float(t * t)) / self.delta)

    def _kl_bernoulli(self, p: float, q: float) -> float:
        p_clip = min(max(p, self.eps), 1.0 - self.eps)
        q_clip = min(max(q, self.eps), 1.0 - self.eps)
        return p_clip * math.log(p_clip / q_clip) + (1.0 - p_clip) * math.log((1.0 - p_clip) / (1.0 - q_clip))

    def _glr_pair(
        self,
        a: int,
        b: int,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
    ) -> float:
        n_a = int(counts[a])
        n_b = int(counts[b])
        if n_a <= 0 or n_b <= 0:
            return 0.0

        mu_a = float(means[a])
        mu_b = float(means[b])
        pooled = (float(n_a) * mu_a + float(n_b) * mu_b) / float(n_a + n_b)
        return float(n_a) * self._kl_bernoulli(mu_a, pooled) + float(n_b) * self._kl_bernoulli(mu_b, pooled)

    def _compute_target_weights(self, means: NDArray[np.float64]) -> NDArray[np.float64]:
        n_arms = int(means.size)
        a_hat = int(np.argmax(means))
        scores = np.zeros(n_arms, dtype=np.float64)

        for arm in range(n_arms):
            if arm == a_hat:
                continue
            gap = max(float(means[a_hat] - means[arm]), self.eps)
            scores[arm] = 1.0 / (gap * gap)

        if np.all(scores == 0.0):
            return np.ones(n_arms, dtype=np.float64) / float(n_arms)

        # Best arm should also be tracked against its hardest challengers.
        scores[a_hat] = float(np.max(scores))
        total = float(np.sum(scores))
        if total <= 0.0:
            return np.ones(n_arms, dtype=np.float64) / float(n_arms)
        return scores / total

    def _select_arm(
        self,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
        total_pulls: int,
    ) -> int:
        n_arms = int(means.size)
        min_required = max(1, int(self.forced_exploration_scale * math.sqrt(float(total_pulls + 1))))
        under = np.where(counts < min_required)[0]
        if under.size > 0:
            under_counts = counts[under]
            return int(under[int(np.argmin(under_counts))])

        target = self._compute_target_weights(means)
        tracked = counts.astype(np.float64) / float(max(1, total_pulls))
        deficits = target - tracked
        return int(np.argmax(deficits))

    def run(
        self,
        env: BernoulliBandit,
        track_history: bool = True,
        on_round: RoundCallback | None = None,
    ) -> BAIResult:
        n_arms = env.n_arms
        if n_arms == 0:
            raise ValueError("bandit must have at least one arm")

        counts = np.zeros(n_arms, dtype=np.int_)
        sums = np.zeros(n_arms, dtype=np.float64)
        means = np.zeros(n_arms, dtype=np.float64)
        lcbs = np.zeros(n_arms, dtype=np.float64)
        ucbs = np.ones(n_arms, dtype=np.float64)

        total_pulls = 0
        bound_step = 0
        history: list[HistoryRecord] = []
        all_arms = tuple(range(n_arms))

        init_selected: list[int] = []
        for arm in range(n_arms):
            if total_pulls >= self.max_pulls:
                break
            reward = env.pull(arm)
            total_pulls += 1
            counts[arm] += 1
            sums[arm] += float(reward)
            means[arm] = sums[arm] / float(counts[arm])
            init_selected.append(arm)

        bound_step += 1
        self._update_bounds(means, counts, lcbs, ucbs, n_arms, bound_step)

        if track_history or on_round is not None:
            init_record = HistoryRecord(
                round_id=0,
                total_pulls=total_pulls,
                selected_arms=tuple(init_selected),
                active_arms=all_arms,
                counts=counts.copy(),
                means=means.copy(),
                lcbs=lcbs.copy(),
                ucbs=ucbs.copy(),
                a_hat=int(np.argmax(means)),
                challenger=None,
            )
            if track_history:
                history.append(init_record)
            if on_round is not None:
                on_round(init_record)

        round_id = 0
        while total_pulls < self.max_pulls:
            round_id += 1
            a_hat = int(np.argmax(means))
            beta_t = self._beta(total_pulls, n_arms)

            stop_ok = True
            hardest = -1
            hardest_stat = float("inf")
            for arm in range(n_arms):
                if arm == a_hat:
                    continue
                stat = self._glr_pair(a_hat, arm, means, counts)
                if stat < hardest_stat:
                    hardest_stat = stat
                    hardest = arm
                if stat <= beta_t:
                    stop_ok = False

            if stop_ok:
                if track_history or on_round is not None:
                    stop_record = HistoryRecord(
                        round_id=round_id,
                        total_pulls=total_pulls,
                        selected_arms=tuple(),
                        active_arms=all_arms,
                        counts=counts.copy(),
                        means=means.copy(),
                        lcbs=lcbs.copy(),
                        ucbs=ucbs.copy(),
                        a_hat=a_hat,
                        challenger=hardest if hardest >= 0 else None,
                    )
                    if track_history:
                        history.append(stop_record)
                    if on_round is not None:
                        on_round(stop_record)
                break

            arm_to_pull = self._select_arm(means, counts, total_pulls)
            reward = env.pull(arm_to_pull)
            total_pulls += 1
            counts[arm_to_pull] += 1
            sums[arm_to_pull] += float(reward)
            means[arm_to_pull] = sums[arm_to_pull] / float(counts[arm_to_pull])

            bound_step += 1
            self._update_bounds(means, counts, lcbs, ucbs, n_arms, bound_step)

            if track_history or on_round is not None:
                round_record = HistoryRecord(
                    round_id=round_id,
                    total_pulls=total_pulls,
                    selected_arms=(arm_to_pull,),
                    active_arms=all_arms,
                    counts=counts.copy(),
                    means=means.copy(),
                    lcbs=lcbs.copy(),
                    ucbs=ucbs.copy(),
                    a_hat=a_hat,
                    challenger=hardest if hardest >= 0 else None,
                )
                if track_history:
                    history.append(round_record)
                if on_round is not None:
                    on_round(round_record)

        recommend_arm = int(np.argmax(means))
        return BAIResult(
            recommend_arm=recommend_arm,
            total_pulls=total_pulls,
            pulls_per_arm=counts.copy(),
            history=history,
        )
