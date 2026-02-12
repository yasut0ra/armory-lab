from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult, BanditLike, HistoryRecord, RoundCallback
from armory_lab.confidence import delta_i_t, hoeffding_radius


@dataclass(slots=True)
class LUCB:
    delta: float
    max_pulls: int = 1_000_000

    def __post_init__(self) -> None:
        if self.delta <= 0.0 or self.delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.max_pulls <= 0:
            raise ValueError("max_pulls must be positive")

    def _update_bounds(
        self,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
        lcbs: NDArray[np.float64],
        ucbs: NDArray[np.float64],
        n_arms: int,
        bound_step: int,
        reward_min: float,
        reward_max: float,
        reward_range: float,
    ) -> None:
        local_delta = delta_i_t(self.delta, n_arms, bound_step)
        for arm in range(n_arms):
            n = int(counts[arm])
            if n == 0:
                lcbs[arm] = reward_min
                ucbs[arm] = reward_max
                continue
            radius = hoeffding_radius(n, local_delta, reward_range=reward_range)
            lcbs[arm] = max(reward_min, float(means[arm] - radius))
            ucbs[arm] = min(reward_max, float(means[arm] + radius))

    def run(
        self,
        env: BanditLike,
        track_history: bool = True,
        on_round: RoundCallback | None = None,
    ) -> BAIResult:
        n_arms = env.n_arms
        reward_min = float(env.reward_min)
        reward_max = float(env.reward_max)
        reward_range = float(env.reward_range)
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

        if n_arms == 0:
            raise ValueError("bandit must have at least one arm")

        bound_step += 1
        self._update_bounds(
            means,
            counts,
            lcbs,
            ucbs,
            n_arms,
            bound_step,
            reward_min,
            reward_max,
            reward_range,
        )

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
        stopped = False

        while total_pulls < self.max_pulls:
            round_id += 1
            a_hat = int(np.argmax(means))

            challenger = -1
            challenger_ucb = -1.0
            for arm in range(n_arms):
                if arm == a_hat:
                    continue
                candidate_ucb = float(ucbs[arm])
                if candidate_ucb > challenger_ucb:
                    challenger_ucb = candidate_ucb
                    challenger = arm

            if challenger == -1:
                stopped = True
                break

            if float(lcbs[a_hat]) >= float(ucbs[challenger]):
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
                        challenger=challenger,
                    )
                    if track_history:
                        history.append(stop_record)
                    if on_round is not None:
                        on_round(stop_record)
                stopped = True
                break

            selected = [a_hat, challenger]
            for arm in selected:
                if total_pulls >= self.max_pulls:
                    break
                reward = env.pull(arm)
                total_pulls += 1
                counts[arm] += 1
                sums[arm] += float(reward)
                means[arm] = sums[arm] / float(counts[arm])

            bound_step += 1
            self._update_bounds(
                means,
                counts,
                lcbs,
                ucbs,
                n_arms,
                bound_step,
                reward_min,
                reward_max,
                reward_range,
            )

            if track_history or on_round is not None:
                round_record = HistoryRecord(
                    round_id=round_id,
                    total_pulls=total_pulls,
                    selected_arms=tuple(selected),
                    active_arms=all_arms,
                    counts=counts.copy(),
                    means=means.copy(),
                    lcbs=lcbs.copy(),
                    ucbs=ucbs.copy(),
                    a_hat=a_hat,
                    challenger=challenger,
                )
                if track_history:
                    history.append(round_record)
                if on_round is not None:
                    on_round(round_record)

        recommend_arm = int(np.argmax(means))
        if stopped and n_arms > 0:
            recommend_arm = int(np.argmax(means))

        return BAIResult(
            recommend_arm=recommend_arm,
            total_pulls=total_pulls,
            pulls_per_arm=counts.copy(),
            history=history,
        )
