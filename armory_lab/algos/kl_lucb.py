from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult, BanditLike, HistoryRecord, RoundCallback
from armory_lab.confidence import delta_i_t, kl_lcb, kl_ucb


@dataclass(slots=True)
class KLLUCB:
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
    ) -> None:
        local_delta = delta_i_t(self.delta, n_arms, bound_step)
        for arm in range(n_arms):
            n = int(counts[arm])
            if n == 0:
                lcbs[arm] = 0.0
                ucbs[arm] = 1.0
                continue
            mean = float(np.clip(means[arm], 0.0, 1.0))
            lcbs[arm] = kl_lcb(mean, n, local_delta)
            ucbs[arm] = kl_ucb(mean, n, local_delta)

    def run(
        self,
        env: BanditLike,
        track_history: bool = True,
        on_round: RoundCallback | None = None,
    ) -> BAIResult:
        if env.reward_min < -1e-12 or env.reward_max > 1.0 + 1e-12:
            raise ValueError("KLLUCB supports only Bernoulli-style rewards in [0, 1]")

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
            reward = float(env.pull(arm))
            if reward < -1e-12 or reward > 1.0 + 1e-12:
                raise ValueError("KLLUCB received non-Bernoulli reward outside [0, 1]")

            total_pulls += 1
            counts[arm] += 1
            sums[arm] += reward
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

            challenger = -1
            challenger_ucb = float("-inf")
            for arm in range(n_arms):
                if arm == a_hat:
                    continue
                candidate_ucb = float(ucbs[arm])
                if candidate_ucb > challenger_ucb:
                    challenger_ucb = candidate_ucb
                    challenger = arm

            if challenger == -1:
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
                break

            selected = [a_hat, challenger]
            for arm in selected:
                if total_pulls >= self.max_pulls:
                    break
                reward = float(env.pull(arm))
                if reward < -1e-12 or reward > 1.0 + 1e-12:
                    raise ValueError("KLLUCB received non-Bernoulli reward outside [0, 1]")

                total_pulls += 1
                counts[arm] += 1
                sums[arm] += reward
                means[arm] = sums[arm] / float(counts[arm])

            bound_step += 1
            self._update_bounds(means, counts, lcbs, ucbs, n_arms, bound_step)

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

        return BAIResult(
            recommend_arm=int(np.argmax(means)),
            total_pulls=total_pulls,
            pulls_per_arm=counts.copy(),
            history=history,
        )
