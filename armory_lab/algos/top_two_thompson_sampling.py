from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult, BanditLike, HistoryRecord, RoundCallback
from armory_lab.confidence import delta_i_t, hoeffding_radius


@dataclass(slots=True)
class TopTwoThompsonSampling:
    delta: float
    max_pulls: int = 1_000_000
    beta: float = 0.5
    forced_exploration_scale: float = 0.5
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if self.delta <= 0.0 or self.delta >= 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.max_pulls <= 0:
            raise ValueError("max_pulls must be positive")
        if self.beta <= 0.0 or self.beta >= 1.0:
            raise ValueError("beta must be in (0, 1)")
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

    def _posterior_params(
        self,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
        reward_min: float,
        reward_range: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        scaled = (means - reward_min) / reward_range
        scaled = np.clip(scaled, self.eps, 1.0 - self.eps)
        alpha = 1.0 + counts.astype(np.float64) * scaled
        beta_param = 1.0 + counts.astype(np.float64) * (1.0 - scaled)
        return alpha, beta_param

    def _sample_top_two(
        self,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
        reward_min: float,
        reward_range: float,
        rng: np.random.Generator,
    ) -> tuple[int, int | None]:
        n_arms = int(means.size)
        alpha, beta_param = self._posterior_params(means, counts, reward_min, reward_range)
        theta = rng.beta(alpha, beta_param)
        leader = int(np.argmax(theta))
        if n_arms <= 1:
            return leader, None

        challenger: int | None = None
        for _ in range(8):
            theta_second = rng.beta(alpha, beta_param)
            candidate = int(np.argmax(theta_second))
            if candidate != leader:
                challenger = candidate
                break

        if challenger is None:
            order = np.argsort(theta)[::-1]
            for arm_idx in order:
                candidate = int(arm_idx)
                if candidate != leader:
                    challenger = candidate
                    break

        return leader, challenger

    def _select_arm(
        self,
        means: NDArray[np.float64],
        counts: NDArray[np.int_],
        total_pulls: int,
        reward_min: float,
        reward_range: float,
        rng: np.random.Generator,
    ) -> tuple[int, int, int | None]:
        n_arms = int(means.size)
        min_required = max(1, int(self.forced_exploration_scale * np.sqrt(float(total_pulls + 1))))
        under = np.where(counts < min_required)[0]
        if under.size > 0:
            arm_to_pull = int(under[int(np.argmin(counts[under]))])
            leader = int(np.argmax(means))
            challenger = arm_to_pull if arm_to_pull != leader else None
            return arm_to_pull, leader, challenger

        leader, challenger = self._sample_top_two(means, counts, reward_min, reward_range, rng)
        if challenger is None:
            return leader, leader, None
        if float(rng.random()) < self.beta:
            return leader, leader, challenger
        return challenger, leader, challenger

    def run(
        self,
        env: BanditLike,
        track_history: bool = True,
        on_round: RoundCallback | None = None,
    ) -> BAIResult:
        n_arms = env.n_arms
        if n_arms == 0:
            raise ValueError("bandit must have at least one arm")

        reward_min = float(env.reward_min)
        reward_max = float(env.reward_max)
        reward_range = float(env.reward_range)
        if reward_range <= 0.0:
            raise ValueError("reward_range must be positive")

        base_env = getattr(env, "base_env", env)
        rng_obj = getattr(base_env, "rng", None)
        if isinstance(rng_obj, np.random.Generator):
            rng = rng_obj
        else:
            rng = np.random.default_rng(0)

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

            arm_to_pull, sampled_leader, sampled_challenger = self._select_arm(
                means,
                counts,
                total_pulls,
                reward_min,
                reward_range,
                rng,
            )

            reward = env.pull(arm_to_pull)
            total_pulls += 1
            counts[arm_to_pull] += 1
            sums[arm_to_pull] += float(reward)
            means[arm_to_pull] = sums[arm_to_pull] / float(counts[arm_to_pull])

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
                    selected_arms=(arm_to_pull,),
                    active_arms=all_arms,
                    counts=counts.copy(),
                    means=means.copy(),
                    lcbs=lcbs.copy(),
                    ucbs=ucbs.copy(),
                    a_hat=sampled_leader,
                    challenger=sampled_challenger,
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
