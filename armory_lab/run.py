from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import BAIResult
from armory_lab.algos.lucb import LUCB
from armory_lab.algos.successive_elimination import SuccessiveElimination
from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.plotting import plot_history


@dataclass(slots=True)
class RunConfig:
    algo: str
    k: int
    delta: float
    means_spec: str
    seed: int
    max_pulls: int
    plot: bool
    save_plot: str | None
    no_show: bool


def _ensure_unique_best(means: NDArray[np.float64]) -> NDArray[np.float64]:
    adjusted = means.copy()
    adjusted += np.linspace(0.0, 1e-8, adjusted.size)
    adjusted = np.asarray(np.clip(adjusted, 0.0, 1.0), dtype=np.float64)

    best = int(np.argmax(adjusted))
    max_val = float(adjusted[best])
    tied = np.where(np.isclose(adjusted, max_val))[0]
    if tied.size > 1:
        adjusted[best] = min(1.0, adjusted[best] + 1e-4)
    return np.asarray(adjusted, dtype=np.float64)


def generate_means(spec: str, k: int, rng: np.random.Generator) -> NDArray[np.float64]:
    if k <= 0:
        raise ValueError("K must be positive")

    if spec == "random":
        means = rng.uniform(0.0, 1.0, size=k)
        return _ensure_unique_best(means.astype(np.float64))

    if spec.startswith("topgap:"):
        try:
            gap = float(spec.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("topgap format must be topgap:<float>") from exc

        means = np.empty(k, dtype=np.float64)
        means[0] = np.clip(0.5 + gap, 0.0, 1.0)
        if k > 1:
            means[1] = 0.5
        if k > 2:
            means[2:] = 0.5 - rng.uniform(0.0, 0.2, size=k - 2)
        means = np.clip(means, 0.0, 1.0)
        return _ensure_unique_best(means)

    if spec == "two-groups":
        n_top = max(1, k // 3)
        n_bottom = k - n_top
        top_group = rng.uniform(0.65, 0.9, size=n_top)
        bottom_group = rng.uniform(0.15, 0.55, size=n_bottom)
        means = np.concatenate([top_group, bottom_group]).astype(np.float64)
        rng.shuffle(means)
        return _ensure_unique_best(means)

    raise ValueError(f"unknown means regime: {spec}")


def build_algo(name: str, delta: float, max_pulls: int) -> LUCB | SuccessiveElimination:
    normalized = name.lower()
    if normalized == "lucb":
        return LUCB(delta=delta, max_pulls=max_pulls)
    if normalized in {"se", "successive_elimination", "successive-elimination"}:
        return SuccessiveElimination(delta=delta, max_pulls=max_pulls)
    raise ValueError(f"unknown algorithm: {name}")


def run_once(config: RunConfig) -> tuple[BAIResult, NDArray[np.float64], int]:
    rng = np.random.default_rng(config.seed)
    means = generate_means(config.means_spec, config.k, rng)
    env = BernoulliBandit.from_means(means, rng=rng)
    algo = build_algo(config.algo, config.delta, config.max_pulls)

    result = algo.run(env, track_history=True)
    true_best = int(np.argmax(means))
    return result, means, true_best


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Fixed-confidence BAI runner")
    parser.add_argument("--algo", default="lucb", choices=["lucb", "se", "successive_elimination"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--means", type=str, default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pulls", type=int, default=200_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    return RunConfig(
        algo=args.algo,
        k=args.K,
        delta=args.delta,
        means_spec=args.means,
        seed=args.seed,
        max_pulls=args.max_pulls,
        plot=bool(args.plot),
        save_plot=args.save_plot,
        no_show=bool(args.no_show),
    )


def main() -> None:
    config = parse_args()
    result, means, true_best = run_once(config)

    is_correct = result.recommend_arm == true_best
    print(f"algo={config.algo} K={config.k} delta={config.delta} seed={config.seed}")
    print(f"means_regime={config.means_spec}")
    print(f"recommended_arm={result.recommend_arm} true_best={true_best} correct={is_correct}")
    print(f"total_pulls={result.total_pulls}")
    print(f"pulls_per_arm={result.pulls_per_arm.tolist()}")

    if config.plot:
        plot_history(
            history=result.history,
            pulls_per_arm=result.pulls_per_arm,
            true_means=means,
            title=f"{config.algo.upper()} | K={config.k} delta={config.delta}",
            save_path=config.save_plot,
            show=not config.no_show,
        )
        if config.save_plot is not None:
            print(f"saved_plot={config.save_plot}")


if __name__ == "__main__":
    main()
