from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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
    means_list: str | None
    seed: int
    seed_step: int
    trials: int
    max_pulls: int
    plot: bool
    save_plot: str | None
    no_show: bool
    output_csv: str | None
    output_json: bool


@dataclass(slots=True)
class TrialRun:
    trial_id: int
    seed: int
    result: BAIResult
    means: NDArray[np.float64]
    true_best: int


@dataclass(slots=True)
class TrialSummary:
    n_trials: int
    mean_total_pulls: float
    std_total_pulls: float
    misidentification_rate: float
    mean_pulls_per_arm: NDArray[np.float64]


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


def parse_means_list(raw: str) -> NDArray[np.float64]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise ValueError("means-list must contain at least one numeric value")

    values: list[float] = []
    for token in parts:
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"invalid value in means-list: {token}") from exc

    means = np.asarray(values, dtype=np.float64)
    if np.any((means < 0.0) | (means > 1.0)):
        raise ValueError("means-list values must be in [0, 1]")
    return _ensure_unique_best(means)


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


def _resolve_means(config: RunConfig, rng: np.random.Generator) -> NDArray[np.float64]:
    if config.means_list is not None:
        return parse_means_list(config.means_list)
    return generate_means(config.means_spec, config.k, rng)


def run_once(
    config: RunConfig,
    seed_override: int | None = None,
    track_history: bool = True,
) -> tuple[BAIResult, NDArray[np.float64], int]:
    seed = config.seed if seed_override is None else seed_override
    rng = np.random.default_rng(seed)
    means = _resolve_means(config, rng)
    env = BernoulliBandit.from_means(means, rng=rng)
    algo = build_algo(config.algo, config.delta, config.max_pulls)

    result = algo.run(env, track_history=track_history)
    true_best = int(np.argmax(means))
    return result, means, true_best


def run_trials(config: RunConfig) -> list[TrialRun]:
    if config.trials <= 0:
        raise ValueError("trials must be positive")

    trials: list[TrialRun] = []
    for trial_id in range(config.trials):
        seed = config.seed + trial_id * config.seed_step
        track_history = bool(config.plot and trial_id == 0)
        result, means, true_best = run_once(config, seed_override=seed, track_history=track_history)
        trials.append(
            TrialRun(
                trial_id=trial_id,
                seed=seed,
                result=result,
                means=means,
                true_best=true_best,
            )
        )
    return trials


def summarize_trial_runs(trials: Sequence[TrialRun]) -> TrialSummary:
    if not trials:
        raise ValueError("trials must be non-empty")

    total_pulls = np.asarray([trial.result.total_pulls for trial in trials], dtype=np.float64)
    errors = np.asarray(
        [trial.result.recommend_arm != trial.true_best for trial in trials],
        dtype=np.float64,
    )
    pulls_matrix = np.stack([trial.result.pulls_per_arm.astype(np.float64) for trial in trials], axis=0)

    return TrialSummary(
        n_trials=len(trials),
        mean_total_pulls=float(np.mean(total_pulls)),
        std_total_pulls=float(np.std(total_pulls)),
        misidentification_rate=float(np.mean(errors)),
        mean_pulls_per_arm=np.mean(pulls_matrix, axis=0),
    )


def _trial_to_json_record(trial: TrialRun) -> dict[str, Any]:
    return {
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "recommend_arm": trial.result.recommend_arm,
        "true_best": trial.true_best,
        "correct": bool(trial.result.recommend_arm == trial.true_best),
        "total_pulls": trial.result.total_pulls,
        "pulls_per_arm": trial.result.pulls_per_arm.tolist(),
        "means": [float(v) for v in trial.means.tolist()],
    }


def write_trials_csv(path: str, trials: Sequence[TrialRun]) -> None:
    if not trials:
        raise ValueError("trials must be non-empty")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "trial_id",
        "seed",
        "recommend_arm",
        "true_best",
        "correct",
        "total_pulls",
        "pulls_per_arm",
        "means",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow(
                {
                    "trial_id": trial.trial_id,
                    "seed": trial.seed,
                    "recommend_arm": trial.result.recommend_arm,
                    "true_best": trial.true_best,
                    "correct": int(trial.result.recommend_arm == trial.true_best),
                    "total_pulls": trial.result.total_pulls,
                    "pulls_per_arm": json.dumps(trial.result.pulls_per_arm.tolist()),
                    "means": json.dumps([round(float(v), 6) for v in trial.means.tolist()]),
                }
            )


def parse_args() -> RunConfig:
    parser = argparse.ArgumentParser(description="Fixed-confidence BAI runner")
    parser.add_argument("--algo", default="lucb", choices=["lucb", "se", "successive_elimination"])
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--means", type=str, default="random")
    parser.add_argument("--means-list", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--max-pulls", type=int, default=200_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    return RunConfig(
        algo=args.algo,
        k=args.K,
        delta=args.delta,
        means_spec=args.means,
        means_list=args.means_list,
        seed=args.seed,
        seed_step=args.seed_step,
        trials=args.trials,
        max_pulls=args.max_pulls,
        plot=bool(args.plot),
        save_plot=args.save_plot,
        no_show=bool(args.no_show),
        output_csv=args.output_csv,
        output_json=bool(args.json),
    )


def _print_single_trial(config: RunConfig, trial: TrialRun) -> None:
    is_correct = trial.result.recommend_arm == trial.true_best
    k = int(trial.means.size)
    print(f"algo={config.algo} K={k} delta={config.delta} seed={trial.seed}")
    if config.means_list is not None:
        print(f"means_list={config.means_list}")
    else:
        print(f"means_regime={config.means_spec}")
    print(f"recommended_arm={trial.result.recommend_arm} true_best={trial.true_best} correct={is_correct}")
    print(f"total_pulls={trial.result.total_pulls}")
    print(f"pulls_per_arm={trial.result.pulls_per_arm.tolist()}")


def _print_multi_trial(config: RunConfig, trials: Sequence[TrialRun], summary: TrialSummary) -> None:
    k = int(trials[0].means.size)
    print(f"algo={config.algo} K={k} delta={config.delta} trials={summary.n_trials}")
    if config.means_list is not None:
        print(f"means_list={config.means_list}")
    else:
        print(f"means_regime={config.means_spec}")
    print(f"seed_start={config.seed} seed_step={config.seed_step}")
    print(
        "summary "
        f"misidentification_rate={summary.misidentification_rate:.4f} "
        f"mean_total_pulls={summary.mean_total_pulls:.2f} "
        f"std_total_pulls={summary.std_total_pulls:.2f}"
    )
    print(f"mean_pulls_per_arm={[round(float(x), 3) for x in summary.mean_pulls_per_arm.tolist()]}")


def _print_json_payload(config: RunConfig, trials: Sequence[TrialRun], summary: TrialSummary | None) -> None:
    payload: dict[str, Any] = {
        "algo": config.algo,
        "delta": config.delta,
        "trials": config.trials,
        "seed": config.seed,
        "seed_step": config.seed_step,
        "means_regime": config.means_spec,
        "means_list": config.means_list,
        "max_pulls": config.max_pulls,
        "results": [_trial_to_json_record(trial) for trial in trials],
    }

    if summary is not None:
        payload["summary"] = {
            "n_trials": summary.n_trials,
            "misidentification_rate": summary.misidentification_rate,
            "mean_total_pulls": summary.mean_total_pulls,
            "std_total_pulls": summary.std_total_pulls,
            "mean_pulls_per_arm": [float(x) for x in summary.mean_pulls_per_arm.tolist()],
        }

    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    config = parse_args()
    trials = run_trials(config)

    if config.output_csv is not None:
        write_trials_csv(config.output_csv, trials)

    if config.trials == 1:
        first = trials[0]
        if config.output_json:
            _print_json_payload(config, trials, summary=None)
        else:
            _print_single_trial(config, first)

        if config.plot:
            plot_history(
                history=first.result.history,
                pulls_per_arm=first.result.pulls_per_arm,
                true_means=first.means,
                title=f"{config.algo.upper()} | K={int(first.means.size)} delta={config.delta}",
                save_path=config.save_plot,
                show=not config.no_show,
            )
            if config.save_plot is not None:
                print(f"saved_plot={config.save_plot}")

        if config.output_csv is not None:
            print(f"saved_csv={config.output_csv}")
        return

    summary = summarize_trial_runs(trials)
    if config.output_json:
        _print_json_payload(config, trials, summary=summary)
    else:
        _print_multi_trial(config, trials, summary)

    if config.plot:
        first = trials[0]
        plot_history(
            history=first.result.history,
            pulls_per_arm=first.result.pulls_per_arm,
            true_means=first.means,
            title=(
                f"{config.algo.upper()} | trial=0 | K={int(first.means.size)} "
                f"delta={config.delta}"
            ),
            save_path=config.save_plot,
            show=not config.no_show,
        )
        if config.save_plot is not None:
            print(f"saved_plot={config.save_plot}")

    if config.output_csv is not None:
        print(f"saved_csv={config.output_csv}")


if __name__ == "__main__":
    main()
