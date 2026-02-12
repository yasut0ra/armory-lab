from __future__ import annotations

import argparse
import math

import numpy as np

from armory_lab.algos.base import HistoryRecord
from armory_lab.plotting import plot_history
from armory_lab.run import RunConfig, build_algo, build_problem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weapon appraisal demo with fixed-confidence BAI")
    parser.add_argument("--env", default="weapon_damage", choices=["bernoulli", "weapon_damage"])
    parser.add_argument("--algo", default="lucb", choices=["lucb", "se", "tas"])
    parser.add_argument("--objective", default="dps", choices=["dps", "oneshot"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--means", type=str, default="topgap:0.08")
    parser.add_argument("--pack", type=str, default="archetypes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pulls", type=int, default=200_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _metric_label(env_name: str, objective: str) -> str:
    if objective == "oneshot":
        return "hit-rate"
    if env_name == "weapon_damage":
        return "expected damage"
    return "mean reward"


def _weapon_names(k: int) -> list[str]:
    base = [
        "Sword",
        "Spear",
        "Axe",
        "Bow",
        "Dagger",
        "Hammer",
        "Halberd",
        "Mace",
        "Crossbow",
        "Katana",
        "Rapier",
        "Scythe",
        "Whip",
        "Pike",
        "Gauntlet",
        "Lance",
    ]
    if k <= len(base):
        return base[:k]
    extra = [f"Weapon-{i}" for i in range(len(base), k)]
    return base + extra


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _gauge_text(gauge: float, width: int = 20) -> str:
    filled = int(round(gauge * width))
    filled = min(max(filled, 0), width)
    return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"


def main() -> None:
    args = parse_args()
    if args.objective == "oneshot" and args.threshold is None:
        raise ValueError("oneshot objective requires --threshold")

    config = RunConfig(
        algo=args.algo,
        env_name=args.env,
        objective=args.objective,
        threshold=args.threshold,
        k=args.K,
        delta=args.delta,
        means_spec=args.means,
        weapon_spec=args.pack,
        seed=args.seed,
        max_pulls=args.max_pulls,
    )

    problem = build_problem(config, seed=args.seed)
    algo = build_algo(args.algo, args.delta, args.max_pulls)

    k = problem.bandit.n_arms
    names = _weapon_names(k)
    reward_range = max(problem.bandit.reward_range, 1e-9)
    metric_label = _metric_label(args.env, args.objective)

    print("=== Armory Lab: 武器鑑定デモ ===")
    print(
        f"env={args.env} algo={args.algo} objective={args.objective} "
        f"delta={args.delta} seed={args.seed}"
    )
    if args.objective == "oneshot":
        print(f"threshold={args.threshold}")

    def round_logger(record: HistoryRecord) -> None:
        tested = ", ".join(names[arm] for arm in record.selected_arms) if record.selected_arms else "(停止判定のみ)"

        a_hat = int(np.argmax(record.means)) if record.means.size > 0 else -1
        max_other_ucb = float("-inf")
        challenger = -1
        for arm in range(k):
            if arm == a_hat:
                continue
            ucb = float(record.ucbs[arm])
            if ucb > max_other_ucb:
                max_other_ucb = ucb
                challenger = arm

        gap = float(record.lcbs[a_hat] - max_other_ucb) if challenger >= 0 else 0.0
        gauge = _sigmoid(12.0 * (gap / reward_range))

        print(f"\n[Round {record.round_id:03d}] 試した武器: {tested}")
        if args.algo == "lucb" and challenger >= 0 and a_hat >= 0:
            print(f"  leader vs challenger: {names[a_hat]} vs {names[challenger]}")

        top = np.argsort(record.means)[::-1][: min(3, k)]
        for idx in top:
            idx_i = int(idx)
            print(
                "  "
                f"{names[idx_i]}: {metric_label}={record.means[idx_i]:.3f} "
                f"CI=[{record.lcbs[idx_i]:.3f}, {record.ucbs[idx_i]:.3f}] "
                f"n={int(record.counts[idx_i])}"
            )

        print(f"  鑑定ゲージ: {_gauge_text(gauge)} {gauge:.3f} (gap={gap:.4f})")

    result = algo.run(problem.bandit, track_history=True, on_round=round_logger)

    best_idx = result.recommend_arm
    print("\n=== 鑑定結果 ===")
    if args.objective == "oneshot":
        print(
            f"最強武器（objective={args.objective}, threshold={args.threshold}）: "
            f"{names[best_idx]}"
        )
    else:
        print(f"最強武器（objective={args.objective}）: {names[best_idx]}")

    print(f"試行回数(total pulls): {result.total_pulls}")
    allocation = {names[i]: int(n) for i, n in enumerate(result.pulls_per_arm.tolist())}
    print(f"サンプル配分: {allocation}")

    true_best = int(np.argmax(problem.true_values))
    print(f"真の最良武器: {names[true_best]} (arm={true_best})")

    if args.plot:
        plot_history(
            history=result.history,
            pulls_per_arm=result.pulls_per_arm,
            true_means=problem.true_values,
            arm_labels=names,
            title=(
                f"Weapon Appraisal ({args.algo}) | env={args.env} "
                f"objective={args.objective}"
            ),
            metric_label=metric_label,
            save_path=args.save_plot,
            show=not args.no_show,
        )
        if args.save_plot is not None:
            print(f"plot saved: {args.save_plot}")


if __name__ == "__main__":
    main()
