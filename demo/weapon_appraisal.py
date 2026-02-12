from __future__ import annotations

import argparse

import numpy as np

from armory_lab.algos.base import HistoryRecord
from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.plotting import plot_history
from armory_lab.run import build_algo, generate_means


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weapon appraisal demo with fixed-confidence BAI")
    parser.add_argument("--algo", default="lucb", choices=["lucb", "se"])
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--means", type=str, default="topgap:0.08")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pulls", type=int, default=200_000)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-plot", type=str, default=None)
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    weapon_names = [
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
    ]

    rng = np.random.default_rng(args.seed)
    k = len(weapon_names)
    means = generate_means(args.means, k, rng)

    env = BernoulliBandit.from_means(means, rng=rng)
    algo = build_algo(args.algo, args.delta, args.max_pulls)

    print("=== Armory Lab: 武器鑑定デモ ===")
    print(f"algorithm={args.algo} delta={args.delta} seed={args.seed}")

    def round_logger(record: HistoryRecord) -> None:
        tested = (
            ", ".join(weapon_names[arm] for arm in record.selected_arms)
            if record.selected_arms
            else "(停止判定のみ)"
        )
        active = ", ".join(weapon_names[arm] for arm in record.active_arms)

        top = np.argsort(record.means)[::-1][: min(3, k)]
        print(f"\n[Round {record.round_id:03d}] 試した武器: {tested}")
        for idx in top:
            print(
                "  "
                f"{weapon_names[int(idx)]}: mean={record.means[int(idx)]:.3f} "
                f"CI=[{record.lcbs[int(idx)]:.3f}, {record.ucbs[int(idx)]:.3f}] "
                f"n={int(record.counts[int(idx)])}"
            )
        print(f"  候補武器: {active}")

    result = algo.run(env, track_history=True, on_round=round_logger)

    best_idx = result.recommend_arm
    print("\n=== 鑑定結果 ===")
    print(f"最強武器を鑑定完了（delta={args.delta}）: {weapon_names[best_idx]}")
    print(f"試行回数(total pulls): {result.total_pulls}")
    allocation = {weapon_names[i]: int(n) for i, n in enumerate(result.pulls_per_arm.tolist())}
    print(f"サンプル配分: {allocation}")

    if args.plot:
        plot_history(
            history=result.history,
            pulls_per_arm=result.pulls_per_arm,
            true_means=means,
            arm_labels=weapon_names,
            title=f"Weapon Appraisal ({args.algo})",
            save_path=args.save_plot,
            show=not args.no_show,
        )
        if args.save_plot is not None:
            print(f"plot saved: {args.save_plot}")


if __name__ == "__main__":
    main()
