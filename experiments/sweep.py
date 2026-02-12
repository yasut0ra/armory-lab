from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from armory_lab.envs.weapon_damage import ENEMY_NAMES
from armory_lab.run import RunConfig, run_trials, summarize_trial_runs



def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]



def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BAI parameter sweep and export CSV")
    parser.add_argument("--env", default="bernoulli", choices=["bernoulli", "weapon_damage"])
    parser.add_argument("--objective", default="dps", choices=["dps", "oneshot"])
    parser.add_argument("--algo", default="lucb", choices=["kllucb", "lucb", "se", "tas", "ttts"])

    parser.add_argument("--K-values", type=str, default="10,20")
    parser.add_argument("--deltas", type=str, default="0.05,0.1")
    parser.add_argument("--gaps", type=str, default="0.05,0.1")
    parser.add_argument("--thresholds", type=str, default="100")

    parser.add_argument("--means-regime", type=str, default="topgap")
    parser.add_argument("--pack-regime", type=str, default="topgap")
    parser.add_argument("--enemy", type=str, default="none", choices=list(ENEMY_NAMES))

    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pulls", type=int, default=200_000)
    parser.add_argument("--output", type=str, default="experiments/sweep_results.csv")
    return parser.parse_args()



def _format_regime(base: str, gap: float) -> str:
    if base == "topgap":
        return f"topgap:{gap}"
    return base



def main() -> None:
    args = parse_args()

    k_values = parse_int_list(args.K_values)
    deltas = parse_float_list(args.deltas)
    gaps = parse_float_list(args.gaps)
    thresholds = parse_float_list(args.thresholds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "env",
        "objective",
        "algo",
        "K",
        "delta",
        "gap",
        "threshold",
        "means_regime",
        "pack_regime",
        "enemy",
        "trials",
        "mean_total_pulls",
        "misidentification_rate",
        "mean_pulls_per_arm",
    ]

    combo_idx = 0
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for k in k_values:
            for delta in deltas:
                if args.objective == "oneshot":
                    for threshold in thresholds:
                        combo_idx += 1
                        trial_seed = args.seed + 100_000 * combo_idx

                        config = RunConfig(
                            algo=args.algo,
                            env_name=args.env,
                            objective="oneshot",
                            threshold=threshold,
                            k=k,
                            delta=delta,
                            means_spec=args.means_regime,
                            weapon_spec=args.pack_regime,
                            enemy=args.enemy,
                            seed=trial_seed,
                            trials=args.trials,
                            seed_step=1,
                            max_pulls=args.max_pulls,
                        )
                        trials = run_trials(config)
                        summary = summarize_trial_runs(trials)

                        writer.writerow(
                            {
                                "env": args.env,
                                "objective": "oneshot",
                                "algo": args.algo,
                                "K": k,
                                "delta": delta,
                                "gap": "",
                                "threshold": threshold,
                                "means_regime": args.means_regime,
                                "pack_regime": args.pack_regime,
                                "enemy": args.enemy,
                                "trials": args.trials,
                                "mean_total_pulls": f"{summary.mean_total_pulls:.3f}",
                                "misidentification_rate": f"{summary.misidentification_rate:.4f}",
                                "mean_pulls_per_arm": json.dumps(
                                    [round(float(x), 3) for x in summary.mean_pulls_per_arm.tolist()]
                                ),
                            }
                        )

                        print(
                            "finished "
                            f"env={args.env} objective=oneshot algo={args.algo} K={k} "
                            f"delta={delta} threshold={threshold} "
                            f"enemy={args.enemy} "
                            f"mean_T={summary.mean_total_pulls:.1f} err={summary.misidentification_rate:.3f}"
                        )
                else:
                    for gap in gaps:
                        combo_idx += 1
                        trial_seed = args.seed + 100_000 * combo_idx

                        means_regime = _format_regime(args.means_regime, gap)
                        pack_regime = _format_regime(args.pack_regime, gap)

                        config = RunConfig(
                            algo=args.algo,
                            env_name=args.env,
                            objective="dps",
                            threshold=None,
                            k=k,
                            delta=delta,
                            means_spec=means_regime,
                            weapon_spec=pack_regime,
                            enemy=args.enemy,
                            seed=trial_seed,
                            trials=args.trials,
                            seed_step=1,
                            max_pulls=args.max_pulls,
                        )
                        trials = run_trials(config)
                        summary = summarize_trial_runs(trials)

                        writer.writerow(
                            {
                                "env": args.env,
                                "objective": "dps",
                                "algo": args.algo,
                                "K": k,
                                "delta": delta,
                                "gap": gap,
                                "threshold": "",
                                "means_regime": means_regime,
                                "pack_regime": pack_regime,
                                "enemy": args.enemy,
                                "trials": args.trials,
                                "mean_total_pulls": f"{summary.mean_total_pulls:.3f}",
                                "misidentification_rate": f"{summary.misidentification_rate:.4f}",
                                "mean_pulls_per_arm": json.dumps(
                                    [round(float(x), 3) for x in summary.mean_pulls_per_arm.tolist()]
                                ),
                            }
                        )

                        print(
                            "finished "
                            f"env={args.env} objective=dps algo={args.algo} K={k} delta={delta} gap={gap} enemy={args.enemy} "
                            f"mean_T={summary.mean_total_pulls:.1f} err={summary.misidentification_rate:.3f}"
                        )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
