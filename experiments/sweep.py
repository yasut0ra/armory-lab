from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.run import build_algo, generate_means


def parse_int_list(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BAI parameter sweep and export CSV")
    parser.add_argument("--algo", default="lucb", choices=["lucb", "se"])
    parser.add_argument("--K-values", type=str, default="10,20")
    parser.add_argument("--deltas", type=str, default="0.05,0.1")
    parser.add_argument("--gaps", type=str, default="0.05,0.1")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-pulls", type=int, default=200_000)
    parser.add_argument("--output", type=str, default="experiments/sweep_results.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    k_values = parse_int_list(args.K_values)
    deltas = parse_float_list(args.deltas)
    gaps = parse_float_list(args.gaps)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "algo",
        "K",
        "delta",
        "gap",
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
                for gap in gaps:
                    combo_idx += 1
                    total_pulls_list: list[float] = []
                    errors: list[float] = []
                    pulls_list: list[np.ndarray] = []

                    for trial in range(args.trials):
                        trial_seed = args.seed + 100_000 * combo_idx + trial
                        rng = np.random.default_rng(trial_seed)
                        means = generate_means(f"topgap:{gap}", k, rng)
                        env = BernoulliBandit.from_means(means, rng=rng)

                        algo = build_algo(args.algo, delta, args.max_pulls)
                        result = algo.run(env, track_history=False)

                        true_best = int(np.argmax(means))
                        total_pulls_list.append(float(result.total_pulls))
                        errors.append(float(result.recommend_arm != true_best))
                        pulls_list.append(result.pulls_per_arm.astype(np.float64))

                    mean_total = float(np.mean(np.asarray(total_pulls_list, dtype=np.float64)))
                    error_rate = float(np.mean(np.asarray(errors, dtype=np.float64)))
                    mean_pulls = np.mean(np.stack(pulls_list, axis=0), axis=0)

                    writer.writerow(
                        {
                            "algo": args.algo,
                            "K": k,
                            "delta": delta,
                            "gap": gap,
                            "trials": args.trials,
                            "mean_total_pulls": f"{mean_total:.3f}",
                            "misidentification_rate": f"{error_rate:.4f}",
                            "mean_pulls_per_arm": json.dumps([round(float(x), 3) for x in mean_pulls.tolist()]),
                        }
                    )

                    print(
                        "finished "
                        f"algo={args.algo} K={k} delta={delta} gap={gap} "
                        f"mean_T={mean_total:.1f} err={error_rate:.3f}"
                    )

    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
