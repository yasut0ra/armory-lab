from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from armory_lab.algos.base import HistoryRecord


def plot_history(
    history: Sequence[HistoryRecord],
    pulls_per_arm: NDArray[np.int_],
    true_means: NDArray[np.float64] | None = None,
    arm_labels: Sequence[str] | None = None,
    title: str | None = None,
    metric_label: str = "mean",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    if not history:
        raise ValueError("history is empty; run with track_history=True")

    n_arms = int(pulls_per_arm.size)
    labels = list(arm_labels) if arm_labels is not None else [f"arm-{i}" for i in range(n_arms)]

    x = np.asarray([record.total_pulls for record in history], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), constrained_layout=True)
    ax_ci, ax_alloc, ax_active = axes

    for arm in range(n_arms):
        means = np.asarray([record.means[arm] for record in history], dtype=np.float64)
        lcbs = np.asarray([record.lcbs[arm] for record in history], dtype=np.float64)
        ucbs = np.asarray([record.ucbs[arm] for record in history], dtype=np.float64)

        ax_ci.plot(x, means, linewidth=1.2, label=labels[arm])
        ax_ci.fill_between(x, lcbs, ucbs, alpha=0.10)

        if true_means is not None:
            ax_ci.hlines(
                y=float(true_means[arm]),
                xmin=float(x[0]),
                xmax=float(x[-1]),
                linestyles="dashed",
                linewidth=0.8,
                alpha=0.25,
            )

    stop_x = float(x[-1])
    ax_ci.axvline(stop_x, color="red", linestyle="--", linewidth=1.2)
    ax_ci.set_ylabel(f"{metric_label} and CI")
    ax_ci.set_ylim(-0.02, 1.02)
    ax_ci.set_title("Confidence Interval Trajectories")
    if n_arms <= 12:
        ax_ci.legend(fontsize=8, ncol=2)

    ax_alloc.bar(np.arange(n_arms), pulls_per_arm)
    ax_alloc.set_xticks(np.arange(n_arms), labels=labels, rotation=45, ha="right")
    ax_alloc.set_ylabel("pull count")
    ax_alloc.set_title("Sample Allocation")

    active_counts = np.asarray([len(record.active_arms) for record in history], dtype=np.float64)
    ax_active.step(x, active_counts, where="post", linewidth=1.5)
    ax_active.axvline(stop_x, color="red", linestyle="--", linewidth=1.2)
    ax_active.set_xlabel("total pulls")
    ax_active.set_ylabel("# active arms")
    ax_active.set_title("Active Arms and Stopping Time")

    if title is not None:
        fig.suptitle(title)

    if save_path is not None:
        fig.savefig(save_path, dpi=160)

    if show:
        plt.show()
    else:
        plt.close(fig)
