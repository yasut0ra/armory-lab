from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from armory_lab.run import RunConfig, parse_means_list, run_trials, summarize_trial_runs, write_trials_csv


def _base_config() -> RunConfig:
    return RunConfig(
        algo="lucb",
        k=5,
        delta=0.1,
        means_spec="random",
        means_list="0.9,0.2,0.1,0.1",
        seed=0,
        seed_step=1,
        trials=3,
        max_pulls=50_000,
        plot=False,
        save_plot=None,
        no_show=True,
        output_csv=None,
        output_json=False,
    )


def test_parse_means_list_valid() -> None:
    means = parse_means_list("0.9, 0.3,0.1")
    assert means.shape == (3,)
    assert np.all((means >= 0.0) & (means <= 1.0))


def test_run_trials_and_summary() -> None:
    config = _base_config()
    trials = run_trials(config)

    assert len(trials) == 3
    assert all(trial.means.shape == (4,) for trial in trials)
    assert all(0 <= trial.result.recommend_arm < 4 for trial in trials)
    assert all(0 < trial.result.total_pulls <= 50_000 for trial in trials)

    summary = summarize_trial_runs(trials)
    assert summary.n_trials == 3
    assert summary.mean_total_pulls > 0
    assert 0.0 <= summary.misidentification_rate <= 1.0
    assert summary.mean_pulls_per_arm.shape == (4,)


def test_write_trials_csv(tmp_path: Path) -> None:
    config = _base_config()
    trials = run_trials(config)
    out_path = tmp_path / "trial_report.csv"
    write_trials_csv(str(out_path), trials)

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 3
    assert reader[0]["trial_id"] == "0"
    assert "pulls_per_arm" in reader[0]
