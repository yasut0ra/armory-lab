from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from armory_lab.run import _trial_to_json_record, RunConfig, parse_means_list, run_trials, summarize_trial_runs, write_trials_csv


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


def _weapon_config() -> RunConfig:
    return RunConfig(
        algo="lucb",
        env_name="weapon_damage",
        objective="dps",
        k=6,
        delta=0.1,
        weapon_spec="archetypes",
        seed=4,
        seed_step=1,
        trials=2,
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
    assert "weapon_name" in reader[0]
    assert reader[0]["weapon_name"] == ""


def test_write_trials_csv_weapon_has_names(tmp_path: Path) -> None:
    config = _weapon_config()
    trials = run_trials(config)
    out_path = tmp_path / "weapon_trial_report.csv"
    write_trials_csv(str(out_path), trials)

    with out_path.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    names_raw = reader[0]["weapon_name"]
    assert names_raw != ""
    names = json.loads(names_raw)
    assert len(names) == config.k
    assert len(set(names)) == config.k


def test_trial_json_has_arm_metadata_with_weapon_names() -> None:
    config = _weapon_config()
    config.trials = 1
    trials = run_trials(config)
    record = _trial_to_json_record(trials[0])

    assert "arm_metadata" in record
    arm_meta = record["arm_metadata"]
    assert isinstance(arm_meta, list)
    assert len(arm_meta) == config.k
    first = arm_meta[0]
    assert "arm" in first
    assert "weapon_name" in first
    assert "d0" in first
    assert "d1" in first
    assert "p" in first
    assert "accuracy" in first
    assert "crit_multiplier" in first
    assert "enemy" in record
