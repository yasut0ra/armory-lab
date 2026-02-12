import numpy as np
import pytest

from armory_lab.algos.lucb import LUCB
from armory_lab.algos.successive_elimination import SuccessiveElimination
from armory_lab.envs.weapon_damage import WeaponDamageBandit
from armory_lab.objective import ObjectiveBandit


@pytest.mark.parametrize("algo_name", ["lucb", "se"])
def test_weapon_damage_dps_identifies_clear_best(algo_name: str) -> None:
    env = WeaponDamageBandit.from_params(
        d0=np.asarray([90.0, 70.0, 50.0], dtype=np.float64),
        d1=np.asarray([110.0, 95.0, 80.0], dtype=np.float64),
        p=np.asarray([0.2, 0.2, 0.2], dtype=np.float64),
        seed=0,
    )
    obj_env = ObjectiveBandit(base_env=env, objective="dps")

    if algo_name == "lucb":
        algo = LUCB(delta=0.05, max_pulls=100_000)
    elif algo_name == "se":
        algo = SuccessiveElimination(delta=0.05, max_pulls=100_000)
    else:
        raise AssertionError("unsupported algo")
    result = algo.run(obj_env, track_history=False)
    assert result.recommend_arm == 0


@pytest.mark.parametrize("algo_name", ["lucb", "se"])
def test_weapon_damage_oneshot_identifies_clear_best(algo_name: str) -> None:
    env = WeaponDamageBandit.from_params(
        d0=np.asarray([80.0, 80.0, 70.0], dtype=np.float64),
        d1=np.asarray([110.0, 130.0, 120.0], dtype=np.float64),
        p=np.asarray([0.1, 0.5, 0.2], dtype=np.float64),
        seed=9,
    )
    obj_env = ObjectiveBandit(base_env=env, objective="oneshot", threshold=100.0)

    if algo_name == "lucb":
        algo = LUCB(delta=0.05, max_pulls=100_000)
    else:
        algo = SuccessiveElimination(delta=0.05, max_pulls=100_000)

    result = algo.run(obj_env, track_history=False)
    assert result.recommend_arm == 1


def test_weapon_damage_oneshot_smoke_stops() -> None:
    env = WeaponDamageBandit.from_params(
        d0=np.asarray([75.0, 72.0, 68.0, 60.0], dtype=np.float64),
        d1=np.asarray([120.0, 118.0, 105.0, 95.0], dtype=np.float64),
        p=np.asarray([0.30, 0.22, 0.18, 0.12], dtype=np.float64),
        seed=13,
    )
    obj_env = ObjectiveBandit(base_env=env, objective="oneshot", threshold=100.0)
    algo = LUCB(delta=0.1, max_pulls=80_000)

    result = algo.run(obj_env, track_history=False)
    assert 0 <= result.recommend_arm < env.n_arms
    assert 0 < result.total_pulls <= 80_000


def test_weapon_damage_enemy_type_can_flip_best() -> None:
    env_slime = WeaponDamageBandit.from_params(
        d0=np.asarray([50.0, 88.0], dtype=np.float64),
        d1=np.asarray([260.0, 108.0], dtype=np.float64),
        p=np.asarray([0.45, 0.10], dtype=np.float64),
        accuracy=np.asarray([0.90, 0.95], dtype=np.float64),
        enemy="slime",
        seed=5,
    )
    env_golem = WeaponDamageBandit.from_params(
        d0=np.asarray([50.0, 88.0], dtype=np.float64),
        d1=np.asarray([260.0, 108.0], dtype=np.float64),
        p=np.asarray([0.45, 0.10], dtype=np.float64),
        accuracy=np.asarray([0.90, 0.95], dtype=np.float64),
        enemy="golem",
        seed=5,
    )

    best_slime = int(np.argmax(env_slime.expected_damages))
    best_golem = int(np.argmax(env_golem.expected_damages))

    assert best_slime == 1
    assert best_golem == 0
