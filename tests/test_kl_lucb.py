import numpy as np

from armory_lab.algos.kl_lucb import KLLUCB
from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.objective import ObjectiveBandit


def test_kl_lucb_identifies_clear_best() -> None:
    means = np.asarray([0.86, 0.35, 0.2, 0.1], dtype=np.float64)
    env = BernoulliBandit.from_means(means, seed=23)
    algo = KLLUCB(delta=0.05, max_pulls=120_000)
    result = algo.run(env, track_history=True)

    assert result.recommend_arm == 0
    assert 0 < result.total_pulls <= 120_000
    assert int(result.pulls_per_arm[0]) >= 1
    assert len(result.history) >= 1


def test_kl_lucb_works_on_oneshot_objective() -> None:
    means = np.asarray([0.75, 0.55, 0.35], dtype=np.float64)
    base_env = BernoulliBandit.from_means(means, seed=9)
    env = ObjectiveBandit(base_env=base_env, objective="oneshot", threshold=1.0)
    algo = KLLUCB(delta=0.1, max_pulls=80_000)
    result = algo.run(env, track_history=False)
    assert result.recommend_arm == 0


def test_kl_lucb_rejects_non_bernoulli_rewards() -> None:
    class DummyEnv:
        n_arms = 2
        reward_min = 0.0
        reward_max = 10.0
        reward_range = 10.0

        def pull(self, arm: int) -> float:
            return 5.0 + float(arm)

    algo = KLLUCB(delta=0.1, max_pulls=1000)
    try:
        algo.run(DummyEnv(), track_history=False)
    except ValueError as exc:
        assert "Bernoulli-style rewards" in str(exc)
    else:
        raise AssertionError("expected ValueError")
