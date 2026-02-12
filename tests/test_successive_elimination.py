import numpy as np

from armory_lab.algos.successive_elimination import SuccessiveElimination
from armory_lab.envs.bernoulli import BernoulliBandit


def test_successive_elimination_identifies_clear_best() -> None:
    means = np.asarray([0.8, 0.2, 0.2, 0.2], dtype=np.float64)
    env = BernoulliBandit.from_means(means, seed=0)

    algo = SuccessiveElimination(delta=0.05, max_pulls=100_000)
    result = algo.run(env, track_history=True)

    assert result.recommend_arm == 0
    assert 0 < result.total_pulls <= 100_000
    assert int(result.pulls_per_arm[0]) >= 1
    assert len(result.history) >= 1
