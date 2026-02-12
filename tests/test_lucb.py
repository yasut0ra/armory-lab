import numpy as np

from armory_lab.algos.lucb import LUCB
from armory_lab.envs.bernoulli import BernoulliBandit


def test_lucb_identifies_clear_best() -> None:
    means = np.asarray([0.85, 0.3, 0.2, 0.1], dtype=np.float64)
    env = BernoulliBandit.from_means(means, seed=7)

    algo = LUCB(delta=0.05, max_pulls=100_000)
    result = algo.run(env, track_history=True)

    assert result.recommend_arm == 0
    assert 0 < result.total_pulls <= 100_000
    assert int(result.pulls_per_arm[0]) >= 1
    assert len(result.history) >= 1
