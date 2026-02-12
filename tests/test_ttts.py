import numpy as np

from armory_lab.algos.top_two_thompson_sampling import TopTwoThompsonSampling
from armory_lab.envs.bernoulli import BernoulliBandit


def test_ttts_identifies_clear_best() -> None:
    means = np.asarray([0.92, 0.40, 0.25, 0.10], dtype=np.float64)
    env = BernoulliBandit.from_means(means, seed=19)

    algo = TopTwoThompsonSampling(delta=0.05, max_pulls=120_000)
    result = algo.run(env, track_history=True)

    assert result.recommend_arm == 0
    assert 0 < result.total_pulls <= 120_000
    assert int(result.pulls_per_arm[0]) >= 1
    assert len(result.history) >= 1
