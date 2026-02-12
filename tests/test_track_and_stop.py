import numpy as np

from armory_lab.algos.track_and_stop import TrackAndStop
from armory_lab.envs.bernoulli import BernoulliBandit


def test_track_and_stop_identifies_clear_best() -> None:
    means = np.asarray([0.9, 0.35, 0.2, 0.1], dtype=np.float64)
    env = BernoulliBandit.from_means(means, seed=11)

    algo = TrackAndStop(delta=0.05, max_pulls=120_000)
    result = algo.run(env, track_history=True)

    assert result.recommend_arm == 0
    assert 0 < result.total_pulls <= 120_000
    assert int(result.pulls_per_arm[0]) >= 1
    assert len(result.history) >= 1
