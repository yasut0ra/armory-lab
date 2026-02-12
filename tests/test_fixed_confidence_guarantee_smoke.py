import numpy as np
import pytest

from armory_lab.algos.kl_lucb import KLLUCB
from armory_lab.algos.lucb import LUCB
from armory_lab.algos.successive_elimination import SuccessiveElimination
from armory_lab.algos.top_two_thompson_sampling import TopTwoThompsonSampling
from armory_lab.algos.track_and_stop import TrackAndStop
from armory_lab.envs.bernoulli import BernoulliBandit


@pytest.mark.parametrize("algo_name", ["kllucb", "lucb", "se", "tas", "ttts"])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_smoke_multiple_seeds_terminates(algo_name: str, seed: int) -> None:
    means = np.asarray([0.7, 0.55, 0.45, 0.35, 0.25], dtype=np.float64)
    env = BernoulliBandit.from_means(means, seed=seed)

    if algo_name == "kllucb":
        algo = KLLUCB(delta=0.1, max_pulls=60_000)
    elif algo_name == "lucb":
        algo = LUCB(delta=0.1, max_pulls=60_000)
    elif algo_name == "se":
        algo = SuccessiveElimination(delta=0.1, max_pulls=60_000)
    elif algo_name == "ttts":
        algo = TopTwoThompsonSampling(delta=0.1, max_pulls=60_000)
    else:
        algo = TrackAndStop(delta=0.1, max_pulls=60_000)

    result = algo.run(env, track_history=False)

    assert 0 <= result.recommend_arm < means.size
    assert 0 < result.total_pulls <= 60_000
