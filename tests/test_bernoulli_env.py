from armory_lab.envs.bernoulli import BernoulliBandit


def test_reproducible_pull_sequence_with_seed() -> None:
    means = [0.2, 0.8, 0.5]
    arm_sequence = [0, 1, 1, 2, 0, 2, 1, 0, 2, 2, 1, 0]

    env_a = BernoulliBandit.from_means(means, seed=42)
    env_b = BernoulliBandit.from_means(means, seed=42)

    out_a = [env_a.pull(arm) for arm in arm_sequence]
    out_b = [env_b.pull(arm) for arm in arm_sequence]

    assert out_a == out_b
