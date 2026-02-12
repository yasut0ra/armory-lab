import numpy as np

from armory_lab.envs.weapon_damage import WeaponDamageBandit, generate_weapon_pack


def test_weapon_damage_reproducible_with_seed() -> None:
    d0 = [80.0, 70.0, 60.0]
    d1 = [120.0, 110.0, 100.0]
    p = [0.2, 0.4, 0.1]
    arm_seq = [0, 1, 2, 1, 0, 2, 1, 1, 0, 2, 0, 1]

    env_a = WeaponDamageBandit.from_params(d0=d0, d1=d1, p=p, seed=42)
    env_b = WeaponDamageBandit.from_params(d0=d0, d1=d1, p=p, seed=42)

    out_a = [env_a.pull(arm) for arm in arm_seq]
    out_b = [env_b.pull(arm) for arm in arm_seq]

    assert out_a == out_b


def test_weapon_pack_names_reproducible_and_unique() -> None:
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    pack_a = generate_weapon_pack(spec="archetypes", k=24, rng=rng_a)
    pack_b = generate_weapon_pack(spec="archetypes", k=24, rng=rng_b)

    assert pack_a.weapon_names is not None
    assert pack_a.weapon_names == pack_b.weapon_names
    assert len(pack_a.weapon_names) == 24
    assert len(set(pack_a.weapon_names)) == 24
    assert all(name.strip() for name in pack_a.weapon_names)
