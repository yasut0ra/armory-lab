from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class EnemyProfile:
    name: str
    hp: float
    evasion: float
    description: str


ENEMY_PROFILES: dict[str, EnemyProfile] = {
    "none": EnemyProfile(name="none", hp=1_000_000.0, evasion=0.0, description="No special defense."),
    "slime": EnemyProfile(name="slime", hp=70.0, evasion=0.05, description="Low HP. Stable weapons shine."),
    "golem": EnemyProfile(name="golem", hp=180.0, evasion=0.10, description="High HP. High expected damage is valuable."),
    "ghost": EnemyProfile(name="ghost", hp=120.0, evasion=0.35, description="High evasion. Accuracy matters."),
}
ENEMY_NAMES: tuple[str, ...] = tuple(ENEMY_PROFILES.keys())


def get_enemy_profile(name: str) -> EnemyProfile:
    normalized = name.strip().lower()
    if normalized not in ENEMY_PROFILES:
        raise ValueError(f"unknown enemy type: {name}")
    return ENEMY_PROFILES[normalized]


@dataclass(slots=True)
class WeaponPack:
    d0: NDArray[np.float64]
    d1: NDArray[np.float64]
    p: NDArray[np.float64]
    accuracy: NDArray[np.float64] | None = None
    crit_multiplier: NDArray[np.float64] | None = None
    enemy_name: str = "none"
    enemy_hp: float = 1_000_000.0
    enemy_evasion: float = 0.0
    weapon_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        n_arms = int(self.d0.size)
        if self.weapon_names is None:
            self.weapon_names = tuple(f"Weapon-{i}" for i in range(n_arms))
        if len(self.weapon_names) != n_arms:
            raise ValueError("weapon_names must have the same length as d0/d1/p")
        if len(set(self.weapon_names)) != len(self.weapon_names):
            raise ValueError("weapon_names must be unique")
        if any(name.strip() == "" for name in self.weapon_names):
            raise ValueError("weapon_names must be non-empty")

        if self.accuracy is None:
            self.accuracy = np.ones(n_arms, dtype=np.float64)
        if self.crit_multiplier is None:
            self.crit_multiplier = np.asarray(self.d1 / np.maximum(self.d0, 1e-9), dtype=np.float64)

        if int(self.accuracy.size) != n_arms:
            raise ValueError("accuracy must have the same length as d0/d1/p")
        if int(self.crit_multiplier.size) != n_arms:
            raise ValueError("crit_multiplier must have the same length as d0/d1/p")
        if np.any((self.accuracy <= 0.0) | (self.accuracy > 1.0)):
            raise ValueError("accuracy must satisfy 0 < accuracy <= 1")
        if np.any(self.crit_multiplier <= 1.0):
            raise ValueError("crit_multiplier must satisfy > 1")
        if self.enemy_evasion < 0.0 or self.enemy_evasion >= 1.0:
            raise ValueError("enemy_evasion must be in [0, 1)")
        if self.enemy_hp <= 0.0:
            raise ValueError("enemy_hp must be positive")

    @property
    def n_arms(self) -> int:
        return int(self.d0.size)

    @property
    def hit_probabilities(self) -> NDArray[np.float64]:
        if self.accuracy is None:
            raise RuntimeError("accuracy is not initialized")
        return np.clip(self.accuracy * (1.0 - self.enemy_evasion), 0.0, 1.0)

    @property
    def capped_d0(self) -> NDArray[np.float64]:
        return np.minimum(self.d0, self.enemy_hp)

    @property
    def capped_d1(self) -> NDArray[np.float64]:
        return np.minimum(self.d1, self.enemy_hp)

    @property
    def expected_damages(self) -> NDArray[np.float64]:
        crit_component = self.capped_d0 * (1.0 - self.p) + self.capped_d1 * self.p
        return self.hit_probabilities * crit_component

    def oneshot_probabilities(self, threshold: float) -> NDArray[np.float64]:
        hit_d0 = (self.capped_d0 >= threshold).astype(np.float64)
        hit_d1 = (self.capped_d1 >= threshold).astype(np.float64)
        miss_hit = 1.0 if threshold <= 0.0 else 0.0
        hit_term = self.p * hit_d1 + (1.0 - self.p) * hit_d0
        return (1.0 - self.hit_probabilities) * miss_hit + self.hit_probabilities * hit_term


@dataclass(slots=True)
class WeaponDamageBandit:
    d0: NDArray[np.float64]
    d1: NDArray[np.float64]
    p: NDArray[np.float64]
    accuracy: NDArray[np.float64]
    crit_multiplier: NDArray[np.float64]
    enemy_name: str
    enemy_hp: float
    enemy_evasion: float
    weapon_names: tuple[str, ...]
    rng: np.random.Generator

    @classmethod
    def from_params(
        cls,
        d0: list[float] | NDArray[np.float64],
        d1: list[float] | NDArray[np.float64],
        p: list[float] | NDArray[np.float64],
        accuracy: list[float] | NDArray[np.float64] | None = None,
        crit_multiplier: list[float] | NDArray[np.float64] | None = None,
        enemy: str = "none",
        weapon_names: list[str] | tuple[str, ...] | None = None,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "WeaponDamageBandit":
        d0_arr = np.asarray(d0, dtype=np.float64)
        d1_arr = np.asarray(d1, dtype=np.float64)
        p_arr = np.asarray(p, dtype=np.float64)

        if d0_arr.ndim != 1 or d1_arr.ndim != 1 or p_arr.ndim != 1:
            raise ValueError("d0, d1, p must be 1D arrays")
        if d0_arr.size == 0 or d0_arr.size != d1_arr.size or d0_arr.size != p_arr.size:
            raise ValueError("d0, d1, p must have same non-zero length")
        if np.any(d1_arr <= d0_arr):
            raise ValueError("must satisfy d1 > d0 for all arms")
        if np.any((p_arr <= 0.0) | (p_arr >= 1.0)):
            raise ValueError("must satisfy 0 < p < 1 for all arms")

        n_arms = int(d0_arr.size)
        if accuracy is None:
            acc_arr = np.ones(n_arms, dtype=np.float64)
        else:
            acc_arr = np.asarray(accuracy, dtype=np.float64)
        if acc_arr.ndim != 1 or int(acc_arr.size) != n_arms:
            raise ValueError("accuracy must have same length as d0/d1/p")
        if np.any((acc_arr <= 0.0) | (acc_arr > 1.0)):
            raise ValueError("must satisfy 0 < accuracy <= 1 for all arms")

        if crit_multiplier is None:
            crit_mult_arr = np.asarray(d1_arr / np.maximum(d0_arr, 1e-9), dtype=np.float64)
        else:
            crit_mult_arr = np.asarray(crit_multiplier, dtype=np.float64)
        if crit_mult_arr.ndim != 1 or int(crit_mult_arr.size) != n_arms:
            raise ValueError("crit_multiplier must have same length as d0/d1/p")
        if np.any(crit_mult_arr <= 1.0):
            raise ValueError("must satisfy crit_multiplier > 1 for all arms")

        enemy_profile = get_enemy_profile(enemy)
        generator = rng if rng is not None else np.random.default_rng(seed)
        names = _normalize_weapon_names(weapon_names, n_arms)
        return cls(
            d0=d0_arr,
            d1=d1_arr,
            p=p_arr,
            accuracy=acc_arr,
            crit_multiplier=crit_mult_arr,
            enemy_name=enemy_profile.name,
            enemy_hp=enemy_profile.hp,
            enemy_evasion=enemy_profile.evasion,
            weapon_names=names,
            rng=generator,
        )

    @classmethod
    def from_pack(
        cls,
        pack: WeaponPack,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "WeaponDamageBandit":
        return cls.from_params(
            pack.d0,
            pack.d1,
            pack.p,
            accuracy=pack.accuracy,
            crit_multiplier=pack.crit_multiplier,
            enemy=pack.enemy_name,
            weapon_names=pack.weapon_names,
            seed=seed,
            rng=rng,
        )

    @property
    def n_arms(self) -> int:
        return int(self.d0.size)

    @property
    def hit_probabilities(self) -> NDArray[np.float64]:
        return np.clip(self.accuracy * (1.0 - self.enemy_evasion), 0.0, 1.0)

    @property
    def reward_min(self) -> float:
        if np.any(self.hit_probabilities < 1.0 - 1e-12):
            return 0.0
        return float(np.min(np.minimum(self.d0, self.enemy_hp)))

    @property
    def reward_max(self) -> float:
        return float(np.max(np.minimum(self.d1, self.enemy_hp)))

    @property
    def reward_range(self) -> float:
        return float(self.reward_max - self.reward_min)

    @property
    def expected_damages(self) -> NDArray[np.float64]:
        d0_eff = np.minimum(self.d0, self.enemy_hp)
        d1_eff = np.minimum(self.d1, self.enemy_hp)
        crit_component = d0_eff * (1.0 - self.p) + d1_eff * self.p
        return self.hit_probabilities * crit_component

    def oneshot_probabilities(self, threshold: float) -> NDArray[np.float64]:
        d0_eff = np.minimum(self.d0, self.enemy_hp)
        d1_eff = np.minimum(self.d1, self.enemy_hp)
        hit_d0 = (d0_eff >= threshold).astype(np.float64)
        hit_d1 = (d1_eff >= threshold).astype(np.float64)
        miss_hit = 1.0 if threshold <= 0.0 else 0.0
        hit_term = self.p * hit_d1 + (1.0 - self.p) * hit_d0
        return (1.0 - self.hit_probabilities) * miss_hit + self.hit_probabilities * hit_term

    def pull(self, arm: int) -> float:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} out of range")

        hit_prob = float(self.hit_probabilities[arm])
        is_hit = bool(self.rng.random() < hit_prob)
        if not is_hit:
            return 0.0

        is_crit = bool(self.rng.random() < float(self.p[arm]))
        reward_raw = float(self.d1[arm] if is_crit else self.d0[arm])
        reward = float(min(reward_raw, self.enemy_hp))
        return reward



def _ensure_unique_best(values: NDArray[np.float64]) -> NDArray[np.float64]:
    out = np.asarray(values, dtype=np.float64).copy()
    out += np.linspace(0.0, 1e-8, out.size)
    best = int(np.argmax(out))
    max_val = float(out[best])
    ties = np.where(np.isclose(out, max_val))[0]
    if ties.size > 1:
        out[best] = out[best] + 1e-4
    return np.asarray(out, dtype=np.float64)


_BASE_WEAPON_NAMES: tuple[str, ...] = (
    "Sword",
    "Spear",
    "Axe",
    "Bow",
    "Dagger",
    "Hammer",
    "Halberd",
    "Mace",
    "Crossbow",
    "Katana",
    "Rapier",
    "Scythe",
    "Whip",
    "Pike",
    "Gauntlet",
    "Lance",
    "Morningstar",
    "Claymore",
    "Falchion",
    "Trident",
    "Glaive",
    "Warpick",
    "Saber",
    "Estoc",
    "Longbow",
    "Shortbow",
    "Flail",
    "Greatsword",
    "Battleaxe",
    "Handaxe",
    "Javelin",
    "Twinblade",
    "Naginata",
    "Khopesh",
    "Cutlass",
    "Broadsword",
    "Warhammer",
    "Shotel",
    "Billhook",
    "Quarterstaff",
    "Poleaxe",
    "Dirk",
    "Katar",
    "Arbalest",
    "Sling",
    "Sai",
    "Tonfa",
    "Urumi",
)


def _normalize_weapon_names(
    weapon_names: list[str] | tuple[str, ...] | None,
    k: int,
) -> tuple[str, ...]:
    if weapon_names is None:
        return tuple(f"Weapon-{i}" for i in range(k))

    names = tuple(name.strip() for name in weapon_names)
    if len(names) != k:
        raise ValueError("weapon_names must have length K")
    if any(name == "" for name in names):
        raise ValueError("weapon_names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("weapon_names must be unique")
    return names


def generate_weapon_names(k: int, rng: np.random.Generator) -> tuple[str, ...]:
    if k <= 0:
        raise ValueError("K must be positive")

    pool = list(_BASE_WEAPON_NAMES)
    seen_counts: dict[str, int] = {}
    out: list[str] = []

    while len(out) < k:
        order = rng.permutation(len(pool))
        for idx in order:
            base = pool[int(idx)]
            seen_counts[base] = seen_counts.get(base, 0) + 1
            count = seen_counts[base]
            if count == 1:
                out.append(base)
            else:
                out.append(f"{base} Mk{count}")

            if len(out) >= k:
                break
    return tuple(out)


def list_enemy_types() -> tuple[str, ...]:
    return ENEMY_NAMES


def _expected_with_enemy(
    d0: NDArray[np.float64],
    d1: NDArray[np.float64],
    p: NDArray[np.float64],
    accuracy: NDArray[np.float64],
    enemy_evasion: float,
    enemy_hp: float,
) -> NDArray[np.float64]:
    hit = np.clip(accuracy * (1.0 - enemy_evasion), 0.0, 1.0)
    d0_eff = np.minimum(d0, enemy_hp)
    d1_eff = np.minimum(d1, enemy_hp)
    return hit * (d0_eff * (1.0 - p) + d1_eff * p)



def generate_weapon_pack(spec: str, k: int, rng: np.random.Generator, enemy: str = "none") -> WeaponPack:
    if k <= 0:
        raise ValueError("K must be positive")
    enemy_profile = get_enemy_profile(enemy)

    if spec == "random":
        d0 = rng.uniform(50.0, 90.0, size=k)
        d1 = rng.uniform(90.0, 140.0, size=k)
        d1 = np.maximum(d1, d0 + 5.0)
        p = rng.uniform(0.05, 0.35, size=k)
        accuracy = rng.uniform(0.70, 0.99, size=k)
        crit_multiplier = d1 / np.maximum(d0, 1e-9)
        weapon_names = generate_weapon_names(k, rng)

        expected = _ensure_unique_best(_expected_with_enemy(d0, d1, p, accuracy, enemy_profile.evasion, enemy_profile.hp))
        best = int(np.argmax(expected))
        if best >= 0:
            # Tiny lift to keep unique best robustly after regeneration noise.
            d1[best] = d1[best] + 0.5
        return WeaponPack(
            d0=d0.astype(np.float64),
            d1=d1.astype(np.float64),
            p=p.astype(np.float64),
            accuracy=accuracy.astype(np.float64),
            crit_multiplier=crit_multiplier.astype(np.float64),
            enemy_name=enemy_profile.name,
            enemy_hp=enemy_profile.hp,
            enemy_evasion=enemy_profile.evasion,
            weapon_names=weapon_names,
        )

    if spec.startswith("topgap:"):
        try:
            gap = float(spec.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("topgap format must be topgap:<float>") from exc
        if gap <= 0.0:
            raise ValueError("topgap gap must be positive")

        p = np.full(k, 0.22, dtype=np.float64)
        accuracy = rng.uniform(0.80, 0.98, size=k)
        crit_multiplier = rng.uniform(1.5, 2.2, size=k)

        expected = np.empty(k, dtype=np.float64)
        expected[0] = 95.0 + gap
        if k > 1:
            expected[1] = 95.0
        if k > 2:
            expected[2:] = 95.0 - rng.uniform(8.0, 26.0, size=k - 2)

        hit = np.clip(accuracy * (1.0 - enemy_profile.evasion), 0.0, 1.0)
        scale = np.maximum(hit * (1.0 + p * (crit_multiplier - 1.0)), 1e-3)
        d0 = expected / scale
        d1 = d0 * crit_multiplier
        d0 = np.clip(d0, 30.0, 150.0)
        d1 = np.maximum(d1, d0 + 1.0)
        crit_multiplier = d1 / np.maximum(d0, 1e-9)
        weapon_names = generate_weapon_names(k, rng)
        best = int(np.argmax(_ensure_unique_best(_expected_with_enemy(d0, d1, p, accuracy, enemy_profile.evasion, enemy_profile.hp))))
        d1[best] = d1[best] + 0.5

        return WeaponPack(
            d0=d0.astype(np.float64),
            d1=d1.astype(np.float64),
            p=p.astype(np.float64),
            accuracy=accuracy.astype(np.float64),
            crit_multiplier=crit_multiplier.astype(np.float64),
            enemy_name=enemy_profile.name,
            enemy_hp=enemy_profile.hp,
            enemy_evasion=enemy_profile.evasion,
            weapon_names=weapon_names,
        )

    if spec == "archetypes":
        archetypes = rng.choice([0, 1, 2], size=k, p=np.asarray([0.40, 0.40, 0.20]))
        d0 = np.zeros(k, dtype=np.float64)
        d1 = np.zeros(k, dtype=np.float64)
        p = np.zeros(k, dtype=np.float64)
        accuracy = np.zeros(k, dtype=np.float64)

        for i in range(k):
            t = int(archetypes[i])
            if t == 0:
                # Stable: higher normal damage, low crit volatility.
                d0_i = rng.uniform(80.0, 95.0)
                p_i = rng.uniform(0.05, 0.12)
                bonus_i = rng.uniform(20.0, 35.0)
                acc_i = rng.uniform(0.90, 0.99)
            elif t == 1:
                # Crit-focused: moderate base with frequent meaningful crits.
                d0_i = rng.uniform(55.0, 75.0)
                p_i = rng.uniform(0.20, 0.35)
                bonus_i = rng.uniform(45.0, 70.0)
                acc_i = rng.uniform(0.80, 0.95)
            else:
                # Gamble: weak base, rare but huge spikes.
                d0_i = rng.uniform(35.0, 60.0)
                p_i = rng.uniform(0.03, 0.08)
                bonus_i = rng.uniform(100.0, 180.0)
                acc_i = rng.uniform(0.65, 0.88)

            d0[i] = d0_i
            p[i] = p_i
            d1[i] = d0_i + bonus_i
            accuracy[i] = acc_i

        crit_multiplier = d1 / np.maximum(d0, 1e-9)
        expected = _ensure_unique_best(_expected_with_enemy(d0, d1, p, accuracy, enemy_profile.evasion, enemy_profile.hp))
        best = int(np.argmax(expected))
        d1[best] = d1[best] + 0.5
        crit_multiplier = d1 / np.maximum(d0, 1e-9)
        weapon_names = generate_weapon_names(k, rng)

        return WeaponPack(
            d0=d0.astype(np.float64),
            d1=d1.astype(np.float64),
            p=p.astype(np.float64),
            accuracy=accuracy.astype(np.float64),
            crit_multiplier=crit_multiplier.astype(np.float64),
            enemy_name=enemy_profile.name,
            enemy_hp=enemy_profile.hp,
            enemy_evasion=enemy_profile.evasion,
            weapon_names=weapon_names,
        )

    raise ValueError(f"unknown weapon pack regime: {spec}")
