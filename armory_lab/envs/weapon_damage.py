from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class WeaponPack:
    d0: NDArray[np.float64]
    d1: NDArray[np.float64]
    p: NDArray[np.float64]
    weapon_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.weapon_names is None:
            return
        if len(self.weapon_names) != int(self.d0.size):
            raise ValueError("weapon_names must have the same length as d0/d1/p")
        if len(set(self.weapon_names)) != len(self.weapon_names):
            raise ValueError("weapon_names must be unique")
        if any(name.strip() == "" for name in self.weapon_names):
            raise ValueError("weapon_names must be non-empty")

    @property
    def n_arms(self) -> int:
        return int(self.d0.size)

    @property
    def expected_damages(self) -> NDArray[np.float64]:
        return self.d0 * (1.0 - self.p) + self.d1 * self.p

    def oneshot_probabilities(self, threshold: float) -> NDArray[np.float64]:
        hit_d0 = (self.d0 >= threshold).astype(np.float64)
        hit_d1 = (self.d1 >= threshold).astype(np.float64)
        return self.p * hit_d1 + (1.0 - self.p) * hit_d0


@dataclass(slots=True)
class WeaponDamageBandit:
    d0: NDArray[np.float64]
    d1: NDArray[np.float64]
    p: NDArray[np.float64]
    weapon_names: tuple[str, ...]
    rng: np.random.Generator

    @classmethod
    def from_params(
        cls,
        d0: list[float] | NDArray[np.float64],
        d1: list[float] | NDArray[np.float64],
        p: list[float] | NDArray[np.float64],
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

        generator = rng if rng is not None else np.random.default_rng(seed)
        names = _normalize_weapon_names(weapon_names, int(d0_arr.size))
        return cls(d0=d0_arr, d1=d1_arr, p=p_arr, weapon_names=names, rng=generator)

    @classmethod
    def from_pack(
        cls,
        pack: WeaponPack,
        seed: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> "WeaponDamageBandit":
        return cls.from_params(pack.d0, pack.d1, pack.p, weapon_names=pack.weapon_names, seed=seed, rng=rng)

    @property
    def n_arms(self) -> int:
        return int(self.d0.size)

    @property
    def reward_min(self) -> float:
        return float(np.min(self.d0))

    @property
    def reward_max(self) -> float:
        return float(np.max(self.d1))

    @property
    def reward_range(self) -> float:
        return float(self.reward_max - self.reward_min)

    @property
    def expected_damages(self) -> NDArray[np.float64]:
        return self.d0 * (1.0 - self.p) + self.d1 * self.p

    def oneshot_probabilities(self, threshold: float) -> NDArray[np.float64]:
        hit_d0 = (self.d0 >= threshold).astype(np.float64)
        hit_d1 = (self.d1 >= threshold).astype(np.float64)
        return self.p * hit_d1 + (1.0 - self.p) * hit_d0

    def pull(self, arm: int) -> float:
        if arm < 0 or arm >= self.n_arms:
            raise IndexError(f"arm index {arm} out of range")

        is_crit = bool(self.rng.random() < float(self.p[arm]))
        reward = float(self.d1[arm] if is_crit else self.d0[arm])
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



def generate_weapon_pack(spec: str, k: int, rng: np.random.Generator) -> WeaponPack:
    if k <= 0:
        raise ValueError("K must be positive")

    if spec == "random":
        d0 = rng.uniform(50.0, 90.0, size=k)
        d1 = rng.uniform(90.0, 140.0, size=k)
        d1 = np.maximum(d1, d0 + 5.0)
        p = rng.uniform(0.05, 0.35, size=k)
        weapon_names = generate_weapon_names(k, rng)

        expected = _ensure_unique_best(d0 * (1.0 - p) + d1 * p)
        best = int(np.argmax(expected))
        if best >= 0:
            # Tiny lift to keep unique best robustly after regeneration noise.
            d1[best] = d1[best] + 0.5
        return WeaponPack(
            d0=d0.astype(np.float64),
            d1=d1.astype(np.float64),
            p=p.astype(np.float64),
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
        bonus = np.full(k, 60.0, dtype=np.float64)

        expected = np.empty(k, dtype=np.float64)
        expected[0] = 95.0 + gap
        if k > 1:
            expected[1] = 95.0
        if k > 2:
            expected[2:] = 95.0 - rng.uniform(8.0, 26.0, size=k - 2)

        d0 = expected - p * bonus
        d1 = d0 + bonus
        d0 = np.clip(d0, 30.0, 150.0)
        d1 = np.maximum(d1, d0 + 1.0)
        weapon_names = generate_weapon_names(k, rng)

        return WeaponPack(
            d0=d0.astype(np.float64),
            d1=d1.astype(np.float64),
            p=p.astype(np.float64),
            weapon_names=weapon_names,
        )

    if spec == "archetypes":
        archetypes = rng.choice([0, 1, 2], size=k, p=np.asarray([0.40, 0.40, 0.20]))
        d0 = np.zeros(k, dtype=np.float64)
        d1 = np.zeros(k, dtype=np.float64)
        p = np.zeros(k, dtype=np.float64)

        for i in range(k):
            t = int(archetypes[i])
            if t == 0:
                # Stable: higher normal damage, low crit volatility.
                d0_i = rng.uniform(80.0, 95.0)
                p_i = rng.uniform(0.05, 0.12)
                bonus_i = rng.uniform(20.0, 35.0)
            elif t == 1:
                # Crit-focused: moderate base with frequent meaningful crits.
                d0_i = rng.uniform(55.0, 75.0)
                p_i = rng.uniform(0.20, 0.35)
                bonus_i = rng.uniform(45.0, 70.0)
            else:
                # Gamble: weak base, rare but huge spikes.
                d0_i = rng.uniform(35.0, 60.0)
                p_i = rng.uniform(0.03, 0.08)
                bonus_i = rng.uniform(100.0, 180.0)

            d0[i] = d0_i
            p[i] = p_i
            d1[i] = d0_i + bonus_i

        expected = _ensure_unique_best(d0 * (1.0 - p) + d1 * p)
        best = int(np.argmax(expected))
        d1[best] = d1[best] + 0.5
        weapon_names = generate_weapon_names(k, rng)

        return WeaponPack(
            d0=d0.astype(np.float64),
            d1=d1.astype(np.float64),
            p=p.astype(np.float64),
            weapon_names=weapon_names,
        )

    raise ValueError(f"unknown weapon pack regime: {spec}")
