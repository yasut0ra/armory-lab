"""Armory Lab: Best Arm Identification toolkit."""

from armory_lab.algos.base import BAIResult, HistoryRecord
from armory_lab.algos.lucb import LUCB
from armory_lab.algos.successive_elimination import SuccessiveElimination
from armory_lab.algos.top_two_thompson_sampling import TopTwoThompsonSampling
from armory_lab.algos.track_and_stop import TrackAndStop
from armory_lab.envs.bernoulli import BernoulliBandit
from armory_lab.envs.weapon_damage import WeaponDamageBandit

__all__ = [
    "BAIResult",
    "HistoryRecord",
    "LUCB",
    "SuccessiveElimination",
    "TopTwoThompsonSampling",
    "TrackAndStop",
    "BernoulliBandit",
    "WeaponDamageBandit",
]
