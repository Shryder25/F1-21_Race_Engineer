"""
Real-race calibration maths (no FastF1 needed here)
===================================================
Pure functions that turn real race lap data into the measured values track_profiles.py
layers over its baseline. tools/build_real_race_calibration.py feeds them with races
loaded through FastF1 (the data source IAmTomShaw/f1-race-replay replays); keeping the
maths here means it is unit-tested without network access.

Method (after f1-race-replay's Bayesian tyre model):
  * drop lap 1, in/out laps, Safety Car / VSC / red-flag laps and outliers;
  * fuel-correct every lap: t - 0.032 s/kg * fuel_mass, fuel burning linearly from 110 kg;
  * skip the warm-up laps of each stint, then take the Theil-Sen slope of lap time
    against tyre age: a robust "seconds lost per lap of age" for that stint;
  * the track's rate for a compound is the median slope over all stints.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

FUEL_EFFECT_S_PER_KG = 0.032
STARTING_FUEL_KG = 110.0
WARMUP_LAPS = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 2, 'WET': 2}
MIN_STINT_LAPS = 6
NEUTRAL_DEG = {'SOFT': 0.100, 'MEDIUM': 0.055, 'HARD': 0.032}   # matches track_analysis.BASE_DEG


def theil_sen_slope(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Median of all pairwise slopes: robust to the odd traffic or mistake lap."""
    xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(xs) < 3:
        return None
    i, j = np.triu_indices(len(xs), k=1)
    dx = xs[j] - xs[i]
    keep = dx != 0
    if not keep.any():
        return None
    return float(np.median((ys[j] - ys[i])[keep] / dx[keep]))


def fuel_mass(lap_number: int, race_laps: int) -> float:
    burn = STARTING_FUEL_KG / max(1, race_laps)
    return max(0.0, STARTING_FUEL_KG - (lap_number - 1) * burn)


def stint_degradation(stint: Dict[str, Any], race_laps: int) -> Optional[float]:
    """Seconds lost per lap of tyre age for one stint, or None if it is unusable.

    ``stint`` = {'compound': 'MEDIUM', 'laps': [(lap_number, tyre_age, lap_time_s), ...]}
    holding only clean green-flag laps.
    """
    compound = stint['compound']
    laps = [lap for lap in stint['laps'] if lap[1] > WARMUP_LAPS.get(compound, 2)]
    if len(laps) < MIN_STINT_LAPS - 2:
        return None
    times = np.array([t for _n, _age, t in laps], dtype=float)
    if np.median(times) <= 0:
        return None
    ok = np.abs(times - np.median(times)) < 0.04 * np.median(times)  # drop traffic / mistakes
    laps = [lap for lap, good in zip(laps, ok) if good]
    if len(laps) < 4:
        return None
    ages = [age for _n, age, _t in laps]
    corrected = [t - FUEL_EFFECT_S_PER_KG * fuel_mass(n, race_laps) for n, _age, t in laps]
    slope = theil_sen_slope(ages, corrected)
    return None if slope is None else max(0.0, slope)


def pit_loss(in_lap_s: float, out_lap_s: float, green_lap_s: float) -> Optional[float]:
    """Time lost to one pit stop compared with two normal laps (None if implausible)."""
    loss = in_lap_s + out_lap_s - 2.0 * green_lap_s
    return loss if 10.0 <= loss <= 45.0 else None


def summarize_races(races: Iterable[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Measured values for one circuit from several real races.

    Each race: {'name': '2021 British Grand Prix', 'race_laps': 52, 'rain': bool,
                'safety_car': bool, 'stints': [stint, ...], 'pit_losses': [s, ...],
                'stops': [stops per classified finisher, ...]}
    """
    races = list(races)
    if not races:
        return None
    slopes: Dict[str, List[float]] = {}
    stint_lengths: Dict[str, List[int]] = {}
    losses: List[float] = []
    stops: List[int] = []
    for race in races:
        for stint in race.get('stints', []):
            compound = stint['compound']
            stint_lengths.setdefault(compound, []).append(int(stint.get('length') or len(stint['laps'])))
            slope = stint_degradation(stint, race['race_laps'])
            if slope is not None:
                slopes.setdefault(compound, []).append(slope)
        losses += [x for x in race.get('pit_losses', []) if x is not None]
        if not race.get('rain'):
            stops += list(race.get('stops', []))
    deg = {c: round(median(v), 4) for c, v in slopes.items() if len(v) >= 3}
    ratios = [deg[c] / NEUTRAL_DEG[c] for c in NEUTRAL_DEG if c in deg]
    return {
        'races': [race['name'] for race in races],
        'deg_s_per_lap': deg,
        'stints_used': {c: len(v) for c, v in slopes.items()},
        'max_stint_laps': {c: int(np.percentile(v, 90)) for c, v in stint_lengths.items() if len(v) >= 3},
        'abrasion_factor': round(float(np.clip(median(ratios), 0.4, 1.8)), 3) if ratios else None,
        'pit_loss_s': round(median(losses), 1) if len(losses) >= 3 else None,
        'typical_stops': int(round(median(stops))) if stops else None,
        'sc_probability': round(sum(1 for r in races if r.get('safety_car')) / len(races), 2),
        'rain_probability': round(sum(1 for r in races if r.get('rain')) / len(races), 2),
        'race_laps': int(median(r['race_laps'] for r in races)),
    }
