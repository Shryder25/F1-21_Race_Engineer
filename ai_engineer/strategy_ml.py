"""Lightweight predictive models for tyre wear, fuel use and pace (numpy.polyfit).

Everything here is a pure function of lap history, so it can be tested without
the game, a database or an AI model.
"""

from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Worst-tyre wear (%) at which the engineer plans the stop / calls it the last sensible lap.
IDEAL_WEAR_PCT = 60.0
LATEST_WEAR_PCT = 75.0
# Typical F1 lap-time benefit of burning 1 kg of fuel (s/kg); used to isolate tyre fall-off.
FUEL_EFFECT_S_PER_KG = 0.03
MIN_WEAR_POINTS = 3
MIN_R2 = 0.5      # below this the wear data is too noisy to plan a stop from
_MAX_LOOKAHEAD_LAPS = 80


@dataclass
class TyreModel:
    """Polynomial wear curve: wear% = f(tyre age in laps)."""
    coeffs: tuple[float, ...]     # highest power first (numpy.polyfit order)
    degree: int
    samples: int
    r2: float

    def wear_at(self, age: float) -> float:
        return float(np.polyval(self.coeffs, age))

    def wear_rate(self, age: float) -> float:
        """Local wear in % per lap at the given age."""
        return self.wear_at(age + 1) - self.wear_at(age)


def fit_wear_curve(samples: Sequence[tuple[float, float]]) -> TyreModel | None:
    """Fit worst-tyre wear against tyre age.

    A brand-new tyre has zero wear, so (0, 0) anchors the curve. Uses a straight
    line for small data sets and a quadratic once there is enough data to show
    the accelerating wear near the cliff; falls back to the line if the quadratic
    is not monotonically increasing over the observed range.
    """
    points = [(float(a), float(w)) for a, w in samples
              if a is not None and w is not None and a >= 0 and 0 <= w <= 100]
    if len({a for a, _ in points}) < MIN_WEAR_POINTS:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        raw_slope = np.polyfit([p[0] for p in points], [p[1] for p in points], 1)[0]
    if raw_slope <= 0:
        return None  # the data itself says wear is not increasing with age
    points.append((0.0, 0.0))
    ages = np.array([p[0] for p in points])
    wears = np.array([p[1] for p in points])

    def fit(degree: int) -> TyreModel | None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                coeffs = np.polyfit(ages, wears, degree)
            except (np.linalg.LinAlgError, ValueError):
                return None
        predicted = np.polyval(coeffs, ages)
        ss_res = float(np.sum((wears - predicted) ** 2))
        ss_tot = float(np.sum((wears - wears.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return TyreModel(tuple(float(c) for c in coeffs), degree, len(points) - 1, r2)

    linear = fit(1)
    if linear is None or linear.coeffs[0] <= 0 or linear.r2 < MIN_R2:
        return None
    if len(points) >= 8:
        quad = fit(2)
        if quad is not None and quad.coeffs[0] > 0:
            vertex = -quad.coeffs[1] / (2 * quad.coeffs[0])
            if vertex <= 0 and quad.r2 >= MIN_R2:  # increasing across the observed range
                return quad
    return linear


def laps_until_wear(model: TyreModel, current_age: float, threshold: float) -> float | None:
    """Laps from now until the modelled worst-tyre wear reaches ``threshold`` %."""
    if model.wear_at(current_age) >= threshold:
        return 0.0
    age = current_age
    while age - current_age < _MAX_LOOKAHEAD_LAPS:
        age += 0.25
        if model.wear_at(age) >= threshold:
            return round(age - current_age, 2)
    return None


@dataclass
class PitPrediction:
    wear_now_pct: float
    wear_per_lap_pct: float
    ideal_lap: int | None
    latest_lap: int | None
    laps_to_ideal: float | None
    laps_to_latest: float | None
    finish_without_stop: bool
    samples: int
    r2: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def predict_pit_window(model: TyreModel | None, current_age: float, current_lap: int,
                       total_laps: int = 0, ideal_pct: float = IDEAL_WEAR_PCT,
                       latest_pct: float = LATEST_WEAR_PCT) -> PitPrediction | None:
    """Turn a wear curve into a pit window expressed in race lap numbers."""
    if model is None:
        return None
    to_ideal = laps_until_wear(model, current_age, ideal_pct)
    to_latest = laps_until_wear(model, current_age, latest_pct)

    def lap(delta):
        return None if delta is None else current_lap + int(delta)

    finish_ok = False
    if total_laps > current_lap:
        wear_at_flag = model.wear_at(current_age + (total_laps - current_lap))
        finish_ok = wear_at_flag <= latest_pct
    return PitPrediction(
        wear_now_pct=round(max(0.0, model.wear_at(current_age)), 1),
        wear_per_lap_pct=round(model.wear_rate(current_age), 2),
        ideal_lap=lap(to_ideal), latest_lap=lap(to_latest),
        laps_to_ideal=to_ideal, laps_to_latest=to_latest,
        finish_without_stop=finish_ok, samples=model.samples, r2=round(model.r2, 3))


def fuel_per_lap(samples: Sequence[float]) -> float | None:
    """Median fuel burn per lap (kg), ignoring implausible values (pit laps, resets)."""
    clean = [s for s in samples if 0.2 < s < 6.0]
    return round(statistics.median(clean), 3) if clean else None


def pace_degradation(laps: Sequence[tuple[float, float, float | None]],
                     fuel_k: float = FUEL_EFFECT_S_PER_KG) -> float | None:
    """Tyre-related lap time loss in seconds per lap.

    ``laps`` holds (stint lap index, lap time ms, fuel kg at lap start). Lap times
    are fuel-corrected to an empty tank before fitting a line; positive means the
    car is getting slower. Needs at least three laps.
    """
    pts = []
    for index, time_ms, fuel in laps:
        if time_ms and time_ms > 0:
            corrected = time_ms / 1000.0 - (fuel_k * fuel if fuel is not None else 0.0)
            pts.append((float(index), corrected))
    if len(pts) < 3 or len({p[0] for p in pts}) < 3:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        slope = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]
    return round(float(slope), 3)
