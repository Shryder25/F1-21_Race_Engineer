"""Pit window from the fitted tyre-wear model, as lap ranges rather than single laps.

The wear model itself (per compound and track, fitted from recorded stints, with ranges) is
in fitted_models.py. This module turns a wear forecast for the current stint into the pit
window the engineer quotes: the lap the worst tyre reaches the target wear and the last
sensible lap, each as (soonest, expected, latest).

Everything here is a pure function of numbers passed in, so it can be tested without the
game, a database or an AI model.
"""

from __future__ import annotations

from dataclasses import dataclass

from fitted_models import DEFAULT_LEVEL, Range, StintWear

# Worst-tyre wear (%) at which the engineer plans the stop / calls it the last sensible lap.
IDEAL_WEAR_PCT = 60.0
LATEST_WEAR_PCT = 75.0


@dataclass
class PitPrediction:
    wear_now_pct: float
    wear_per_lap_pct: float              # expected, this stint
    ideal_lap: int | None                # expected lap numbers (None = after the flag)
    latest_lap: int | None
    laps_to_ideal: float | None
    laps_to_latest: float | None
    finish_without_stop: bool
    samples: int                         # laps behind the model (fit + this stint)
    wear_per_lap_range: list             # [low, mid, high] % per lap
    ideal_lap_range: list                # [soonest, expected, latest] race lap
    latest_lap_range: list
    finish_without_stop_prob: float | None
    basis: str                           # track / pooled / prior (+ this stint's laps)
    stint_laps: int
    level: float = DEFAULT_LEVEL

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _lap(current_lap: int, delta: float | None, total_laps: int) -> int | None:
    if delta is None:
        return None
    target = current_lap + int(delta)
    # A stop after the chequered flag is no stop at all.
    return None if total_laps > 0 and target > total_laps else target


def _laps(current_lap: int, r: Range, total_laps: int) -> list:
    return [_lap(current_lap, r.low, total_laps), _lap(current_lap, r.mid, total_laps),
            _lap(current_lap, r.high, total_laps)]


def predict_pit_window(stint: StintWear | None, tyre_age: float, wear_now: float, current_lap: int,
                       total_laps: int = 0, ideal_pct: float = IDEAL_WEAR_PCT,
                       latest_pct: float = LATEST_WEAR_PCT,
                       level: float = DEFAULT_LEVEL) -> PitPrediction | None:
    """Turn a stint wear forecast into a pit window expressed in race lap numbers."""
    if stint is None:
        return None
    to_ideal = stint.laps_until(ideal_pct, tyre_age, wear_now, level)
    to_latest = stint.laps_until(latest_pct, tyre_age, wear_now, level)
    rate = stint.rate(tyre_age, level)

    prob = None
    if total_laps > current_lap:
        prob = stint.prob_below(latest_pct, tyre_age, wear_now, total_laps - current_lap)
        finish_ok = prob >= 0.5
    else:
        finish_ok = total_laps > 0          # on the last lap (or past the flag): nothing to plan
        prob = 1.0 if finish_ok else None
    return PitPrediction(
        wear_now_pct=round(max(0.0, wear_now), 1),
        wear_per_lap_pct=round(rate.mid, 2),
        ideal_lap=_lap(current_lap, to_ideal.mid, total_laps),
        latest_lap=_lap(current_lap, to_latest.mid, total_laps),
        laps_to_ideal=to_ideal.mid, laps_to_latest=to_latest.mid,
        finish_without_stop=finish_ok,
        samples=stint.model.n_laps + stint.stint_laps,
        wear_per_lap_range=rate.as_list(2),
        ideal_lap_range=_laps(current_lap, to_ideal, total_laps),
        latest_lap_range=_laps(current_lap, to_latest, total_laps),
        finish_without_stop_prob=None if prob is None else round(prob, 3),
        basis=stint.model.basis, stint_laps=stint.stint_laps, level=level)
