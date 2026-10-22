"""Track wetness: how wet the surface is, which tyre is quickest on it, and where it is heading.

F1 2021 does not send how wet the track is, only the weather now and a forecast. A real pit
wall reads the surface two ways, and so does this module:

* **Physics prior.** Rain soaks the track towards a level set by how hard it rains; once it
  stops, the track dries at a rate set by track temperature and cloud. This gives a wetness
  (0 = bone dry, 1 = standing water) between observations and a forward trajectory.
* **Lap times.** The field is the best sensor. When cars on slicks and cars on inters (or inters
  and wets) are lapping at the same time, the gap between them pins down the wetness: slicks a
  second quicker means the track is past the crossover. With one tyre type out, laps against
  the session's dry reference lap do the same, more loosely. Each observation updates the
  estimate on a grid (a proper Bayesian update, so a flat or ambiguous reading moves it little).

Lap time cost of each tyre family against wetness (seconds per lap, for a 90 s dry lap; scaled
to the real lap length) is the table below. The crossovers sit where real F1 sees them: slicks
lose to inters at about 20% wet (a damp track with a drying line), inters lose to full wets at
about 70% (standing water). These are estimates; recordings of real wet races would fit them.

    wetness  0.0  0.1  0.2  0.3  0.5  0.7  0.85  1.0
    slick    0.0  1.8  4.5  9.0  18   28   34    44
    inter    4.0  4.0  4.5  5.5  8.5  13   18    24
    wet      8.0  8.0  8.5  9.0  11   13   16    19

Inters on a dry track also overheat and wear much faster (race_sim applies that).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

import fitted_models as fm
import race_strategy as rs

FAMILIES = ('slick', 'inter', 'wet')
FAMILY_OF = {'SOFT': 'slick', 'MEDIUM': 'slick', 'HARD': 'slick', 'INTERMEDIATE': 'inter', 'WET': 'wet'}

WETNESS_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 1.0])
PENALTY_S = {  # per lap, for a REFERENCE_LAP_S dry lap
    'slick': np.array([0.0, 1.8, 4.5, 9.0, 18.0, 28.0, 34.0, 44.0]),
    'inter': np.array([4.0, 4.0, 4.5, 5.5, 8.5, 13.0, 18.0, 24.0]),
    'wet': np.array([8.0, 8.0, 8.5, 9.0, 11.0, 13.0, 16.0, 19.0]),
}
REFERENCE_LAP_S = 90.0

# ── Rain and drying physics (estimates) ──
WET_TARGET = {3: 0.5, 4: 0.8, 5: 0.95}  # wetness steady rain soaks the track to, by weather code
RAIN_TAU_MIN = 3.0  # minutes for rain to close most of the gap to that level
DRY_RATE_PER_MIN = 0.03  # wetness lost per minute once it stops, at 30 C track temp
DRY_TEMP_COEF = 0.03  # +3% drying rate per degree of track temperature above 30 C
DRY_TEMP_FACTOR_RANGE = (0.4, 2.0)
SKY_DRY_FACTOR = {0: 1.2, 1: 1.0, 2: 0.8}  # clear sky dries faster than overcast
PROCESS_SD_PER_SQRT_MIN = 0.02  # how fast the estimate's uncertainty grows between readings
START_SD = {True: 0.15, False: 0.03}  # first look: raining (unsure how wet) / dry (almost sure)

# ── Lap-time observations ──
OBS_SD_TWO_FAMILIES_S = 1.5  # two tyre types running together: the strong reading
OBS_SD_OWN_DRY_S = 1.8  # one tyre type, each car against its own dry laps
OBS_SD_DRY_REFERENCE_S = 2.5  # one tyre type against the field's dry lap: looser (traffic, car pace)
MIN_CARS_PER_FAMILY = 2
DRY_REFERENCE_MAX_WETNESS = 0.05  # slick laps count towards the dry reference below this
DRY_REFERENCE_LAPS = 12
OBS_WINDOW_LAPS = 1.3  # laps within this many lap-lengths are compared

GRID = np.linspace(0.0, 1.0, 101)


def penalty(family: str, wetness, lap_s: float = REFERENCE_LAP_S):
    """Lap time cost of a tyre family at this wetness (scalar or array), seconds."""
    return np.interp(wetness, WETNESS_GRID, PENALTY_S[family]) * (lap_s / REFERENCE_LAP_S)


def best_family(wetness: float) -> str:
    return min(FAMILIES, key=lambda f: float(penalty(f, wetness)))


def crossover(a: str, b: str) -> float | None:
    """Wetness at which family b becomes quicker than a (first sign change on the grid)."""
    d = penalty(a, GRID) - penalty(b, GRID)
    idx = np.where(np.diff(np.sign(d)) != 0)[0]
    return float(GRID[idx[0] + 1]) if len(idx) else None


def family(compound: str | None) -> str | None:
    """'Soft' / 'C3' / 'Inter' / 'Wet' -> slick / inter / wet."""
    return FAMILY_OF.get(rs.normalise_compound(compound) or '')


def dry_rate_per_min(track_temp: float, weather: int) -> float:
    lo, hi = DRY_TEMP_FACTOR_RANGE
    temp = min(max(1.0 + DRY_TEMP_COEF * ((track_temp or 30.0) - 30.0), lo), hi)
    return DRY_RATE_PER_MIN * temp * SKY_DRY_FACTOR.get(weather, 1.0)


def step(wetness: float, minutes: float, weather: int, track_temp: float) -> float:
    """Wetness after ``minutes`` of this weather."""
    if minutes <= 0:
        return wetness
    if weather >= 3:
        target = WET_TARGET.get(weather, 0.95)
        if wetness < target:
            return wetness + (target - wetness) * (1.0 - math.exp(-minutes / RAIN_TAU_MIN))
        # Lighter rain on a soaked track: it drains slowly back towards the lighter level.
        return max(target, wetness - 0.5 * dry_rate_per_min(track_temp, 2) * minutes)
    return max(0.0, wetness - dry_rate_per_min(track_temp, weather) * minutes)


def rain_probability(sample) -> float:
    """Chance that it is raining at a forecast sample."""
    pct = getattr(sample, 'rain_percentage', 0) / 100.0
    return max(pct, 0.75) if getattr(sample, 'weather', 0) >= 3 else pct


@dataclass
class Reading:
    wetness: float
    sd: float
    basis: str


class TrackWetness:
    """Live wetness estimate: physics between readings, field lap times as the readings."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.wetness: float | None = None
        self.sd = 0.05
        self.weather = 0
        self.track_temp = 30.0
        self.forecast: tuple = ()
        self.last_t: float | None = None
        self.records: deque = deque(maxlen=200)  # (time, car, family, lap seconds)
        self.dry_laps: deque = deque(maxlen=DRY_REFERENCE_LAPS)
        self.lap_s = REFERENCE_LAP_S
        self.new_records = 0
        self.basis = 'weather'
        self.last_reading: Reading | None = None
        self.history: deque = deque(maxlen=60)  # (time, wetness), for the trend
        self._cars: dict[int, tuple] = {}
        self.car_dry: dict[int, deque] = {}  # each car's own dry slick laps (fuel-corrected)

    # ── physics ──
    def update_weather(self, weather: int, track_temp: float, now: float, forecast: tuple = ()):
        self.track_temp = float(track_temp) if track_temp is not None else self.track_temp
        self.forecast = tuple(forecast or ())
        raining = weather >= 3
        if self.wetness is None:
            self.wetness = WET_TARGET.get(weather, 0.95) if raining else 0.0
            self.sd = START_SD[raining]
        elif self.last_t is not None and now > self.last_t:
            minutes = (now - self.last_t) / 60.0
            # Integrate under the weather that was in force since the last update.
            self.wetness = step(self.wetness, minutes, self.weather, self.track_temp)
            self.sd = min(0.4, math.sqrt(self.sd**2 + PROCESS_SD_PER_SQRT_MIN**2 * minutes))
        self.weather = weather
        self.last_t = now
        if not self.history or now - self.history[-1][0] >= 10.0:
            self.history.append((now, self.wetness))

    # ── lap-time readings ──
    def observe_field(self, laps, statuses, now: float, neutralised: bool = False):
        """Feed every car's completed laps; clean ones (no stop, not lap 1, green) are readings."""
        if not laps:
            return
        for index, lap in enumerate(laps):
            if lap is None or getattr(lap, 'result_status', 2) != 2 or lap.car_position == 0:
                continue
            seen = self._cars.get(index)
            state = (lap.current_lap_num, lap.num_pit_stops)
            in_pits = bool(lap.pit_status)
            if seen is None:
                self._cars[index] = (*state, True)
                continue
            prev_lap, prev_stops, dirty = seen
            if lap.current_lap_num > prev_lap:
                status = statuses[index] if statuses and index < len(statuses) else None
                fam = family(getattr(status, 'compound_name', None)) if status else None
                clean = (
                    not dirty
                    and lap.num_pit_stops == prev_stops
                    and prev_lap > 1
                    and lap.last_lap_time_ms
                    and not neutralised
                )
                if clean and fam:
                    fuel = getattr(status, 'fuel_in_tank', 0.0) or 0.0
                    # Fuel-corrected, so laps late in the race compare fairly with early dry laps.
                    self.add_lap(
                        index, fam, lap.last_lap_time_ms / 1000.0 - fm.FUEL_EFFECT_S_PER_KG * fuel, now
                    )
                dirty = in_pits or neutralised or lap.num_pit_stops != prev_stops
            else:
                dirty = dirty or in_pits or neutralised
            self._cars[index] = (*state, dirty)
        self.assimilate(now)

    def add_lap(self, car: int, fam: str, lap_s: float, now: float):
        self.records.append((now, car, fam, lap_s))
        self.new_records += 1
        if fam == 'slick' and self.wetness is not None and self.wetness <= DRY_REFERENCE_MAX_WETNESS:
            self.dry_laps.append(lap_s)
            self.car_dry.setdefault(car, deque(maxlen=5)).append(lap_s)
        if fam == 'slick' and self.dry_laps:
            self.lap_s = float(np.median(self.dry_laps))

    def _recent(self, now: float, with_cars: bool = False) -> dict[str, list]:
        window = OBS_WINDOW_LAPS * self.lap_s
        latest: dict[int, tuple] = {}
        for t, car, fam, lap_s in self.records:
            if now - t <= window:
                latest[car] = (fam, lap_s)  # each car once: its latest lap
        out: dict[str, list] = {f: [] for f in FAMILIES}
        for car, (fam, lap_s) in latest.items():
            out[fam].append((car, lap_s) if with_cars else lap_s)
        return out

    def reading(self, now: float) -> tuple | None:
        """(observed difference, model difference fn, sd, basis), or None without a usable reading."""
        recent = self._recent(now)
        med = {f: float(np.median(v)) for f, v in recent.items() if len(v) >= MIN_CARS_PER_FAMILY}
        lap = self.lap_s
        for a, b in (('slick', 'inter'), ('inter', 'wet'), ('slick', 'wet')):
            if a in med and b in med:
                return (
                    med[a] - med[b],
                    lambda w, a=a, b=b: penalty(a, w, lap) - penalty(b, w, lap),
                    OBS_SD_TWO_FAMILIES_S,
                    f'{a} vs {b} lap times',
                )
        by_car = self._recent(now, with_cars=True)
        for f in ('inter', 'wet', 'slick'):
            # Each car against its own dry laps: removes the pace difference between cars.
            diffs = [
                lap_s - float(np.median(self.car_dry[car]))
                for car, lap_s in by_car[f]
                if self.car_dry.get(car)
            ]
            if len(diffs) >= 3:
                return (
                    float(np.median(diffs)),
                    lambda w, f=f: penalty(f, w, lap),
                    OBS_SD_OWN_DRY_S,
                    f'{f} lap times vs each car\'s dry laps',
                )
        if self.dry_laps and len(self.dry_laps) >= 3:
            ref = float(np.median(self.dry_laps))
            for f in ('inter', 'wet', 'slick'):
                if f in med and len(recent[f]) >= 3:
                    return (
                        med[f] - ref,
                        lambda w, f=f: penalty(f, w, lap),
                        OBS_SD_DRY_REFERENCE_S,
                        f'{f} lap times vs the dry lap',
                    )
        return None

    def assimilate(self, now: float):
        """Fold in the latest lap-time reading once enough new laps are in (about one per car)."""
        if self.wetness is None or self.new_records < max(4, len({r[1] for r in self.records}) // 2):
            return
        obs = self.reading(now)
        self.new_records = 0
        if obs is None:
            return
        raining = self.weather >= 3
        if not raining and self.wetness <= 0.0:
            return  # dry and not raining: a slow lap is fuel or traffic, not water
        d, model, sd, basis = obs
        prior = np.exp(-0.5 * ((GRID - self.wetness) / max(self.sd, 1e-3)) ** 2)
        like = np.exp(-0.5 * ((d - model(GRID)) / sd) ** 2)
        post = prior * like
        if post.sum() <= 1e-12:  # reading far outside the model: trust it, but loosely
            post = like
        post = post / post.sum()
        if not raining:
            # Water only arrives with rain: without it the reading can show the track drier than
            # the physics says (a dry line forming), never wetter.
            post = np.where(GRID <= self.wetness + 1e-9, post, 0.0)
            if post.sum() <= 1e-12:
                return
            post = post / post.sum()
        mean = float((post * GRID).sum())
        self.sd = max(0.02, float(math.sqrt((post * (GRID - mean) ** 2).sum())))
        self.wetness = mean
        self.basis = basis
        self.last_reading = Reading(mean, self.sd, basis)

    # ── outputs ──
    def trajectory(self, laps: int, lap_s: float | None = None) -> list[float]:
        """Expected wetness at the start of each of the next ``laps`` laps (forecast rain >= 50%)."""
        if self.wetness is None:
            return [0.0] * laps
        lap_s = lap_s or self.lap_s
        w, out = self.wetness, []
        for k in range(laps):
            out.append(w)
            weather = self.weather_at(k * lap_s / 60.0)
            w = step(w, lap_s / 60.0, weather, self.track_temp)
        return out

    def weather_at(self, minutes_ahead: float) -> int:
        """Most likely weather at a time ahead, from the forecast (rain if its chance is >= 50%)."""
        sample = None
        for s in self.forecast:
            if s.offset_min <= minutes_ahead + 1e-9:
                sample = s
        if sample is None or not self.forecast:
            return self.weather
        if rain_probability(sample) >= 0.5:
            return max(sample.weather, 3)
        return min(sample.weather, 2)

    def trend(self) -> str:
        if len(self.history) < 2 or self.wetness is None:
            return 'steady'
        t0, w0 = self.history[0] if len(self.history) < 6 else self.history[-6]
        dw = self.wetness - w0
        return 'drying' if dw < -0.02 else ('wetting' if dw > 0.02 else 'steady')

    def laps_until(self, fam: str, current: str, laps: int = 30) -> int | None:
        """Laps until ``fam`` is quicker than ``current`` (0 = already), on the expected trajectory."""
        for k, w in enumerate(self.trajectory(laps)):
            if penalty(fam, w) < penalty(current, w):
                return k
        return None

    def summary(self, current_family: str | None = None) -> dict | None:
        if self.wetness is None:
            return None
        best = best_family(self.wetness)
        out = {
            'wetness': round(self.wetness, 2),
            'sd': round(self.sd, 2),
            'trend': self.trend(),
            'best_tyre': best,
            'basis': self.basis,
            'dry_lap_s': round(self.lap_s, 2),
            'forecast_rain': [
                {'in_min': s.offset_min, 'weather': s.weather, 'rain_pct': s.rain_percentage}
                for s in self.forecast[:6]
            ],
            'switch_in_laps': None,
            'switch_to': None,
        }
        if current_family:
            for fam in FAMILIES:
                if fam == current_family:
                    continue
                k = self.laps_until(fam, current_family)
                if k is not None and (out['switch_in_laps'] is None or k < out['switch_in_laps']):
                    out['switch_in_laps'], out['switch_to'] = k, fam
        return out

    def radio(self, current_family: str | None) -> str | None:
        """'Track is drying, slicks in about three laps.'"""
        s = self.summary(current_family)
        if s is None or (s['wetness'] < 0.03 and s['trend'] != 'wetting' and not s['switch_to']):
            return None
        words = {'slick': 'slicks', 'inter': 'inters', 'wet': 'full wets'}
        pct = int(round(s['wetness'] * 100))
        line = (
            f"Track about {pct}% wet, {s['trend']}." if s['trend'] != 'steady' else f'Track about {pct}% wet.'
        )
        k, to = s['switch_in_laps'], s['switch_to']
        if to is not None:
            line += (
                f' {words[to].capitalize()} are the quicker tyre now.'
                if k == 0
                else f' {words[to].capitalize()} quicker in about {k} lap{"s" if k != 1 else ""}.'
            )
        return line
