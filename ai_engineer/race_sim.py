"""Monte Carlo race simulator: the whole field, lap by lap, many times over.

race_strategy.plan_race finds the fastest stop plan for the player in a race with no one
else on track and nothing going wrong. Real races are decided by what that leaves out:
a Safety Car that hands someone a cheap stop, a slow pit stop, rejoining behind a train,
a track where a car one second a lap faster still cannot get past. This module runs the
rest of the race a few hundred times with those things drawn at random, for each
candidate plan, and counts what happens:

    "Two-stop comes out best 60% of the time. Expected finish P4, P3 to P6."

Per run and per lap, every car's lap time is

    base pace + driver form + compound offset + tyre fall-off * tyre age
              + cliff past the tyre's life + fuel load + lap-to-lap noise
              + dirty air / being held up + pit loss on the lap it stops

Where the inputs come from:

* Tyre fall-off (s/lap) and its spread: the fitted degradation models (fitted_models.py,
  per compound and track, with confidence ranges). Each stint draws its own slope.
* Tyre life: the physics model's life per compound, scaled by a draw from the fitted wear
  model's range (a set that wears faster than average lives proportionally shorter).
* Fuel burn per lap: the fitted fuel model.
* Safety Car / VSC / rain chances, overtaking difficulty: track_profiles.py (real-race
  figures, overridden by FastF1 calibration where it exists).
* Pit loss: race_strategy's per-track game figure as the mean, plus a spread and the odd
  slow stop.
* Rivals: their own stop plans from race_strategy.plan_race, jittered per run, or from a
  rival-stop prediction (any object with stop_lap_mean / stop_lap_sd / will_stop /
  next_compound_probs) when one is supplied.

Every random number is drawn once per run and shared by all candidate plans (common random
numbers), so plans are compared under the same Safety Cars, the same slow stops and the
same rival form; the difference between them is the plan, not the dice.

Simplifications, all marked where they live: no retirements, Safety Car bunching is
instant. Rain follows the game's forecast (or, without one, the track profile's
rain chance and an average shower length); the track soaks and dries per track_wetness.py,
each tyre type pays its lap-time cost for the wetness, and rivals change tyre type at the
crossover. The player's plan decides our own tyre changes, except after a shower the plan
could not foresee, blue flags are implicit (lapped cars never interact
with the leaders), VSC chance is a fixed share of the SC chance. The constants are
educated estimates; recordings of real F1 2021 races (tools/fit_models.py,
tools/backtest.py) are what will sharpen the tyre/fuel inputs, and the SC/overtaking
constants are the next things to fit from them.

Everything is arithmetic on numbers passed in, vectorised over runs with numpy, so it is
testable without the game and fast enough to re-run once a lap.
"""

from __future__ import annotations

import math
import threading
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

import fitted_models as fm
import race_strategy as rs
from physics_models import _compound_params
from track_profiles import get_profile
import track_wetness as tw

COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET')
_CI = {c: i for i, c in enumerate(COMPOUNDS)}
WET_IDX = (_CI['INTERMEDIATE'], _CI['WET'])
DEFAULT_ALLOCATION = {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2', 'INTERMEDIATE': 'INTER', 'WET': 'WET'}
_C_ORDER = ('C5', 'C4', 'C3', 'C2', 'C1', 'C0')

DEFAULT_RUNS = 300
NEVER = 10_000                       # stop lap meaning "no stop"

# ── Safety Car / VSC ──
SC_DURATION_LAPS = (3, 5)            # inclusive range, drawn per Safety Car
VSC_DURATION_LAPS = (1, 3)
VSC_SHARE_OF_SC = 0.7                # estimate: no per-track VSC data yet, so VSC chance = 0.7 x SC chance
LAP_ONE_HAZARD_MULT = 4.0            # first-lap incidents: lap 1 is this many times likelier than any other
SC_LAP_FACTOR = 1.40                 # lap time behind the Safety Car vs a normal lap
VSC_LAP_FACTOR = 1.35
SC_PIT_FACTOR = 0.55                 # share of the green-flag pit loss a stop costs under SC
VSC_PIT_FACTOR = 0.65
SC_GAP_S = 0.8                       # car-to-car gap once the field has bunched behind the SC
SC_PIT_WINDOW_LAPS = 8               # take a planned stop early under SC/VSC if it was this close

# ── Pit stops ──
PIT_LOSS_SD_S = 0.6                  # lap-to-lap spread of a normal stop (entry, stationary, exit)
SLOW_STOP_PROB = 0.06                # chance a stop goes wrong
SLOW_STOP_MEAN_S = 3.0               # mean extra time when it does (exponential)
RIVAL_STOP_JITTER_LAPS = 1.5         # sd of a rival's stop lap around its modelled plan
RIVAL_ALT_PLAN_PROB = 0.25           # chance a rival runs its second-best stop count

# ── Traffic and overtaking ──
DIRTY_AIR_GAP_S = 1.0                # following closer than this costs time
DIRTY_AIR_S = 0.25                   # cost per lap right on a gearbox (fades linearly to 0 at 1 s)
MIN_GAP_S = 0.3                      # a car held up crosses the line this far behind
# Pace advantage (s/lap) at which a pass is a coin flip, by track_profiles overtaking rating
# (1 easy .. 5 Monaco). PASS_SCALE_S sets how quickly the chance rises around that point.
PASS_THRESHOLD_S = {1: 0.3, 2: 0.6, 3: 0.9, 4: 1.3, 5: 2.2}
PASS_SCALE_S = 0.25
PASS_COST_S = 0.3                    # time a pass costs the passer
DEFEND_COST_S = 0.2                  # and the car being passed

# ── Pace ──
DRIVER_FORM_SD_S = 0.15              # per-run spread of a car's pace when measured from clean laps
ASSUMED_FORM_SD_S = 0.40             # when the pace is a guess from track position
POSITION_PACE_STEP_S = 0.08          # guessed pace step per position when a car has no clean laps
DEFAULT_LAP_S = 90.0
LAP_NOISE_RANGE_S = (0.15, 0.6)

# ── Rain, wetness and the tyre crossover (track_wetness.py has the lap-time table) ──
RAIN_MEAN_DURATION_MIN = 20.0        # no forecast: a shower that is falling now lasts this long on average
HEAVY_SHOWER_SHARE = 0.3             # no forecast: share of new showers that are heavy rain
DRY_RATE_RUN_SD = 0.3                # per-run spread (log) of how fast the track dries
WETNESS_START_SD_RAINING = 0.15      # no live estimate: how unsure the starting wetness is in the rain
REACT_GAIN_S = 1.2                   # a car changes tyre type once the other type is this much quicker
REACT_GAIN_SD_S = 0.5                # (per car and run: some gamble early, some wait)
REACT_GAIN_RANGE_S = (0.3, 3.0)
WRONG_TYRE_S = 1.0                   # a planned stop onto a tyre this much slower than the best is changed
WET_TYRE_DRY_DEG_MULT = 3.0          # inters below 15% wet, full wets below 35%, overheat and wear this much faster
FAM_IDX = np.array([0, 0, 0, 1, 2])  # COMPOUNDS -> tyre family (slick, inter, wet)


# ──────────────────────────────────────────────
# Inputs
# ──────────────────────────────────────────────

@dataclass
class SimCar:
    index: int
    position: int
    gap_s: float                      # race time behind the leader (at the last line crossing)
    compound: str                     # SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    tyre_age: int
    pace_s: float | None = None       # fuel- and tyre-corrected base lap time; None = unknown
    pace_laps: int = 0                # clean laps behind pace_s
    fuel_kg: float | None = None
    compounds_used: tuple = ()
    name: str | None = None
    is_player: bool = False
    prediction: object | None = None  # optional rival-stop prediction (duck-typed, see module doc)


@dataclass
class RaceState:
    track_id: int | None
    total_laps: int
    current_lap: int                  # the lap in progress (it is simulated as the first lap)
    cars: list
    safety_car: int = 0               # 0 none, 1 SC, 2 VSC right now
    raining: bool = False
    track_length: int | None = None
    allocation: dict = field(default_factory=lambda: dict(DEFAULT_ALLOCATION))
    # Conditions (track_wetness.py): live wetness estimate and its sd, weather code now, track
    # temperature, the game's forecast (WeatherSample-like) and the dry lap time. None/() = unknown.
    wetness: float | None = None
    wetness_sd: float = 0.05
    weather: int | None = None
    track_temp: float = 30.0
    forecast: tuple = ()
    lap_s: float | None = None

    @property
    def laps_to_go(self) -> int:
        return max(0, self.total_laps - self.current_lap + 1)

    @property
    def player(self) -> SimCar | None:
        return next((c for c in self.cars if c.is_player), None)


@dataclass
class Candidate:
    """One plan for the player: compounds (current set first) and the laps to box at the end of."""
    label: str
    stops: int
    compounds: list
    pit_laps: list

    @classmethod
    def from_option(cls, option: rs.StrategyOption) -> 'Candidate':
        return cls(plan_label(option.stops, option.compounds), option.stops,
                   list(option.compounds), list(option.pit_laps))


def plan_label(stops: int, compounds: Sequence[str]) -> str:
    words = {0: 'No-stop', 1: 'One-stop', 2: 'Two-stop', 3: 'Three-stop'}
    sets = '-'.join(c[0] for c in compounds)
    return f"{words.get(stops, f'{stops}-stop')} ({sets})"


def compound_allocation(pairs: Iterable[tuple[str | None, str | None]]) -> dict:
    """Map SOFT/MEDIUM/HARD to this weekend's C-compounds from (visual, actual) pairs seen on cars.

    One known pair fixes the rest: the three dry sets are consecutive C-compounds.
    """
    alloc = dict(DEFAULT_ALLOCATION)
    for visual, actual in pairs:
        visual, key = rs.normalise_compound(visual), fm.compound_key(actual)
        if visual in ('SOFT', 'MEDIUM', 'HARD') and key in _C_ORDER:
            base = _C_ORDER.index(key) - ('SOFT', 'MEDIUM', 'HARD').index(visual)
            for i, name in enumerate(('SOFT', 'MEDIUM', 'HARD')):
                alloc[name] = _C_ORDER[min(max(base + i, 0), len(_C_ORDER) - 1)]
            break
    return alloc


def candidates_from_plan(plan: rs.StrategyPlan | None) -> list[Candidate]:
    if plan is None:
        return []
    options = sorted([plan.best, *plan.alternatives], key=lambda o: o.stops)
    return [Candidate.from_option(o) for o in options]


def stay_out(compound: str | None) -> Candidate:
    """Run to the flag on the current set. Used when the planner has nothing left to plan (the
    final lap), so the odds still follow the race instead of freezing on an older run."""
    current = rs.normalise_compound(compound) or 'MEDIUM'
    return Candidate(plan_label(0, [current]), 0, [current], [])


# ──────────────────────────────────────────────
# Outputs
# ──────────────────────────────────────────────

@dataclass
class StrategyOutcome:
    label: str
    stops: int
    compounds: list
    pit_laps: list
    p_best: float                     # share of runs where this plan gave the best result of all plans
    p_win: float
    p_podium: float
    p_points: float
    mean_position: float
    position_range: list              # [10th pct, median, 90th pct] finishing position
    position_probs: list              # P(finish P1), P(P2), ...
    time_vs_best_s: list              # race time vs the best plan in the same run, [p10, median, p90]
    is_reference: bool = False        # the plan the others are measured against (the current Plan A)
    # Against the reference plan in the same runs (None without a reference): median and
    # [p10, p90] seconds gained, mean places gained, and the share of runs where this plan
    # finishes ahead of it or in the same place quicker.
    vs_reference: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class SimResult:
    runs: int
    lap: int
    laps_to_go: int
    outcomes: list                    # StrategyOutcome, best p_best first
    field: list                       # per car, under the recommended plan
    p_safety_car: float               # share of runs with at least one SC in the rest of the race
    p_vsc: float
    p_rain: float
    pit_loss_s: list                  # [p10, mean, p90] of a green-flag stop
    overtaking: int
    inputs: dict
    radio: str = ''

    @property
    def recommended(self) -> StrategyOutcome | None:
        return self.outcomes[0] if self.outcomes else None

    def as_dict(self) -> dict:
        out = asdict(self)
        out['recommended'] = self.recommended.label if self.recommended else None
        return out

    def summary(self) -> dict:
        """Compact form for the HUD and the AI prompt."""
        return {
            'runs': self.runs, 'lap': self.lap, 'radio': self.radio,
            'recommended': self.recommended.label if self.recommended else None,
            'options': [{'label': o.label, 'stops': o.stops, 'pit_laps': o.pit_laps,
                         'p_best': o.p_best, 'p_win': o.p_win, 'p_podium': o.p_podium,
                         'p_points': o.p_points, 'mean_position': o.mean_position,
                         'position_range': o.position_range, 'is_reference': o.is_reference,
                         'vs_reference': o.vs_reference} for o in self.outcomes],
            'p_safety_car': self.p_safety_car, 'p_vsc': self.p_vsc, 'p_rain': self.p_rain,
            'pit_loss_s': self.pit_loss_s, 'overtaking': self.overtaking,
            'inputs': self.inputs,
        }


# ──────────────────────────────────────────────
# The simulator
# ──────────────────────────────────────────────

def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


class RaceSimulator:
    """Draws every random input once (``runs`` races), then plays each candidate plan through them."""

    def __init__(self, state: RaceState, models: fm.ModelStore | None = None, runs: int = DEFAULT_RUNS,
                 seed: int | None = None, profile: dict | None = None, pit_loss_mean: float | None = None,
                 player_deg_scale: float = 1.0, player_life_scale: float = 1.0,
                 player_deg: fm.Range | None = None, rain: bool = True):
        if state.player is None:
            raise ValueError('RaceState needs a player car')
        self.state = state
        self.models = models if models is not None else fm.ModelStore()
        self.runs = int(runs)
        self.rng = np.random.default_rng(seed)
        self.profile = profile if profile is not None else (
            get_profile(state.track_id) if state.track_id is not None and state.track_id >= 0 else None) or {}
        self.pit_loss_mean = pit_loss_mean if pit_loss_mean is not None else rs.pit_loss_s(state.track_id)
        self.overtaking = int(self.profile.get('overtaking', 3))
        self.cars = sorted(state.cars, key=lambda c: (c.gap_s, c.position))
        self.n = len(self.cars)
        self.player_slot = next(i for i, c in enumerate(self.cars) if c.is_player)
        self.laps = list(range(state.current_lap, state.total_laps + 1))
        self.max_stops = rs.MAX_STOPS
        self._tyre_inputs(player_deg_scale, player_life_scale, player_deg)
        self._draw(rain)

    # ── inputs ──
    def _tyre_inputs(self, deg_scale, life_scale, player_deg):
        st = self.state
        z = fm.z_score()
        n_c = len(COMPOUNDS)
        self.offset = np.array([rs.COMPOUND_PACE_OFFSET_S.get(c, 0.35) if c in rs.DRY_COMPOUNDS else 0.0
                                for c in COMPOUNDS])
        self.slope_mu = np.zeros(n_c)
        self.slope_sd = np.zeros(n_c)
        self.life_mu = np.zeros(n_c)
        self.life_rel_sd = np.zeros(n_c)
        bases = {}
        sigmas = []
        for i, name in enumerate(COMPOUNDS):
            key = st.allocation.get(name, DEFAULT_ALLOCATION[name])
            deg = self.models.deg_model(st.track_id, key)
            self.slope_mu[i] = deg.mean
            self.slope_sd[i] = math.sqrt(max(deg.var + deg.tau2, 0.0))
            sigmas.append(math.sqrt(max(deg.sigma2, 0.0)))
            self.life_mu[i] = _compound_params(name).expected_life_laps
            rate = self.models.wear_model(st.track_id, key, st.total_laps).rate(5.0)
            rel = (rate.high - rate.low) / (2 * z * rate.mid) if rate.mid and rate.mid > 0 else 0.3
            self.life_rel_sd[i] = min(max(rel, 0.05), 0.5)
            bases[name] = deg.basis
        self.deg_basis = bases
        self.lap_noise_sd = min(max(float(np.median(sigmas)), LAP_NOISE_RANGE_S[0]), LAP_NOISE_RANGE_S[1])
        # Per-car multipliers: the player's own calibration (from strategy_ml / the fitted
        # stint) applies only to the player's car.
        self.slope_mult = np.ones(self.n)
        self.life_mult = np.ones(self.n)
        self.slope_mult[self.player_slot] = deg_scale
        self.life_mult[self.player_slot] = life_scale
        # The player's current stint: the stint-conditioned degradation range, if given.
        self.player_deg = None
        if player_deg is not None and player_deg.mid is not None and player_deg.low is not None:
            self.player_deg = (player_deg.mid, max((player_deg.high - player_deg.low) / (2 * z), 1e-3))
        fuel = self.models.fuel_model(st.track_id, st.track_length, st.total_laps).per_lap()
        self.fuel_mu = fuel.mid if fuel.mid and fuel.mid > 0 else 1.6
        self.fuel_sd = max((fuel.high - fuel.low) / (2 * z), 0.0) if fuel.mid else 0.1
        self.fuel_basis = self.models.fuel_model(st.track_id, st.track_length, st.total_laps).basis

    def _rival_plans(self):
        """Stop laps and compounds per run for every car but the player: (R, N, S) arrays."""
        R, N, S = self.runs, self.n, self.max_stops + 1
        laps = np.full((R, N, S), NEVER, dtype=np.int64)
        comps = np.zeros((R, N, S), dtype=np.int64)
        st = self.state
        first, last = st.current_lap, st.total_laps - 1
        for slot, car in enumerate(self.cars):
            if car.is_player or st.laps_to_go <= 1:
                continue
            plan = rs.plan_race(st.laps_to_go, car.compound, car.tyre_age, st.current_lap,
                                self.pit_loss_mean, car.compounds_used)
            if plan is None:
                continue
            best = plan.best
            alt = min(plan.alternatives, key=lambda o: o.total_time_s, default=None)
            use_alt = (self.rng.random(R) < RIVAL_ALT_PLAN_PROB) if alt is not None else np.zeros(R, bool)
            pred = car.prediction
            for r in range(R):
                option = alt if use_alt[r] else best
                stops = list(option.pit_laps)
                sets = list(option.compounds[1:])
                if pred is not None and getattr(pred, 'stop_lap_mean', None) is not None:
                    if getattr(pred, 'will_stop', True) is False:
                        stops, sets = [], []
                    else:
                        sd = float(getattr(pred, 'stop_lap_sd', None) or RIVAL_STOP_JITTER_LAPS)
                        target = round(float(pred.stop_lap_mean) + self.rng.normal(0, sd))
                        stops = [s + target - stops[0] for s in stops] if stops else [target]
                        sets = sets or [rs.next_compound(car.compound)]
                        probs = getattr(pred, 'next_compound_probs', None)
                        if probs:
                            names = [k for k in probs if k in _CI]
                            if names:
                                p = np.array([probs[k] for k in names], float)
                                if p.sum() > 0:
                                    sets[0] = names[int(self.rng.choice(len(names), p=p / p.sum()))]
                else:
                    stops = [round(s + self.rng.normal(0, RIVAL_STOP_JITTER_LAPS)) for s in stops]
                prev = first - 1
                for s_i, (lap, comp) in enumerate(zip(stops, sets)):
                    lap = max(int(lap), prev + (1 if s_i == 0 else 3), first)
                    if lap > last:
                        break
                    laps[r, slot, s_i] = lap
                    comps[r, slot, s_i] = _CI.get(comp, _CI['MEDIUM'])
                    prev = lap
        return laps, comps

    def _neutralisations(self):
        """(R, laps) track status: 0 green, 1 Safety Car, 2 VSC."""
        R, L = self.runs, len(self.laps)
        total = max(self.state.total_laps, 1)
        p_sc = float(self.profile.get('sc_probability', 0.4))
        p_vsc = float(self.profile.get('vsc_probability', VSC_SHARE_OF_SC * p_sc))
        # Poisson rates per race so P(at least one) matches the profile; lap 1 weighted up.
        per_lap = lambda p: -math.log(max(1e-9, 1.0 - min(p, 0.99))) / (total + LAP_ONE_HAZARD_MULT - 1)
        h_sc, h_vsc = per_lap(p_sc), per_lap(p_vsc)
        status = np.zeros((R, L), dtype=np.int8)
        kind = np.zeros(R, dtype=np.int8)
        remaining = np.zeros(R, dtype=np.int64)
        if self.state.safety_car in (1, 2):
            kind[:] = self.state.safety_car
            lo, hi = (1, 3) if self.state.safety_car == 1 else (1, 2)
            remaining[:] = self.rng.integers(lo, hi + 1, R)
        for li, lap in enumerate(self.laps):
            mult = LAP_ONE_HAZARD_MULT if lap == 1 else 1.0
            free = remaining == 0
            u = self.rng.random(R)
            start_sc = free & (u < 1 - math.exp(-h_sc * mult))
            start_vsc = free & ~start_sc & (u < 1 - math.exp(-(h_sc + h_vsc) * mult))
            kind = np.where(start_sc, 1, np.where(start_vsc, 2, kind)).astype(np.int8)
            remaining = np.where(start_sc, self.rng.integers(SC_DURATION_LAPS[0], SC_DURATION_LAPS[1] + 1, R),
                                 np.where(start_vsc, self.rng.integers(VSC_DURATION_LAPS[0], VSC_DURATION_LAPS[1] + 1, R),
                                          remaining))
            status[:, li] = np.where(remaining > 0, kind, 0)
            remaining = np.maximum(remaining - 1, 0)
        return status

    def _draw(self, rain: bool):
        R, N, L, S = self.runs, self.n, len(self.laps), self.max_stops + 1
        rng = self.rng
        st = self.state
        # Base pace: measured where we can, guessed from track position otherwise.
        known = [c.pace_s for c in self.cars if c.pace_s]
        ref = float(np.median(known)) if known else DEFAULT_LAP_S
        ref_pos = float(np.median([c.position for c in self.cars if c.pace_s])) if known else 1.0
        base = np.array([c.pace_s if c.pace_s else ref + POSITION_PACE_STEP_S * (c.position - ref_pos)
                         for c in self.cars])
        form_sd = np.array([DRIVER_FORM_SD_S if c.pace_s and c.pace_laps >= 3 else ASSUMED_FORM_SD_S
                            for c in self.cars])
        self.base_ref = ref
        self.base = base
        self.pace_assumed = sum(1 for c in self.cars if not c.pace_s)
        self.form = rng.normal(0, 1, (R, N)) * form_sd
        self.form[:, self.player_slot] = rng.normal(0, 1, R) * DRIVER_FORM_SD_S * 0.5   # we know our car best
        self.noise = rng.normal(0, self.lap_noise_sd, (R, L, N))
        slow = rng.random((R, L, N)) < SLOW_STOP_PROB
        self.pit_loss = (self.pit_loss_mean + rng.normal(0, PIT_LOSS_SD_S, (R, L, N))
                         + np.where(slow, rng.exponential(SLOW_STOP_MEAN_S, (R, L, N)), 0.0))
        self.u_pass = rng.random((R, L, N, 2))
        self.z_slope = rng.normal(0, 1, (R, N, S + 2))
        self.z_life = rng.normal(0, 1, (R, N, S + 2))
        self.burn = np.maximum(0.2, self.fuel_mu + rng.normal(0, 1, (R, N)) * self.fuel_sd)
        self.fuel0 = np.array([c.fuel_kg if c.fuel_kg else self.fuel_mu * st.laps_to_go + 1.0
                               for c in self.cars])
        self.status = self._neutralisations()
        self._conditions(rain)
        self.rival_laps, self.rival_comps = self._rival_plans()
        self.gap0 = np.array([c.gap_s for c in self.cars], float)
        self.age0 = np.array([c.tyre_age for c in self.cars], float)
        self.comp0 = np.array([_CI.get(rs.normalise_compound(c.compound) or 'MEDIUM', 1) for c in self.cars])
        self.thr = PASS_THRESHOLD_S.get(min(max(self.overtaking, 1), 5), 0.9)

    def _weather_timeline(self, rain: bool) -> np.ndarray:
        """(R, L) weather code per lap and run: 0-2 dry, 3 light rain, 4 heavy, 5 storm."""
        R, L, rng, st = self.runs, len(self.laps), self.rng, self.state
        now = st.weather if st.weather is not None else (3 if st.raining else 1)
        if st.raining:
            now = max(now, 3)
        elif now >= 3:
            now = 2
        code = np.full((R, L), now, dtype=np.int8)
        if not rain or L <= 1:
            return code
        lap_min = self.lap_s / 60.0
        forecast = [s for s in (st.forecast or ()) if getattr(s, 'offset_min', None) is not None]
        if forecast:
            # The game's forecast: one draw per run, so a run that gets the shower keeps it
            # for as long as the forecast keeps the chance up (rain chance is coherent in time).
            u = rng.random(R)
            for li in range(1, L):
                t = li * lap_min
                sample = None
                for s_ in forecast:
                    if s_.offset_min <= t + 1e-9:
                        sample = s_
                if sample is None:
                    continue
                wet = u < tw.rain_probability(sample)
                code[:, li] = np.where(wet, max(sample.weather, 3), min(sample.weather, 2))
            return code
        if st.raining:
            # No forecast: the shower stops after an exponential time, and the sky stays overcast.
            stop_min = rng.exponential(RAIN_MEAN_DURATION_MIN, R)
            stop_li = np.ceil(stop_min / lap_min).astype(np.int64)
            code = np.where(np.arange(L)[None, :] >= stop_li[:, None], 2, code).astype(np.int8)
            return code
        p_rain = 0.0
        if st.total_laps > 0:
            p_rain = float(self.profile.get('rain_probability', 0.0)) * st.laps_to_go / st.total_laps
        rains = rng.random(R) < p_rain
        start = rng.integers(0, max(L, 1), R)
        heavy = rng.random(R) < HEAVY_SHOWER_SHARE
        shower = np.where(heavy, 4, 3)
        on = rains[:, None] & (np.arange(L)[None, :] >= start[:, None])
        return np.where(on, shower[:, None], code).astype(np.int8)

    def _conditions(self, rain: bool):
        """Wetness per run and lap, each tyre family's lap-time cost on it, and who reacts."""
        R, N, L, rng, st = self.runs, self.n, len(self.laps), self.rng, self.state
        self.lap_s = float(st.lap_s or self.base_ref)
        code = self._weather_timeline(rain)
        if st.wetness is not None:
            w = np.clip(st.wetness + rng.normal(0, 1, R) * max(st.wetness_sd, 0.0), 0.0, 1.0)
        elif st.raining:
            target = tw.WET_TARGET.get(int(code[0, 0]), 0.5)
            w = np.clip(target + rng.normal(0, 1, R) * WETNESS_START_SD_RAINING, 0.0, 1.0)
        else:
            w = np.zeros(R)
        dry_mult = np.exp(rng.normal(0, DRY_RATE_RUN_SD, R))
        lap_min = self.lap_s / 60.0
        temp = st.track_temp if st.track_temp is not None else 30.0
        rate = {c: tw.dry_rate_per_min(temp, c) for c in range(6)}
        W = np.zeros((R, L))
        for li in range(L):
            W[:, li] = w
            c = code[:, li]
            target = np.array([tw.WET_TARGET.get(int(k), 0.0) for k in range(6)])[c]
            soak = w + (target - w) * (1.0 - math.exp(-lap_min / tw.RAIN_TAU_MIN))
            drain = np.maximum(target, w - 0.5 * rate[2] * dry_mult * lap_min)
            dry = np.maximum(0.0, w - np.array([rate[k] for k in range(6)])[c] * dry_mult * lap_min)
            w = np.where(c >= 3, np.where(w < target, soak, drain), dry)
        self.W = W
        self.weather_code = code
        self.pen = np.stack([tw.penalty(f, W, self.lap_s) for f in tw.FAMILIES])      # (3, R, L)
        wet_lap = code >= 3
        self.rain_at = np.where(wet_lap.any(axis=1), wet_lap.argmax(axis=1), -1)
        self.p_rain_input = float(wet_lap.any(axis=1).mean())
        # The player follows the plan, except after a shower that arrives during the run (one the
        # plan could not know about): from then on we react like everyone else.
        new_shower = np.zeros((R, L), dtype=bool)
        new_shower[:, 1:] = wet_lap[:, 1:] & ~wet_lap[:, :-1]
        self.player_reacts = np.cumsum(new_shower, axis=1) > 0
        scale = self.lap_s / tw.REFERENCE_LAP_S
        self.react_thr = np.clip(REACT_GAIN_S + rng.normal(0, 1, (R, N)) * REACT_GAIN_SD_S,
                                 *REACT_GAIN_RANGE_S) * scale
        # Which slick to fit when the track is ready, by laps left (the longest-lasting set that
        # still has pace: softs for a sprint to the flag, else mediums, else hards).
        self.dry_life = {name: self.life_mu[_CI[name]] for name in ('SOFT', 'MEDIUM', 'HARD')}

    def _dry_choice(self, laps_left: int) -> int:
        for name in ('SOFT', 'MEDIUM'):
            if laps_left <= 0.9 * self.dry_life[name]:
                return _CI[name]
        return _CI['HARD']

    # ── one plan through every run ──
    def _simulate(self, candidate: Candidate) -> np.ndarray:
        R, N, S = self.runs, self.n, self.max_stops + 1
        rows = np.arange(R)
        stop_lap = self.rival_laps.copy()
        stop_comp = self.rival_comps.copy()
        p = self.player_slot
        stop_lap[:, p, :] = NEVER
        for s_i, (lap, comp) in enumerate(zip(candidate.pit_laps, candidate.compounds[1:])):
            if s_i < S:
                stop_lap[:, p, s_i] = lap
                stop_comp[:, p, s_i] = _CI.get(comp, 1)
        T = np.broadcast_to(self.gap0, (R, N)).copy()
        age = np.broadcast_to(self.age0, (R, N)).copy()
        comp = np.broadcast_to(self.comp0, (R, N)).copy()
        stint = np.zeros((R, N), dtype=np.int64)
        nxt = np.zeros((R, N), dtype=np.int64)
        out_lap = np.zeros((R, N), dtype=bool)
        fuel = np.broadcast_to(self.fuel0, (R, N)).copy()
        prev_status = np.full(R, self.state.safety_car if self.state.safety_car in (1, 2) else 0)
        rivals = np.ones(N, dtype=bool)
        rivals[p] = False
        fam_idx = FAM_IDX
        base = self.base[None, :] + self.form
        # Per-stint tyre draws, looked up by compound at use.
        r_idx = rows[:, None]
        c_idx = np.arange(N)[None, :]

        for li, lapnum in enumerate(self.laps):
            status = self.status[:, li]
            sc, vsc = status == 1, status == 2
            neutral = sc | vsc
            new_neutral = neutral & (prev_status == 0)
            prev_status = status
            pen_l = self.pen[:, :, li].T                                   # (R, 3)
            fam = fam_idx[comp]
            cur_pen = np.take_along_axis(pen_l, fam, 1)
            best_pen = pen_l.min(axis=1)
            best_f = pen_l.argmin(axis=1)
            laps_left = self.state.total_laps - lapnum

            # ── who boxes at the end of this lap ──
            has_next = nxt < S
            pending = np.where(has_next, np.take_along_axis(stop_lap, np.minimum(nxt, S - 1)[..., None], 2)[..., 0], NEVER)
            planned = pending == lapnum
            early = new_neutral[:, None] & (pending > lapnum) & (pending - lapnum <= SC_PIT_WINDOW_LAPS) \
                & (lapnum < self.laps[-1])
            # Conditions: a car whose tyre type is clearly slower than the best one on this
            # surface changes (its own threshold: some gamble early, some wait), if the laps
            # left repay the stop. The player only does so after an unforeseen shower.
            gain = cur_pen - best_pen[:, None]
            adapts = np.broadcast_to(rivals[None, :], (R, N)).copy()
            adapts[:, p] = self.player_reacts[:, li]
            cond = (adapts & (gain >= self.react_thr) & (gain * laps_left > self.pit_loss_mean)
                    & (laps_left >= 2) & ~out_lap)
            pit = (planned | early | cond) & (lapnum < self.state.total_laps)
            next_set = np.take_along_axis(stop_comp, np.minimum(nxt, S - 1)[..., None], 2)[..., 0]
            cond_comp = np.where(best_f == 0, self._dry_choice(laps_left),
                                 np.where(best_f == 1, _CI['INTERMEDIATE'], _CI['WET']))[:, None]
            wrong = (np.take_along_axis(pen_l, fam_idx[next_set], 1) - best_pen[:, None]) > WRONG_TYRE_S * (
                self.lap_s / tw.REFERENCE_LAP_S)
            new_comp = np.where(cond | (pit & wrong & adapts), cond_comp, next_set)

            # ── lap times ──
            k = np.minimum(stint, self.z_slope.shape[2] - 1)
            zs = self.z_slope[r_idx, c_idx, k]
            zl = self.z_life[r_idx, c_idx, k]
            slope = self.slope_mu[comp] + self.slope_sd[comp] * zs
            if self.player_deg is not None:
                first_stint = stint[:, p] == 0
                slope[:, p] = np.where(first_stint, self.player_deg[0] + self.player_deg[1] * zs[:, p], slope[:, p])
            slope = np.maximum(slope, 0.0) * self.slope_mult[None, :]
            W_l = self.W[:, li][:, None]
            overheat = ((fam == 1) & (W_l < 0.15)) | ((fam == 2) & (W_l < 0.35))
            slope = np.where(overheat, slope * WET_TYRE_DRY_DEG_MULT, slope)
            life = self.life_mu[comp] * self.life_mult[None, :] * np.exp(self.life_rel_sd[comp] * zl)
            past = np.maximum(age - life, 0.0)
            tyre = self.offset[comp] + slope * age + rs.CLIFF_RATE * past * past + out_lap * rs.OUT_LAP_WARMUP_S
            lt = base + tyre + fm.FUEL_EFFECT_S_PER_KG * fuel + self.noise[:, li, :]
            lt = lt + cur_pen
            neutral_lt = np.where(sc, self.base_ref * SC_LAP_FACTOR, self.base_ref * VSC_LAP_FACTOR)
            lt = np.where(neutral[:, None], neutral_lt[:, None], lt)

            # ── traffic: dirty air, then held up or passing, green laps only ──
            T_start = T
            order = np.argsort(T_start, axis=1)
            sorted_T = np.take_along_axis(T_start, order, 1)
            gap_ahead = np.full((R, N), np.inf)
            gaps_sorted = np.diff(sorted_T, axis=1)
            np.put_along_axis(gap_ahead, order[:, 1:], gaps_sorted, 1)
            dirty = np.where(gap_ahead < DIRTY_AIR_GAP_S, DIRTY_AIR_S * (1 - gap_ahead / DIRTY_AIR_GAP_S), 0.0)
            green = ~neutral
            lt = lt + np.where(green[:, None] & ~pit, dirty, 0.0)
            T_new = T_start + lt
            if green.any() and N > 1:
                self._resolve_traffic(T_start, T_new, order, pit, green, li)

            # ── Safety Car bunching (instant) ──
            if sc.any():
                o = np.argsort(T_new, axis=1)
                lead = np.take_along_axis(T_new, o[:, :1], 1)
                bunched = np.empty_like(T_new)
                np.put_along_axis(bunched, o, lead + SC_GAP_S * np.arange(N)[None, :], 1)
                T_new = np.where(sc[:, None], bunched, T_new)

            factor = np.where(sc, SC_PIT_FACTOR, np.where(vsc, VSC_PIT_FACTOR, 1.0))
            T = T_new + pit * self.pit_loss[:, li, :] * factor[:, None]

            # ── after the line ──
            consumed = pit & ~cond
            nxt = nxt + consumed
            nxt = np.where(cond, S, nxt)                # a tyre-type change replaces the dry stop plan
            comp = np.where(pit, new_comp, comp)
            stint = stint + pit
            age = np.where(pit, 0.0, age + 1.0)
            out_lap = pit
            fuel = np.maximum(fuel - self.burn, 0.0)
        self.last_compounds = comp            # every car's tyre at the flag (for checks and tests)
        return T

    def _resolve_traffic(self, T_start, T_new, order, pit, green, li):
        """Walk the field front to back; each car either stays behind the car ahead or passes it.

        A car that would cross the line less than MIN_GAP_S behind (or ahead of) the nearest
        on-track car in front has caught it. It passes with a chance that rises with its pace
        advantage, centred on this track's PASS_THRESHOLD_S; otherwise it is held up. A car
        that gets by may take one more car the same lap. Cars on their in-lap are in the pit
        lane for the crossing and neither block nor get blocked.
        """
        R, N = T_new.shape
        rows = np.arange(R)
        b1 = np.full(R, -1)
        b2 = np.full(R, -1)
        thr, scale = self.thr, PASS_SCALE_S
        for pos in range(N):
            k = order[:, pos]
            Tk = T_new[rows, k]
            in_pit = pit[rows, k]
            lt_k = Tk - T_start[rows, k]
            has1 = b1 >= 0
            j1 = np.where(has1, b1, 0)
            T1 = T_new[rows, j1]
            catch = green & has1 & ~in_pit & (Tk < T1 + MIN_GAP_S)
            adv = (T1 - T_start[rows, j1]) - lt_k
            p_pass = 1.0 / (1.0 + np.exp(-(adv - thr) / scale))
            passed = catch & (self.u_pass[rows, li, k, 0] < p_pass)
            held = catch & ~passed
            Tk = np.where(held, T1 + MIN_GAP_S, Tk)
            Tk = np.where(passed, np.minimum(Tk + PASS_COST_S, T1 - 0.1), Tk)
            if passed.any():
                T_new[rows[passed], j1[passed]] += DEFEND_COST_S
            # A second car the same lap, only for a car that just got by.
            has2 = passed & (b2 >= 0)
            j2 = np.where(has2, b2, 0)
            T2 = T_new[rows, j2]
            catch2 = has2 & (Tk < T2 + MIN_GAP_S)
            adv2 = (T2 - T_start[rows, j2]) - lt_k
            p2 = 1.0 / (1.0 + np.exp(-(adv2 - thr) / scale))
            passed2 = catch2 & (self.u_pass[rows, li, k, 1] < p2)
            held2 = catch2 & ~passed2
            Tk = np.where(held2, T2 + MIN_GAP_S, Tk)
            Tk = np.where(passed2, np.minimum(Tk + PASS_COST_S, T2 - 0.1), Tk)
            if passed2.any():
                T_new[rows[passed2], j2[passed2]] += DEFEND_COST_S
            T_new[rows, k] = Tk
            # Nearest two on-track cars ahead of the next car in the start order.
            on_track = ~in_pit
            new_b1 = np.where(passed, b1, k)
            new_b2 = np.where(passed, np.where(passed2, b2, k), b1)
            b1 = np.where(on_track, new_b1, b1)
            b2 = np.where(on_track, new_b2, b2)

    # ── all plans ──
    def evaluate(self, candidates: Sequence[Candidate], reference: int | None = None) -> SimResult:
        """Play every plan through the same races. ``reference`` (an index into ``candidates``,
        usually the current Plan A) adds a paired comparison of each plan against it."""
        if reference is not None and not 0 <= reference < len(candidates):
            raise ValueError('reference must index a candidate')
        if reference is not None and candidates[reference] is None:
            raise ValueError('reference candidate is None')
        if reference is not None:
            reference = sum(1 for c in candidates[:reference] if c is not None)
        candidates = [c for c in candidates if c is not None]
        if not candidates:
            raise ValueError('no candidate plans to simulate')
        R, N = self.runs, self.n
        p = self.player_slot
        finish_T, finish_pos, all_pos = [], [], []
        for cand in candidates:
            T = self._simulate(cand)
            ranks = np.argsort(np.argsort(T, axis=1), axis=1) + 1
            finish_T.append(T[:, p])
            finish_pos.append(ranks[:, p])
            all_pos.append(ranks)
        pos = np.stack(finish_pos)                     # (C, R)
        tim = np.stack(finish_T)                       # player's race time, comparable across plans
        # Best plan per run: best position, then the quicker race. Ties share the credit.
        score = pos * 1e6 + (tim - tim.min())
        best = score.min(axis=0)
        is_best = np.isclose(score, best[None, :], rtol=0, atol=1e-6)
        share = is_best / is_best.sum(axis=0, keepdims=True)
        p_best = share.mean(axis=1)
        # Race time vs the best plan in the same run (absolute, comparable thanks to shared draws).
        abs_T = tim - tim.min(axis=0, keepdims=True)
        outcomes = []
        for ci, cand in enumerate(candidates):
            x = pos[ci]
            vs_ref = None
            if reference is not None and ci != reference:
                gain_t = tim[reference] - tim[ci]                 # +: this plan is quicker
                ahead = score[ci] < score[reference] - 1e-6
                vs_ref = {'time_gain_s': round(float(np.median(gain_t)), 2),
                          'time_gain_range_s': [round(_pct(gain_t, 10), 1), round(_pct(gain_t, 90), 1)],
                          'position_gain': round(float((pos[reference] - x).mean()), 2),
                          'p_better': round(float(ahead.mean()), 3)}
            probs = np.bincount(x, minlength=N + 1)[1:N + 1] / R
            outcomes.append(StrategyOutcome(
                cand.label, cand.stops, list(cand.compounds), list(cand.pit_laps),
                round(float(p_best[ci]), 3), round(float((x == 1).mean()), 3),
                round(float((x <= 3).mean()), 3), round(float((x <= 10).mean()), 3),
                round(float(x.mean()), 2),
                [int(np.percentile(x, 10)), int(np.median(x)), int(np.percentile(x, 90))],
                [round(float(v), 3) for v in probs],
                [round(_pct(abs_T[ci], 10), 1), round(_pct(abs_T[ci], 50), 1), round(_pct(abs_T[ci], 90), 1)],
                ci == reference, vs_ref))
        order = sorted(range(len(outcomes)), key=lambda i: (-outcomes[i].p_best, outcomes[i].mean_position))
        outcomes = [outcomes[i] for i in order]
        rec_ranks = all_pos[order[0]]
        field_out = []
        for slot, car in enumerate(self.cars):
            x = rec_ranks[:, slot]
            field_out.append({'car_index': car.index, 'name': car.name, 'is_player': car.is_player,
                              'position_now': car.position, 'mean_position': round(float(x.mean()), 2),
                              'p_win': round(float((x == 1).mean()), 3),
                              'p_podium': round(float((x <= 3).mean()), 3),
                              'position_range': [int(np.percentile(x, 10)), int(np.median(x)),
                                                 int(np.percentile(x, 90))]})
        field_out.sort(key=lambda f: f['mean_position'])
        green_loss = self.pit_loss[:, 0, :].ravel()
        result = SimResult(
            runs=R, lap=self.state.current_lap, laps_to_go=self.state.laps_to_go, outcomes=outcomes,
            field=field_out,
            p_safety_car=round(float((self.status == 1).any(axis=1).mean()), 3),
            p_vsc=round(float((self.status == 2).any(axis=1).mean()), 3),
            p_rain=round(float((self.rain_at >= 0).mean()), 3),
            pit_loss_s=[round(_pct(green_loss, 10), 1), round(float(green_loss.mean()), 1),
                        round(_pct(green_loss, 90), 1)],
            overtaking=self.overtaking,
            inputs={'deg_basis': self.deg_basis, 'fuel_basis': self.fuel_basis,
                    'simulated_fit': bool(self.models.simulated_only),
                    'cars_with_guessed_pace': self.pace_assumed,
                    'track_profile': self.profile.get('source', 'default')})
        result.radio = describe(result)
        return result


def describe(result: SimResult) -> str:
    """Radio line: which plan comes out best how often, and where we finish."""
    if not result.outcomes:
        return ''
    top = result.outcomes[0]
    name = top.label.split(' (')[0].lower()
    line = f'{name.capitalize()} comes out best {round(top.p_best * 100)}% of the time'
    if len(result.outcomes) > 1:
        second = result.outcomes[1]
        line += f", {second.label.split(' (')[0].lower()} {round(second.p_best * 100)}%"
    lo, mid, hi = top.position_range
    line += f'. Expected finish P{mid}' + (f', P{lo} to P{hi}.' if lo != hi else '.')
    return line


def simulate(state: RaceState, candidates: Sequence[Candidate], reference: int | None = None,
             **kwargs) -> SimResult:
    return RaceSimulator(state, **kwargs).evaluate(candidates, reference)


def settled_result(position: int, cars: int, lap: int, compound: str | None = None) -> SimResult:
    """The odds once we have taken the chequered flag: the result is known, so they are certain."""
    cars = max(int(cars), int(position), 1)
    position = max(int(position), 1)
    current = rs.normalise_compound(compound) or 'MEDIUM'
    probs = [1.0 if i + 1 == position else 0.0 for i in range(cars)]
    outcome = StrategyOutcome('Finished', 0, [current], [], 1.0, float(position == 1), float(position <= 3),
                              float(position <= 10), float(position), [position] * 3, probs, [0.0, 0.0, 0.0],
                              True)
    result = SimResult(runs=0, lap=lap, laps_to_go=0, outcomes=[outcome], field=[], p_safety_car=0.0,
                       p_vsc=0.0, p_rain=0.0, pit_loss_s=[0.0, 0.0, 0.0], overtaking=0,
                       inputs={'settled': True})
    result.radio = f'Chequered flag. We finished P{position}.'
    return result


# ──────────────────────────────────────────────
# Live field tracking (fed from the Lap Data packet every frame)
# ──────────────────────────────────────────────

CLEAN_LAPS_KEPT = 6


class FieldTracker:
    """Per-car line-crossing times and clean-lap pace, for building a RaceState live.

    A lap counts as clean when the car neither pitted on it nor started it in the pit lane
    (in-laps and out-laps are left out), it was not neutralised by a SC/VSC, and it is not
    lap 1. Pace is the median of the clean laps, corrected for tyre age and fuel, so it is
    the "fresh tyres, empty tank" lap time the simulator builds on.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.crossings: dict[int, dict[int, float]] = {}
        self.last_lap: dict[int, int] = {}
        self.dirty: dict[int, bool] = {}
        self.clean: dict[int, list] = {}
        self.compounds: dict[int, set] = {}
        self.stops_at_start: dict[int, int] = {}
        self.pitted_prev: dict[int, bool] = {}
        self.last_session_time = None

    def update(self, laps, session_time: float, statuses=None, neutralised: bool = False):
        if self.last_session_time is not None and session_time < self.last_session_time - 1.0:
            self.reset()                 # flashback / restart
        self.last_session_time = session_time
        for index, lap in enumerate(laps):
            if lap.result_status != 2 or lap.car_position == 0:
                continue
            lap_num = lap.current_lap_num
            prev = self.last_lap.get(index)
            if prev is None:
                self.last_lap[index] = lap_num
                self.dirty[index] = True     # joined mid-lap: we can't vouch for this lap
                self.stops_at_start[index] = lap.num_pit_stops
                self.pitted_prev[index] = False
            elif lap_num > prev:
                self.crossings.setdefault(index, {})[lap_num] = session_time
                pitted = lap.num_pit_stops != self.stops_at_start.get(index, lap.num_pit_stops)
                done_ms = lap.last_lap_time_ms
                if (done_ms and not self.dirty.get(index) and not pitted
                        and not self.pitted_prev.get(index) and prev > 1):
                    status = statuses[index] if statuses and index < len(statuses) else None
                    age = (status.tyres_age_laps - 1) if status else None
                    fuel = status.fuel_in_tank if status else None
                    self.clean.setdefault(index, []).append((done_ms / 1000.0, age, fuel))
                    del self.clean[index][:-CLEAN_LAPS_KEPT]
                self.last_lap[index] = lap_num
                self.pitted_prev[index] = pitted          # next lap is an out-lap
                self.stops_at_start[index] = lap.num_pit_stops
                self.dirty[index] = bool(lap.pit_status) or neutralised
            if lap.pit_status or neutralised:
                self.dirty[index] = True
            if statuses and index < len(statuses) and statuses[index] is not None:
                name = rs.normalise_compound(statuses[index].compound_name)
                if name:
                    self.compounds.setdefault(index, set()).add(name)

    def pace(self, index: int, slope_s_per_lap: float) -> tuple[float | None, int]:
        laps = self.clean.get(index) or []
        if not laps:
            return None, 0
        vals = [t - slope_s_per_lap * max(age or 0, 0) - fm.FUEL_EFFECT_S_PER_KG * (fuel or 0.0)
                for t, age, fuel in laps]
        return float(np.median(vals)), len(vals)

    def gap_to_leader(self, laps) -> dict[int, float]:
        """Race-time gap of each running car to the leader at its most recent line crossing."""
        running = {i: l for i, l in enumerate(laps) if l.result_status == 2 and l.car_position > 0}
        if not running:
            return {}
        leader = min(running, key=lambda i: running[i].car_position)
        lead_cross = self.crossings.get(leader, {})
        gaps = {}
        for index, lap in running.items():
            mine = self.crossings.get(index, {})
            lap_num = max(mine) if mine else None
            if lap_num is not None and lap_num in lead_cross:
                gaps[index] = max(0.0, mine[lap_num] - lead_cross[lap_num])
        return gaps


def build_state(laps, statuses, player_index: int, session, tracker: FieldTracker,
                names: dict | None = None, predictions: dict | None = None,
                models: fm.ModelStore | None = None, conditions: dict | None = None) -> RaceState | None:
    """RaceState from the latest Lap Data / Car Status packets and the field tracker."""
    if session is None or not laps or player_index >= len(laps):
        return None
    me = laps[player_index]
    if me.result_status != 2 or session.total_laps <= 0 or me.current_lap_num > session.total_laps:
        return None
    models = models if models is not None else fm.ModelStore()
    statuses = statuses or []
    alloc = compound_allocation((s.compound_name, s.actual_tyre_compound) for s in statuses if s is not None)
    gaps = tracker.gap_to_leader(laps)
    cars = []
    for index, lap in enumerate(laps):
        if lap.result_status != 2 or lap.car_position == 0:
            continue
        status = statuses[index] if index < len(statuses) else None
        compound = rs.normalise_compound(status.compound_name) if status else None
        compound = compound or 'MEDIUM'
        slope = models.deg_model(session.track_id, alloc.get(compound, 'C3')).mean
        pace, n = tracker.pace(index, slope)
        gap = gaps.get(index)
        if gap is None:          # no shared line crossing yet: space cars by position
            gap = (lap.car_position - 1) * 1.0
        cars.append(SimCar(index, lap.car_position, gap, compound,
                           int(status.tyres_age_laps) if status else 0, pace, n,
                           (status.fuel_in_tank if status and status.fuel_in_tank > 0 else None),
                           tuple(sorted(tracker.compounds.get(index, ()))),
                           (names or {}).get(index), index == player_index,
                           (predictions or {}).get(index)))
    if not any(c.is_player for c in cars):
        return None
    return RaceState(session.track_id, session.total_laps, me.current_lap_num, cars,
                     session.safety_car_status if session.safety_car_status in (1, 2) else 0,
                     session.weather >= 3, session.track_length, alloc, **(conditions or {}))


class SimRunner:
    """Runs the simulation off the telemetry thread and keeps the latest result.

    ``mode`` 'async' (live: a daemon thread, the result appears a moment later), 'sync'
    (tests: runs inline) or 'off'.
    """

    def __init__(self, mode: str = 'off', runs: int = DEFAULT_RUNS):
        self.mode = mode
        self.runs = runs
        self.latest: SimResult | None = None
        self.latest_key = None            # key of the last run, finished or failed
        self.result_key = None            # key that ``latest`` was computed for
        self.error: str | None = None
        self._busy = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.mode in ('sync', 'async')

    def reset(self):
        with self._lock:
            self.latest = None
            self.latest_key = None
            self.result_key = None

    def settle(self, key, result: SimResult):
        """Store a known result (the race is over) in place of a simulated one."""
        with self._lock:
            self.latest, self.latest_key, self.result_key, self.error = result, key, key, None

    def submit(self, key, state: RaceState, candidates: list, **kwargs) -> bool:
        if not self.enabled or not candidates or key == self.latest_key:
            return False
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        kwargs.setdefault('runs', self.runs)

        def work():
            try:
                result = simulate(state, candidates, **kwargs)
                with self._lock:
                    self.latest, self.latest_key, self.result_key, self.error = result, key, key, None
            except Exception as error:       # never take the telemetry loop down
                with self._lock:
                    self.error, self.latest_key = repr(error), key
            finally:
                with self._lock:
                    self._busy = False

        if self.mode == 'sync':
            work()
        else:
            threading.Thread(target=work, name='race-sim', daemon=True).start()
        return True
