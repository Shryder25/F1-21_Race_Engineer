"""Rival strategy models: when will each rival box next, and onto which tyre?

A real pit wall doesn't wait for a rival to turn into the pit lane: it keeps a running guess
of every car's next stop and plans the undercut, overcut or cover around it. This module
does that for every car in the field, from what the game shows about each one:

* **Tyre age and compound** (Car Status packet, all 22 cars).
* **Observed tyre wear** (Car Damage packet, all 22 cars; hidden for online players with
  restricted telemetry, in which case the model falls back to age alone).
* **The track's typical strategy** (track_profiles + track_analysis) and the race plan the
  car would follow if it ran the optimal strategy from where it is now (race_strategy).

Three estimates of the stop lap are blended by how sure each one is:

  1. *Plan*: the cheapest strategy for the rest of the race from the rival's tyre age, with
     tyre life scaled to this track. Always available.
  2. *Wear*: the fitted wear model (fitted_models, per compound and track) conditioned on
     the rival's own wear this stint, run forward to the wear at which cars are boxing.
  3. *Game window*: the game's own pit window (sent for the player) is a good guide to the
     AI's first stop, scaled for compound.

Rain overrides the stint maths: slicks in heavy rain are due in now, and the next set is
an intermediate (or a wet). The field teaches the model as the race goes on: each real rival stop updates the wear at
which cars box and how much longer or shorter they run than the plan said
(``FieldStats``). The next compound is weighed by what the rest of the race costs on each
set (respecting the two-compound rule) and nudged towards the track's typical strategy.

Everything that predicts is a pure function of numbers passed in; ``RivalTracker`` only
collects per-car stints from the packets. The Monte Carlo race simulator can sample from a
prediction with ``sample_stop_lap`` / ``sample_next_compound``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist

import fitted_models
import race_strategy
import track_analysis
import track_profiles
from physics_models import _compound_params
from track_data import resolve_track_id

DRY = race_strategy.DRY_COMPOUNDS
WET = race_strategy.WET_COMPOUNDS

# Wear (worst tyre, %) at which cars box, before any rival has stopped. Matches the
# player's own ideal-wear target (strategy_ml.IDEAL_WEAR_PCT); wide on purpose.
STOP_WEAR_PRIOR_PCT = 60.0
STOP_WEAR_PRIOR_SD = 10.0
PRIOR_PSEUDO_STOPS = 2.0            # how many real stops the priors are worth
LIFE_FACTOR_PRIOR_SD = 0.2
# Plan estimate: every first-stop lap is weighted by exp(-extra race time / this), so a flat
# cost curve gives a wide spread. A no-stop plan gets this fraction of its stint as spread.
PLAN_TEMPERATURE_S = 2.0
PLAN_SD_FRACTION = 0.2
PLAN_SD_MIN = 1.5
# The plan is a generic tyre model, not this car: its stint length carries this relative
# uncertainty on top, so the rival's own wear takes over once it has been measured.
PLAN_MODEL_REL_SD = 0.2
# Once the rival's own wear has been measured the generic plan is only a weak prior.
PLAN_SD_WITH_WEAR = 2.0
GAME_WINDOW_SD = 3.0                # same compound as the player; more when it differs
MIN_SD = 0.8
OVERDUE_SD = 1.0
MIN_WEAR_LAPS = 2                   # wear increments needed before the wear estimate counts
HIDDEN_WEAR_AGE = 2                 # wear still zero after this many laps = restricted telemetry
COMPOUND_TEMPERATURE_S = 4.0        # race-time difference that makes one set e^1 times likelier
RAIN_WEATHER = 3                    # session weather: 3 light rain, 4 heavy rain, 5 storm
HEAVY_RAIN_WEATHER = 4
DRY_WEATHER_MAX = 1                 # clear or light cloud: inters/wets are due to come off
TRANSITION_PSEUDO = 2.0             # model's next-compound guess is worth this many observed stops
SNAPSHOT_DISTANCE_M = fitted_models.SNAPSHOT_DISTANCE_M
LEVEL = fitted_models.DEFAULT_LEVEL


# ──────────────────────────────────────────────
# Track priors
# ──────────────────────────────────────────────

_typical_cache: dict = {}


def track_life_scale(track_id: int | None, track_length: int | None) -> float:
    """How much longer (>1) or shorter tyres last here than the model's reference track."""
    if track_id is None or track_id < 0:
        return 1.0
    profile = track_profiles.get_profile(resolve_track_id(track_id))
    if profile is None:
        return 1.0
    length = float(track_length or track_analysis.REFERENCE_LAP_M)
    scale = (track_analysis.REFERENCE_LAP_M / max(2500.0, length)) / track_analysis.demand_factor(profile)
    return round(min(1.6, max(0.6, scale)), 3)


def position_cost_s(track_id: int | None) -> float:
    """Extra cost teams put on a stop where passing is hard (track position), as track_analysis does."""
    if track_id is None or track_id < 0:
        return 0.0
    profile = track_profiles.get_profile(resolve_track_id(track_id))
    return max(0, profile['overtaking'] - 2) * 3.0 if profile else 0.0


def typical_sequence(track_id: int | None, track_length: int | None, total_laps: int) -> list[str]:
    """Compound order of the track's typical dry strategy for this race distance ([] if unknown)."""
    if track_id is None or track_id < 0 or total_laps <= 0:
        return []
    key = (resolve_track_id(track_id), track_length, total_laps)
    if key not in _typical_cache:
        sequence = []
        profile = track_profiles.get_profile(key[0])
        if profile is not None:
            length = float(track_length or track_analysis.REFERENCE_LAP_M)
            plan = track_analysis.plan_strategy(
                profile, length, track_analysis.degradation_rates(profile),
                track_analysis.compound_life(profile, length), wet=False,
                race_fraction=total_laps / max(1, profile['race_laps']))
            if plan.get('best'):
                sequence = [s['compound'] for s in plan['best']['stints']]
        _typical_cache[key] = sequence
    return list(_typical_cache[key])


def typical_next(sequence: list[str], compound: str, stops_done: int) -> str | None:
    """The set the typical strategy fits after ``compound``, or None."""
    if not sequence:
        return None
    if stops_done + 1 < len(sequence) and sequence[stops_done] == compound:
        return sequence[stops_done + 1]
    if compound in sequence[:-1]:
        return sequence[sequence.index(compound) + 1]
    return None


# ──────────────────────────────────────────────
# What the field has shown so far
# ──────────────────────────────────────────────

@dataclass
class FieldStats:
    """Learned from rival stops this race: wear at the stop and stint length against the plan."""
    stop_wear: list = field(default_factory=list)      # worst-tyre wear % when a car boxed
    life_ratio: list = field(default_factory=list)     # actual stint / planned stint at its start
    transitions: dict = field(default_factory=dict)    # (from, to) compound -> stops seen

    def stop_wear_pct(self) -> tuple[float, float]:
        """(mean, sd) wear at which cars box, prior blended with what has been seen."""
        k, n = PRIOR_PSEUDO_STOPS, len(self.stop_wear)
        mean = (STOP_WEAR_PRIOR_PCT * k + sum(self.stop_wear)) / (k + n)
        ss = STOP_WEAR_PRIOR_SD ** 2 * k + sum((w - mean) ** 2 for w in self.stop_wear)
        return mean, max(3.0, math.sqrt(ss / (k + n)))

    def life_factor(self) -> tuple[float, float]:
        """(mean, sd) of how long cars run compared with the plan (1.0 = as planned)."""
        k, n = PRIOR_PSEUDO_STOPS, len(self.life_ratio)
        mean = (k + sum(self.life_ratio)) / (k + n)
        ss = LIFE_FACTOR_PRIOR_SD ** 2 * k + sum((r - mean) ** 2 for r in self.life_ratio)
        return mean, math.sqrt(ss / (k + n))

    def next_compound_counts(self, compound: str) -> dict:
        return {to: n for (frm, to), n in self.transitions.items() if frm == compound}

    def record_stop(self, wear_pct: float | None, life_ratio: float | None,
                    from_compound: str | None = None, to_compound: str | None = None):
        if from_compound in DRY and to_compound in DRY:
            key = (from_compound, to_compound)
            self.transitions[key] = self.transitions.get(key, 0) + 1
        if wear_pct is not None and 15.0 <= wear_pct <= 100.0:
            self.stop_wear.append(float(wear_pct))
        if life_ratio is not None and math.isfinite(life_ratio):
            self.life_ratio.append(min(2.0, max(0.4, float(life_ratio))))

    def as_dict(self) -> dict:
        wear, wear_sd = self.stop_wear_pct()
        factor, factor_sd = self.life_factor()
        return {'stops_seen': max(len(self.stop_wear), len(self.life_ratio)),
                'stop_wear_pct': round(wear, 1), 'stop_wear_sd': round(wear_sd, 1),
                'life_factor': round(factor, 3), 'life_factor_sd': round(factor_sd, 3),
                'transitions': {f'{a}>{b}': n for (a, b), n in sorted(self.transitions.items())}}


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

@dataclass
class RivalState:
    """Everything the predictor needs to know about one car."""
    car_index: int
    compound: str | None                 # SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    tyre_age: int
    current_lap: int
    actual_compound: str | None = None   # 'C3', 'INTER', ...
    name: str | None = None
    position: int | None = None
    lap_fraction: float = 0.0            # how far round the current lap (0-1)
    wear_pct: float | None = None        # worst tyre now; None when hidden
    wear_ages: list = field(default_factory=list)        # tyre age at the start of each lap...
    wear_increments: list = field(default_factory=list)  # ...and the wear added over it
    stops_done: int = 0
    compounds_used: tuple = ()


@dataclass
class RaceContext:
    track_id: int | None
    total_laps: int
    track_length: int | None = None
    pit_loss_s: float = race_strategy.DEFAULT_PIT_LOSS_S
    game_window: tuple | None = None     # (ideal, latest) the game gives the player
    player_start_compound: str | None = None
    player_stops: int = 0
    weather: int | None = None           # session weather code (0 clear .. 5 storm)


@dataclass
class RivalPrediction:
    car_index: int
    name: str | None
    position: int | None
    compound: str
    actual_compound: str | None
    tyre_age: int
    wear_pct: float | None
    stops_done: int
    current_lap: int
    total_laps: int
    will_stop: bool                      # another stop expected before the flag
    must_stop: bool                      # still owes the two-compound stop
    stop_lap: int | None                 # expected in-lap of the next stop
    stop_lap_range: list                 # [soonest, expected, latest] at LEVEL (latest None = after the flag)
    stop_lap_mean: float | None          # normal approximation, for sampling
    stop_lap_sd: float | None
    stop_before_flag_prob: float
    next_compound: str | None
    next_compound_probs: dict
    stops_left: int
    basis: str                           # 'wear' when the rival's own wear is used, else 'age'
    sources: dict                        # each estimate: {'plan': [mean, sd], 'wear': ..., 'game_window': ...}
    summary: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)

    def prob_stop_by(self, lap: int) -> float:
        """Chance the next stop comes at the end of ``lap`` or earlier."""
        if not self.will_stop or self.stop_lap_mean is None:
            return 0.0
        if self.must_stop and lap >= self.total_laps - 1:
            return 1.0
        return round(NormalDist(self.stop_lap_mean, max(self.stop_lap_sd or 0.0, 1e-6)).cdf(lap + 0.5), 3)

    def laps_to_stop(self, lap: int | None = None) -> int | None:
        if not self.will_stop or self.stop_lap is None:
            return None
        return self.stop_lap - (self.current_lap if lap is None else lap)


def _blend(estimates: list[tuple[float, float]]) -> tuple[float, float]:
    """Precision-weighted mean; the spread grows when the estimates disagree."""
    weights = [1.0 / (sd * sd) for _m, sd in estimates]
    total = sum(weights)
    mean = sum(w * m for w, (m, _sd) in zip(weights, estimates)) / total
    spread = sum(w * (m - mean) ** 2 for w, (m, _sd) in zip(weights, estimates)) / total
    return mean, max(MIN_SD, math.sqrt(1.0 / total + spread))


def _must_stop(compound: str, used) -> bool:
    used = {c for c in (race_strategy.normalise_compound(u) for u in used) if c} | {compound}
    return not (used & set(WET)) and len(used & set(DRY)) < 2


def first_stop_distribution(plan, tyre_age: float, current_lap: int, life_scale: float,
                            temperature_s: float = 0.0) -> tuple[float, float] | None:
    """(mean, sd) of the first stop lap over every lap that is nearly as good as the best.

    The cost of a strategy is often flat around its best stop lap (low degradation, short
    races), so the single best lap is a poor guess of what a team will do. Each possible
    first-stop lap is weighed by exp(-extra race time / temperature), keeping the rest of the
    plan's compounds and re-allocating their stints.
    """
    temperature_s = temperature_s or PLAN_TEMPERATURE_S
    best = plan.best
    if not best.stops:
        return None
    curves = [race_strategy.tyre_curve(c, 1.0, life_scale) for c in best.compounds]
    first, rest = curves[0], curves[1:]
    laps_remaining = plan.laps_remaining
    costs = {}
    cost_first = 0.0
    for n in range(1, laps_remaining - len(rest) + 1):
        age = tyre_age + n - 1
        if age + 1 > first.max_age:
            break
        cost_first += first.lap_cost(age)
        allocation = race_strategy._allocate(rest, [0.0] * len(rest), laps_remaining - n)
        if allocation is not None:
            costs[current_lap - 1 + n] = cost_first + allocation[1]
    if not costs:
        return float(best.pit_laps[0]), PLAN_SD_MIN
    low = min(costs.values())
    weights = {lap: math.exp(-(cost - low) / temperature_s) for lap, cost in costs.items()}
    total = sum(weights.values())
    mean = sum(lap * w for lap, w in weights.items()) / total
    var = sum(w * (lap - mean) ** 2 for lap, w in weights.items()) / total
    return mean, max(PLAN_SD_MIN, math.sqrt(var))


def _stop_cost(ctx: RaceContext) -> float:
    return ctx.pit_loss_s + position_cost_s(ctx.track_id)


def _plan_estimate(state: RivalState, ctx: RaceContext, compound: str, life_scale: float,
                   laps_remaining: int):
    plan = race_strategy.plan_race(laps_remaining, compound, state.tyre_age, state.current_lap,
                                   _stop_cost(ctx), state.compounds_used, 1.0, life_scale)
    if plan is None:
        return None, None
    if plan.best.stops:
        mean, sd = first_stop_distribution(plan, state.tyre_age, state.current_lap, life_scale)
        stint_age = state.tyre_age + max(0.0, mean - state.current_lap + 1)
        return plan, (mean, math.hypot(sd, PLAN_MODEL_REL_SD * stint_age))
    # Plan says no stop: the tyre would last about to the end of its life.
    sd = max(PLAN_SD_MIN, PLAN_SD_FRACTION * (state.tyre_age + plan.best.stint_laps[0]))
    life = race_strategy.tyre_curve(compound, 1.0, life_scale).life_laps
    end = state.current_lap - 1 + max(laps_remaining + 1.0, life - state.tyre_age)
    return plan, (end, sd)


def _wear_estimate(state: RivalState, ctx: RaceContext, fitted, field_stats: FieldStats):
    if (state.wear_pct is None or fitted is None or len(state.wear_increments) < MIN_WEAR_LAPS
            or fitted_models.compound_key(state.actual_compound) is None):
        return None
    model = fitted.wear_model(ctx.track_id, state.actual_compound, ctx.total_laps)
    stint = model.condition(state.wear_ages, state.wear_increments)
    threshold, threshold_sd = field_stats.stop_wear_pct()
    laps = stint.laps_until(threshold, state.tyre_age, state.wear_pct, LEVEL)
    if laps.mid is None:
        return None                     # not within the lookahead: no information
    z = fitted_models.z_score(LEVEL)
    sd_model = ((laps.high - laps.low) / (2 * z) if laps.low is not None and laps.high is not None
                else max(3.0, laps.mid * 0.3))
    rate = max(0.3, stint.rate(state.tyre_age, LEVEL).mid or 0.3)
    sd = math.hypot(sd_model, threshold_sd / rate)
    mean = state.current_lap + state.lap_fraction + laps.mid - 0.5
    return mean, max(MIN_SD, sd)


def _window_estimate(state: RivalState, ctx: RaceContext, compound: str):
    window = ctx.game_window
    if (not window or not window[0] or state.stops_done or ctx.player_stops
            or compound in WET or not ctx.player_start_compound):
        return None
    player = race_strategy.normalise_compound(ctx.player_start_compound)
    if player not in DRY:
        return None
    ratio = _compound_params(compound).expected_life_laps / _compound_params(player).expected_life_laps
    sd = GAME_WINDOW_SD if ratio == 1.0 else GAME_WINDOW_SD * 1.7
    return float(window[0]) * ratio, sd


def _compound_choice(state: RivalState, ctx: RaceContext, compound: str, stop_lap: int,
                     life_scale: float, typical: str | None,
                     field_counts: dict | None = None) -> tuple[dict, int]:
    """Probability of each next set, and the stops left after this one on the likeliest.

    Three equal-weight views of a dry stop, then what the field has actually fitted:
      * cost: race time for the rest of the race on each set (softmax over the difference),
      * habit: the usual next tyre (soft to medium, medium to hard, hard to medium),
      * track: the next set in this track's typical strategy, when there is one.
    Rain overrides all of it: slicks in the rain go to intermediates (or full wets in heavy rain).
    """
    raining = ctx.weather is not None and ctx.weather >= RAIN_WEATHER
    if compound not in WET and raining:
        heavy = ctx.weather >= HEAVY_RAIN_WEATHER
        return ({'INTERMEDIATE': 0.7, 'WET': 0.3} if heavy else {'INTERMEDIATE': 0.9, 'WET': 0.1}), 0
    if compound in WET and (ctx.weather is None or raining):
        return {compound: 1.0}, 0
    laps_after = ctx.total_laps - stop_lap
    if laps_after <= 0:
        return {race_strategy.next_compound(compound): 1.0}, 0
    used = {c for c in state.compounds_used if c in DRY} | ({compound} if compound in DRY else set())
    if compound in WET:
        used |= set(WET)                    # back to slicks: the two-compound rule is waived
    costs, stops_after = {}, {}
    for option in DRY:
        plan = race_strategy.plan_race(laps_after, option, 0, stop_lap + 1, _stop_cost(ctx),
                                       used | {option}, 1.0, life_scale)
        if plan is not None:
            costs[option] = plan.best.total_time_s
            stops_after[option] = plan.best.stops
    if not costs:
        return {race_strategy.next_compound(compound): 1.0}, 0
    best = min(costs.values())
    raw = {c: math.exp(-(cost - best) / COMPOUND_TEMPERATURE_S) for c, cost in costs.items()}
    views = [{c: v / sum(raw.values()) for c, v in raw.items()}]
    habit = race_strategy.next_compound(compound) if compound in DRY else 'MEDIUM'
    for choice in (habit, typical):
        if choice in costs:
            views.append({c: float(c == choice) for c in costs})
    probs = {c: sum(v[c] for v in views) / len(views) for c in costs}
    # What the field has actually fitted off this compound so far (only sets still allowed).
    counts = {c: n for c, n in (field_counts or {}).items() if c in probs}
    seen = sum(counts.values())
    if seen:
        probs = {c: (TRANSITION_PSEUDO * p + counts.get(c, 0)) / (TRANSITION_PSEUDO + seen)
                 for c, p in probs.items()}
    probs = {c: round(p, 3) for c, p in sorted(probs.items(), key=lambda kv: -kv[1])}
    likeliest = next(iter(probs))
    return probs, stops_after[likeliest]


def _weather_estimate(state: RivalState, ctx: RaceContext, compound: str):
    """Slicks in the rain (or wets on a dry track) bring the stop forward: the crossover."""
    if ctx.weather is None:
        return None
    raining = ctx.weather >= RAIN_WEATHER
    if compound not in WET and raining:
        due = 0.0 if ctx.weather >= HEAVY_RAIN_WEATHER else 2.0
        return state.current_lap + due, 1.0 if due == 0 else 2.0
    if compound in WET and ctx.weather <= DRY_WEATHER_MAX:
        return state.current_lap + 2.0, 2.0
    return None


def predict_rival(state: RivalState, ctx: RaceContext, fitted=None,
                  field_stats: FieldStats | None = None,
                  typical: list[str] | None = None) -> RivalPrediction | None:
    """Next stop lap and compound for one rival, or None when there is nothing to predict."""
    compound = race_strategy.normalise_compound(state.compound)
    laps_remaining = ctx.total_laps - state.current_lap + 1
    if compound is None or ctx.total_laps <= 0 or laps_remaining <= 0:
        return None
    field_stats = field_stats or FieldStats()
    if typical is None:
        typical = typical_sequence(ctx.track_id, ctx.track_length, ctx.total_laps)
    factor, _factor_sd = field_stats.life_factor()
    life_scale = track_life_scale(ctx.track_id, ctx.track_length)
    must_stop = _must_stop(compound, state.compounds_used)

    sources = {}
    plan, plan_est = _plan_estimate(state, ctx, compound, life_scale, laps_remaining)
    if plan_est:
        # Cars in this race have been running their sets ``factor`` times as long as planned.
        planned_age = state.tyre_age + max(0.0, plan_est[0] - state.current_lap)
        mean = state.current_lap + planned_age * factor - state.tyre_age
        sources['plan'] = (mean, plan_est[1] * max(1.0, factor))
    wear_est = _wear_estimate(state, ctx, fitted, field_stats)
    if wear_est:
        sources['wear'] = wear_est
        if 'plan' in sources:
            sources['plan'] = (sources['plan'][0], sources['plan'][1] * PLAN_SD_WITH_WEAR)
    window_est = _window_estimate(state, ctx, compound)
    if window_est:
        sources['game_window'] = window_est
    if not sources:
        return None
    mean, sd = _blend(list(sources.values()))
    weather_est = _weather_estimate(state, ctx, compound)
    if weather_est:
        # A tyre that is wrong for the conditions overrides the stint maths.
        sources['weather'] = weather_est
        mean, sd = min(mean, weather_est[0]), weather_est[1]
    if mean < state.current_lap:          # should have stopped already: expect it any moment
        mean, sd = float(state.current_lap), min(sd, OVERDUE_SD)
    flag = ctx.total_laps + 0.5
    if must_stop:
        # The two-compound rule forces a stop before the flag.
        last = max(state.current_lap, ctx.total_laps - 1)
        mean = min(mean, float(last))
        before_flag = 1.0
    else:
        before_flag = NormalDist(mean, sd).cdf(flag)
    will_stop = must_stop or (mean <= flag and laps_remaining > 1)

    z = fitted_models.z_score(LEVEL)
    stop_lap, stop_range, next_set, probs, stops_left = None, [None, None, None], None, {}, 0
    if will_stop:
        stop_lap = int(min(ctx.total_laps, max(state.current_lap, round(mean))))
        low = int(min(stop_lap, max(state.current_lap, round(mean - z * sd))))
        high = round(mean + z * sd)
        high = None if high > ctx.total_laps and not must_stop else int(max(stop_lap, min(high, ctx.total_laps)))
        stop_range = [low, stop_lap, high]
        stops_done = state.stops_done
        typical_set = typical_next(typical or [], compound, stops_done)
        probs, stops_after = _compound_choice(state, ctx, compound, stop_lap, life_scale, typical_set,
                                              field_stats.next_compound_counts(compound))
        next_set = next(iter(probs))
        stops_left = 1 + stops_after

    prediction = RivalPrediction(
        car_index=state.car_index, name=state.name, position=state.position, compound=compound,
        actual_compound=state.actual_compound, tyre_age=state.tyre_age,
        wear_pct=None if state.wear_pct is None else round(state.wear_pct, 1),
        stops_done=state.stops_done, current_lap=state.current_lap, total_laps=ctx.total_laps,
        will_stop=will_stop, must_stop=must_stop, stop_lap=stop_lap, stop_lap_range=stop_range,
        stop_lap_mean=round(mean, 2), stop_lap_sd=round(sd, 2),
        stop_before_flag_prob=round(before_flag, 3), next_compound=next_set,
        next_compound_probs=probs, stops_left=stops_left,
        basis='wear' if 'wear' in sources else 'age',
        sources={k: [round(m, 2), round(s, 2)] for k, (m, s) in sources.items()}, summary='')
    prediction.summary = describe(prediction)
    return prediction


def describe(p: RivalPrediction) -> str:
    """Short radio line, e.g. 'Hamilton likely to box lap 22, onto hards.'"""
    who = p.name or (f'Car in P{p.position}' if p.position else 'Rival')
    if not p.will_stop:
        return f'{who} looks to be going to the end on {p.compound.lower()}s.'
    soonest, mid, latest = p.stop_lap_range
    line = f'{who} likely to box lap {mid}'
    if p.next_compound:
        line += f', onto {p.next_compound.lower()}s'
    line += '.'
    if latest is None:
        line += ' Could run to the end.'
    elif latest - soonest >= 3:
        line += f' Window lap {soonest} to {latest}.'
    return line


def sample_stop_lap(p: RivalPrediction, rng: random.Random | None = None) -> int | None:
    """One draw of the rival's next stop lap (None = no stop before the flag)."""
    if p.stop_lap_mean is None or p.stop_lap_sd is None:
        return None
    rng = rng or random
    lap = max(p.current_lap, round(rng.gauss(p.stop_lap_mean, p.stop_lap_sd)))
    if lap > p.total_laps:
        return max(p.current_lap, p.total_laps - 1) if p.must_stop else None
    return int(lap)


def sample_next_compound(p: RivalPrediction, rng: random.Random | None = None) -> str | None:
    if not p.next_compound_probs:
        return p.next_compound
    rng = rng or random
    x, total = rng.random(), 0.0
    for compound, prob in p.next_compound_probs.items():
        total += prob
        if x <= total:
            return compound
    return next(reversed(p.next_compound_probs))


# ──────────────────────────────────────────────
# Collecting each car's stints from the packets
# ──────────────────────────────────────────────

@dataclass
class Stint:
    compound: str
    actual: str | None
    start_lap: int
    start_age: int
    observations: dict = field(default_factory=dict)    # lap -> (tyre age, worst wear or None)
    planned_age: float | None = None                    # stop age the first prediction expected
    planned_factor: float = 1.0                         # field life factor that prediction used
    end_lap: int | None = None
    age_at_stop: int | None = None
    wear_at_stop: float | None = None


@dataclass
class CarTrack:
    stints: list = field(default_factory=list)
    lap_num: int = 0
    lap_distance: float = 0.0
    position: int = 0
    stops: int = 0
    in_pits: bool = False
    pit_entry_lap: int | None = None
    active: bool = False
    age: int | None = None
    wear: float | None = None              # worst tyre now (None until seen)
    wear_hidden: bool = False

    @property
    def stint(self) -> Stint | None:
        return self.stints[-1] if self.stints else None

    def compounds_used(self) -> tuple:
        return tuple(sorted({s.compound for s in self.stints}))


class RivalTracker:
    """Per-car stints, wear and stops for every car, built from Lap Data, Status and Damage."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.cars: dict[int, CarTrack] = {}
        self.player_index: int | None = None
        self.track_length: int | None = None
        self.field = FieldStats()
        self.stops_log: list[dict] = []     # every stop seen: car, lap, from/to compound, age, wear
        self._cache: dict[int, tuple] = {}

    def _car(self, index) -> CarTrack:
        return self.cars.setdefault(index, CarTrack())

    def update_laps(self, laps, player_index: int | None = None):
        self.player_index = player_index
        for index, lap in enumerate(laps):
            car = self._car(index)
            car.active = lap.result_status == 2 and lap.car_position > 0
            if not car.active:
                continue
            car.lap_num, car.position, car.stops = lap.current_lap_num, lap.car_position, lap.num_pit_stops
            car.lap_distance = lap.lap_distance
            in_pits = lap.pit_status in (1, 2)
            if in_pits and not car.in_pits:
                car.pit_entry_lap = lap.current_lap_num
            car.in_pits = in_pits
            stint = car.stint
            if (stint is not None and not in_pits and car.age is not None
                    and lap.lap_distance >= SNAPSHOT_DISTANCE_M and lap.current_lap_num not in stint.observations):
                stint.observations[lap.current_lap_num] = (car.age, None if car.wear_hidden else car.wear)

    def update_status(self, all_status):
        for index, status in enumerate(all_status or ()):
            compound = race_strategy.normalise_compound(status.compound_name)
            if compound is None:
                continue
            actual = fitted_models.compound_key(status.actual_tyre_compound)
            car = self._car(index)
            age = int(status.tyres_age_laps)
            stint = car.stint
            if stint is None:
                car.stints.append(Stint(compound, actual, max(1, car.lap_num), age))
            elif compound != stint.compound or (car.age is not None and age < car.age):
                self._close_stint(index, car, compound, actual, age)
            car.age = age

    def update_damage(self, all_damage):
        for index, damage in enumerate(all_damage or ()):
            if not all(math.isfinite(w) and 0.0 <= w <= 100.0 for w in damage.tyres_wear):
                continue
            car = self._car(index)
            wear = max(damage.tyres_wear)
            car.wear = wear
            car.wear_hidden = wear <= 0.0 and (car.age or 0) >= HIDDEN_WEAR_AGE

    def _close_stint(self, index, car: CarTrack, compound, actual, age):
        old = car.stint
        old.end_lap = car.pit_entry_lap or max(old.start_lap, car.lap_num - 1)
        old.age_at_stop = car.age
        wears = [w for _a, w in old.observations.values() if w is not None]
        old.wear_at_stop = max(wears) if wears else None
        # Stint length against the first prediction, expressed against an unscaled plan.
        ratio = (old.age_at_stop / old.planned_age * old.planned_factor
                 if old.planned_age and old.age_at_stop is not None and old.planned_age > 0 else None)
        if index != self.player_index:
            self.field.record_stop(old.wear_at_stop, ratio, old.compound, compound)
            self.stops_log.append({'car_index': index, 'lap': old.end_lap, 'from': old.compound,
                                   'to': compound, 'age': old.age_at_stop, 'wear': old.wear_at_stop})
        car.pit_entry_lap = None
        car.stints.append(Stint(compound, actual, max(1, car.lap_num), age))

    def state(self, index, name=None, track_length=None) -> RivalState | None:
        car = self.cars.get(index)
        if car is None or not car.active or car.stint is None or car.age is None:
            return None
        stint = car.stint
        laps = sorted(stint.observations)
        ages, incs = [], []
        for a, b in zip(laps, laps[1:]):
            (age_a, wear_a), (_age_b, wear_b) = stint.observations[a], stint.observations[b]
            if b == a + 1 and wear_a is not None and wear_b is not None and wear_b >= wear_a:
                ages.append(age_a)
                incs.append(wear_b - wear_a)
        fraction = 0.0
        if track_length:
            fraction = min(1.0, max(0.0, car.lap_distance / track_length))
        return RivalState(
            car_index=index, compound=stint.compound, tyre_age=car.age, current_lap=max(1, car.lap_num),
            actual_compound=stint.actual, name=name, position=car.position, lap_fraction=fraction,
            wear_pct=None if car.wear_hidden or car.wear is None else car.wear,
            wear_ages=ages, wear_increments=incs, stops_done=len(car.stints) - 1,
            compounds_used=car.compounds_used())

    def predict(self, index, ctx: RaceContext, fitted=None, name=None) -> RivalPrediction | None:
        """Prediction for one car, cached until its lap, stint or observations change."""
        car = self.cars.get(index)
        if car is None or not car.active or car.in_pits or car.stint is None:
            return None
        key = (car.lap_num, len(car.stints), len(car.stint.observations), car.age, car.stops,
               name, ctx.total_laps, ctx.track_id, ctx.game_window, ctx.player_stops,
               len(self.field.stop_wear), len(self.field.life_ratio))
        cached = self._cache.get(index)
        if cached and cached[0] == key:
            return cached[1]
        state = self.state(index, name, ctx.track_length)
        prediction = None
        if state is not None:
            typical = typical_sequence(ctx.track_id, ctx.track_length, ctx.total_laps)
            prediction = predict_rival(state, ctx, fitted, self.field, typical)
        if prediction is not None and car.stint.planned_age is None and prediction.will_stop:
            # Tyre age only ticks over at the line, so on the in-lap it is age + laps until then.
            car.stint.planned_age = state.tyre_age + (prediction.stop_lap - state.current_lap)
            car.stint.planned_factor = self.field.life_factor()[0]
        self._cache[index] = (key, prediction)
        return prediction

    def last_prediction(self, index) -> RivalPrediction | None:
        """The most recent prediction made for a car (kept while it is in the pit lane)."""
        cached = self._cache.get(index)
        return cached[1] if cached else None

    def predict_all(self, ctx: RaceContext, fitted=None, names: dict | None = None,
                    exclude: int | None = None) -> list[RivalPrediction]:
        names = names or {}
        out = []
        for index in sorted(self.cars):
            if index == exclude:
                continue
            prediction = self.predict(index, ctx, fitted, names.get(index))
            if prediction is not None:
                out.append(prediction)
        out.sort(key=lambda p: p.position or 99)
        return out
