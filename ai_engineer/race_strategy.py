"""Race strategy: number of stops, compound choice, undercut and overcut calls.

Builds on the two tyre models instead of replacing them:
  * physics_models gives each compound's degradation curve and expected life, so a plan
    exists from lap one;
  * strategy_ml (observed wear and fuel-corrected pace loss) recalibrates that curve to how
    the player's car is actually using its tyres once a few laps have been logged.

Everything here is a pure function of numbers passed in, so it can be tested without the
game. Lap times are modelled as ``compound pace offset + degradation(tyre age)``; anything
common to every plan (fuel burn, driver pace) cancels out when plans are compared.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import asdict, dataclass, field

from physics_models import _compound_params
from track_data import resolve_track_id

DRY_COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')
WET_COMPOUNDS = ('INTERMEDIATE', 'WET')

# Fresh-tyre pace gap to the soft, s/lap (typical F1 2021 step between allocated compounds).
COMPOUND_PACE_OFFSET_S = {'SOFT': 0.0, 'MEDIUM': 0.35, 'HARD': 0.7}
# Extra time on the out-lap while a cold new set comes up to temperature.
OUT_LAP_WARMUP_S = 0.6
# Past its expected life a tyre falls off a cliff: extra s/lap = CLIFF_RATE * laps_past_life^2.
CLIFF_RATE = 0.05
# Hard limit: no stint is planned longer than this multiple of the compound's life.
MAX_LIFE_MULTIPLE = 1.6
MAX_STOPS = 3
# An undercut/overcut has to beat the gap by this much to be worth calling (pit-stop variance).
CALL_MARGIN_S = 0.5
# Only call an undercut when the planned stop is this close, so it never brings a stop forward
# by half a stint.
UNDERCUT_WINDOW_LAPS = 4

# Approximate pit-lane time loss (pit entry to exit versus staying on track), seconds.
DEFAULT_PIT_LOSS_S = 21.0
PIT_LOSS_S = {
    0: 21.0, 1: 22.0, 2: 21.5, 3: 22.5, 4: 21.5, 5: 19.5, 6: 18.5, 7: 20.5, 8: 17.5,
    9: 20.5, 10: 18.5, 11: 23.5, 12: 28.0, 13: 22.5, 14: 21.0, 15: 21.0, 16: 21.5,
    17: 19.5, 18: 26.0, 19: 22.0, 20: 20.0, 25: 20.5, 26: 18.0, 27: 27.5, 28: 25.0, 29: 19.5,
}


def pit_loss_s(track_id: int | None) -> float:
    if track_id is None or track_id < 0:
        return DEFAULT_PIT_LOSS_S
    return PIT_LOSS_S.get(resolve_track_id(track_id), DEFAULT_PIT_LOSS_S)


def normalise_compound(name: str | None) -> str | None:
    """Map the game's compound names ('Soft', 'C3', 'Inter', ...) onto SOFT/MEDIUM/HARD/INTERMEDIATE/WET."""
    if not name:
        return None
    key = name.strip().upper()
    if key.startswith('INTER'):
        return 'INTERMEDIATE'
    if key in DRY_COMPOUNDS or key == 'WET':
        return key
    # Actual C-compound without a visual name: the two softest are softs, and so on.
    return {'C5': 'SOFT', 'C4': 'SOFT', 'C3': 'MEDIUM', 'C2': 'HARD', 'C1': 'HARD', 'C0': 'HARD'}.get(key)


@dataclass
class TyreCurve:
    """Lap-time model for one compound: offset + scaled degradation + cliff past its life."""
    compound: str
    offset_s: float
    alpha: float
    beta: float
    life_laps: float

    def lap_cost(self, age: float, out_lap: bool = False) -> float:
        past = max(0.0, age - self.life_laps)
        cost = self.offset_s + self.alpha * age + self.beta * age * age + CLIFF_RATE * past * past
        return cost + (OUT_LAP_WARMUP_S if out_lap else 0.0)

    @property
    def max_age(self) -> float:
        return self.life_laps * MAX_LIFE_MULTIPLE


def tyre_curve(compound: str, deg_scale: float = 1.0, life_scale: float = 1.0) -> TyreCurve:
    params = _compound_params(compound)
    return TyreCurve(compound, COMPOUND_PACE_OFFSET_S.get(compound, 0.35),
                     params.alpha * deg_scale, params.beta * deg_scale,
                     params.expected_life_laps * life_scale)


def calibration(compound: str | None, tyre_age: float, observed_deg_s_per_lap: float | None = None,
                observed_laps_to_limit: float | None = None) -> tuple[float, float]:
    """(deg_scale, life_scale) that bend the physics curves towards what strategy_ml has seen.

    ``observed_deg_s_per_lap`` is strategy_ml.pace_degradation (fuel-corrected lap time lost
    per lap of tyre age); ``observed_laps_to_limit`` is the ML prediction of laps until the
    worst tyre reaches the wear limit. Both are optional; with neither the scale is 1.0.
    """
    compound = normalise_compound(compound)
    if compound not in DRY_COMPOUNDS:
        return 1.0, 1.0
    params = _compound_params(compound)
    deg_scale = life_scale = 1.0
    if observed_deg_s_per_lap is not None and observed_deg_s_per_lap > 0:
        model_rate = params.alpha + params.beta * max(tyre_age, 1.0)   # mean slope over the stint so far
        deg_scale = min(3.0, max(0.5, observed_deg_s_per_lap / model_rate))
    if observed_laps_to_limit is not None and observed_laps_to_limit >= 0:
        observed_life = tyre_age + observed_laps_to_limit
        life_scale = min(2.0, max(0.5, observed_life / params.expected_life_laps))
    return round(deg_scale, 3), round(life_scale, 3)


# ──────────────────────────────────────────────
# Stop-count and compound planning
# ──────────────────────────────────────────────

@dataclass
class StrategyOption:
    stops: int
    compounds: list[str]          # current set first, then each new set in order
    stint_laps: list[int]         # laps on each set, including the lap in progress
    pit_laps: list[int]           # race lap at the end of which to box, one per stop
    total_time_s: float           # modelled tyre + pit cost; only meaningful relative to other options

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class StrategyPlan:
    best: StrategyOption
    alternatives: list[StrategyOption] = field(default_factory=list)   # best plan for each other stop count
    pit_loss_s: float = DEFAULT_PIT_LOSS_S
    laps_remaining: int = 0

    def margin_to(self, stops: int) -> float | None:
        """How much slower the best plan with ``stops`` stops is, in seconds."""
        for option in self.alternatives:
            if option.stops == stops:
                return round(option.total_time_s - self.best.total_time_s, 1)
        return 0.0 if self.best.stops == stops else None

    def as_dict(self) -> dict:
        return {'best': self.best.as_dict(),
                'alternatives': [{'stops': o.stops, 'compounds': o.compounds,
                                  'pit_laps': o.pit_laps,
                                  'slower_by_s': round(o.total_time_s - self.best.total_time_s, 1)}
                                 for o in self.alternatives],
                'pit_loss_s': self.pit_loss_s, 'laps_remaining': self.laps_remaining}


def _allocate(curves: list[TyreCurve], start_ages: list[float], laps: int) -> tuple[list[int], float] | None:
    """Split ``laps`` across stints to minimise total cost.

    Each stint's cost is convex in its length (a lap added later on a set is never cheaper than
    the one before it), so handing out laps one at a time to whichever stint is cheapest to extend
    is optimal. Every stint gets at least one lap (the current set always gets the lap in
    progress); no stint may run a set past its compound's age limit. None if that is impossible.
    """
    n = len(curves)
    if laps < n:
        return None
    lengths = [1] * n
    total = sum(curve.lap_cost(start_ages[i], out_lap=i > 0) for i, curve in enumerate(curves))
    heap: list[tuple[float, int]] = []

    def offer(i: int):
        age = start_ages[i] + lengths[i]          # age at the start of the lap being added
        if age + 1 <= curves[i].max_age:
            heapq.heappush(heap, (curves[i].lap_cost(age), i))

    for i in range(n):
        offer(i)
    for _ in range(laps - n):
        if not heap:
            return None
        cost, i = heapq.heappop(heap)
        total += cost
        lengths[i] += 1
        offer(i)
    return lengths, total


def plan_race(laps_remaining: int, current_compound: str | None, tyre_age: float, current_lap: int,
              pit_loss: float = DEFAULT_PIT_LOSS_S, compounds_used: set[str] | tuple = (),
              deg_scale: float = 1.0, life_scale: float = 1.0,
              max_stops: int = MAX_STOPS) -> StrategyPlan | None:
    """Best number of stops and compound sequence for the rest of a dry race.

    ``laps_remaining`` counts the lap in progress. Enforces the dry-race rule that two different
    dry compounds must be used (waived once a wet or intermediate set has been run). Returns None
    for a wet race or when there is nothing left to plan.
    """
    current = normalise_compound(current_compound) or 'MEDIUM'
    if laps_remaining <= 0 or current in WET_COMPOUNDS:
        return None
    used = {c for c in (normalise_compound(u) for u in compounds_used) if c} | {current}
    rule_waived = bool(used & set(WET_COMPOUNDS))
    curves = {c: tyre_curve(c, deg_scale, life_scale) for c in DRY_COMPOUNDS}

    best_by_stops: dict[int, StrategyOption] = {}
    for stops in range(0, max_stops + 1):
        if stops >= laps_remaining:
            break
        for sequence in itertools.product(DRY_COMPOUNDS, repeat=stops):
            if not rule_waived and len(used | set(sequence)) < 2:
                continue
            stint_curves = [curves[current]] + [curves[c] for c in sequence]
            allocation = _allocate(stint_curves, [float(tyre_age)] + [0.0] * stops, laps_remaining)
            if allocation is None:
                continue
            lengths, tyre_time = allocation
            total = tyre_time + stops * pit_loss
            pit_laps, lap = [], current_lap - 1
            for length in lengths[:-1]:
                lap += length
                pit_laps.append(lap)
            option = StrategyOption(stops, [current, *sequence], lengths, pit_laps, round(total, 2))
            if stops not in best_by_stops or option.total_time_s < best_by_stops[stops].total_time_s:
                best_by_stops[stops] = option
    if not best_by_stops:
        return None
    best = min(best_by_stops.values(), key=lambda o: o.total_time_s)
    others = sorted((o for o in best_by_stops.values() if o is not best), key=lambda o: o.stops)
    return StrategyPlan(best, others, pit_loss, laps_remaining)


# ──────────────────────────────────────────────
# Undercut / overcut
# ──────────────────────────────────────────────

def undercut_gain(stayer_compound: str | None, stayer_age: float, new_compound: str | None = None,
                  laps_before_cover: int = 1, deg_scale: float = 1.0, life_scale: float = 1.0) -> float:
    """Time the car that pits first gains on the car that stays out, through the second car's out-lap.

    The pitter runs ``laps_before_cover`` laps on new tyres (the first one cold) while the stayer
    runs on its worn set; then the stayer pits and does its own cold out-lap while the pitter is
    on slightly used tyres. Both in-laps are on old tyres, so they cancel. Positive = the early
    stop gains time. ``new_compound`` (assumed the same for both cars) defaults to the stayer's
    next sensible set.
    """
    stayer = tyre_curve(normalise_compound(stayer_compound) or 'MEDIUM', deg_scale, life_scale)
    new = tyre_curve(normalise_compound(new_compound) or next_compound(stayer.compound), deg_scale, life_scale)
    gain = 0.0
    for i in range(laps_before_cover):
        gain += stayer.lap_cost(stayer_age + i) - new.lap_cost(i, out_lap=i == 0)
    # Stayer's cold out-lap against the pitter's warmed-up set.
    gain += new.lap_cost(0, out_lap=True) - new.lap_cost(laps_before_cover)
    return round(gain, 2)


def next_compound(current: str | None) -> str:
    """The usual next dry set: off softs onto mediums, otherwise onto hards, off hards onto mediums."""
    return {'SOFT': 'MEDIUM', 'MEDIUM': 'HARD', 'HARD': 'MEDIUM'}.get(normalise_compound(current), 'MEDIUM')


@dataclass
class StrategyCall:
    kind: str            # 'undercut', 'overcut', 'cover', 'stay_out', 'plan'
    message: str          # short radio line
    gain_s: float | None = None
    gap_s: float | None = None
    car_index: int | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def undercut_opportunity(gap_ahead_s: float | None, my_compound: str | None, my_age: float,
                         rival_compound: str | None, rival_age: float | None, laps_to_planned_stop: int | None,
                         laps_remaining: int, deg_scale: float = 1.0, life_scale: float = 1.0,
                         rival_index: int | None = None) -> StrategyCall | None:
    """Should we box now to jump the car ahead? Returns an 'undercut' call or None."""
    if (gap_ahead_s is None or rival_age is None or laps_to_planned_stop is None
            or laps_to_planned_stop > UNDERCUT_WINDOW_LAPS or laps_remaining <= 3):
        return None
    my_next = next_compound(my_compound)
    gain = undercut_gain(rival_compound, rival_age, my_next, 1, deg_scale, life_scale)
    if gain - gap_ahead_s < CALL_MARGIN_S:
        return None
    return StrategyCall('undercut', f'Undercut is on the car ahead. Box this lap, box this lap. '
                                    f'Gain about {gain:.1f}, gap {gap_ahead_s:.1f}.',
                        gain, gap_ahead_s, rival_index)


def react_to_rival_stop(rival_side: str, gap_s: float, my_compound: str | None, my_age: float,
                        rival_old_compound: str | None, rival_new_compound: str | None, laps_remaining: int, my_stops_left: int,
                        deg_scale: float = 1.0, life_scale: float = 1.0,
                        rival_index: int | None = None) -> StrategyCall:
    """Radio call when a nearby rival boxes.

    ``rival_side`` is 'ahead' or 'behind'; ``gap_s`` is the absolute gap before they pitted.
    A car behind stopping is an undercut threat (cover it or stay out); a car ahead stopping
    opens the overcut (stay out and push) unless their fresh tyres are too strong.
    """
    gap = abs(gap_s)
    rival_new = normalise_compound(rival_new_compound) or next_compound(rival_old_compound)
    if my_stops_left <= 0 or laps_remaining <= 2:
        return StrategyCall('stay_out', f'Car {rival_side} has boxed. We are not stopping again, '
                                        'stay out and push to the flag.', None, gap, rival_index)
    if rival_side == 'behind':
        # They are the pitter; we would cover next lap.
        threat = undercut_gain(my_compound, my_age, rival_new, 1, deg_scale, life_scale)
        if threat >= gap - CALL_MARGIN_S:
            return StrategyCall('cover', f'Car behind has boxed, undercut threat. '
                                         f'Box this lap to cover, box box.', threat, gap, rival_index)
        return StrategyCall('stay_out', f'Car behind has boxed. Gap is {gap:.1f}, '
                                        'undercut will not work. Stay out.', threat, gap, rival_index)
    # Rival ahead pitted: we stay out one lap longer and they do a cold out-lap.
    overcut = -undercut_gain(my_compound, my_age, rival_new, 1, deg_scale, life_scale)
    if overcut >= gap + CALL_MARGIN_S:
        return StrategyCall('overcut', 'Car ahead has boxed. Overcut is on, push now, '
                                       'big lap please.', overcut, gap, rival_index)
    return StrategyCall('stay_out', f'Car ahead has boxed for {rival_new.lower()}s. '
                                    'Overcut is marginal, we stay on plan.', overcut, gap, rival_index)


def describe_plan(plan: StrategyPlan | None) -> str:
    """Short spoken summary of the best plan."""
    if plan is None:
        return 'No dry strategy to plan right now.'
    best = plan.best
    if best.stops == 0:
        return f'No more stops. Bring the {best.compounds[0].lower()}s home.'
    sets = ', then '.join(c.lower() for c in best.compounds[1:])
    stops = 'one stop' if best.stops == 1 else f'{best.stops} stops'
    line = f'Plan is {stops}. Box lap {best.pit_laps[0]} for {sets}.'
    nearest = min(plan.alternatives, key=lambda o: o.total_time_s, default=None)
    if nearest is not None:
        delta = nearest.total_time_s - best.total_time_s
        other = 'no stop' if nearest.stops == 0 else ('one stop' if nearest.stops == 1 else f'{nearest.stops} stops')
        line += f' {other.capitalize()} is {delta:.1f} slower.'
    return line
