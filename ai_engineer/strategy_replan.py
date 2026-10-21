"""Live strategy re-planning: Plan A, Plan B, and when to make the call.

The Monte Carlo simulator (race_sim.py) already re-runs once a lap. This module gives
those runs a memory:

* **Plan A** is the plan we are actually on. It is adopted from the first race plan,
  moved along as we make our stops, and only replaced when the engineer makes a call.
* Every re-run plays Plan A through the same simulated races as the alternatives (the
  simulator's common random numbers), so the comparison is paired: same Safety Cars,
  same slow stops, same rival form. The difference is the plan.
* A re-run happens every lap and straight away on the events that change a race: a
  Safety Car or VSC, rain arriving, or a nearby rival boxing.
* **Plan B** is called only when an alternative beats Plan A by a clear margin
  (``ReplanConfig``): enough seconds or places, in enough of the runs. On a routine lap
  re-run it also has to win two re-runs in a row and respect a cooldown, so the call
  does not flip-flop on Monte Carlo noise. After an event (SC, rain, rival stop) one run
  is enough, because those windows close within a lap.

Everything here is plain bookkeeping on numbers passed in, so it is testable without
the game; analysis.py wires it to the live telemetry and the radio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import race_sim
import race_strategy as rs

# ── The margin a Plan B must clear (tunable; main.py exposes the first two as flags) ──
REPLAN_MARGIN_S = 3.0            # median race time gained over Plan A, seconds ...
REPLAN_MARGIN_POSITIONS = 0.5    # ... or mean finishing places gained (either one will do)
REPLAN_MIN_CONFIDENCE = 0.6      # and Plan B must come out ahead in at least this share of runs
REPLAN_CONFIRM_RUNS = 2          # routine lap re-runs: Plan B must win this many in a row
REPLAN_COOLDOWN_LAPS = 3         # routine lap re-runs: no new Plan B call this soon after the last
# Two plans with the same tyres whose stops are this close (laps) are the same plan.
SAME_PLAN_LAPS = 2
# A rival's stop triggers a re-run when they are this many places either side of us.
RIVAL_TRIGGER_POSITIONS = 3
# Event re-runs come at most this often (seconds): a pit-stop cascade must not queue a
# simulation per car. A routine lap re-run is never held back by it.
MIN_EVENT_INTERVAL_S = 3.0
# Keep each re-run cheap: Plan A plus at most this many alternatives.
MAX_CANDIDATES = 5
# Past this many places the number stops meaning much on the radio (staying on slicks in the
# rain, say): the call says it costs a lot instead.
BIG_GAIN_PLACES = 4
# The HUD keeps showing a Plan B call for this many laps.
CALL_SHOWN_LAPS = 2

EVENT_REASONS = ('safety_car', 'vsc', 'rain', 'rival_stop')
WET_WEATHER = 3                  # session weather code from which the track is wet (light rain)


@dataclass
class ReplanConfig:
    margin_s: float = REPLAN_MARGIN_S
    margin_positions: float = REPLAN_MARGIN_POSITIONS
    min_confidence: float = REPLAN_MIN_CONFIDENCE
    confirm_runs: int = REPLAN_CONFIRM_RUNS
    cooldown_laps: int = REPLAN_COOLDOWN_LAPS


@dataclass
class PlanBCall:
    message: str
    plan: race_sim.Candidate          # the new Plan A
    previous: race_sim.Candidate
    lap: int
    reasons: list
    time_gain_s: float
    position_gain: float
    p_better: float

    def as_dict(self) -> dict:
        return {'kind': 'plan_b', 'message': self.message, 'lap': self.lap, 'reasons': list(self.reasons),
                'plan': plan_dict(self.plan), 'previous': plan_dict(self.previous),
                'time_gain_s': self.time_gain_s, 'position_gain': self.position_gain,
                'p_better': self.p_better}


def plan_dict(c: race_sim.Candidate | None) -> dict | None:
    return None if c is None else {'label': c.label, 'stops': c.stops, 'compounds': list(c.compounds),
                                   'pit_laps': list(c.pit_laps)}


def _plan_key(c: race_sim.Candidate) -> tuple:
    return tuple(c.compounds), tuple(c.pit_laps)


def same_plan(a: race_sim.Candidate, b: race_sim.Candidate, lap_num: int) -> bool:
    """Same tyres and stops within a couple of laps, unless one of them says box this lap."""
    if list(a.compounds) != list(b.compounds) or len(a.pit_laps) != len(b.pit_laps):
        return False
    if a.pit_laps and (a.pit_laps[0] == lap_num) != (b.pit_laps[0] == lap_num):
        return False
    return all(abs(x - y) <= SAME_PLAN_LAPS for x, y in zip(a.pit_laps, b.pit_laps))


def _candidate(compounds, pit_laps, suffix: str = '') -> race_sim.Candidate:
    compounds, pit_laps = list(compounds), list(pit_laps)
    label = race_sim.plan_label(len(pit_laps), compounds) + suffix
    return race_sim.Candidate(label, len(pit_laps), compounds, pit_laps)


def pit_now(c: race_sim.Candidate, lap_num: int) -> race_sim.Candidate | None:
    """The same plan with its next stop brought forward to the end of this lap."""
    if not c.pit_laps or c.pit_laps[0] <= lap_num:
        return None
    laps, prev = [], lap_num - 1
    for p in [lap_num, *c.pit_laps[1:]]:
        prev = max(p, prev + 1)
        laps.append(prev)
    return _candidate(c.compounds, laps, ', box now')


def _words(stops: int) -> str:
    return {0: 'no stop', 1: 'one stop', 2: 'two stops', 3: 'three stops'}.get(stops, f'{stops} stops')


def plan_b_message(new: race_sim.Candidate, old: race_sim.Candidate, lap_num: int, time_gain_s: float,
                   position_gain: float, reasons, rival_name: str | None, config: ReplanConfig) -> str:
    """Short radio line in the engineer's style: why, the call, what it is worth."""
    why = ''
    if 'safety_car' in reasons:
        why = 'Safety car. '
    elif 'vsc' in reasons:
        why = 'VSC. '
    elif 'rain' in reasons:
        why = 'Rain is here. '
    elif 'rival_stop' in reasons:
        why = f'{rival_name} has boxed. ' if rival_name else 'Car near us has boxed. '
    if not new.pit_laps:
        action = f'No more stops, bring the {new.compounds[0].lower()}s home.'
    else:
        first = new.pit_laps[0]
        when = 'this lap' if first <= lap_num else ('next lap' if first == lap_num + 1 else f'lap {first}')
        action = f'Box {when} for {new.compounds[1].lower()}s.'
        if new.stops != old.stops:
            action = f'We go {_words(new.stops)}. ' + action
        if first <= lap_num:
            action += ' Box, box.'
    places = int(round(position_gain))
    if position_gain >= config.margin_positions and places > BIG_GAIN_PLACES:
        worth = 'Staying out costs us a lot of places.' if new.pit_laps else 'Stopping again costs us a lot.'
    elif position_gain >= config.margin_positions and places >= 1:
        worth = f'Worth {places} place{"s" if places > 1 else ""}.'
    else:
        worth = f'Gains about {max(1, int(round(time_gain_s)))} seconds.'
    return f'{why}Plan B, Plan B. {action} {worth}'


class Replanner:
    """Keeps Plan A, spots re-plan triggers and decides when Plan B is worth a call."""

    def __init__(self, config: ReplanConfig | None = None):
        self.config = config or ReplanConfig()
        self.reset()

    def reset(self):
        self.plan_a: race_sim.Candidate | None = None
        self.stops_done: int | None = None
        self.seen_lap: int | None = None
        self.seen_sc = 0
        self.seen_wet: bool | None = None
        self.seen_stops: dict[int, int] = {}
        self.events = 0                     # bumps on every event trigger (part of the sim key)
        self.pending: list[str] = []        # reasons gathered since the last submitted re-run
        self.rival_name: str | None = None  # the last rival whose stop triggered a re-run
        self.submitted: dict = {}           # sim key -> (Plan A, reasons, rival name)
        self.last_result = None
        self.streak_key = None
        self.streak = 0
        self.last_call: PlanBCall | None = None
        self.challenger: dict | None = None  # best alternative in the last re-run, for the HUD
        self.status = 'waiting'              # waiting / holding / watching / cooldown / switched
        self.last_run: dict | None = None

    # ── triggers ──
    def observe(self, lap_num: int, safety_car: int, weather: int, field_laps=None,
                player_index: int = 0, name_of=None) -> list[str]:
        """Note what changed since the last frame; returns the new re-plan reasons."""
        reasons = []
        wet = weather >= WET_WEATHER
        sc = safety_car if safety_car in (1, 2) else 0
        if self.seen_lap is None:
            self.seen_lap, self.seen_sc, self.seen_wet = lap_num, sc, wet
        else:
            if lap_num > self.seen_lap:
                reasons.append('lap')
            if sc != self.seen_sc:
                reasons.append({1: 'safety_car', 2: 'vsc', 0: 'green_flag'}[sc])
            if wet != self.seen_wet:
                reasons.append('rain' if wet else 'drying')
            self.seen_lap, self.seen_sc, self.seen_wet = max(lap_num, self.seen_lap), sc, wet
        if field_laps:
            me = field_laps[player_index] if 0 <= player_index < len(field_laps) else None
            for index, car in enumerate(field_laps):
                if car is None or index == player_index or getattr(car, 'result_status', 2) != 2:
                    continue
                stops = car.num_pit_stops
                before = self.seen_stops.get(index)
                self.seen_stops[index] = stops
                if before is None or stops <= before or me is None:
                    continue
                if abs(car.car_position - me.car_position) <= RIVAL_TRIGGER_POSITIONS:
                    reasons.append('rival_stop')
                    self.rival_name = name_of(index) if name_of else None
        if any(r in EVENT_REASONS for r in reasons):
            self.events += 1
        for r in reasons:
            if r not in self.pending:
                self.pending.append(r)
        return reasons

    # ── Plan A ──
    def adopt(self, candidate: race_sim.Candidate, stops_done: int):
        self.plan_a = replace(candidate, compounds=list(candidate.compounds), pit_laps=list(candidate.pit_laps))
        self.stops_done = stops_done

    def sync_plan_a(self, plan: rs.StrategyPlan, lap_num: int, stops_done: int, compound: str | None,
                    laps_remaining: int) -> race_sim.Candidate | None:
        """Bring Plan A up to date with the race: stops made, a stop we skipped, our tyres."""
        if plan is None:
            return self.plan_a
        current = rs.normalise_compound(compound) or plan.best.compounds[0]
        if self.plan_a is None:
            self.adopt(race_sim.Candidate.from_option(plan.best), stops_done)
            return self.plan_a
        a = self.plan_a
        if stops_done > self.stops_done:
            made = stops_done - self.stops_done
            if a.stops >= made:
                a = _candidate([current, *a.compounds[made + 1:]], a.pit_laps[made:])
            else:                               # a stop the plan didn't have (damage, a gamble)
                a = race_sim.Candidate.from_option(plan.best)
            self.stops_done = stops_done
        elif stops_done < self.stops_done:      # flashback past a stop: start again
            a = race_sim.Candidate.from_option(plan.best)
            self.stops_done = stops_done
        if a.compounds and a.compounds[0] != current:
            a = _candidate([current, *a.compounds[1:]], a.pit_laps)
        # A stop we didn't make slides to this lap; nothing can be later than the last lap but one.
        last = lap_num + max(laps_remaining, 1) - 2
        laps, prev = [], lap_num - 1
        for p in a.pit_laps:
            prev = max(p, prev + 1)
            laps.append(prev)
        if laps != list(a.pit_laps):
            keep = [p for p in laps if p <= last]
            a = _candidate(a.compounds[:len(keep) + 1], keep)
        self.plan_a = a
        return a

    def candidates(self, plan: rs.StrategyPlan | None, lap_num: int, neutralised: bool, raining: bool,
                   compound: str | None) -> list[race_sim.Candidate]:
        """Plan A first, then the planner's options and the event-specific ones (box now, inters)."""
        if self.plan_a is None:
            return race_sim.candidates_from_plan(plan)
        out, seen = [self.plan_a], {_plan_key(self.plan_a)}

        def add(c):
            if c is not None and _plan_key(c) not in seen and len(out) < MAX_CANDIDATES + 1:
                seen.add(_plan_key(c))
                out.append(c)

        current = rs.normalise_compound(compound) or self.plan_a.compounds[0]
        if raining and current not in rs.WET_COMPOUNDS:
            add(_candidate([current, 'INTERMEDIATE'], [lap_num], ', box now'))
        cheap_stop = neutralised or 'rival_stop' in self.pending
        if cheap_stop and self.plan_a.pit_laps and \
                self.plan_a.pit_laps[0] - lap_num <= race_sim.SC_PIT_WINDOW_LAPS:
            add(pit_now(self.plan_a, lap_num))
        for c in race_sim.candidates_from_plan(plan):
            add(c)
            if neutralised and c.pit_laps and c.stops != self.plan_a.stops:
                add(pit_now(c, lap_num))
        return out

    def note_submitted(self, key):
        """A re-run went off with Plan A first: remember what it was asked."""
        self.submitted[key] = (self.plan_a, list(self.pending), self.rival_name)
        if len(self.submitted) > 8:
            self.submitted.pop(next(iter(self.submitted)))
        self.pending = []

    @property
    def has_event_pending(self) -> bool:
        return any(r in EVENT_REASONS for r in self.pending)

    # ── the decision ──
    def consider(self, result, key, lap_num: int) -> PlanBCall | None:
        """Look at a finished re-run; returns a Plan B call when one is due."""
        if result is None or result is self.last_result or key not in self.submitted:
            return None
        self.last_result = result
        plan_a, reasons, rival_name = self.submitted.pop(key)
        if plan_a is None or plan_a is not self.plan_a or lap_num - result.lap > 1:
            return None                  # Plan A moved on (a stop, a call) while it ran
        cfg = self.config
        ref = next((o for o in result.outcomes if o.is_reference), None)
        self.last_run = {'lap': result.lap, 'runs': result.runs, 'reasons': reasons}
        if ref is None:
            return None
        best, best_score = None, None
        for o in result.outcomes:
            vs = o.vs_reference
            if o.is_reference or not vs:
                continue
            c = race_sim.Candidate(o.label, o.stops, list(o.compounds), list(o.pit_laps))
            if same_plan(c, plan_a, lap_num):
                continue
            score = (vs['position_gain'], vs['time_gain_s'])
            if best_score is None or score > best_score:
                best, best_score = (o, c), score
        if best is None:
            self.challenger, self.status, self.streak_key, self.streak = None, 'holding', None, 0
            return None
        o, c = best
        vs = o.vs_reference
        clears = ((vs['time_gain_s'] >= cfg.margin_s or vs['position_gain'] >= cfg.margin_positions)
                  and vs['p_better'] >= cfg.min_confidence)
        self.challenger = {**plan_dict(c), **vs, 'clears_margin': clears}
        if not clears:
            self.status, self.streak_key, self.streak = 'holding', None, 0
            return None
        event = any(r in EVENT_REASONS for r in reasons)
        streak_key = (tuple(c.compounds), len(c.pit_laps))
        self.streak = self.streak + 1 if streak_key == self.streak_key else 1
        self.streak_key = streak_key
        if not event:
            if self.last_call is not None and lap_num - self.last_call.lap < cfg.cooldown_laps:
                self.status = 'cooldown'
                return None
            if self.streak < cfg.confirm_runs:
                self.status = 'watching'
                return None
        call = PlanBCall(plan_b_message(c, plan_a, lap_num, vs['time_gain_s'], vs['position_gain'], reasons,
                                        rival_name, cfg),
                         c, plan_a, lap_num, [r for r in reasons if r != 'lap'] or ['lap'],
                         vs['time_gain_s'], vs['position_gain'], vs['p_better'])
        self.adopt(c, self.stops_done if self.stops_done is not None else 0)
        self.last_call, self.status = call, 'switched'
        self.streak_key, self.streak, self.challenger = None, 0, None
        return call

    # ── for the HUD, the AI prompt and the strategy answer ──
    def summary(self, lap_num: int | None) -> dict | None:
        if self.plan_a is None:
            return None
        call = self.last_call
        active = call is not None and lap_num is not None and lap_num - call.lap < CALL_SHOWN_LAPS
        a = self.plan_a
        if a.pit_laps:
            line = f"Plan A: {_words(a.stops)}, box lap {a.pit_laps[0]} for {a.compounds[1].lower()}s."
        else:
            line = f"Plan A: no more stops, {a.compounds[0].lower()}s to the flag."
        return {'plan_a': plan_dict(a), 'status': self.status, 'challenger': self.challenger,
                'last_call': call.as_dict() if call else None, 'call_active': active,
                'last_run': self.last_run, 'radio': line,
                'margin': asdict(self.config)}
