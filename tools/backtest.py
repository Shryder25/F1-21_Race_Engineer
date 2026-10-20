"""Backtest the engineer's predictions against what actually happened in recorded races.

Each recording (the `.f1rec` format from `main.py --record`, see docs/udp-recording-format.md)
is replayed offline through the real engineer, the same way the golden-session tests do it
(tests/golden/harness.py: simulated clock, silent voice, throwaway history database). Once
per lap the engineer's predictions are written down; afterwards they are compared with what
the recording shows really happened:

  * pit lap     the lap the engineer expects the next stop on (ML wear model, race-strategy
                plan, and the game's own pit window as a baseline) vs the lap you pitted
  * tyre wear   worst-tyre wear the ML wear curve predicts 1, 3 and 5 laps ahead vs the wear
                the game reported then
  * lap time    the engineer's race pace (average of the last three clean laps) vs the lap
                you then drove
  * gaps        the live gap to the cars ahead and behind vs the true gap, and the gap 1 and
                3 laps later predicted from the last-lap pace difference vs the true gap then
  * rival stops every rival's predicted next stop lap and next compound (rival_strategy.py)
                vs the lap that car really pitted on and the tyre it fitted, against a
                plan-from-tyre-age-only baseline and the "usual next tyre" rule

The ground truth is read straight from the datagrams (every car's distance and time, pit
stops, lap times, tyre wear), independently of the engineer's own bookkeeping.

Usage:
    python tools/backtest.py recordings/*.f1rec.gz
    python tools/backtest.py recordings/*.f1rec.gz --out reports/backtest.md --json reports/backtest.json
    python tools/backtest.py --simulated            # the built-in simulated races only

Simulated recordings (the golden sessions and the longer backtest races generated here)
only prove the pipeline works: the simulator is simple and its "driver" pits on scripted
laps. Real error numbers need recordings of your own races.
"""

from __future__ import annotations

import argparse
import bisect
import contextlib
import json
import math
import statistics
import struct
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'golden'
for _path in (ROOT / 'ai_engineer', GOLDEN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import analysis as analysis_module                  # noqa: E402
import race_strategy                                # noqa: E402
import rival_strategy                               # noqa: E402
import harness                                      # noqa: E402
import recording                                    # noqa: E402
import telemetry_stream as stream_module            # noqa: E402
from analysis import AnalysisEngine                 # noqa: E402
from history import TelemetryHistory                # noqa: E402
from packets import (HEADER_SIZE, TRACK_NAMES, PacketHeader, parse_all_car_status,  # noqa: E402
                     parse_all_lap_data, parse_player_car_damage, parse_session_data)
from telemetry_stream import PACKET_SIZES, TelemetryStream   # noqa: E402

RACE_SESSION_TYPES = (10, 11, 12)
# Predictions are written down this far into each lap, once the car status for the new lap
# (tyre age) has arrived; the pit lane exit is about 250 m past the line.
SNAPSHOT_DISTANCE_M = 400.0
WEAR_HORIZONS = (1, 3, 5)
GAP_HORIZONS = (1, 3)
PIT_HORIZON_BUCKETS = ((0, 3, '0-3 laps before'), (4, 7, '4-7 laps before'), (8, 999, '8+ laps before'))

METRICS = {
    'pit_lap': ('Pit lap', 'laps'),
    'tyre_wear': ('Tyre wear (worst tyre)', '% points'),
    'lap_time': ('Lap time', 's'),
    'gap_now': ('Gap, live reading', 's'),
    'gap': ('Gap forecast', 's'),
    'rival_pit_lap': ('Rival next stop lap', 'laps'),
}
PREDICTORS = {
    'ml_ideal': 'Engineer fitted wear model, target lap (60% wear)',
    'ml_latest': 'Engineer fitted wear model, latest lap (75% wear)',
    'strategy': 'Engineer race-strategy plan, next stop',
    'game_window': "Game's own pit window, ideal lap (baseline)",
    'ml_curve': 'Engineer fitted wear model (expected value)',
    'last_lap_rate': "Last lap's wear rate carried on (baseline)",
    'race_pace': 'Engineer race pace (last 3 clean laps)',
    'last_lap': 'Previous lap time (baseline)',
    'engineer': 'Engineer',
    'pace_delta': 'Engineer gap + last-lap pace difference',
    'hold': 'Gap stays the same (baseline)',
    'rival_model': 'Rival stop model (wear + plan + game window, learns from the field)',
    'rival_plan_only': 'Rival race plan from tyre age only (baseline)',
    'rival_rule': 'Usual next tyre: soft to medium, medium to hard (baseline)',
}
RIVAL_PREDICTORS = ('rival_model', 'rival_plan_only')


# ──────────────────────────────────────────────
# Ground truth, straight from the datagrams
# ──────────────────────────────────────────────

class GroundTruth:
    """What really happened in one session, without going through the engineer."""

    def __init__(self, session_uid: int):
        self.session_uid = session_uid
        self.session = None
        self.player_index = None
        self.trajectories: dict[int, list[tuple[float, float]]] = defaultdict(list)
        self.positions: dict[int, float] = {}            # car -> latest total distance
        self.stop_counts: dict[int, int] = {}
        self.pit_status: dict[int, int] = {}
        self.stops: dict[int, list[int]] = defaultdict(list)   # car -> in-lap of each stop
        self.compounds: dict[int, list[str]] = defaultdict(list)  # car -> each set it ran, in order
        self._ages: dict[int, int] = {}
        self._pit_entry_lap: dict[int, int] = {}
        self.player_lap = None
        self.player_lap_distance = 0.0
        self.player_last_lap_ms = 0
        self.lap_times_ms: dict[int, int] = {}
        self.safety_car_laps: set[int] = set()
        self.worst_wear = None
        self.session_time = 0.0
        self.max_lap_seen = 0

    def on_session(self, session, player_index):
        self.session = session
        self.player_index = player_index
        if session.safety_car_status and self.player_lap:
            self.safety_car_laps.add(self.player_lap)

    def on_laps(self, laps, session_time, player_index):
        self.player_index = player_index
        self.session_time = session_time
        for index, lap in enumerate(laps):
            if lap.car_position == 0 or lap.result_status not in (2, 3):
                continue
            path = self.trajectories[index]
            distance = float(lap.total_distance)
            if not math.isfinite(distance):
                continue
            if path and distance < path[-1][0] - 50:
                # Flashback: the timeline was rewound, forget the future that never happened.
                cut = bisect.bisect_left(path, (distance, -math.inf))
                del path[cut:]
            if not path or (distance > path[-1][0] and session_time > path[-1][1]):
                path.append((distance, session_time))
            self.positions[index] = distance
            if lap.pit_status:
                self._pit_entry_lap.setdefault(index, lap.current_lap_num)
            previous = self.stop_counts.get(index)
            if previous is not None and lap.num_pit_stops > previous:
                self.stops[index].append(self._pit_entry_lap.get(index, lap.current_lap_num))
            if not lap.pit_status:
                self._pit_entry_lap.pop(index, None)
            self.stop_counts[index] = lap.num_pit_stops
            self.pit_status[index] = lap.pit_status

        me = laps[player_index]
        lap_num = me.current_lap_num
        if self.player_lap is not None and lap_num < self.player_lap:       # flashback
            for lap in [n for n in self.lap_times_ms if n >= lap_num]:
                del self.lap_times_ms[lap]
            self.stops[player_index] = [n for n in self.stops[player_index] if n < lap_num]
        elif self.player_lap is not None and lap_num == self.player_lap + 1 and me.last_lap_time_ms:
            self.lap_times_ms[self.player_lap] = me.last_lap_time_ms
        elif (self.player_lap == lap_num and me.result_status == 3 and me.last_lap_time_ms
              and me.last_lap_time_ms != self.player_last_lap_ms and lap_num not in self.lap_times_ms):
            self.lap_times_ms[lap_num] = me.last_lap_time_ms            # chequered flag lap
        self.player_lap = lap_num
        self.max_lap_seen = max(self.max_lap_seen, lap_num)
        self.player_lap_distance = me.lap_distance
        self.player_last_lap_ms = me.last_lap_time_ms
        if self.session and self.session.safety_car_status:
            self.safety_car_laps.add(lap_num)

    def on_damage(self, damage):
        self.worst_wear = max(damage.tyres_wear)

    def on_status(self, all_status):
        """Every car's tyre sets in order (a new set = new compound or the tyre age dropping)."""
        for index, status in enumerate(all_status):
            compound = race_strategy.normalise_compound(status.compound_name)
            if compound is None:
                continue
            sets, age = self.compounds[index], status.tyres_age_laps
            if not sets or sets[-1] != compound or age < self._ages.get(index, age):
                sets.append(compound)
            self._ages[index] = age

    @staticmethod
    def _crossing(path, distance):
        """Session time at which a car passed ``distance`` (interpolated), or None."""
        if not path or not path[0][0] <= distance <= path[-1][0]:
            return None
        i = bisect.bisect_left(path, (distance, -math.inf))
        if path[i][0] == distance or i == 0:
            return path[i][1]
        (d0, t0), (d1, t1) = path[i - 1], path[i]
        return t0 + (distance - d0) / (d1 - d0) * (t1 - t0)

    def gap(self, car: int) -> float | None:
        """True gap between the player and ``car`` now, in seconds: positive = car is ahead."""
        me = self.player_index
        if me is None or car not in self.positions or me not in self.positions or car == me:
            return None
        mine, theirs = self.positions[me], self.positions[car]
        if theirs >= mine:
            passed = self._crossing(self.trajectories[car], mine)
            return None if passed is None else self.session_time - passed
        passed = self._crossing(self.trajectories[me], theirs)
        return None if passed is None else -(self.session_time - passed)

    def all_gaps(self) -> dict[int, float]:
        gaps = {}
        for car in self.positions:
            value = self.gap(car)
            if value is not None:
                gaps[car] = round(value, 3)
        return gaps


# ──────────────────────────────────────────────
# Replay: snapshots of the engineer's predictions, once per lap
# ──────────────────────────────────────────────

def _pit_value(lap, total_laps):
    """A predicted pit lap, or 'none' when the prediction is that no (further) stop is needed."""
    if lap is None or (total_laps and lap > total_laps):
        return 'none'
    return int(lap)


def engineer_snapshot(engine: AnalysisEngine, truth: GroundTruth) -> dict:
    lap, status, session = engine.latest_lap, engine.latest_status, engine.session
    total = session.total_laps if session else 0
    snap = {
        'lap': lap.current_lap_num, 't': round(truth.session_time, 3),
        'stops_done': lap.num_pit_stops, 'pit_status': lap.pit_status,
        'compound': status.compound_name if status else None,
        'tyre_age': status.tyres_age_laps if status else None,
        'worst_wear': None if truth.worst_wear is None else round(truth.worst_wear, 2),
        'safety_car': bool(session and session.safety_car_status),
        'truth_gaps': truth.all_gaps(),
        'stop_counts': dict(truth.stop_counts),
        'pit_status_all': {car: s for car, s in truth.pit_status.items() if s},
    }
    pit = {}
    prediction = engine.pit_prediction()
    if prediction is not None:
        pit['ml_ideal'] = _pit_value(prediction.ideal_lap, total)
        pit['ml_latest'] = _pit_value(prediction.latest_lap, total)
    plan = engine.strategy_plan()
    if plan is not None:
        pit['strategy'] = plan.best.pit_laps[0] if plan.best.stops else 'none'
    window = engine.radio.pit_window
    if window:
        pit['game_window'] = _pit_value(window[0], total)
    snap['pit'] = pit

    wear, wear_range = {}, {}
    model = engine.wear_forecast() if status else None
    if model is not None and status is not None:
        wear_now = max(engine.latest_damage.tyres_wear) if engine.latest_damage else 0.0
        for k in WEAR_HORIZONS:
            forecast = model.wear_after(status.tyres_age_laps, wear_now, k)
            wear[k] = round(min(100.0, max(0.0, forecast.mid)), 2)
            wear_range[k] = [round(forecast.low, 2), round(forecast.high, 2)]
    snap['wear_pred'] = wear
    snap['wear_range'] = wear_range
    snap['race_pace_ms'] = engine.radio.race_pace_ms

    radio = engine.radio
    gaps = []
    me = truth.player_index
    for side, gap, car in (('ahead', radio.gap_ahead_s, radio.car_ahead_index),
                           ('behind', radio.gap_behind_s, radio.car_behind_index)):
        if gap is None or car is None:
            continue
        mine, theirs = radio.last_lap_times_ms.get(me), radio.last_lap_times_ms.get(car)
        # Seconds per lap the gap is expected to grow, from the last completed laps.
        rate = None
        if mine and theirs:
            rate = (mine - theirs) / 1000.0 if side == 'ahead' else (theirs - mine) / 1000.0
        gaps.append({'side': side, 'car': car, 'engineer_gap': round(gap, 3),
                     'rate_s_per_lap': None if rate is None else round(rate, 3),
                     'name': radio.driver_name(car)})
    snap['gaps'] = gaps
    snap['rivals'] = rival_snapshot(engine)
    return snap


def _rival_value(prediction):
    if prediction is None:
        return None
    if not prediction.will_stop:
        return {'lap': 'none'}
    low, mid, high = prediction.stop_lap_range
    return {'lap': mid, 'low': low, 'high': high, 'next': prediction.next_compound}


def rival_snapshot(engine: AnalysisEngine) -> dict:
    """Each rival's predicted next stop, from the full model and from the age-only baseline."""
    ctx = engine._race_context()
    if ctx is None:
        return {}
    bare_ctx = rival_strategy.RaceContext(ctx.track_id, ctx.total_laps, ctx.track_length, ctx.pit_loss_s)
    out = {}
    for prediction in engine.rival_predictions():
        index = prediction.car_index
        state = engine.rivals.state(index, prediction.name, ctx.track_length)
        baseline = rival_strategy.predict_rival(state, bare_ctx) if state else None
        out[index] = {'sets_done': len(engine.rivals.cars[index].stints) - 1,
                      'compound': prediction.compound,
                      'rival_model': _rival_value(prediction),
                      'rival_plan_only': _rival_value(baseline)}
    return out


@dataclass
class Segment:
    """One session found in a recording, with its ground truth and the engineer's snapshots."""
    session_uid: int
    truth: GroundTruth
    snapshots: dict = field(default_factory=dict)   # lap -> snapshot


@contextlib.contextmanager
def no_hud():
    """Skip the web HUD state updates during a replay: nobody is watching, and deep-copying
    the HUD state every frame is most of the replay time."""
    modules = (analysis_module, stream_module)
    saved = [(m, m.update_web_state) for m in modules]
    try:
        for module in modules:
            module.update_web_state = lambda data: None
        yield
    finally:
        for module, original in saved:
            module.update_web_state = original


def replay(path) -> list[Segment]:
    """Replay a recording through the engineer and collect per-lap snapshots and ground truth."""
    clock = harness.SimClock()
    voice = harness.RecordingVoice(clock)
    segments: dict[int, Segment] = {}
    with tempfile.TemporaryDirectory() as tmp, harness.simulated_time(clock), no_hud():
        history = TelemetryHistory(Path(tmp) / 'history.sqlite3')
        try:
            engine = AnalysisEngine(voice, history=history, on_demand=True)
            stream = TelemetryStream(engine)
            for record in recording.read_records(path):
                clock.t = record.t
                data = record.data
                stream.handle(data, now=clock.monotonic())
                segment = _feed_truth(segments, data)
                if segment is None or engine.latest_lap is None or stream.session_uid != segment.session_uid:
                    continue
                truth = segment.truth
                lap = engine.latest_lap
                if (lap.current_lap_num not in segment.snapshots and lap.current_lap_num == truth.player_lap
                        and lap.lap_distance >= SNAPSHOT_DISTANCE_M and lap.driver_status != 0
                        and lap.result_status == 2 and engine.session is not None
                        and engine.session.session_type in RACE_SESSION_TYPES):
                    segment.snapshots[lap.current_lap_num] = engineer_snapshot(engine, truth)
        finally:
            history.close()
    return [s for s in segments.values()
            if s.truth.session and s.truth.session.session_type in RACE_SESSION_TYPES and s.snapshots]


def _feed_truth(segments: dict, data: bytes):
    if len(data) < HEADER_SIZE:
        return None
    try:
        header = PacketHeader.from_bytes(data)
    except (struct.error, ValueError):
        return None
    if header.packet_format != 2021 or not 0 <= header.player_car_index < 22:
        return None
    packet_id = header.packet_id
    if packet_id not in (1, 2, 7, 10) or len(data) != PACKET_SIZES[packet_id]:
        return segments.get(header.session_uid)
    if packet_id == 1:
        session = parse_session_data(data)
        if session.track_id < 0 or session.track_length <= 0:
            return segments.get(header.session_uid)
        segment = segments.setdefault(header.session_uid,
                                      Segment(header.session_uid, GroundTruth(header.session_uid)))
        segment.truth.on_session(session, header.player_car_index)
        return segment
    segment = segments.get(header.session_uid)
    if segment is None or segment.truth.session is None:
        return segment
    if packet_id == 2:
        segment.truth.on_laps(parse_all_lap_data(data), header.session_time, header.player_car_index)
    elif packet_id == 7:
        segment.truth.on_status(parse_all_car_status(data))
    else:
        segment.truth.on_damage(parse_player_car_damage(data, header.player_car_index))
    return segment


# ──────────────────────────────────────────────
# Comparing predictions with what happened
# ──────────────────────────────────────────────

def _sample(metric, predictor, horizon, lap, predicted, actual, low=None, high=None):
    sample = {'metric': metric, 'predictor': predictor, 'horizon': horizon, 'lap': lap,
              'predicted': predicted, 'actual': actual, 'error': round(predicted - actual, 3)}
    if low is not None and high is not None:
        sample['range'] = [low, high]
        sample['in_range'] = low <= actual <= high
    return sample


def evaluate(segment: Segment) -> dict:
    """Error samples plus pit-call outcome counts for one race."""
    truth, snaps = segment.truth, segment.snapshots
    player = truth.player_index
    my_stops = sorted(truth.stops.get(player, []))
    total = truth.session.total_laps if truth.session else 0
    samples, outcomes = [], defaultdict(lambda: defaultdict(int))

    for lap, snap in sorted(snaps.items()):
        # ── pit lap
        next_stop = next((s for s in my_stops if s >= lap), None)
        for predictor in ('ml_ideal', 'ml_latest', 'strategy', 'game_window'):
            counts = outcomes[predictor]
            counts['laps'] += 1
            value = snap['pit'].get(predictor)
            if value is None:
                continue
            counts['predicted'] += 1
            if value == 'none':
                counts['missed_stop' if next_stop else 'correct_no_stop'] += 1
            elif next_stop is None:
                counts['stop_that_never_came'] += 1
            else:
                counts['lap_compared'] += 1
                laps_before = next_stop - lap
                bucket = next(label for lo, hi, label in PIT_HORIZON_BUCKETS if lo <= laps_before <= hi)
                samples.append(_sample('pit_lap', predictor, bucket, lap, value, next_stop))

        # ── tyre wear, same stint only
        for k in WEAR_HORIZONS:
            target = snaps.get(lap + k)
            if (target is None or target['stops_done'] != snap['stops_done']
                    or target['worst_wear'] is None or snap['worst_wear'] is None):
                continue
            if k in snap['wear_pred']:
                low, high = (snap.get('wear_range') or {}).get(k, (None, None))
                samples.append(_sample('tyre_wear', 'ml_curve', f'+{k} lap{"s" * (k > 1)}', lap,
                                       snap['wear_pred'][k], target['worst_wear'], low, high))
            before = snaps.get(lap - 1)
            if before and before['stops_done'] == snap['stops_done'] and before['worst_wear'] is not None:
                rate = snap['worst_wear'] - before['worst_wear']
                samples.append(_sample('tyre_wear', 'last_lap_rate', f'+{k} lap{"s" * (k > 1)}', lap,
                                       round(snap['worst_wear'] + k * rate, 2), target['worst_wear']))

        # ── lap time (racing laps only)
        def clean(n):
            return (n > 1 and n not in my_stops and n - 1 not in my_stops
                    and n not in truth.safety_car_laps and n in truth.lap_times_ms)
        if clean(lap):
            actual = truth.lap_times_ms[lap] / 1000.0
            if snap['race_pace_ms']:
                samples.append(_sample('lap_time', 'race_pace', 'this lap', lap,
                                       snap['race_pace_ms'] / 1000.0, actual))
            if clean(lap - 1):
                samples.append(_sample('lap_time', 'last_lap', 'this lap', lap,
                                       truth.lap_times_ms[lap - 1] / 1000.0, actual))

        # ── gaps
        for gap in snap['gaps']:
            car, side = gap['car'], gap['side']
            sign = 1 if side == 'ahead' else -1       # truth gaps are positive when the car is ahead
            now = snap['truth_gaps'].get(car)
            if now is not None:
                samples.append(_sample('gap_now', 'engineer', side, lap, gap['engineer_gap'], sign * now))
            for k in GAP_HORIZONS:
                target = snaps.get(lap + k)
                if target is None or car not in target['truth_gaps']:
                    continue
                if _gap_disturbed(snap, target, car, player, truth, lap, k):
                    continue
                actual = sign * target['truth_gaps'][car]
                horizon = f'+{k} lap{"s" * (k > 1)}'
                samples.append(_sample('gap', 'hold', horizon, lap, gap['engineer_gap'], actual))
                if gap['rate_s_per_lap'] is not None:
                    samples.append(_sample('gap', 'pace_delta', horizon, lap,
                                           round(gap['engineer_gap'] + k * gap['rate_s_per_lap'], 3), actual))

        # ── every rival's next stop and compound
        _evaluate_rivals(snap, truth, lap, samples, outcomes)

    return {'samples': samples, 'pit_outcomes': {p: dict(c) for p, c in outcomes.items()},
            'player_stops': my_stops, 'laps_snapshotted': len(snaps), 'total_laps': total}


def _evaluate_rivals(snap, truth, lap, samples, outcomes):
    total = truth.session.total_laps if truth.session else 0
    for car, rival in (snap.get('rivals') or {}).items():
        car = int(car)
        stops = sorted(truth.stops.get(car, []))
        next_stop = next((s for s in stops if s >= lap), None)
        for predictor in RIVAL_PREDICTORS:
            value = rival.get(predictor)
            counts = outcomes[predictor]
            counts['laps'] += 1
            if value is None:
                continue
            counts['predicted'] += 1
            if value['lap'] == 'none':
                counts['missed_stop' if next_stop else 'correct_no_stop'] += 1
                continue
            if next_stop is None:
                if lap < total:
                    counts['stop_that_never_came'] += 1
                continue
            counts['lap_compared'] += 1
            laps_before = next_stop - lap
            bucket = next(label for lo, hi, label in PIT_HORIZON_BUCKETS if lo <= laps_before <= hi)
            high = value['high'] if value.get('high') is not None else total + 1
            samples.append(_sample('rival_pit_lap', predictor, bucket, lap, value['lap'], next_stop,
                                   value.get('low'), high))
        # Next compound: the set this car really fitted at that stop.
        sets = truth.compounds.get(car, [])
        actual = sets[rival['sets_done'] + 1] if next_stop and len(sets) > rival['sets_done'] + 1 else None
        model = rival.get('rival_model') or {}
        if actual is None or model.get('lap') in (None, 'none'):
            continue
        for predictor, guess in (('rival_model', model.get('next')),
                                 ('rival_rule', race_strategy.next_compound(rival['compound']))):
            counts = outcomes[f'{predictor}_compound']
            counts['compared'] += 1
            counts['correct'] += int(guess == actual)


def _gap_disturbed(snap, target, car, player, truth, lap, k) -> bool:
    """A pit stop or Safety Car between the two laps makes a pace-based gap forecast moot."""
    for who in (car, player):
        if snap['stop_counts'].get(who) != target['stop_counts'].get(who):
            return True
        if who in snap['pit_status_all'] or who in target['pit_status_all']:
            return True
    return any(n in truth.safety_car_laps for n in range(lap, lap + k + 1))


def summarize(errors) -> dict:
    errors = [e for e in errors if e is not None and math.isfinite(e)]
    if not errors:
        return {'n': 0}
    absolute = [abs(e) for e in errors]
    return {
        'n': len(errors),
        'mae': round(statistics.fmean(absolute), 3),
        'median_ae': round(statistics.median(absolute), 3),
        'rmse': round(math.sqrt(statistics.fmean(e * e for e in errors)), 3),
        'bias': round(statistics.fmean(errors), 3),
        'max_ae': round(max(absolute), 3),
    }


def summary_rows(samples) -> list[dict]:
    """One row per (metric, predictor, horizon), plus an all-horizons row per (metric, predictor)."""
    groups, ranges = defaultdict(list), defaultdict(list)
    for s in samples:
        for horizon in (s['horizon'], 'all'):
            groups[(s['metric'], s['predictor'], horizon)].append(s['error'])
            if 'in_range' in s:
                ranges[(s['metric'], s['predictor'], horizon)].append(s['in_range'])
    order = list(METRICS)
    rows = []
    for (metric, predictor, horizon), errors in groups.items():
        if horizon == 'all' and len({s['horizon'] for s in samples
                                     if s['metric'] == metric and s['predictor'] == predictor}) == 1:
            continue   # a single horizon already says it all
        row = {'metric': metric, 'predictor': predictor, 'horizon': horizon, **summarize(errors)}
        hits = ranges.get((metric, predictor, horizon))
        if hits:
            row['in_range'] = round(sum(hits) / len(hits), 3)
        rows.append(row)
    rows.sort(key=lambda r: (order.index(r['metric']), list(PREDICTORS).index(r['predictor']),
                             r['horizon'] == 'all', _horizon_key(r['horizon'])))
    return rows


def _horizon_key(horizon):
    buckets = [label for _lo, _hi, label in PIT_HORIZON_BUCKETS]
    if horizon in buckets:
        return buckets.index(horizon)
    digits = ''.join(ch for ch in horizon if ch.isdigit())
    return int(digits) if digits else 0


def merge_outcomes(outcome_dicts) -> dict:
    merged = defaultdict(lambda: defaultdict(int))
    for outcomes in outcome_dicts:
        for predictor, counts in outcomes.items():
            for key, value in counts.items():
                merged[predictor][key] += value
    return {p: dict(c) for p, c in merged.items()}


# ──────────────────────────────────────────────
# Running and reporting
# ──────────────────────────────────────────────

def is_simulated(metadata: dict) -> bool:
    return bool(metadata.get('simulated')) or metadata.get('source') == 'synthetic'


def backtest_file(path) -> list[dict]:
    path = Path(path)
    metadata = recording.read_metadata(path)
    segments = replay(path)
    races = []
    for number, segment in enumerate(segments, start=1):
        result = evaluate(segment)
        session = segment.truth.session
        name = path.name.removesuffix('.gz').removesuffix('.f1rec')
        if len(segments) > 1:
            name += f' (race {number})'
        races.append({
            'name': metadata.get('scenario') or name if len(segments) == 1 else name,
            'path': _display_path(path), 'simulated': is_simulated(metadata), 'metadata': metadata,
            'track': TRACK_NAMES.get(session.track_id, f'Track {session.track_id}'),
            **result, 'summary': summary_rows(result['samples']),
        })
    return races


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def backtest(paths) -> dict:
    races, skipped = [], []
    for path in paths:
        try:
            found = backtest_file(path)
        except (OSError, recording.RecordingError, ValueError) as error:
            skipped.append({'path': str(path), 'reason': str(error)})
            continue
        if not found:
            skipped.append({'path': str(path), 'reason': 'no race session with completed laps'})
        races.extend(found)
    samples = [s for race in races for s in race['samples']]
    return {
        'generated_at': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        'races': races, 'skipped': skipped,
        'aggregate': {'summary': summary_rows(samples),
                      'pit_outcomes': merge_outcomes(r['pit_outcomes'] for r in races)},
        'simulated_races': sum(r['simulated'] for r in races),
        'real_races': sum(not r['simulated'] for r in races),
    }


def _fmt(value, places=2):
    return '–' if value is None else f'{value:.{places}f}'


def _summary_table(rows) -> list[str]:
    if not rows:
        return ['_No comparable laps._', '']
    lines = ['| What | Predictor | Horizon | n | Mean abs error | Median abs error | Bias | Worst | '
             'Inside 80% range |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        label, unit = METRICS[r['metric']]
        coverage = '–' if r.get('in_range') is None else f"{r['in_range'] * 100:.0f}%"
        lines.append(f"| {label} ({unit}) | {PREDICTORS[r['predictor']]} | {r['horizon']} | {r['n']} | "
                     f"{_fmt(r.get('mae'))} | {_fmt(r.get('median_ae'))} | {_fmt(r.get('bias'), 2)} | "
                     f"{_fmt(r.get('max_ae'))} | {coverage} |")
    return lines + ['']


def _pit_table(outcomes) -> list[str]:
    lines = ['| Predictor | Laps | Made a call | Lap compared | Said no stop, car pitted | '
             'Called a stop that never came | Correct "no more stops" |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for predictor in ('ml_ideal', 'ml_latest', 'strategy', 'game_window', *RIVAL_PREDICTORS):
        c = outcomes.get(predictor)
        if not c:
            continue
        lines.append(f"| {PREDICTORS[predictor]} | {c.get('laps', 0)} | {c.get('predicted', 0)} | "
                     f"{c.get('lap_compared', 0)} | {c.get('missed_stop', 0)} | "
                     f"{c.get('stop_that_never_came', 0)} | {c.get('correct_no_stop', 0)} |")
    return lines + ['']


def _compound_table(outcomes) -> list[str]:
    rows = [(p, outcomes.get(f'{p}_compound')) for p in ('rival_model', 'rival_rule')]
    rows = [(p, c) for p, c in rows if c and c.get('compared')]
    if not rows:
        return []
    lines = ['| Predictor | Rival stops compared | Right tyre |', '|---|---:|---:|']
    for predictor, c in rows:
        lines.append(f"| {PREDICTORS[predictor]} | {c['compared']} | {c['correct'] / c['compared'] * 100:.0f}% |")
    return lines + ['']


def render_markdown(report: dict) -> str:
    races = report['races']
    out = ['# Race engineer backtest', '',
           f"Generated {report['generated_at']} from {len(races)} race(s): "
           f"{report['real_races']} recorded in F1 2021, {report['simulated_races']} simulated.", '']
    if report['real_races'] == 0:
        out += ['> **These numbers come from simulated races only.** They show the backtest works '
                'end to end, not how accurate the engineer is in the real game: the simulator is '
                'simple and its driver pits on scripted laps. To get real error numbers, record '
                'your own races (`python main.py --record recordings/<name>.f1rec.gz`) and run '
                '`python tools/backtest.py recordings/*.f1rec.gz`.', '']
    elif report['simulated_races']:
        out += ['> Simulated races are included in the totals below; the per-race sections say '
                'which is which. Run the tool on your recordings only for real-game numbers.', '']
    out += ['## How to read this', '',
            '* **Error = predicted − actual.** Mean/median abs error is the typical miss; '
            '**bias** is the average signed miss (positive = predicted too late / too much wear / '
            'too slow / too big a gap).',
            '* Predictions are taken once per lap, about 400 m after the line, using only what the '
            'engineer knew at that moment. "Horizon" is how far ahead the prediction looks.',
            '* **Pit lap** is compared with the lap you actually pitted on (the in-lap). It measures '
            'agreement with what you did, which is not always the optimal lap.',
            '* **Lap time** excludes lap 1, in- and out-laps and Safety Car laps. **Gap forecasts** '
            'skip any window where either car pitted or a Safety Car came out.',
            '* Rows marked baseline are simple rules of thumb; the engineer should beat them.',
            "* **Rival next stop** is each rival's predicted next in-lap against the lap that car "
            'really pitted on; its range is the model\'s 80% stop window.',
            '* **Inside 80% range**: how often the truth landed inside the range the fitted model '
            'gave. About 80% means the ranges are honest; much lower means they are too narrow.', '',
            '## All races', '']
    out += _summary_table(report['aggregate']['summary'])
    out += ['### Pit-stop calls', '',
            'Rival rows count one prediction per rival per lap.', '']
    out += _pit_table(report['aggregate']['pit_outcomes'])
    compounds = _compound_table(report['aggregate']['pit_outcomes'])
    if compounds:
        out += ['### Rivals: next tyre', ''] + compounds
    out += ['## Per race', '']
    for race in races:
        kind = 'simulated' if race['simulated'] else 'recorded in F1 2021'
        stops = ', '.join(str(s) for s in race['player_stops']) or 'no stop'
        out += [f"### {race['name']} ({kind})", '',
                f"{race['track']}, {race['total_laps']} laps, you pitted on lap {stops}; "
                f"{race['laps_snapshotted']} laps with predictions. `{race['path']}`", '']
        out += _summary_table(race['summary'])
        out += _pit_table(race['pit_outcomes'])
        out += _compound_table(race['pit_outcomes'])
    if report['skipped']:
        out += ['## Skipped files', '']
        out += [f"* `{s['path']}`: {s['reason']}" for s in report['skipped']]
        out.append('')
    return '\n'.join(out)


# ──────────────────────────────────────────────
# Built-in simulated races
# ──────────────────────────────────────────────

GOLDEN_RACES = ('race_dry_red_bull_ring', 'race_wet_hungaroring', 'race_safety_car_zandvoort')
SIMULATED_DIR = ROOT / 'recordings' / 'simulated'


def backtest_scenarios():
    """Longer simulated races than the golden sessions, so wear and pit predictions have room."""
    from synth import Car, Scenario
    from scenarios import FIELD, NUMBERS

    def field_of(first_stint, pit_laps, start_index=1):
        cars = []
        compounds = ('soft', 'medium', 'medium', 'hard')
        for index in range(start_index, len(FIELD)):
            compound = compounds[index % len(compounds)]
            second = 'hard' if compound != 'hard' else 'medium'
            stop = pit_laps[index % len(pit_laps)]
            cars.append(Car(FIELD[index], pace=0.25 * (index - 1) - 0.4, compound=compound,
                            grid=index + 1, pits={stop: second}, race_number=NUMBERS[index]))
        return cars

    spain = Scenario(
        name='backtest_spain_one_stop', track_id=4, track_length=4655, session_type=10,
        total_laps=24, cars=[Car('Player', pace=0.6, compound='soft', grid=6, pits={15: 'hard'},
                                 race_number=NUMBERS[0])] + field_of('soft', (12, 14, 16, 18)),
        fuel_kg=24 * 1.7 + 3, pit_window=(13, 17),
        description='24-lap dry one-stop race at Barcelona, soft to hard.')
    monza = Scenario(
        name='backtest_monza_two_stop', track_id=11, track_length=5793, session_type=10,
        total_laps=26, cars=[Car('Player', pace=0.3, compound='soft', grid=4,
                                 pits={9: 'medium', 18: 'soft'}, race_number=NUMBERS[0])]
        + field_of('soft', (10, 13, 15, 17)),
        fuel_kg=26 * 1.7 + 3, pit_window=(8, 12),
        description='26-lap dry two-stop race at Monza, soft-medium-soft.')
    return [spain, monza, wear_driven_race()]


def wear_driven_race(seed: int = 21):
    """A race where every rival boxes on tyre wear, the way game AI roughly does.

    Each rival has its own wear threshold (50-70%) and boxes at the end of the lap that takes
    its worst tyre past it, then picks a tyre for the laps left (short run: soft, medium run:
    medium, long run: hard), stopping again if that set would also wear out. Unlike the
    scripted races, the right answer is in the data the rival stop model sees.
    """
    import random
    from synth import COMPOUNDS, Car, Scenario
    from scenarios import FIELD, NUMBERS

    rng = random.Random(seed)
    total, wear_scale = 32, 1.15
    starts = ('soft', 'medium', 'medium', 'hard')

    def plan(compound, threshold):
        pits, lap, current, used = {}, 0, compound, {compound}
        while True:
            rate = COMPOUNDS[current][4] * wear_scale
            stop = lap + math.ceil(threshold / rate)
            if stop >= total and len(used) < 2:
                stop = total - 2                    # the mandatory stop, late
            if stop >= total:
                return pits
            left = total - stop
            choice = 'soft' if left <= 9 else 'medium' if left <= 18 else 'hard'
            last_stint = left * COMPOUNDS[choice][4] * wear_scale < threshold
            if last_stint and used == {choice}:     # two dry compounds are mandatory
                choice = {'soft': 'medium', 'hard': 'medium'}.get(choice, 'hard' if left > 13 else 'soft')
            current = choice
            used.add(current)
            pits[stop] = current
            lap = stop

    cars = [Car('Player', pace=0.6, compound='medium', grid=8, pits={17: 'hard'}, race_number=NUMBERS[0])]
    for index in range(1, len(FIELD)):
        compound = starts[index % len(starts)]
        cars.append(Car(FIELD[index], pace=0.2 * (index - 1) - 0.8, compound=compound,
                        grid=index if index < 8 else index + 1,
                        pits=plan(compound, rng.uniform(50.0, 70.0)), race_number=NUMBERS[index]))
    return Scenario(
        name='backtest_silverstone_wear_stops', track_id=7, track_length=5891, session_type=10,
        total_laps=total, cars=cars, fuel_kg=total * 1.7 + 3, wear_scale=wear_scale,
        description=f'{total}-lap dry race at Silverstone where rivals box on tyre wear.')


def simulated_recordings(regenerate=False) -> list[Path]:
    """Golden race recordings plus the backtest races (generated once into recordings/simulated/)."""
    paths = [harness.session_paths(name)[0] for name in GOLDEN_RACES]
    paths = [p for p in paths if p.exists()]
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)
    for scenario in backtest_scenarios():
        path = SIMULATED_DIR / f'{scenario.name}.f1rec.gz'
        if regenerate or not path.exists():
            write_scenario(scenario, path)
        paths.append(path)
    return paths


def write_scenario(scenario, path):
    import synth
    metadata = {'source': 'synthetic', 'scenario': scenario.name,
                'generator': 'tests/golden/synth.py via tools/backtest.py',
                'description': scenario.description}
    with recording.ReproducibleWriter(path, metadata) as writer:
        for t, data in synth.simulate(scenario):
            writer.write(data, round(t * 1e9))
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the engineer's predictions with what happened in recorded races")
    parser.add_argument('recordings', nargs='*', type=Path, help='.f1rec / .f1rec.gz files')
    parser.add_argument('--simulated', action='store_true',
                        help='Also run the built-in simulated races (golden sessions + longer races)')
    parser.add_argument('--regenerate', action='store_true',
                        help='Rebuild the simulated backtest recordings first')
    parser.add_argument('--out', type=Path, help='Write the Markdown report here (default: print it)')
    parser.add_argument('--json', type=Path, help='Also write every number, per lap, as JSON')
    args = parser.parse_args(argv)

    paths = list(args.recordings)
    if args.simulated or not paths:
        if not paths:
            print('[BACKTEST] No recordings given, running the built-in simulated races.', file=sys.stderr)
        paths += simulated_recordings(args.regenerate)
    for path in paths:
        print(f'[BACKTEST] {path}', file=sys.stderr)
    report = backtest(paths)
    text = render_markdown(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding='utf-8')
        print(f'[BACKTEST] Report written to {args.out}', file=sys.stderr)
    else:
        print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
        print(f'[BACKTEST] JSON written to {args.json}', file=sys.stderr)
    return 0 if report['races'] else 1


if __name__ == '__main__':
    sys.exit(main())
