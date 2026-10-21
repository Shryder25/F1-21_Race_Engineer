"""Replay a recorded session through the real engineer offline and capture what it did.

The replay drives the same objects main.py wires together (TelemetryStream ->
AnalysisEngine -> RaceRadio / RaceControl / strategy), with three substitutions so the
result depends only on the recording:

  * a recording "voice" that logs every callout the engineer queues, applying the same
    per-topic cooldown the real VoiceEngine applies, instead of speaking;
  * a simulated clock: every time.time()/time.monotonic() the engine reads is the
    recording's own arrival time, so cooldowns and timers behave as they did live;
  * a fresh, throwaway history database, so earlier sessions on this PC can't leak in.

What comes out is a "transcript": the radio calls in order, the rule-based answers to the
driver's voice commands at checkpoints, and the key numbers (lap times, positions, gaps,
fuel, tyre and strategy predictions) at the end of every lap. compare() diffs two
transcripts with small numeric tolerances and returns readable differences.
"""

from __future__ import annotations

import contextlib
import copy
import json
import difflib
import math
import struct
import sys
import tempfile
import time as real_time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT / 'ai_engineer', Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import analysis as analysis_module          # noqa: E402
import race_radio as race_radio_module      # noqa: E402
import telemetry_stream as stream_module    # noqa: E402
import web_hud                              # noqa: E402
from analysis import AnalysisEngine, PROACTIVE_KEYS   # noqa: E402
from engineer_brain import rule_based_answer          # noqa: E402
from history import TelemetryHistory                  # noqa: E402
from telemetry_stream import TelemetryStream          # noqa: E402
from voice_commands import COMMANDS                   # noqa: E402

import recording                                       # noqa: E402

TRANSCRIPT_VERSION = 1
# Fixed origins for the simulated clocks: far from zero, like a real PC, so "seconds since
# the last alert" checks that start from 0.0 behave as they do live.
WALL_CLOCK_ORIGIN = 1_700_000_000.0
MONOTONIC_ORIGIN = 100_000.0
DEFAULT_TOLERANCES = {'time_s': 0.25, 'number_abs': 0.011, 'number_rel': 0.0}


# ──────────────────────────────────────────────
# Simulated clock and recording voice
# ──────────────────────────────────────────────

class SimClock:
    """Stands in for the `time` module inside the engine modules during a replay."""

    def __init__(self):
        self.t = 0.0

    def time(self):
        return WALL_CLOCK_ORIGIN + self.t

    def monotonic(self):
        return MONOTONIC_ORIGIN + self.t

    def perf_counter(self):
        return self.monotonic()

    def __getattr__(self, name):              # sleep, strftime, ... fall through
        return getattr(real_time, name)


class RecordingVoice:
    """Logs callouts instead of speaking them, with VoiceEngine's per-topic cooldown."""

    is_running = True

    def __init__(self, clock: SimClock, default_cooldown: float = 30.0):
        self.clock = clock
        self.default_cooldown = default_cooldown
        self.calls: list[dict] = []
        self._last: dict[str, float] = {}
        self.session_time = 0.0

    def say(self, message, priority=None, *, key=None, ttl=15, cooldown=None):
        if not message or ttl <= 0:
            return False
        topic = message if key is None else key
        interval = self.default_cooldown if cooldown is None else cooldown
        now = self.clock.t
        last = self._last.get(topic)
        if last is not None and now - last < interval:
            return False
        self._last[topic] = now
        short_key = topic.split(':', 1)[1] if ':' in topic else topic
        self.calls.append({
            't': round(self.session_time, 2), 'key': short_key,
            'priority': getattr(priority, 'name', str(priority)),
            'proactive': short_key in PROACTIVE_KEYS, 'text': message,
        })
        return True

    def say_immediate(self, message):
        return self.say(message, None, cooldown=0)

    def clear(self):
        pass


@contextlib.contextmanager
def simulated_time(clock: SimClock):
    modules = (analysis_module, race_radio_module, stream_module, web_hud)
    saved = [(m, m.time) for m in modules]
    saved_state = copy.deepcopy(web_hud.dashboard_state)
    try:
        for module in modules:
            module.time = clock
        yield
    finally:
        for module, original in saved:
            module.time = original
        web_hud.replace_web_state(saved_state)


# ──────────────────────────────────────────────
# Transcript
# ──────────────────────────────────────────────

def _round(value, places=3):
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, places)
    if isinstance(value, dict):
        return {str(k): _round(v, places) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_round(v, places) for v in value]
    return value


def key_numbers(engine: AnalysisEngine) -> dict:
    """The numbers a race engineer would quote, read from the engine's own context."""
    ctx = engine.build_context()
    pit = ctx.get('pit_prediction') or {}
    fuel = ctx.get('fuel') or {}
    tire = ctx.get('tire_health') or {}
    strategy = ctx.get('strategy') or {}
    best = strategy.get('best') if isinstance(strategy, dict) else None
    threats = ctx.get('opponent_threats') or []
    return _round({
        'lap': ctx.get('lap'), 'position': ctx.get('position'),
        'weather': ctx.get('weather'), 'safety_car': ctx.get('safety_car'),
        'last_lap_ms': ctx.get('last_lap_ms'), 'best_lap_ms': ctx.get('best_lap_ms'),
        'race_pace_ms': ctx.get('race_pace_ms'),
        'gap_ahead_s': ctx.get('gap_ahead_s'), 'gap_behind_s': ctx.get('gap_behind_s'),
        'car_ahead': ctx.get('car_ahead_name'), 'car_behind': ctx.get('car_behind_name'),
        'compound': ctx.get('compound'), 'tyre_age_laps': ctx.get('tyre_age_laps'),
        'worst_wear_pct': max((ctx.get('tyre_wear_pct') or {}).values(), default=None),
        'fuel_kg': fuel.get('fuel_kg'), 'fuel_per_lap_kg': fuel.get('per_lap_kg'),
        'fuel_margin_laps': fuel.get('margin_laps'),
        'ml_pit_window': [pit.get('ideal_lap'), pit.get('latest_lap')] if pit else None,
        'ml_wear_per_lap_pct': pit.get('wear_per_lap_pct'),
        'tyre_health': tire.get('health_score'), 'tyre_remaining_laps': tire.get('remaining_laps'),
        'pace_trend_s_per_lap': ctx.get('pace_trend_s_per_lap'),
        'strategy_stops': best.get('stops') if isinstance(best, dict) else None,
        'strategy_plan': best.get('compounds') if isinstance(best, dict) else None,
        'strategy_pit_laps': best.get('pit_laps') if isinstance(best, dict) else None,
        'top_threat': (threats[0]['driver_name'], threats[0]['threat_level']) if threats else None,
        'race_report_laps': len(engine.session_lap_stats),
    }, 3)


def answers(engine: AnalysisEngine) -> dict:
    """What the engineer says to each voice command, from the rule-based fallback brain."""
    ctx = engine.build_context()
    return {command: rule_based_answer(command, ctx) for command in COMMANDS}


def replay(path, checkpoints_every_lap=True) -> dict:
    """Replay a recording through the engine and return its transcript."""
    clock = SimClock()
    voice = RecordingVoice(clock)
    with tempfile.TemporaryDirectory() as tmp, simulated_time(clock):
        history = TelemetryHistory(Path(tmp) / 'history.sqlite3')
        try:
            engine = AnalysisEngine(voice, history=history, on_demand=False)
            stream = TelemetryStream(engine)
            laps: list[dict] = []
            mid_answers = None
            last_lap = None
            datagrams = accepted = 0
            for record in recording.read_records(path):
                clock.t = record.t
                datagrams += 1
                if len(record.data) >= 24:
                    voice.session_time = struct.unpack_from('<f', record.data, 14)[0]
                if stream.handle(record.data, now=clock.monotonic()):
                    accepted += 1
                lap = engine.latest_lap
                if lap is not None and lap.current_lap_num != last_lap:
                    if last_lap is not None and checkpoints_every_lap:
                        entry = {'lap_completed': last_lap, 't': round(voice.session_time, 2)}
                        entry.update(key_numbers(engine))
                        entry['answers'] = answers(engine)
                        laps.append(entry)
                    last_lap = lap.current_lap_num
            final = key_numbers(engine) if engine.latest_lap else {}
            final_answers = answers(engine) if engine.latest_lap else {}
            race_control = [dict(e) for e in engine.race_control.recent(30)][::-1]
            report = engine.race_report()
        finally:
            history.close()
    return {
        'transcript_version': TRANSCRIPT_VERSION,
        'datagrams': datagrams, 'paired_frames': accepted,
        'calls': voice.calls,
        'race_control_feed': _round(race_control, 2),
        'laps': laps,
        'race_report': _round(report, 1),
        'final': final,
        'answers': final_answers,
    }


# ──────────────────────────────────────────────
# Comparison
# ──────────────────────────────────────────────

def _numbers_equal(a, b, tol) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= max(tol['number_abs'], tol['number_rel'] * max(abs(a), abs(b)))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_numbers_equal(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_numbers_equal(a[k], b[k], tol) for k in a)
    return a == b


def _fmt_call(call):
    return f"t={call['t']:>7.2f}  [{call['key']}] {call['text']}"


def compare(expected: dict, actual: dict, tolerances: dict | None = None, limit=25) -> list[str]:
    """Readable differences between two transcripts; empty when they match."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    problems: list[str] = []

    exp_calls, act_calls = expected.get('calls', []), actual.get('calls', [])
    exp_lines = [f"[{c['key']}] {c['text']}" for c in exp_calls]
    act_lines = [f"[{c['key']}] {c['text']}" for c in act_calls]
    if exp_lines != act_lines:
        diff = list(difflib.unified_diff(exp_lines, act_lines, 'expected calls', 'actual calls',
                                         lineterm='', n=1))
        problems.append('Radio calls changed:\n  ' + '\n  '.join(diff[:limit * 2]))
    else:
        for e, a in zip(exp_calls, act_calls):
            if abs(e['t'] - a['t']) > tol['time_s']:
                problems.append(f"Call timing moved: expected {_fmt_call(e)}  got t={a['t']:.2f}")
                if len(problems) >= limit:
                    break

    def compare_block(label, exp, act):
        if not _numbers_equal(exp, act, tol):
            if isinstance(exp, dict) and isinstance(act, dict):
                for key in sorted(set(exp) | set(act)):
                    if not _numbers_equal(exp.get(key), act.get(key), tol):
                        problems.append(f'{label} {key}: expected {exp.get(key)!r}, got {act.get(key)!r}')
            else:
                problems.append(f'{label}: expected {exp!r}, got {act!r}')

    exp_laps, act_laps = expected.get('laps', []), actual.get('laps', [])
    if len(exp_laps) != len(act_laps):
        problems.append(f'Lap checkpoints: expected {len(exp_laps)}, got {len(act_laps)}')
    def compare_answers(label, exp, act):
        for command in sorted(set(exp) | set(act)):
            if exp.get(command) != act.get(command):
                problems.append(f"{label} answer to '{command}' changed:\n    expected: {exp.get(command)!r}"
                                f"\n    got:      {act.get(command)!r}")

    for e, a in zip(exp_laps, act_laps):
        label = f"After lap {e.get('lap_completed')}:"
        e2 = {k: v for k, v in e.items() if k not in ('t', 'answers')}
        a2 = {k: v for k, v in a.items() if k not in ('t', 'answers')}
        compare_block(label, e2, a2)
        compare_answers(label, e.get('answers', {}), a.get('answers', {}))
    compare_block('Final', expected.get('final', {}), actual.get('final', {}))
    compare_block('Race report', expected.get('race_report', []), actual.get('race_report', []))
    compare_answers('Final', expected.get('answers', {}), actual.get('answers', {}))
    exp_feed = [f"{e.get('kind')}: {e.get('text')}" for e in expected.get('race_control_feed', [])]
    act_feed = [f"{e.get('kind')}: {e.get('text')}" for e in actual.get('race_control_feed', [])]
    if exp_feed != act_feed:
        diff = list(difflib.unified_diff(exp_feed, act_feed, 'expected feed', 'actual feed',
                                         lineterm='', n=0))
        problems.append('Race control feed changed:\n  ' + '\n  '.join(diff[:limit * 2]))
    return problems[:limit] + ([f'... and {len(problems) - limit} more'] if len(problems) > limit else [])


# ──────────────────────────────────────────────
# Store
# ──────────────────────────────────────────────

STORE = Path(__file__).resolve().parent / 'sessions'
MANIFEST = STORE / 'manifest.json'


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding='utf-8'))


def save_manifest(manifest: dict):
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')


def session_paths(name: str):
    folder = STORE / name
    return folder / 'session.f1rec.gz', folder / 'expected.json'


ONE_PER_LINE = ('calls', 'race_control_feed', 'race_report')


def dumps_transcript(transcript: dict) -> str:
    """JSON with one radio call per line, so a behaviour change reads as a small git diff."""
    parts = []
    for key, value in transcript.items():
        if key in ONE_PER_LINE and isinstance(value, list):
            rows = ',\n'.join('  ' + json.dumps(item, ensure_ascii=False) for item in value)
            text = '[\n' + rows + '\n ]' if value else '[]'
        else:
            text = json.dumps(value, indent=1, ensure_ascii=False).replace('\n', '\n ')
        parts.append(f' {json.dumps(key)}: {text}')
    return '{\n' + ',\n'.join(parts) + '\n}\n'


def save_expected(name: str, transcript: dict):
    _, expected = session_paths(name)
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text(dumps_transcript(transcript), encoding='utf-8')


def load_expected(name: str) -> dict:
    _, expected = session_paths(name)
    return json.loads(expected.read_text(encoding='utf-8'))
