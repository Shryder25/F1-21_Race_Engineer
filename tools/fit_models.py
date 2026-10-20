"""Fit the tyre-wear, degradation and fuel models to the game from recorded stints.

Reads `.f1rec` recordings (`main.py --record`, docs/udp-recording-format.md) and, optionally,
the engineer's history database (lap_stats), fits one model per tyre compound and track
(see ai_engineer/fitted_models.py), saves the parameters as JSON for the engineer to load,
and writes a report with every fit and a hold-out check of how good the ranges are.

Usage:
    python tools/fit_models.py recordings/*.f1rec.gz
    python tools/fit_models.py recordings/*.f1rec.gz --history telemetry-history/telemetry.sqlite3
    python tools/fit_models.py --simulated            # built-in noisy simulator stints only

Default output: telemetry-history/fitted_models.json (the file main.py loads at start-up)
and the report on stdout (or --report FILE).

The hold-out check fits on every session but one, then replays the one left out lap by lap
the way the engineer would see it live: before each lap it forecasts the wear 1, 3 and 5 laps
ahead and the fuel needed, and counts how often the truth lands inside the range. An 80%
range should catch about 80% of them; much less means the ranges are too narrow.

Simulated stints only prove the pipeline works and that the ranges are honest on data with
known answers. Real fits need recordings of your own sessions in F1 2021.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / 'tests' / 'golden'
for _path in (ROOT / 'ai_engineer', GOLDEN):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import fitted_models as fm                          # noqa: E402
from packets import TRACK_NAMES                     # noqa: E402

DEFAULT_OUT = ROOT / 'telemetry-history' / 'fitted_models.json'
SIMULATED_DIR = ROOT / 'recordings' / 'simulated' / 'fit'
WEAR_HORIZONS = (1, 3, 5)
FUEL_HORIZONS = (5, 10)
BASIS_NAMES = {'track': 'this track', 'pooled': 'other tracks (pooled)', 'prior': 'built-in estimate'}


# ──────────────────────────────────────────────
# Collecting lap records
# ──────────────────────────────────────────────

def collect(recordings, history_path=None, exclude_captures=(), rivals=True):
    """(records, sources, history capture ids used) from recordings and a history database.

    rivals adds every other car's laps from Session History (fall-off fit only).
    """
    records, sources, captures = [], [], set()
    for path in recordings:
        path = Path(path)
        try:
            recs = fm.laps_from_recording(path, rivals=rivals)
        except Exception as error:                  # a broken file should not stop the rest
            print(f'[FIT] Skipping {path}: {error}', file=sys.stderr)
            sources.append({'path': _display(path), 'error': str(error)})
            continue
        records += recs
        sources.append({'path': _display(path), 'laps': sum(r.car_index is None for r in recs),
                        'rival_laps': sum(r.car_index is not None for r in recs),
                        'simulated': bool(recs) and all(r.simulated for r in recs)})
    if history_path:
        from history import TelemetryHistory
        history = TelemetryHistory(Path(history_path))
        try:
            recs, captures = fm.laps_from_history(history, exclude_captures, rivals=rivals)
        finally:
            history.close()
        records += recs
        sources.append({'history': _display(Path(history_path)),
                        'laps': sum(r.car_index is None for r in recs),
                        'rival_laps': sum(r.car_index is not None for r in recs),
                        'sessions': len(captures), 'simulated': False})
    return records, [s for s in sources if 'error' not in s], captures


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


# ──────────────────────────────────────────────
# Hold-out validation
# ──────────────────────────────────────────────

@dataclass
class Check:
    kind: str           # 'wear' or 'fuel'
    horizon: int
    basis: str
    covered: bool
    error: float
    width: float


def validate(records, level=fm.DEFAULT_LEVEL) -> list[Check]:
    """Leave one session out: fit on the rest, forecast the left-out session lap by lap."""
    sessions = sorted({r.session for r in records})
    if len(sessions) < 2:
        return []
    checks = []
    for held in sessions:
        train = [r for r in records if r.session != held]
        test = [r for r in records if r.session == held]
        store = fm.ModelStore.fit(train)
        checks += _check_wear(store, test, level)
        checks += _check_fuel(store, test, level)
    return checks


def _check_wear(store, test, level):
    checks = []
    stints = defaultdict(list)
    for r in test:
        if r.wear_increment is not None and not r.wet:
            stints[r.stint].append(r)
    for laps in stints.values():
        laps.sort(key=lambda r: r.tyre_age)
        first = laps[0]
        model = store.wear_model(first.track_id, first.compound, first.race_laps)
        ages = [float(r.tyre_age) for r in laps]
        incs = [r.wear_increment for r in laps]
        for j in range(len(laps)):
            stint = model.condition(ages[:j], incs[:j])
            for k in WEAR_HORIZONS:
                if j + k > len(laps) or ages[j + k - 1] - ages[j] != k - 1:
                    continue
                forecast = stint.wear_after(ages[j], 0.0, k, level)
                actual = sum(incs[j:j + k])
                checks.append(Check('wear', k, model.basis, forecast.contains(actual),
                                    forecast.mid - actual, forecast.width))
    return checks


def _check_fuel(store, test, level):
    checks = []
    by_session = defaultdict(list)
    for r in test:
        if r.fuel_used is not None:
            by_session[r.session].append(r)
    for laps in by_session.values():
        laps.sort(key=lambda r: r.lap)
        first = laps[0]
        model = store.fuel_model(first.track_id, first.track_length, first.race_laps)
        used = [r.fuel_used for r in laps]
        for j in range(len(laps)):
            fuel = model.condition(used[:j])
            for k in FUEL_HORIZONS:
                if j + k > len(laps):
                    continue
                forecast = fuel.needed(k, level)
                actual = sum(used[j:j + k])
                checks.append(Check('fuel', k, model.basis, forecast.contains(actual),
                                    forecast.mid - actual, forecast.width))
    return checks


def summarize_checks(checks) -> list[dict]:
    groups = defaultdict(list)
    for c in checks:
        groups[(c.kind, c.basis, c.horizon)].append(c)
    rows = []
    for (kind, basis, horizon), cs in sorted(groups.items()):
        rows.append({'kind': kind, 'basis': basis, 'horizon': horizon, 'n': len(cs),
                     'coverage': sum(c.covered for c in cs) / len(cs),
                     'mae': statistics.fmean(abs(c.error) for c in cs),
                     'bias': statistics.fmean(c.error for c in cs),
                     'width': statistics.fmean(c.width for c in cs)})
    return rows


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────

def _track(track_id) -> str:
    if track_id is None:
        return 'All tracks (pooled)'
    return TRACK_NAMES.get(track_id, f'Track {track_id}')


def _r(r, places=2, unit='') -> str:
    lo, mid, hi = r.as_list(places)
    return f'{mid}{unit} ({lo} to {hi})'


def render_report(store: fm.ModelStore, checks, records, truth: dict | None = None,
                  level=fm.DEFAULT_LEVEL) -> str:
    pct = round(level * 100)
    out = ['# Fitted tyre and fuel models', '',
           f"Fitted {store.created} from {len(store.sources)} source(s), "
           f"{sum(1 for r in records if r.wear_increment is not None)} laps of tyre wear, "
           f"{sum(1 for r in records if r.fuel_used is not None)} laps of fuel use, "
           f"{sum(1 for r in records if r.corrected_time_s is not None)} clean laps for fall-off "
           f"({sum(1 for r in records if r.corrected_time_s is not None and r.car_index is not None)} "
           f"of them by other cars, from Session History).", '']
    if store.simulated_only:
        out += ['> **Simulator data only.** These fits show the fitting and the ranges work on '
                'stints whose true wear and fuel use are known. They say nothing about F1 2021 '
                'itself: record your own sessions (`python main.py --record recordings/<name>.f1rec.gz`) '
                'and run `python tools/fit_models.py recordings/*.f1rec.gz`.', '']
    out += [f'Every number is **expected (low to high)**, the low and high ends being an {pct}% range.',
            'Wear is the worst tyre, % per lap. Fall-off is fuel-corrected lap time lost per lap of '
            'tyre age. A track with no fit of its own uses the pooled fit scaled to its tyre '
            'severity; a compound with no data at all uses a wide built-in estimate.', '',
            '## Tyre wear and fall-off', '',
            '| Track | Compound | Laps | Stints | Wear/lap, new tyre | Wear/lap at 15 laps old | '
            'Lap-to-lap scatter | Fall-off s/lap | Used |',
            '|---|---|---:|---:|---|---|---:|---|---|']
    for key, model in sorted(store.wear.items(), key=lambda kv: (kv[0].startswith('*'), kv[0])):
        deg = store.deg.get(key)
        track_id = None if key.startswith('*') else model.track_id
        out.append(
            f"| {_track(track_id)} | {model.compound} | {model.n_laps} | {model.n_stints} | "
            f"{_r(model.rate(0, level))} | {_r(model.rate(15, level))} | "
            f"{math.sqrt(model.sigma2):.2f} | {_r(deg.slope(level), 3) if deg else '–'} | "
            f"{'simulated' if model.simulated else 'game'} |")
    if not store.wear:
        out.append('| – | – | 0 | 0 | – | – | – | – | – |')
    out += ['', '## Fuel', '', '| Track | Laps | Sessions | kg per lap | Session-to-session sd | Used |',
            '|---|---:|---:|---|---:|---|']
    for key, model in sorted(store.fuel.items(), key=lambda kv: (kv[0] == '*', kv[0])):
        unit = ' per km' if key == '*' else ''
        out.append(f"| {_track(None if key == '*' else model.track_id)} | {model.n_laps} | "
                   f"{model.n_sessions} | {_r(model.per_lap(level), 3)}{unit} | "
                   f"{math.sqrt(model.tau2):.3f} | {'simulated' if model.simulated else 'game'} |")
    if truth:
        out += ['', '## Fitted vs the simulator\'s true values', '',
                '| Track | Compound | True wear/lap, new | Fitted | Inside range? |', '|---|---|---:|---|---|']
        for (track_id, compound), true_rate in sorted(truth['wear'].items()):
            model = store.wear_model(track_id, compound)
            r = model.rate(0, level)
            out.append(f"| {_track(track_id)} | {compound} | {true_rate:.2f} | {_r(r)} | "
                       f"{'yes' if r.contains(true_rate) else 'no'} ({BASIS_NAMES[model.basis]}) |")
        for track_id, true_fuel in sorted(truth['fuel'].items()):
            length = truth['lengths'][track_id]
            r = store.fuel_model(track_id, length).per_lap(level)
            out.append(f"| {_track(track_id)} | fuel kg/lap | {true_fuel:.2f} | {_r(r, 3)} | "
                       f"{'yes' if r.contains(true_fuel) else 'no'} |")
    out += ['', '## Hold-out check', '']
    if not checks:
        out += ['_Needs at least two sessions._', '']
    else:
        out += [f'Fit on every session but one, then forecast the one left out lap by lap. '
                f'"Inside range" should be close to {pct}%.', '',
                '| What | Model used | Horizon | n | Inside range | Mean abs error | Bias | Range width |',
                '|---|---|---|---:|---:|---:|---:|---:|']
        for row in summarize_checks(checks):
            unit = '% wear' if row['kind'] == 'wear' else 'kg'
            out.append(f"| {'Tyre wear' if row['kind'] == 'wear' else 'Fuel needed'} | "
                       f"{BASIS_NAMES[row['basis']]} | +{row['horizon']} laps | {row['n']} | "
                       f"{row['coverage'] * 100:.0f}% | {row['mae']:.2f} {unit} | {row['bias']:+.2f} | "
                       f"{row['width']:.2f} |")
    out += ['', '## Sources', '']
    for s in store.sources:
        name = s.get('path') or s.get('history')
        rivals = f" + {s['rival_laps']} laps by other cars" if s.get('rival_laps') else ''
        out.append(f"* `{name}`: {s.get('laps', 0)} laps{rivals}{' (simulated)' if s.get('simulated') else ''}")
    out.append('')
    return '\n'.join(out)


# ──────────────────────────────────────────────
# Built-in noisy simulator stints, with known true values
# ──────────────────────────────────────────────

# (track id, length, wear scale, fuel kg/lap, [(player compound, pits)], seeds)
SIM_TRACKS = (
    (4, 4655, 1.15, 1.75, [('soft', {7: 'medium'}), ('medium', {9: 'hard'}), ('soft', {6: 'medium', 12: 'soft'})]),
    (11, 5793, 0.8, 1.95, [('medium', {10: 'hard'}), ('soft', {7: 'hard'}), ('hard', {12: 'medium'})]),
    (7, 5891, 1.3, 1.85, [('medium', {8: 'hard'})]),       # one race only: the pooled fallback
)
SIM_LAPS = 16


def simulated_scenarios():
    from synth import Car, Scenario
    from scenarios import FIELD, NUMBERS
    out = []
    for track_id, length, scale, fuel, races in SIM_TRACKS:
        for n, (compound, pits) in enumerate(races):
            seed = 1000 * track_id + n
            cars = [Car('Player', pace=0.2, compound=compound, grid=3, pits=pits, race_number=NUMBERS[0])]
            cars += [Car(FIELD[i], pace=0.3 * i - 0.5, compound='medium', grid=i if i < 3 else i + 1,
                         pits={SIM_LAPS // 2: 'hard'}, race_number=NUMBERS[i]) for i in (1, 2, 3)]
            out.append(Scenario(
                name=f'fit_{track_id}_{n}', track_id=track_id, track_length=length, session_type=10,
                total_laps=SIM_LAPS, cars=cars, fuel_kg=SIM_LAPS * fuel + 4, wear_scale=scale,
                wear_growth=0.01, wear_noise=0.08, stint_noise=0.10, fuel_per_lap_kg=fuel,
                fuel_noise=0.04, lap_noise_s=0.15, seed=seed,
                description=f'{SIM_LAPS}-lap noisy race for model fitting, {compound} start'))
    return out


def simulated_truth() -> dict:
    """True new-tyre worst-wear rate (% per lap) per (track, compound) and fuel per lap."""
    from synth import COMPOUNDS
    wear, fuel, lengths = {}, {}, {}
    for track_id, length, scale, kg, _races in SIM_TRACKS:
        for name in ('soft', 'medium', 'hard'):
            _visual, actual, _offset, _deg, rate = COMPOUNDS[name]
            # The damage packet's worst wheel is the rear right, 1.08 x the car's wear.
            wear[(track_id, fm.compound_key(actual))] = rate * scale * 1.08
        fuel[track_id] = kg
        lengths[track_id] = length
    return {'wear': wear, 'fuel': fuel, 'lengths': lengths}


def simulated_recordings(regenerate=False) -> list[Path]:
    import recording
    import synth
    SIMULATED_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario in simulated_scenarios():
        path = SIMULATED_DIR / f'{scenario.name}.f1rec.gz'
        if regenerate or not path.exists():
            metadata = {'source': 'synthetic', 'scenario': scenario.name,
                        'generator': 'tests/golden/synth.py via tools/fit_models.py',
                        'description': scenario.description}
            with recording.ReproducibleWriter(path, metadata) as writer:
                for t, data in synth.simulate(scenario):
                    writer.write(data, round(t * 1e9))
        paths.append(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description='Fit tyre and fuel models to recorded F1 2021 stints')
    parser.add_argument('recordings', nargs='*', type=Path, help='.f1rec / .f1rec.gz files')
    parser.add_argument('--history', type=Path, help="The engineer's history database "
                        '(telemetry-history/telemetry.sqlite3) to fit from as well')
    parser.add_argument('--simulated', action='store_true',
                        help='Use the built-in noisy simulator stints (no game data needed)')
    parser.add_argument('--regenerate', action='store_true', help='Rebuild the simulator stints first')
    parser.add_argument('--out', type=Path, default=None,
                        help=f'Where to save the fitted parameters (default: {_display(DEFAULT_OUT)}; '
                             'with only --simulated: recordings/simulated/fit/fitted_models.json)')
    parser.add_argument('--report', type=Path, help='Write the Markdown report here (default: print it)')
    parser.add_argument('--level', type=float, default=fm.DEFAULT_LEVEL,
                        help='Range level for the report (default 0.8 = 10th to 90th percentile)')
    parser.add_argument('--no-validate', action='store_true', help='Skip the hold-out check')
    parser.add_argument('--player-only', action='store_true',
                        help="Fit from your own laps only, not other cars' laps from Session History")
    args = parser.parse_args(argv)

    paths = list(args.recordings)
    truth = None
    if args.simulated:
        paths += simulated_recordings(args.regenerate)
        truth = simulated_truth()
    if not paths and not args.history:
        parser.error('give recordings, --history, or --simulated')
    records, sources, captures = collect(paths, args.history, rivals=not args.player_only)
    if not records:
        print('[FIT] No usable laps found.', file=sys.stderr)
        return 1
    store = fm.ModelStore.fit(records, sources, captures)
    out = args.out
    if out is None:
        out = SIMULATED_DIR / 'fitted_models.json' if store.simulated_only else DEFAULT_OUT
    store.save(out)
    print(f'[FIT] {len(store.wear)} wear, {len(store.deg)} fall-off, {len(store.fuel)} fuel models '
          f'saved to {_display(out)}', file=sys.stderr)
    checks = [] if args.no_validate else validate(records, args.level)
    text = render_report(store, checks, records, truth, args.level)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding='utf-8')
        print(f'[FIT] Report written to {args.report}', file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
