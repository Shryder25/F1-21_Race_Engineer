"""Refit the tyre, degradation and fuel models from your own laps after every session.

``tools/fit_models.py`` does the same fit by hand. This runs it automatically when the
engineer shuts down, so the wear, fuel and pit-window predictions for each track and
compound get sharper the more you drive. It keeps every recording the last fit used
(so a fit you made from recordings isn't lost) and adds everything in the history
database. Nothing is written when there are no new laps since the last fit.
"""

from __future__ import annotations

import time
from pathlib import Path

import fitted_models as fm

ROOT = Path(__file__).resolve().parents[1]


def _recording_paths(store: fm.ModelStore, extra=()) -> list[Path]:
    paths = []
    for source in store.sources:
        if 'path' in source and not source.get('simulated'):
            path = Path(source['path'])
            path = path if path.is_absolute() else ROOT / path
            if path.exists():
                paths.append(path)
    for path in extra:
        path = Path(path)
        if path.exists() and path not in paths:
            paths.append(path)
    return paths


def _history_laps(store: fm.ModelStore) -> int:
    return sum(s.get('laps', 0) for s in store.sources if 'history' in s)


def refit(history, models_path, extra_recordings=(), log=print) -> fm.ModelStore | None:
    """Refit from the history database (+ recordings) and save. Returns the new store or None."""
    started = time.monotonic()
    models_path = Path(models_path)
    previous = fm.ModelStore.load_or_empty(models_path)
    try:
        history_records, captures = fm.laps_from_history(history, rivals=True)
    except Exception as error:
        log(f'[Learn] Could not read lap history: {error}')
        return None
    player_laps = history.connection.execute('SELECT COUNT(*) FROM lap_stats').fetchone()[0]
    recordings = _recording_paths(previous, extra_recordings)
    if player_laps == 0 and not recordings:
        return None
    if player_laps <= _history_laps(previous) and not extra_recordings:
        return None  # nothing new since the last fit
    records = list(history_records)
    sources = []
    for path in recordings:
        try:
            recs = fm.laps_from_recording(path, rivals=True)
        except Exception as error:
            log(f'[Learn] Skipping recording {path.name}: {error}')
            continue
        records += recs
        try:
            shown = str(path.resolve().relative_to(ROOT))
        except ValueError:
            shown = str(path)
        sources.append(
            {
                'path': shown,
                'laps': sum(r.car_index is None for r in recs),
                'rival_laps': sum(r.car_index is not None for r in recs),
                'simulated': False,
            }
        )
    sources.append(
        {
            'history': 'telemetry.sqlite3',
            'laps': player_laps,
            'rival_laps': sum(r.car_index is not None for r in history_records),
            'sessions': len(captures),
            'simulated': False,
        }
    )
    if not records:
        return None
    store = fm.ModelStore.fit(records, sources, captures)
    store.save(models_path)
    log(
        f'[Learn] Tyre/fuel models updated from {player_laps} of your laps '
        f'({len(store.wear)} wear, {len(store.fuel)} fuel fits) in {time.monotonic() - started:.1f}s.'
    )
    return store
