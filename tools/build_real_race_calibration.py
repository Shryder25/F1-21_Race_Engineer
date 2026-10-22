"""
Build ai_engineer/data/real_race_calibration.json from real Formula 1 races.

Loads race sessions through FastF1 - the same timing data IAmTomShaw/f1-race-replay
replays - for every circuit in the F1 2021 game, and measures what the track really does
to tyres: fuel-corrected degradation per compound, how long stints last, pit-lane time
loss, how many stops winners made, and how often a Safety Car or rain appeared.

The HUD's Track Database page and the DeepSeek track analyst then use these measured
values instead of the built-in baseline ratings.

Usage (needs internet access; first run downloads ~1-2 GB into the FastF1 cache):
    pip install fastf1
    python tools/build_real_race_calibration.py                    # 2018-2024, all tracks
    python tools/build_real_race_calibration.py --years 2021 2022 2023 --tracks 7 10
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ai_engineer'))

from race_calibration import pit_loss, summarize_races  # noqa: E402
from track_data import TRACK_DATABASE  # noqa: E402
from track_profiles import BASELINE_PROFILES, CALIBRATION_PATH  # noqa: E402

NOT_GREEN = set('4567')  # FastF1 TrackStatus: 4 SC, 5 red, 6 VSC, 7 VSC ending


def race_from_session(session, name: str) -> dict:
    """Convert one FastF1 race session into the plain dict race_calibration expects."""
    import pandas as pd

    laps = session.laps
    race_laps = int(laps['LapNumber'].max())
    statuses = laps['TrackStatus'].fillna('').astype(str)
    safety_car = bool(statuses.str.contains('4').any())
    rain = False
    weather = getattr(session, 'weather_data', None)
    if weather is not None and 'Rainfall' in weather:
        rain = bool(weather['Rainfall'].astype(bool).any())
    if laps['Compound'].isin(['INTERMEDIATE', 'WET']).any():
        rain = True

    bad_status = statuses.apply(lambda s: bool(set(s) & NOT_GREEN))
    clean_all = laps[
        (laps['LapNumber'] > 1)
        & laps['LapTime'].notna()
        & laps['PitInTime'].isna()
        & laps['PitOutTime'].isna()
        & ~bad_status
    ]
    stints, pit_losses, stops = [], [], []
    for driver, driver_laps in laps.groupby('Driver'):
        driver_laps = driver_laps.sort_values('LapNumber')
        clean = clean_all[clean_all['Driver'] == driver]
        if clean.empty:
            continue
        green_median = clean['LapTime'].dt.total_seconds().median()
        stint_sizes = driver_laps.groupby('Stint').size()
        for stint_no, stint_laps in clean.groupby('Stint'):
            compound = str(stint_laps['Compound'].iloc[0]).upper()
            rows = [
                (
                    int(r.LapNumber),
                    int(r.TyreLife) if not pd.isna(r.TyreLife) else i + 1,
                    r.LapTime.total_seconds(),
                )
                for i, r in enumerate(stint_laps.itertuples())
            ]
            stints.append(
                {'compound': compound, 'laps': rows, 'length': int(stint_sizes.get(stint_no, len(rows)))}
            )
        # pit loss: in-lap + out-lap vs two green laps
        for _, in_lap in driver_laps[driver_laps['PitInTime'].notna()].iterrows():
            out = driver_laps[driver_laps['LapNumber'] == in_lap['LapNumber'] + 1]
            if out.empty or pd.isna(in_lap['LapTime']) or pd.isna(out['LapTime'].iloc[0]):
                continue
            if set(str(in_lap['TrackStatus'])) & NOT_GREEN:
                continue
            pit_losses.append(
                pit_loss(
                    in_lap['LapTime'].total_seconds(), out['LapTime'].iloc[0].total_seconds(), green_median
                )
            )
        if driver_laps['LapNumber'].max() >= 0.9 * race_laps:
            stops.append(int(driver_laps['Stint'].max()) - 1)
    return {
        'name': name,
        'race_laps': race_laps,
        'rain': rain,
        'safety_car': safety_car,
        'stints': stints,
        'pit_losses': pit_losses,
        'stops': stops,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--years', type=int, nargs='+', default=list(range(2018, 2025)))
    parser.add_argument(
        '--tracks', type=int, nargs='+', default=None, help='TRACK_DATABASE ids (default: all)'
    )
    parser.add_argument('--cache', type=Path, default=ROOT / '.fastf1-cache')
    parser.add_argument('--out', type=Path, default=CALIBRATION_PATH)
    args = parser.parse_args()

    try:
        import fastf1
    except ImportError:
        print('FastF1 is not installed: pip install fastf1')
        return 1
    args.cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(args.cache))

    wanted = args.tracks or sorted(BASELINE_PROFILES)
    names = {name: tid for tid in wanted for name in BASELINE_PROFILES[tid]['fastf1_names']}
    races = {tid: [] for tid in wanted}
    for year in args.years:
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as error:
            print(f'{year}: schedule unavailable ({error})')
            continue
        for _, event in schedule.iterrows():
            tid = names.get(event['EventName'])
            if tid is None:
                continue
            label = f"{year} {event['EventName']}"
            try:
                session = fastf1.get_session(year, event['RoundNumber'], 'R')
                session.load(laps=True, telemetry=False, weather=True, messages=False)
                races[tid].append(race_from_session(session, label))
                print(f'  loaded {label}')
            except Exception as error:
                print(f'  skipped {label}: {error}')

    existing = {}
    if args.out.exists():
        existing = json.loads(args.out.read_text(encoding='utf-8')).get('tracks', {})
    for tid, track_races in races.items():
        summary = summarize_races(track_races)
        if summary:
            existing[str(tid)] = summary
            print(
                f"{TRACK_DATABASE[tid]['short_name']}: {summary['deg_s_per_lap']} from {len(track_races)} races"
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                'generated_at': datetime.now(UTC).isoformat(timespec='seconds'),
                'source': 'FastF1 race timing data (as used by IAmTomShaw/f1-race-replay)',
                'years': args.years,
                'tracks': existing,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    print(f'Wrote {args.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
