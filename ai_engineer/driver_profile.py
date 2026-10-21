"""What the engineer knows about how *you* drive a track, built from your saved laps.

Every clean lap is already saved in the history database (lap time, sectors, and each
corner's entry/exit times and minimum speed). ``build_profile`` turns all of your laps at
the current track (same layout and formula, dry or wet) into a short profile, e.g.:

    12 clean laps over 3 sessions here. Best 1:27.412, typical 1:28.105 (spread 0.4 s).
    Most time left in sector 2: typical 0.31 s off your best sector.
    Corners where you lose most vs your best: Maggots-Becketts 0.18 s, Stowe 0.11 s.

That goes to DeepSeek with every question, so answers can be about you ("you usually
lose time in sector 2 here"). It grows with every lap you drive: no retraining needed.
"""

from __future__ import annotations

import json
import statistics

MIN_LAPS = 3                 # fewer clean laps than this and there's nothing to say yet
SLOW_LAP_FACTOR = 1.07       # laps slower than 107% of best (traffic, mistakes) are left out
TOP_CORNERS = 2
MIN_CORNER_LOSS_S = 0.05


def _fmt_lap(ms: int) -> str:
    seconds = ms / 1000
    return f'{int(seconds // 60)}:{seconds % 60:06.3f}'


def build_profile(history, session) -> dict | None:
    """Profile from every clean lap at this track/layout/formula/weather, or None if too few."""
    if history is None or session is None:
        return None
    rows = history.connection.execute('''
        SELECT l.capture_id, l.lap_time_ms, json_extract(l.recording, '$.sectors'),
               json_extract(l.recording, '$.corners')
        FROM laps l JOIN sessions s USING(capture_id)
        WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND s.weather_group=?
          AND l.eligible=1 AND l.lap_time_ms > 0
    ''', (session.track_id, session.track_length, session.formula,
          history.weather_group(session.weather))).fetchall()
    if not rows:
        return None
    best_ms = min(row[1] for row in rows)
    clean = [row for row in rows if row[1] <= best_ms * SLOW_LAP_FACTOR]
    if len(clean) < MIN_LAPS:
        return None
    times = [row[1] for row in clean]
    profile = {
        'laps': len(clean),
        'sessions': len({row[0] for row in clean}),
        'best_lap_ms': best_ms,
        'typical_lap_ms': round(statistics.median(times)),
        'spread_s': round(statistics.pstdev(times) / 1000, 2),
    }

    sector_times: dict[int, list[int]] = {}
    corner_times: dict[str, list[int]] = {}
    for _, _, sectors_json, corners_json in clean:
        for sector, ms in (json.loads(sectors_json or '{}') or {}).items():
            if ms and ms > 0:
                sector_times.setdefault(int(sector), []).append(ms)
        for name, data in (json.loads(corners_json or '{}') or {}).items():
            start, end = (data or {}).get('start_time_ms'), (data or {}).get('end_time_ms')
            if start is not None and end is not None and end > start:
                corner_times.setdefault(name, []).append(end - start)

    sector_gaps = {sector: (statistics.median(values) - min(values)) / 1000
                   for sector, values in sector_times.items() if len(values) >= MIN_LAPS}
    if sector_gaps:
        sector = max(sector_gaps, key=sector_gaps.get)
        profile['weakest_sector'] = {'sector': sector, 'typical_loss_s': round(sector_gaps[sector], 2)}

    corner_gaps = [(name, (statistics.median(values) - min(values)) / 1000)
                   for name, values in corner_times.items() if len(values) >= MIN_LAPS]
    corner_gaps = [(name, gap) for name, gap in corner_gaps if gap >= MIN_CORNER_LOSS_S]
    corner_gaps.sort(key=lambda item: item[1], reverse=True)
    if corner_gaps:
        profile['weakest_corners'] = [{'corner': name, 'typical_loss_s': round(gap, 2)}
                                      for name, gap in corner_gaps[:TOP_CORNERS]]
    return profile


def describe(profile: dict | None) -> str | None:
    """The profile as one compact line for the AI prompt."""
    if not profile:
        return None
    parts = [f"{profile['laps']} clean laps over {profile['sessions']} session(s) here; "
             f"best {_fmt_lap(profile['best_lap_ms'])}, typical {_fmt_lap(profile['typical_lap_ms'])} "
             f"(lap-to-lap spread {profile['spread_s']} s)"]
    weak = profile.get('weakest_sector')
    if weak:
        parts.append(f"most time left in sector {weak['sector']} "
                     f"(typically {weak['typical_loss_s']} s off their best sector)")
    corners = profile.get('weakest_corners')
    if corners:
        parts.append('corners losing most vs their best: ' + ', '.join(
            f"{c['corner']} {c['typical_loss_s']} s" for c in corners))
    return '; '.join(parts)
