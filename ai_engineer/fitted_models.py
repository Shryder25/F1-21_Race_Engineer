"""Tyre wear, tyre degradation and fuel models fitted to the game, with ranges.

The engineer used to quote single numbers ("2.5% wear a lap", "1.7 kg a lap") from a curve
fitted to whatever laps it happened to have. This module replaces those with models fitted
per tyre compound and per track from recorded stints, and every answer is a range
(by default the 10th to 90th percentile, an 80% interval) so strategy can reason about how
sure it is.

Three models, all Bayesian linear models with a random effect per stint or session:

* **Wear**: worst-tyre wear added over one lap = r0 + r1 * tyre age + stint effect + noise.
  The stint effect captures that two stints on the same tyre never wear quite alike
  (driving, fuel load, setup); the noise is lap-to-lap scatter.
* **Degradation**: fuel-corrected lap time lost per lap of tyre age (s/lap), with its own
  stint-to-stint spread.
* **Fuel**: kg burned per racing lap, with a session-to-session spread (fuel mix, lift and
  coast, race length).

Fitting falls back in steps, so there is always an answer and it is only as confident as
the data behind it:

  built-in prior  ->  compound fit pooled over every track  ->  this track's fit
                  ->  conditioned on the laps already done in this stint / session

A track with no data uses the pooled compound fit (scaled to the track's tyre severity) with
extra between-track uncertainty; a compound with no data anywhere uses the wide built-in
prior. Within a stint, every lap driven narrows the range for the rest of that stint.

Fitted parameters are saved as JSON (see ModelStore) by tools/fit_models.py, built from
`.f1rec` recordings (docs/udp-recording-format.md) and the history database's lap_stats.
Everything here is plain arithmetic on numbers passed in, so it is testable without the game.
"""

from __future__ import annotations

import gzip
import json
import math
import statistics
import struct
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np

from packets import (ACTUAL_COMPOUNDS, HEADER_SIZE, VISUAL_COMPOUNDS, PacketHeader,
                     parse_player_car_damage, parse_player_car_status, parse_player_lap_data,
                     parse_session_data)
from physics_models import _compound_params
from track_data import get_track_info, resolve_track_id

STORE_VERSION = 1
DEFAULT_LEVEL = 0.80            # 10th-90th percentile
FUEL_EFFECT_S_PER_KG = 0.03     # lap time cost of 1 kg of fuel, used to fuel-correct lap times
MAX_LOOKAHEAD_LAPS = 80
SNAPSHOT_DISTANCE_M = 400.0     # read tyre age/wear/fuel this far into each lap (past pit exit)

# ── Built-in priors (used only where there is no data; deliberately wide) ──
# Worst-tyre wear, % per lap, on a track of medium tyre severity in a dry race.
WEAR_PRIOR_PCT = {'C5': 3.6, 'C4': 3.0, 'C3': 2.5, 'C2': 2.0, 'C1': 1.6, 'C0': 1.3,
                  'INTER': 2.6, 'WET': 2.0}
WEAR_PRIOR_REL_SD = 0.5             # +/- 50% on the base rate
WEAR_PRIOR_GROWTH_SD = 0.008        # r1 sd, as a fraction of the base rate per lap of age
WEAR_SIGMA0 = (0.10, 0.10)          # lap-to-lap noise sd = a + b * rate
WEAR_TAU0_REL = 0.15                # stint-to-stint sd of the rate, fraction of the rate
TRACK_SEVERITY = {'Low': 0.8, 'Medium': 1.0, 'High': 1.15, 'Very High': 1.3}
BETWEEN_TRACK_REL_SD = 0.25         # how far one track's wear can sit from the pooled scaling

DEG_PRIOR_SD = 0.06                 # s/lap around the physics model's ageing rate
DEG_SIGMA0 = 0.35                   # lap-time scatter, s
DEG_TAU0 = 0.03                     # stint-to-stint spread of the slope, s/lap
DEG_BETWEEN_TRACK_SD = 0.03

FUEL_PRIOR_KG_PER_KM = 0.33
FUEL_PRIOR_REL_SD = 0.35
FUEL_SIGMA0_REL = 0.05
FUEL_TAU0_REL = 0.06
FUEL_BETWEEN_TRACK_REL_SD = 0.12
FUEL_RESERVE_KG = 0.0               # margins are quoted against an empty tank

RACE_DISTANCE_REL_SD = 0.25         # extra spread when racing a distance the fit has not seen

# Pseudo-counts that pull noisy variance estimates towards the priors.
NU0 = 4.0          # for the lap-to-lap noise
KAPPA0 = 2.0       # for the between-stint/session spread

RACE_SESSION_TYPES = (10, 11, 12)
SLICKS = ('C0', 'C1', 'C2', 'C3', 'C4', 'C5')


def z_score(level: float = DEFAULT_LEVEL) -> float:
    return NormalDist().inv_cdf(0.5 + level / 2.0)


def compound_key(compound) -> str | None:
    """'C1'..'C5', 'INTER' or 'WET' from an actual-compound id (16-20, 7, 8) or a name."""
    if compound is None:
        return None
    if isinstance(compound, (int, np.integer)):
        compound = ACTUAL_COMPOUNDS.get(int(compound))
        if compound is None:
            return None
    key = str(compound).strip().upper()
    if key.startswith('INTER'):
        return 'INTER'
    if key in WEAR_PRIOR_PCT:
        return key
    return None


def track_severity(track_id: int | None) -> float:
    info = get_track_info(resolve_track_id(track_id)) if track_id is not None and track_id >= 0 else None
    return TRACK_SEVERITY.get((info or {}).get('tyre_wear'), 1.0)


def _num(x, places=3):
    return None if x is None or not math.isfinite(x) else round(float(x), places)


@dataclass(frozen=True)
class Range:
    """A central estimate with a lower and upper bound (``level`` two-sided interval)."""
    low: float | None
    mid: float | None
    high: float | None
    level: float = DEFAULT_LEVEL

    def as_list(self, places: int = 2) -> list:
        return [_num(self.low, places), _num(self.mid, places), _num(self.high, places)]

    def contains(self, value: float) -> bool:
        return self.low is not None and self.high is not None and self.low <= value <= self.high

    @property
    def width(self) -> float | None:
        if self.low is None or self.high is None:
            return None
        return self.high - self.low


def _normal_range(mean: float, var: float, level: float, offset: float = 0.0) -> Range:
    sd = math.sqrt(max(var, 0.0))
    z = z_score(level)
    return Range(offset + mean - z * sd, offset + mean, offset + mean + z * sd, level)


# ──────────────────────────────────────────────
# Lap records: one per lap, from recordings, the history database or the live session
# ──────────────────────────────────────────────

@dataclass
class LapRecord:
    source: str                   # recording path, 'history:<capture>' or 'live'
    session: str                  # identifies one session (for the fuel session effect)
    stint: str                    # identifies one set of tyres in one session
    track_id: int
    track_length: int
    session_type: int
    race_laps: int                # session total laps (0 outside a race)
    compound: str                 # compound_key(...)
    lap: int
    tyre_age: int                 # tyre age at the start of the lap
    wear_start: float | None = None   # worst-tyre wear %, start and end of the lap (same stint)
    wear_end: float | None = None
    fuel_start: float | None = None
    fuel_end: float | None = None
    lap_time_ms: int | None = None
    fuel_ok: bool = False         # racing lap: no pit stop, no Safety Car
    time_ok: bool = False         # fuel_ok, a valid complete lap time, and not a standing start
    wet: bool = False             # slick tyres on a wet track (kept out of the slick fits)
    simulated: bool = False
    visual: str | None = None     # 'Soft' / 'Medium' / ... as the game showed it

    @property
    def wear_increment(self) -> float | None:
        if self.wear_start is None or self.wear_end is None:
            return None
        inc = self.wear_end - self.wear_start
        return inc if -0.05 <= inc <= 25.0 else None    # resets, punctures, garbage

    @property
    def fuel_used(self) -> float | None:
        if not self.fuel_ok or self.fuel_start is None or self.fuel_end is None:
            return None
        used = self.fuel_start - self.fuel_end
        return used if 0.05 < used < 8.0 else None

    @property
    def corrected_time_s(self) -> float | None:
        if not self.time_ok or not self.lap_time_ms or self.fuel_start is None:
            return None
        return self.lap_time_ms / 1000.0 - FUEL_EFFECT_S_PER_KG * self.fuel_start


def records_from_lap_rows(rows: Sequence[dict], *, source: str, session: str, track_id: int,
                          track_length: int, session_type: int = 10, race_laps: int = 0,
                          wet: bool = False, simulated: bool = False) -> list[LapRecord]:
    """Lap records from lap-end rows (the engine's session_lap_stats or history lap_stats).

    Each row is a snapshot at the end of a lap: lap_num, compound (actual id), tyre_age_laps,
    wear (4 wheels) or worst_wear, fuel_start, fuel_end, lap_time_ms, eligible. A snapshot
    taken right on the line can still show the previous lap's tyre age and wear, so wear per
    lap comes from the distinct (tyre age, wear) points of each set: one lap apart gives one
    lap's wear, two laps apart is split evenly over both. Fuel and lap time come from each row.
    """
    out: list[LapRecord] = []
    stint_no = 0
    prev = None
    points: list[tuple[int, float]] = []          # (tyre age, worst wear) for the set in use

    def wear_records(stint, compound, lap):
        recs = []
        for (a, w0), (b, w1) in zip(points, points[1:]):
            if b - a not in (1, 2):
                continue
            share = (w1 - w0) / (b - a)
            for k in range(b - a):
                recs.append(LapRecord(
                    source=source, session=session, stint=stint, track_id=track_id,
                    track_length=track_length, session_type=session_type, race_laps=race_laps,
                    compound=compound, lap=lap, tyre_age=a + k, wear_start=w0 + k * share,
                    wear_end=w0 + (k + 1) * share, wet=wet and compound in SLICKS,
                    simulated=simulated))
        return recs

    for row in sorted(rows, key=lambda r: r['lap_num']):
        compound = compound_key(row.get('compound'))
        age = row.get('tyre_age_laps')
        wear = row.get('worst_wear')
        if wear is None and row.get('wear'):
            wear = max(row['wear'])
        if compound is None or age is None:
            continue
        age = int(age)
        if prev is not None and not (prev[0] == compound and age >= prev[1]):
            out.extend(wear_records(f'{session}#{stint_no}', prev[0], prev[2]))
            stint_no += 1
            points = []
        if wear is not None and not points and age == 1:
            points.append((0, 0.0))             # a set one lap old was new at the start of it
        if wear is not None and (not points or age > points[-1][0]):
            points.append((age, float(wear)))
        lap_num = row['lap_num']
        eligible = bool(row.get('eligible'))
        out.append(LapRecord(
            source=source, session=session, stint=f'{session}#{stint_no}', track_id=track_id,
            track_length=track_length, session_type=session_type, race_laps=race_laps,
            compound=compound, lap=lap_num, tyre_age=max(0, age - 1),
            fuel_start=row.get('fuel_start'), fuel_end=row.get('fuel_end'),
            lap_time_ms=row.get('lap_time_ms'), fuel_ok=eligible,
            time_ok=eligible and lap_num >= 2 and bool(row.get('lap_time_ms')),
            wet=wet and compound in SLICKS, simulated=simulated))
        prev = (compound, age, lap_num)
    if prev is not None:
        out.extend(wear_records(f'{session}#{stint_no}', prev[0], prev[2]))
    _drop_time_outliers(out)
    return out


def _drop_time_outliers(records: list[LapRecord]):
    """Mark laps far off the stint's typical time (traffic, mistakes, Safety Car) as not clean."""
    by_stint = defaultdict(list)
    for r in records:
        if r.time_ok and r.lap_time_ms:
            by_stint[r.stint].append(r)
    for laps in by_stint.values():
        if len(laps) < 3:
            continue
        median = statistics.median(r.lap_time_ms for r in laps)
        for r in laps:
            if r.lap_time_ms > median + 2500 or r.lap_time_ms < median - 2500:
                r.time_ok = False


class _SessionScan:
    """Per-session state while reading a recording."""

    def __init__(self, uid):
        self.uid = uid
        self.session = None
        self.snaps: dict[int, dict] = {}       # lap -> snapshot at SNAPSHOT_DISTANCE_M
        self.flags: dict[int, dict] = defaultdict(lambda: {'pit': False, 'sc': False, 'invalid': False})
        self.lap_times: dict[int, int] = {}
        self.status = None
        self.damage = None
        self.last_lap = None


def laps_from_recording(path, simulated: bool | None = None) -> list[LapRecord]:
    """Lap records for the player's car from a `.f1rec` recording (gzipped or not).

    Tyre age, worst-tyre wear and fuel are read once per lap at SNAPSHOT_DISTANCE_M into it,
    so the numbers are steady (the line itself is where packets disagree about which lap it
    is). Flashbacks rewind: laps at or after the one rewound to are thrown away and re-read.
    """
    from udp_recording import RecordingReader

    path = Path(path)
    scans: dict[int, _SessionScan] = {}
    with RecordingReader(path) as reader:
        metadata = dict(reader.metadata)
        if simulated is None:
            simulated = bool(metadata.get('simulated')) or metadata.get('source') == 'synthetic'
        for record in reader:
            _scan_datagram(scans, record.data)
    out = []
    for scan in scans.values():
        out.extend(_records_from_scan(scan, str(path), simulated))
    return out


def _scan_datagram(scans: dict, data: bytes):
    if len(data) < HEADER_SIZE:
        return
    try:
        header = PacketHeader.from_bytes(data)
    except (struct.error, ValueError):
        return
    if header.packet_format != 2021 or not 0 <= header.player_car_index < 22:
        return
    pid, me = header.packet_id, header.player_car_index
    sizes = {1: 625, 2: 970, 7: 1058, 10: 882}
    if pid not in sizes or len(data) != sizes[pid]:
        return
    scan = scans.get(header.session_uid)
    if pid == 1:
        session = parse_session_data(data)
        if session.track_id < 0 or session.track_length <= 0:
            return
        scan = scans.setdefault(header.session_uid, _SessionScan(header.session_uid))
        scan.session = session
        if scan.last_lap is not None and session.safety_car_status:
            scan.flags[scan.last_lap]['sc'] = True
        return
    if scan is None or scan.session is None:
        return
    if pid == 7:
        scan.status = parse_player_car_status(data, me)
        return
    if pid == 10:
        scan.damage = parse_player_car_damage(data, me)
        return
    lap = parse_player_lap_data(data, me)
    n = lap.current_lap_num
    if n <= 0:
        return
    if scan.last_lap is not None and n < scan.last_lap:          # flashback across the line
        for store in (scan.snaps, scan.flags, scan.lap_times):
            for key in [k for k in store if k >= n]:
                del store[key]
    if scan.last_lap is not None and n == scan.last_lap + 1 and lap.last_lap_time_ms > 0:
        scan.lap_times[scan.last_lap] = lap.last_lap_time_ms
    scan.last_lap = n
    flags = scan.flags[n]
    if lap.pit_status:
        flags['pit'] = True
    if lap.current_lap_invalid:
        flags['invalid'] = True
    if scan.session.safety_car_status:
        flags['sc'] = True
    if (n not in scan.snaps and lap.lap_distance >= SNAPSHOT_DISTANCE_M and lap.driver_status != 0
            and lap.result_status == 2 and scan.status is not None and scan.damage is not None):
        status = scan.status
        scan.snaps[n] = {
            'age': status.tyres_age_laps, 'actual': status.actual_tyre_compound,
            'visual': VISUAL_COMPOUNDS.get(status.visual_tyre_compound),
            'wear': max(scan.damage.tyres_wear), 'fuel': status.fuel_in_tank,
            'pit': lap.pit_status != 0, 'weather': scan.session.weather,
            'stops': lap.num_pit_stops,
        }


def _records_from_scan(scan: _SessionScan, source: str, simulated: bool) -> list[LapRecord]:
    session = scan.session
    if session is None or session.formula in (1, 2):      # classic cars and F2 are other models
        return []
    is_race = session.session_type in RACE_SESSION_TYPES
    sid = f'{source}#{scan.uid}'
    out = []
    stint_no = 0
    prev_key = None
    for n in sorted(scan.snaps):
        snap, nxt = scan.snaps[n], scan.snaps.get(n + 1)
        compound = compound_key(snap['actual'])
        if compound is None:
            continue
        key = (compound, snap['stops'])
        if prev_key is not None and key != prev_key:
            stint_no += 1
        prev_key = key
        same_set = (nxt is not None and compound_key(nxt['actual']) == compound
                    and nxt['age'] == snap['age'] + 1 and nxt['stops'] == snap['stops'])
        flags = scan.flags.get(n, {})
        racing = (not flags.get('pit') and not flags.get('sc') and not snap['pit']
                  and not (nxt and nxt['pit']))
        lap_time = scan.lap_times.get(n)
        out.append(LapRecord(
            source=source, session=sid, stint=f'{sid}#{stint_no}', track_id=session.track_id,
            track_length=session.track_length, session_type=session.session_type,
            race_laps=session.total_laps if is_race else 0, compound=compound, lap=n,
            tyre_age=int(snap['age']),
            wear_start=snap['wear'] if same_set else None,
            wear_end=nxt['wear'] if same_set else None,
            fuel_start=snap['fuel'] if nxt is not None else None,
            fuel_end=nxt['fuel'] if nxt is not None else None,
            lap_time_ms=lap_time, fuel_ok=racing and nxt is not None,
            time_ok=(racing and bool(lap_time) and not flags.get('invalid')
                     and (n >= 2 or not is_race)),      # a standing start is no pace lap
            wet=snap['weather'] >= 3 and compound in SLICKS, simulated=simulated,
            visual=snap['visual']))
    _drop_time_outliers(out)
    return out


def laps_from_history(history, exclude_captures: Iterable[str] = (),
                      track_id: int | None = None) -> tuple[list[LapRecord], set[str]]:
    """Lap records from a TelemetryHistory database, and the capture ids they came from."""
    excluded = set(exclude_captures)
    where, args = ('AND s.track_id=?', (track_id,)) if track_id is not None else ('', ())
    rows = history.connection.execute(f'''
        SELECT s.capture_id, s.track_id, s.track_length, s.formula, s.weather_group, s.session_type,
               l.lap_num, l.lap_time_ms, l.eligible, l.compound, l.tyre_age_laps,
               MAX(l.wear_rl, l.wear_rr, l.wear_fl, l.wear_fr), l.fuel_start, l.fuel_end
        FROM lap_stats l JOIN sessions s USING(capture_id)
        WHERE l.compound IS NOT NULL AND l.tyre_age_laps IS NOT NULL {where}
        ORDER BY s.capture_id, l.lap_num
    ''', args).fetchall()
    grouped = defaultdict(list)
    meta = {}
    for (capture, track_id, length, formula, weather, stype, lap_num, lap_ms, eligible, compound,
         age, wear, fuel_start, fuel_end) in rows:
        if capture in excluded or formula in (1, 2):
            continue
        meta[capture] = (track_id, length, weather, stype)
        grouped[capture].append({'lap_num': lap_num, 'lap_time_ms': lap_ms, 'eligible': eligible,
                                 'compound': compound, 'tyre_age_laps': age, 'worst_wear': wear,
                                 'fuel_start': fuel_start, 'fuel_end': fuel_end})
    out = []
    for capture, laps in grouped.items():
        track_id, length, weather, stype = meta[capture]
        out.extend(records_from_lap_rows(
            laps, source=f'history:{capture}', session=f'history:{capture}', track_id=track_id,
            track_length=length, session_type=stype, race_laps=0, wet=weather != 'dry'))
    return out, set(grouped)


# ──────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────

def _prior_wear(compound: str, track_id: int | None) -> 'WearModel':
    rate = WEAR_PRIOR_PCT.get(compound, 2.5) * track_severity(track_id)
    sigma = WEAR_SIGMA0[0] + WEAR_SIGMA0[1] * rate
    return WearModel(compound, track_id, [rate, 0.0],
                     [[(WEAR_PRIOR_REL_SD * rate) ** 2, 0.0], [0.0, (WEAR_PRIOR_GROWTH_SD * rate) ** 2]],
                     sigma * sigma, (WEAR_TAU0_REL * rate) ** 2, basis='prior')


@dataclass
class WearModel:
    """Worst-tyre wear % added per lap = r0 + r1 * tyre age (+ stint effect + noise)."""
    compound: str
    track_id: int | None              # None for a pooled (all-track, severity-normalised) fit
    mean: list                        # [r0, r1]
    cov: list                         # 2x2 posterior covariance of [r0, r1]
    sigma2: float                     # lap-to-lap noise variance
    tau2: float                       # stint-to-stint variance of the rate
    n_laps: int = 0
    n_stints: int = 0
    basis: str = 'prior'              # prior / pooled / track
    simulated: bool = False
    race_laps: list = field(default_factory=list)

    def rate(self, age: float, level: float = DEFAULT_LEVEL) -> Range:
        """Expected wear per lap at this tyre age for a new stint (includes the stint spread)."""
        g = np.array([1.0, age])
        var = float(g @ np.array(self.cov) @ g) + self.tau2
        return _normal_range(float(g @ np.array(self.mean)), var, level)

    def scaled(self, factor: float, extra_rel_sd: float = 0.0, track_id=None, basis=None) -> 'WearModel':
        m = [self.mean[0] * factor, self.mean[1] * factor]
        cov = (np.array(self.cov) * factor * factor)
        cov[0][0] += (extra_rel_sd * m[0]) ** 2
        return WearModel(self.compound, track_id, m, cov.tolist(), self.sigma2 * factor * factor,
                         self.tau2 * factor * factor, self.n_laps, self.n_stints,
                         basis or self.basis, self.simulated, list(self.race_laps))

    def condition(self, ages: Sequence[float], increments: Sequence[float]) -> 'StintWear':
        """Combine the model with the laps already done on this set of tyres."""
        m = np.array([self.mean[0], self.mean[1], 0.0])
        V = np.zeros((3, 3))
        V[:2, :2] = np.array(self.cov)
        V[2, 2] = self.tau2
        if len(increments):
            X = np.array([[1.0, a, 1.0] for a in ages])
            y = np.array(increments, dtype=float)
            P0 = np.linalg.pinv(V)
            P = P0 + X.T @ X / self.sigma2
            V = np.linalg.pinv(P)
            m = V @ (P0 @ m + X.T @ y / self.sigma2)
        return StintWear(self, m, V, len(increments))


@dataclass
class StintWear:
    """A wear model conditioned on the current stint: forecasts for the rest of it."""
    model: WearModel
    beta: np.ndarray                  # [r0, r1, this stint's effect]
    cov: np.ndarray
    stint_laps: int

    def rate(self, age: float, level: float = DEFAULT_LEVEL) -> Range:
        g = np.array([1.0, age, 1.0])
        return _normal_range(float(g @ self.beta), float(g @ self.cov @ g), level)

    def _sum(self, age: float, laps: int) -> tuple[float, float]:
        if laps <= 0:
            return 0.0, 0.0
        ages = age + np.arange(laps)
        g = np.array([float(laps), float(ages.sum()), float(laps)])
        return float(g @ self.beta), float(g @ self.cov @ g) + laps * self.model.sigma2

    def wear_after(self, age: float, wear_now: float, laps: int, level: float = DEFAULT_LEVEL) -> Range:
        """Worst-tyre wear % after ``laps`` more laps, starting at this age and wear."""
        mean, var = self._sum(age, laps)
        r = _normal_range(mean, var, level, wear_now)
        return Range(max(wear_now, r.low), r.mid, r.high, level)

    def laps_until(self, threshold: float, age: float, wear_now: float,
                   level: float = DEFAULT_LEVEL) -> Range:
        """Laps until worst-tyre wear reaches ``threshold`` %: (soonest, expected, latest)."""
        if wear_now >= threshold:
            return Range(0.0, 0.0, 0.0, level)
        z = z_score(level)
        need = threshold - wear_now
        found = [None, None, None]                    # high wear first -> soonest
        prev = [0.0, 0.0, 0.0]
        for k in range(1, MAX_LOOKAHEAD_LAPS + 1):
            mean, var = self._sum(age, k)
            sd = math.sqrt(max(var, 0.0))
            vals = [mean + z * sd, mean, mean - z * sd]
            for i, v in enumerate(vals):
                if found[i] is None and v >= need:
                    step = v - prev[i]
                    frac = (need - prev[i]) / step if step > 0 else 1.0
                    found[i] = round(k - 1 + min(max(frac, 0.0), 1.0), 2)
            prev = vals
            if all(f is not None for f in found):
                break
        return Range(found[0], found[1], found[2], level)

    def prob_below(self, threshold: float, age: float, wear_now: float, laps: int) -> float:
        """Chance worst-tyre wear is still under ``threshold`` after ``laps`` laps."""
        if laps <= 0:
            return 1.0 if wear_now < threshold else 0.0
        mean, var = self._sum(age, laps)
        sd = math.sqrt(max(var, 1e-12))
        return NormalDist(wear_now + mean, sd).cdf(threshold)


def _prior_deg(compound: str) -> 'DegModel':
    params = _compound_params({'INTER': 'INTERMEDIATE'}.get(compound, compound))
    return DegModel(compound, None, params.alpha + params.beta * 10.0, DEG_PRIOR_SD ** 2,
                    DEG_SIGMA0 ** 2, DEG_TAU0 ** 2, basis='prior')


@dataclass
class DegModel:
    """Fuel-corrected lap time lost per lap of tyre age, s/lap."""
    compound: str
    track_id: int | None
    mean: float
    var: float
    sigma2: float                     # lap-time scatter about each stint's line
    tau2: float                       # stint-to-stint spread of the slope
    n_laps: int = 0
    n_stints: int = 0
    basis: str = 'prior'
    simulated: bool = False

    def slope(self, level: float = DEFAULT_LEVEL) -> Range:
        return _normal_range(self.mean, self.var + self.tau2, level)

    def condition(self, ages: Sequence[float], times_s: Sequence[float],
                  level: float = DEFAULT_LEVEL) -> Range:
        """This stint's slope, combining the model with the stint's own laps."""
        mean, var = self.mean, self.var + self.tau2
        if len(ages) >= 3:
            x = np.array(ages, float)
            sxx = float(((x - x.mean()) ** 2).sum())
            if sxx > 0:
                y = np.array(times_s, float)
                b = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
                prec = 1.0 / var + sxx / self.sigma2
                mean = (mean / var + b * sxx / self.sigma2) / prec
                var = 1.0 / prec
        return _normal_range(mean, var, level)


def _prior_fuel(track_id: int | None, track_length: int | None) -> 'FuelModel':
    km = (track_length or 5000) / 1000.0
    mu = FUEL_PRIOR_KG_PER_KM * km
    return FuelModel(track_id, track_length or 0, mu, (FUEL_PRIOR_REL_SD * mu) ** 2,
                     (FUEL_SIGMA0_REL * mu) ** 2, (FUEL_TAU0_REL * mu) ** 2, basis='prior')


@dataclass
class FuelModel:
    """kg of fuel burned per racing lap (+ session effect + noise)."""
    track_id: int | None              # None for pooled (kg per km)
    track_length: int
    mean: float
    var: float
    sigma2: float
    tau2: float                       # session-to-session spread
    n_laps: int = 0
    n_sessions: int = 0
    basis: str = 'prior'
    simulated: bool = False
    race_laps: list = field(default_factory=list)

    def per_lap(self, level: float = DEFAULT_LEVEL) -> Range:
        return _normal_range(self.mean, self.var + self.tau2, level)

    def scaled(self, factor: float, extra_rel_sd: float = 0.0, track_id=None, track_length=0,
               basis=None) -> 'FuelModel':
        mean = self.mean * factor
        return FuelModel(track_id, track_length, mean,
                         self.var * factor * factor + (extra_rel_sd * mean) ** 2,
                         self.sigma2 * factor * factor, self.tau2 * factor * factor, self.n_laps,
                         self.n_sessions, basis or self.basis, self.simulated, list(self.race_laps))

    def condition(self, used: Sequence[float]) -> 'SessionFuel':
        mean, var = self.mean, self.var + self.tau2
        used = list(used)
        if len(used) >= 3:          # drop laps far off this session's own typical burn
            med = statistics.median(used)
            used = [u for u in used if abs(u - med) <= 0.25 * med]
        if used:
            n = len(used)
            prec = 1.0 / var + n / self.sigma2
            mean = (mean / var + sum(used) / self.sigma2) / prec
            var = 1.0 / prec
        return SessionFuel(self, mean, var, len(used))


@dataclass
class SessionFuel:
    model: FuelModel
    mean: float
    var: float
    session_laps: int

    def per_lap(self, level: float = DEFAULT_LEVEL) -> Range:
        return _normal_range(self.mean, self.var, level)

    def needed(self, laps: float, level: float = DEFAULT_LEVEL) -> Range:
        laps = max(0.0, laps)
        return _normal_range(laps * self.mean, laps * laps * self.var + laps * self.model.sigma2, level)

    def margin(self, fuel_kg: float, laps: float, level: float = DEFAULT_LEVEL) -> tuple[Range, Range]:
        """(kg left at the flag, the same in laps of fuel): low end = the worrying case."""
        need = self.needed(laps, level)
        kg = Range(fuel_kg - need.high, fuel_kg - need.mid, fuel_kg - need.low, level)
        per = self.mean if self.mean > 0 else 1.0
        return kg, Range(kg.low / per, kg.mid / per, kg.high / per, level)

    def laps_of_fuel(self, fuel_kg: float, level: float = DEFAULT_LEVEL) -> Range:
        per = self.per_lap(level)
        lo = fuel_kg / per.high if per.high and per.high > 0 else None
        hi = fuel_kg / per.low if per.low and per.low > 0 else None
        mid = fuel_kg / per.mid if per.mid and per.mid > 0 else None
        return Range(lo, mid, hi, level)


# ──────────────────────────────────────────────
# Fitting
# ──────────────────────────────────────────────

def _variance_components(groups: list[tuple[np.ndarray, np.ndarray]], theta: np.ndarray,
                         sigma2_0: float, tau2_0: float) -> tuple[float, float]:
    """Lap-to-lap noise and stint-to-stint spread from residuals, pulled towards the priors."""
    ssw, dof_w, means, inv_n = 0.0, 0, [], []
    for X, y in groups:
        r = y - X @ theta
        ssw += float(((r - r.mean()) ** 2).sum())
        dof_w += len(r) - 1
        means.append(float(r.mean()))
        inv_n.append(1.0 / len(r))
    sigma2 = (NU0 * sigma2_0 + ssw) / (NU0 + dof_w)
    tau2 = tau2_0
    if len(means) >= 2:
        between = statistics.variance(means) - sigma2 * statistics.fmean(inv_n)
        tau_hat = max(0.0, between)
        k = len(means) - 1
        tau2 = (KAPPA0 * tau2_0 + k * tau_hat) / (KAPPA0 + k)
    return max(sigma2, 1e-6), max(tau2, 1e-8)


def _gls_posterior(m0, V0, groups, sigma2, tau2):
    """Posterior of a linear model with a random intercept per group (exact GLS update)."""
    m0, V0 = np.array(m0, float), np.array(V0, float)
    P0 = np.linalg.pinv(V0)
    P, b = P0.copy(), P0 @ m0
    for X, y in groups:
        n = len(y)
        c = tau2 / (sigma2 + n * tau2)
        s_x, s_y = X.sum(axis=0), y.sum()
        P += (X.T @ X - c * np.outer(s_x, s_x)) / sigma2
        b += (X.T @ y - c * s_x * s_y) / sigma2
    V = np.linalg.pinv(P)
    return V @ b, V


def fit_wear(records: Sequence[LapRecord], prior: WearModel, scale: dict | None = None) -> WearModel:
    """Fit a wear model to records (one compound). ``scale`` maps track id -> severity divisor."""
    groups = defaultdict(lambda: ([], []))
    for r in records:
        inc = r.wear_increment
        if inc is None or r.wet:
            continue
        s = (scale or {}).get(r.track_id, 1.0)
        groups[r.stint][0].append([1.0, float(r.tyre_age)])
        groups[r.stint][1].append(inc / s)
    data = [(np.array(X), np.array(y)) for X, y in groups.values() if y]
    n = sum(len(y) for _, y in data)
    if n == 0:
        return prior
    sigma2, tau2 = prior.sigma2, prior.tau2
    theta, _ = _gls_posterior(prior.mean, prior.cov, data, sigma2, tau2)
    for _ in range(2):
        sigma2, tau2 = _variance_components(data, theta, prior.sigma2, prior.tau2)
        theta, V = _gls_posterior(prior.mean, prior.cov, data, sigma2, tau2)
    race_laps = sorted({r.race_laps for r in records if r.race_laps and r.wear_increment is not None})
    return WearModel(prior.compound, prior.track_id, theta.tolist(), V.tolist(), sigma2, tau2,
                     n, len(data), 'track' if prior.track_id is not None else 'pooled',
                     all(r.simulated for r in records if r.wear_increment is not None), race_laps)


def fit_deg(records: Sequence[LapRecord], prior: DegModel) -> DegModel:
    """Within-stint slope of fuel-corrected lap time vs tyre age, pooled across stints."""
    stints = defaultdict(list)
    for r in records:
        t = r.corrected_time_s
        if t is not None and not r.wet:
            stints[r.stint].append((float(r.tyre_age), t))
    slopes, ssr, dof = [], 0.0, 0
    for laps in stints.values():
        if len(laps) < 3:
            continue
        x = np.array([a for a, _ in laps])
        y = np.array([t for _, t in laps])
        sxx = float(((x - x.mean()) ** 2).sum())
        if sxx <= 0:
            continue
        b = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
        resid = (y - y.mean()) - b * (x - x.mean())
        ssr += float((resid ** 2).sum())
        dof += len(laps) - 2
        slopes.append((b, sxx, len(laps)))
    if not slopes:
        return prior
    sigma2 = (NU0 * prior.sigma2 + ssr) / (NU0 + dof)
    tau2 = prior.tau2
    if len(slopes) >= 2:
        bs = [b for b, _, _ in slopes]
        between = statistics.variance(bs) - statistics.fmean(sigma2 / sxx for _, sxx, _ in slopes)
        k = len(slopes) - 1
        tau2 = (KAPPA0 * prior.tau2 + k * max(0.0, between)) / (KAPPA0 + k)
    prec, acc = 1.0 / prior.var, prior.mean / prior.var
    for b, sxx, _ in slopes:
        w = 1.0 / (sigma2 / sxx + tau2)
        prec += w
        acc += w * b
    used = [r for r in records if r.corrected_time_s is not None and not r.wet]
    return DegModel(prior.compound, prior.track_id, acc / prec, 1.0 / prec, sigma2, tau2,
                    sum(n for _, _, n in slopes), len(slopes),
                    'track' if prior.track_id is not None else 'pooled',
                    all(r.simulated for r in used))


def fit_fuel(records: Sequence[LapRecord], prior: FuelModel, per_km: bool = False) -> FuelModel:
    """Mean burn per racing lap with a session effect; per_km pools tracks of different lengths."""
    sessions = defaultdict(list)
    for r in records:
        used = r.fuel_used
        if used is None:
            continue
        sessions[r.session].append(used / (r.track_length / 1000.0) if per_km else used)
    # Drop Safety Car-like and garbage laps the flags missed: far from the session median.
    groups = []
    for laps in sessions.values():
        med = statistics.median(laps)
        kept = [u for u in laps if abs(u - med) <= 0.25 * med]
        if kept:
            groups.append((np.ones((len(kept), 1)), np.array(kept)))
    n = sum(len(y) for _, y in groups)
    if n == 0:
        return prior
    sigma2, tau2 = prior.sigma2, prior.tau2
    theta, _ = _gls_posterior([prior.mean], [[prior.var]], groups, sigma2, tau2)
    for _ in range(2):
        sigma2, tau2 = _variance_components(groups, theta, prior.sigma2, prior.tau2)
        theta, V = _gls_posterior([prior.mean], [[prior.var]], groups, sigma2, tau2)
    used = [r for r in records if r.fuel_used is not None]
    return FuelModel(prior.track_id, prior.track_length, float(theta[0]), float(V[0][0]), sigma2,
                     tau2, n, len(groups), 'track' if prior.track_id is not None else 'pooled',
                     all(r.simulated for r in used),
                     sorted({r.race_laps for r in used if r.race_laps}))


# ──────────────────────────────────────────────
# Store: fitted parameters per compound and track, saved as JSON
# ──────────────────────────────────────────────

class ModelStore:
    """Fitted models, looked up with fallbacks: track fit -> pooled compound fit -> prior.

    Keys: wear and degradation by '<track_id>/<compound>' (pooled: '*/<compound>'), fuel by
    '<track_id>' (pooled: '*', in kg per km).
    """

    def __init__(self):
        self.wear: dict[str, WearModel] = {}
        self.deg: dict[str, DegModel] = {}
        self.fuel: dict[str, FuelModel] = {}
        self.consumed_captures: set[str] = set()
        self.sources: list[dict] = []
        self.created: str | None = None

    # ── lookups ──
    @property
    def simulated_only(self) -> bool:
        return bool(self.sources) and all(s.get('simulated') for s in self.sources)

    def wear_model(self, track_id: int | None, compound, race_laps: int = 0) -> WearModel:
        key = compound_key(compound) or 'C3'
        model = self.wear.get(f'{track_id}/{key}')
        if model is None:
            pooled = self.wear.get(f'*/{key}')
            if pooled is not None:
                model = pooled.scaled(track_severity(track_id), BETWEEN_TRACK_REL_SD, track_id, 'pooled')
            else:
                model = _prior_wear(key, track_id)
        return _distance_check(model, race_laps)

    def deg_model(self, track_id: int | None, compound) -> DegModel:
        key = compound_key(compound) or 'C3'
        model = self.deg.get(f'{track_id}/{key}')
        if model is not None:
            return model
        pooled = self.deg.get(f'*/{key}')
        if pooled is not None:
            return DegModel(key, track_id, pooled.mean, pooled.var + DEG_BETWEEN_TRACK_SD ** 2,
                            pooled.sigma2, pooled.tau2, pooled.n_laps, pooled.n_stints, 'pooled',
                            pooled.simulated)
        return _prior_deg(key)

    def fuel_model(self, track_id: int | None, track_length: int | None, race_laps: int = 0) -> FuelModel:
        model = self.fuel.get(str(track_id))
        if model is None:
            pooled = self.fuel.get('*')
            if pooled is not None and track_length:
                model = pooled.scaled(track_length / 1000.0, FUEL_BETWEEN_TRACK_REL_SD, track_id,
                                      track_length, 'pooled')
            else:
                model = _prior_fuel(track_id, track_length)
        return _distance_check(model, race_laps)

    # ── fitting ──
    @classmethod
    def fit(cls, records: Sequence[LapRecord], sources: list[dict] | None = None,
            consumed_captures: Iterable[str] = ()) -> 'ModelStore':
        store = cls()
        store.sources = list(sources or [])
        store.consumed_captures = set(consumed_captures)
        store.created = datetime.now(timezone.utc).isoformat(timespec='seconds')
        by_compound = defaultdict(list)
        for r in records:
            by_compound[r.compound].append(r)
        for compound, recs in by_compound.items():
            tracks = sorted({r.track_id for r in recs})
            severity = {t: track_severity(t) for t in tracks}
            pooled = fit_wear(recs, _prior_wear(compound, None), severity)
            if pooled.n_laps:
                store.wear[f'*/{compound}'] = pooled
            pooled_deg = fit_deg(recs, _prior_deg(compound))
            if pooled_deg.n_laps:
                store.deg[f'*/{compound}'] = pooled_deg
            for t in tracks:
                track_recs = [r for r in recs if r.track_id == t]
                prior = (pooled.scaled(severity[t], BETWEEN_TRACK_REL_SD, t)
                         if pooled.n_laps else _prior_wear(compound, t))
                prior.basis, prior.track_id = 'prior', t
                fit = fit_wear(track_recs, prior)
                if fit.n_laps:
                    store.wear[f'{t}/{compound}'] = fit
                dprior = store.deg_model(t, compound)
                dprior = DegModel(compound, t, dprior.mean, dprior.var, dprior.sigma2, dprior.tau2)
                dfit = fit_deg(track_recs, dprior)
                if dfit.n_laps:
                    store.deg[f'{t}/{compound}'] = dfit
        pooled_fuel = fit_fuel(records, FuelModel(None, 1000, FUEL_PRIOR_KG_PER_KM,
                                                  (FUEL_PRIOR_REL_SD * FUEL_PRIOR_KG_PER_KM) ** 2,
                                                  (FUEL_SIGMA0_REL * FUEL_PRIOR_KG_PER_KM) ** 2,
                                                  (FUEL_TAU0_REL * FUEL_PRIOR_KG_PER_KM) ** 2), per_km=True)
        if pooled_fuel.n_laps:
            store.fuel['*'] = pooled_fuel
        by_track = defaultdict(list)
        for r in records:
            by_track[(r.track_id, r.track_length)].append(r)
        for (t, length), recs in by_track.items():
            prior = store.fuel_model(t, length)
            prior = FuelModel(t, length, prior.mean, prior.var, prior.sigma2, prior.tau2)
            fit = fit_fuel(recs, prior)
            if fit.n_laps:
                store.fuel[str(t)] = fit
        return store

    # ── persistence ──
    def to_dict(self) -> dict:
        return {
            'version': STORE_VERSION, 'created': self.created,
            'simulated_only': self.simulated_only, 'sources': self.sources,
            'consumed_captures': sorted(self.consumed_captures),
            'wear': {k: asdict(v) for k, v in sorted(self.wear.items())},
            'deg': {k: asdict(v) for k, v in sorted(self.deg.items())},
            'fuel': {k: asdict(v) for k, v in sorted(self.fuel.items())},
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(json.dumps(self.to_dict(), indent=1, allow_nan=False), encoding='utf-8')
        tmp.replace(path)

    @classmethod
    def from_dict(cls, data: dict) -> 'ModelStore':
        if data.get('version', 0) > STORE_VERSION:
            raise ValueError(f"fitted model file version {data.get('version')} is newer than this code")
        store = cls()
        store.created = data.get('created')
        store.sources = list(data.get('sources') or [])
        store.consumed_captures = set(data.get('consumed_captures') or [])
        store.wear = {k: WearModel(**v) for k, v in (data.get('wear') or {}).items()}
        store.deg = {k: DegModel(**v) for k, v in (data.get('deg') or {}).items()}
        store.fuel = {k: FuelModel(**v) for k, v in (data.get('fuel') or {}).items()}
        return store

    @classmethod
    def load(cls, path) -> 'ModelStore':
        path = Path(path)
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def load_or_empty(cls, path) -> 'ModelStore':
        """The saved fits, or an empty store (built-in priors) if there are none or they are unreadable."""
        try:
            return cls.load(path)
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f'[Models] Ignoring unreadable fitted models {path}: {error}')
            return cls()


def _distance_check(model, race_laps: int):
    """Widen a fit taken at other race distances: the game may scale wear and fuel with length."""
    if not race_laps or not model.race_laps or race_laps in model.race_laps:
        return model
    if isinstance(model, WearModel):
        return model.scaled(1.0, RACE_DISTANCE_REL_SD, model.track_id, model.basis)
    return model.scaled(1.0, RACE_DISTANCE_REL_SD, model.track_id, model.track_length, model.basis)


# ──────────────────────────────────────────────
# Live forecasts for the engine
# ──────────────────────────────────────────────

def current_stint(records: Sequence[LapRecord], compound, tyre_age: int) -> list[LapRecord]:
    """The records of the stint in progress (same set as the car is on now)."""
    key = compound_key(compound)
    if not records or key is None:
        return []
    last = records[-1]
    if last.compound != key or last.tyre_age >= tyre_age:
        return []
    return [r for r in records if r.stint == last.stint]


def stint_increments(stint: Sequence[LapRecord]) -> tuple[list[float], list[float]]:
    ages, incs = [], []
    for r in stint:
        inc = r.wear_increment
        if inc is not None:
            ages.append(float(r.tyre_age))
            incs.append(inc)
    return ages, incs


def stint_times(stint: Sequence[LapRecord]) -> tuple[list[float], list[float]]:
    ages, times = [], []
    for r in stint:
        t = r.corrected_time_s
        if t is not None:
            ages.append(float(r.tyre_age))
            times.append(t)
    return ages, times


def refine(model, records: Sequence[LapRecord]):
    """Update a stored fit with extra records (history laps the store has not seen yet)."""
    if not records:
        return model
    if isinstance(model, WearModel):
        fit = fit_wear(records, model)
    elif isinstance(model, DegModel):
        fit = fit_deg(records, model)
    else:
        fit = fit_fuel(records, model)
    if not fit.n_laps:
        return model
    if model.n_laps:                    # the stored fit's data still counts
        fit.n_laps += model.n_laps
        fit.simulated = fit.simulated and model.simulated
    return fit
