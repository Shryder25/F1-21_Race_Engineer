"""Per-sample telemetry channels for the player's car, stored in Parquet, keyed by lap distance.

Every time the game sends a Lap Data + Car Telemetry pair for the same frame, the logger
writes one row: the lap number and lap distance (the alignment key), the session time,
and every channel we read from the Telemetry, Motion, Car Status, Car Damage and Session
packets (the latest value of each). Because each row carries `lap` and `lap_distance`,
laps can be overlaid and compared point by point (`align_laps`), and MoTeC i2 can be fed
from the same file (`motec.export_session`).

This store sits next to the other ones and replaces none of them:
  - udp_recording.py  raw datagrams, for exact replay (any channel store can be rebuilt
                      from a recording: `python tools/channels.py build rec.f1rec.gz`)
  - history.py        one row per lap (times, wear, fuel) in SQLite, for the ML models
  - fitted_models     fitted tyre/fuel curves
  - channel_store.py  every sample, every channel, for lap-by-lap analysis

One Parquet file per session, one row group per lap. Rows of the current and previous lap
stay in memory until the lap after them starts, so a flashback (which rewinds a few
seconds) can drop the samples it undid before they reach the file. The file footer is
written when the session ends or the engineer shuts down; a hard crash leaves the open
file unreadable, which is why the raw recording remains the source of truth.

Columns use the game's own units (throttle 0..1, ERS in joules, wheels ordered as named).
`CHANNELS` lists them with their unit; the unit is also stored in each Parquet field's
metadata. Needs pyarrow (`pip install pyarrow`); without it the logger refuses to start
with a clear message and nothing else in the engineer is affected.

Format details: docs/channel-store.md.
"""

import datetime
import json
import math
import re
import struct
import time
from collections import namedtuple
from pathlib import Path

import numpy as np

from packets import (HEADER_SIZE, PacketHeader, parse_all_car_damage, parse_all_car_status,
                     parse_all_lap_data, parse_participants, parse_player_motion,
                     parse_player_motion_extra, parse_player_telemetry, parse_session_data)
from session_modes import mode_label
from track_data import get_track_info

STORE_VERSION = 1
PACKET_SIZES = {0: 1464, 1: 625, 2: 970, 4: 1257, 6: 1347, 7: 1058, 10: 882}
FLASHBACK_EVENT = b'FLBK'
WHEELS = ('rl', 'rr', 'fl', 'fr')  # the game's array order
TEAMS = {0: 'Mercedes', 1: 'Ferrari', 2: 'Red Bull Racing', 3: 'Williams', 4: 'Aston Martin',
         5: 'Alpine', 6: 'AlphaTauri', 7: 'Haas', 8: 'McLaren', 9: 'Alfa Romeo'}

# kind: 'linear' interpolates between samples, 'step' holds the previous value (gear, lap,
# flags), 'angle' interpolates the short way round (yaw). motec: (name, unit, factor) used
# by the MoTeC export, or None to leave the column out of it.
Channel = namedtuple('Channel', 'name type unit kind motec')


def _wheel(prefix, motec_prefix, type_, unit, kind='linear', motec_unit=None, factor=1.0):
    return [Channel(f'{prefix}_{w}', type_, unit, kind,
                    (f'{motec_prefix} {w.upper()}', motec_unit or unit, factor)) for w in WHEELS]


CHANNELS = [
    # alignment / timing
    Channel('session_time', 'float64', 's', 'linear', None),
    Channel('frame', 'uint32', '', 'step', None),
    Channel('lap', 'int16', '', 'step', ('Lap Number', '', 1.0)),
    Channel('lap_distance', 'float32', 'm', 'linear', ('Lap Distance', 'm', 1.0)),
    Channel('total_distance', 'float64', 'm', 'linear', ('Session Distance', 'm', 1.0)),
    Channel('lap_time', 'float32', 's', 'linear', ('Lap Time', 's', 1.0)),
    Channel('last_lap_time', 'float32', 's', 'step', ('Last Lap Time', 's', 1.0)),
    Channel('sector', 'int8', '', 'step', ('Sector', '', 1.0)),
    Channel('position', 'int8', '', 'step', ('Race Position', '', 1.0)),
    Channel('pit_status', 'int8', '', 'step', ('Pit Status', '', 1.0)),
    Channel('num_pit_stops', 'int8', '', 'step', ('Pit Stops', '', 1.0)),
    Channel('lap_invalid', 'int8', '', 'step', ('Lap Invalid', '', 1.0)),
    Channel('driver_status', 'int8', '', 'step', ('Driver Status', '', 1.0)),
    # car telemetry (packet 6)
    Channel('speed', 'float32', 'km/h', 'linear', ('Ground Speed', 'km/h', 1.0)),
    Channel('throttle', 'float32', '0-1', 'linear', ('Throttle Pos', '%', 100.0)),
    Channel('brake', 'float32', '0-1', 'linear', ('Brake Pos', '%', 100.0)),
    Channel('steer', 'float32', '-1..1', 'linear', ('Steering', '%', 100.0)),
    Channel('clutch', 'int8', '%', 'linear', ('Clutch Pos', '%', 1.0)),
    Channel('gear', 'int8', '', 'step', ('Gear', '', 1.0)),
    Channel('rpm', 'int16', 'rpm', 'linear', ('Engine RPM', 'rpm', 1.0)),
    Channel('drs', 'int8', '', 'step', ('DRS', '', 1.0)),
    Channel('engine_temp', 'int16', 'C', 'linear', ('Engine Temp', 'C', 1.0)),
    *_wheel('brake_temp', 'Brake Temp', 'int16', 'C'),
    *_wheel('tyre_surface_temp', 'Tyre Temp Surface', 'int16', 'C'),
    *_wheel('tyre_inner_temp', 'Tyre Temp Inner', 'int16', 'C'),
    *_wheel('tyre_pressure', 'Tyre Pressure', 'float32', 'psi'),
    *_wheel('surface', 'Surface Type', 'int8', '', kind='step'),
    # motion (packet 0, the player's car plus the player-only extra block)
    Channel('pos_x', 'float32', 'm', 'linear', ('Car Pos X', 'm', 1.0)),
    Channel('pos_y', 'float32', 'm', 'linear', ('Car Pos Y', 'm', 1.0)),
    Channel('pos_z', 'float32', 'm', 'linear', ('Car Pos Z', 'm', 1.0)),
    Channel('g_lat', 'float32', 'G', 'linear', ('G Force Lat', 'G', 1.0)),
    Channel('g_long', 'float32', 'G', 'linear', ('G Force Long', 'G', 1.0)),
    Channel('g_vert', 'float32', 'G', 'linear', ('G Force Vert', 'G', 1.0)),
    Channel('yaw', 'float32', 'rad', 'angle', ('Yaw', 'deg', math.degrees(1.0))),
    Channel('pitch', 'float32', 'rad', 'linear', ('Pitch', 'deg', math.degrees(1.0))),
    Channel('roll', 'float32', 'rad', 'linear', ('Roll', 'deg', math.degrees(1.0))),
    Channel('yaw_rate', 'float32', 'rad/s', 'linear', ('Yaw Rate', 'deg/s', math.degrees(1.0))),
    Channel('front_wheels_angle', 'float32', 'rad', 'linear', ('Steered Angle', 'deg', math.degrees(1.0))),
    *_wheel('susp_pos', 'Susp Pos', 'float32', 'mm'),
    *_wheel('susp_vel', 'Susp Vel', 'float32', 'mm/s'),
    *_wheel('wheel_speed', 'Wheel Speed', 'float32', 'm/s', motec_unit='km/h', factor=3.6),
    *_wheel('wheel_slip', 'Wheel Slip', 'float32', 'ratio'),
    # car status (packet 7)
    Channel('fuel', 'float32', 'kg', 'linear', ('Fuel Level', 'kg', 1.0)),
    Channel('fuel_laps', 'float32', 'laps', 'linear', ('Fuel Laps Remaining', 'laps', 1.0)),
    Channel('ers_store', 'float32', 'J', 'linear', ('ERS Store', 'MJ', 1e-6)),
    Channel('ers_mode', 'int8', '', 'step', ('ERS Deploy Mode', '', 1.0)),
    Channel('ers_deployed_lap', 'float32', 'J', 'linear', ('ERS Deployed Lap', 'MJ', 1e-6)),
    Channel('tyre_compound', 'int8', '', 'step', ('Tyre Compound', '', 1.0)),
    Channel('tyre_age', 'int8', 'laps', 'step', ('Tyre Age', 'laps', 1.0)),
    # car damage (packet 10)
    *_wheel('tyre_wear', 'Tyre Wear', 'float32', '%'),
    Channel('front_left_wing_damage', 'int8', '%', 'step', ('Front Wing Dmg L', '%', 1.0)),
    Channel('front_right_wing_damage', 'int8', '%', 'step', ('Front Wing Dmg R', '%', 1.0)),
    Channel('rear_wing_damage', 'int8', '%', 'step', ('Rear Wing Dmg', '%', 1.0)),
    # session (packet 1)
    Channel('track_temp', 'int8', 'C', 'step', ('Track Temp', 'C', 1.0)),
    Channel('air_temp', 'int8', 'C', 'step', ('Air Temp', 'C', 1.0)),
    Channel('weather', 'int8', '', 'step', ('Weather', '', 1.0)),
]
CHANNELS_BY_NAME = {c.name: c for c in CHANNELS}
_INT_RANGES = {'int8': (-128, 127), 'int16': (-32768, 32767), 'uint32': (0, 2 ** 32 - 1)}


def _require_pyarrow():
    try:
        import pyarrow
        import pyarrow.parquet
    except ImportError as error:
        raise RuntimeError('The channel store needs pyarrow: pip install pyarrow '
                           '(it is in ai_engineer/requirements.txt)') from error
    return pyarrow, pyarrow.parquet


def arrow_schema():
    pa, _ = _require_pyarrow()
    return pa.schema([pa.field(c.name, getattr(pa, c.type)(), metadata={'unit': c.unit, 'kind': c.kind})
                      for c in CHANNELS],
                     metadata={'f1_channel_store_version': str(STORE_VERSION)})


def _wheels(prefix, values):
    return {f'{prefix}_{w}': v for w, v in zip(WHEELS, values)}


def _clean(row):
    """Non-finite floats become nulls, out-of-range integers are clamped to their column."""
    out = {}
    for key, value in row.items():
        channel = CHANNELS_BY_NAME[key]
        if value is None:
            out[key] = None
        elif channel.type in _INT_RANGES:
            lo, hi = _INT_RANGES[channel.type]
            out[key] = min(max(int(value), lo), hi)
        else:
            out[key] = float(value) if math.isfinite(value) else None
    return out


def _slug(text):
    return re.sub(r'[^A-Za-z0-9]+', '-', text).strip('-') or 'session'


class ChannelLogger:
    """Turns raw F1 2021 datagrams into channel rows and writes them to Parquet.

    It reads datagrams on its own (it does not depend on TelemetryStream), so it can be fed
    live from the UDP socket or offline from a recording, and a bug in one cannot break the
    other. Each new session (a new session UID in a Session packet) gets its own file in
    `out_dir`. `clock` gives the wall time stored as the session start.
    """

    KEEP_LAPS = 2  # laps held in memory so a flashback can still undo them
    # After a flashback the new timeline restarts at the restored time and moves forward in
    # small steps; datagrams still in flight from the undone timeline are seconds ahead of it.
    REWIND_MAX_STEP_S = 1.0

    def __init__(self, out_dir, clock=time.time, compression='zstd', on_file=None):
        self.pa, self.pq = _require_pyarrow()
        self.out_dir = Path(out_dir)
        self.clock = clock
        self.compression = compression
        self.on_file = on_file          # called with the path of each finished session file
        self.schema = arrow_schema()
        self.files = []                 # every session file written (finished or open)
        self.rows_written = 0
        self.flashbacks = 0
        self._reset_session(None)

    # ── session lifecycle ──
    def _reset_session(self, uid):
        self.session_uid = uid
        self.meta = {}
        self.player_index = None
        self.paused = False
        self._session_values = {}
        self._motion = {}
        self._status = {}
        self._damage = {}
        self._pending = {}
        self._buffer = []
        self._last_time = None
        self._last_frame = None
        self._rewind = None
        self._writer = None
        self._path = None
        self._session_rows = 0
        self._row_groups = 0

    def _open(self, wall_time):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.fromtimestamp(wall_time).strftime('%Y%m%d-%H%M%S')
        name = f"{stamp}_{_slug(self.meta.get('track', 'track'))}_{_slug(self.meta.get('session', ''))}"
        path = self.out_dir / f'{name}.parquet'
        n = 2
        while path.exists():
            path = self.out_dir / f'{name}-{n}.parquet'
            n += 1
        self.meta['start_unix'] = wall_time
        self._path = path
        self._writer = self.pq.ParquetWriter(path, self.schema, compression=self.compression)
        self.files.append(path)

    def _finish_session(self):
        self._flush_laps(keep=0)
        if self._writer is not None:
            meta = dict(self.meta, rows=self._session_rows, laps=self._row_groups,
                        store_version=STORE_VERSION,
                        channels={c.name: c.unit for c in CHANNELS})
            self._writer.add_key_value_metadata({'f1_session': json.dumps(meta, sort_keys=True)})
            self._writer.close()
            if self.on_file:
                self.on_file(self._path)

    def close(self):
        """Write everything still in memory and finish the current session's file."""
        self._finish_session()
        self._reset_session(None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── input ──
    def handle(self, data: bytes, wall_time=None) -> bool:
        """Feed one datagram. Returns True when it produced a row. Never raises on bad input.

        `wall_time` (unix seconds) overrides the clock for the session start time; a replay
        passes the recording's own time so rebuilt files are named and dated like the live run.
        """
        try:
            return self._handle(data, wall_time)
        except (ValueError, struct.error):
            return False

    def _handle(self, data, wall_time):
        if len(data) < HEADER_SIZE:
            return False
        header = PacketHeader.from_bytes(data)
        pid = header.packet_id
        if header.packet_format != 2021 or not 0 <= header.player_car_index < 22:
            return False
        if pid == 3:
            if bytes(data[24:28]) == FLASHBACK_EVENT and len(data) >= 36 \
                    and header.session_uid == self.session_uid:
                self._flashback(*struct.unpack_from('<If', data, 28))
            return False
        if PACKET_SIZES.get(pid) != len(data) or not math.isfinite(header.session_time):
            return False
        if pid == 1:
            self._on_session(header, parse_session_data(data))
            return False
        if header.session_uid != self.session_uid or header.player_car_index != self.player_index:
            return False
        p = self.player_index
        if pid == 0:
            m, ex = parse_player_motion(data, p), parse_player_motion_extra(data)
            self._motion = {
                'pos_x': m.world_position_x, 'pos_y': m.world_position_y, 'pos_z': m.world_position_z,
                'g_lat': m.g_force_lateral, 'g_long': m.g_force_longitudinal,
                'g_vert': m.g_force_vertical, 'yaw': m.yaw, 'pitch': m.pitch, 'roll': m.roll,
                'yaw_rate': ex.angular_velocity[1], 'front_wheels_angle': ex.front_wheels_angle,
                **_wheels('susp_pos', ex.suspension_position),
                **_wheels('susp_vel', ex.suspension_velocity),
                **_wheels('wheel_speed', ex.wheel_speed),
                **_wheels('wheel_slip', ex.wheel_slip)}
        elif pid == 4:
            people = parse_participants(data)
            if p < len(people):
                self.meta['driver'] = people[p].name
                self.meta['team'] = TEAMS.get(people[p].team_id, f'Team {people[p].team_id}')
        elif pid == 7:
            s = parse_all_car_status(data)[p]
            self._status = {'fuel': s.fuel_in_tank, 'fuel_laps': s.fuel_remaining_laps,
                            'ers_store': s.ers_store_energy, 'ers_mode': s.ers_deploy_mode,
                            'ers_deployed_lap': s.ers_deployed_this_lap,
                            'tyre_compound': s.visual_tyre_compound, 'tyre_age': s.tyres_age_laps}
        elif pid == 10:
            d = parse_all_car_damage(data)[p]
            self._damage = {**_wheels('tyre_wear', d.tyres_wear),
                            'front_left_wing_damage': d.front_left_wing,
                            'front_right_wing_damage': d.front_right_wing,
                            'rear_wing_damage': d.rear_wing}
        elif pid in (2, 6):
            if self.paused:
                return False
            frame = header.frame_identifier
            pair = self._pending.setdefault(frame, {})
            pair[pid] = data
            while len(self._pending) > 8:
                self._pending.pop(next(iter(self._pending)))
            if 2 in pair and 6 in pair:
                self._pending.pop(frame)
                return self._add_row(header, pair[2], pair[6], wall_time)
        return False

    def _on_session(self, header, session):
        if header.session_uid != self.session_uid:
            self._finish_session()
            self._reset_session(header.session_uid)
            track = get_track_info(session.track_id) or {}
            self.meta = {'session_uid': f'{header.session_uid:016x}', 'track_id': session.track_id,
                         'track': track.get('short_name') or f'Track {session.track_id}',
                         'track_length_m': session.track_length, 'session_type': session.session_type,
                         'session': mode_label(session.session_type), 'total_laps': session.total_laps,
                         'formula': session.formula,
                         'game_version': f'{header.game_major_version}.{header.game_minor_version:02d}'}
        self.player_index = header.player_car_index
        self.paused = bool(session.game_paused or session.is_spectating)
        if self.paused:
            self._pending.clear()
        self._session_values = {'track_temp': session.track_temperature,
                                'air_temp': session.air_temperature, 'weather': session.weather}

    def _flashback(self, frame, restored_time):
        if not math.isfinite(restored_time):
            return
        self.flashbacks += 1
        self.meta['flashbacks'] = self.meta.get('flashbacks', 0) + 1
        old_max = self._last_time if self._last_time is not None else restored_time
        self._buffer = [r for r in self._buffer if r['session_time'] <= restored_time]
        self._pending.clear()
        self._last_time = min(old_max, restored_time)
        self._last_frame = None
        self._rewind = old_max

    def _add_row(self, header, lap_bytes, tel_bytes, wall_time):
        t = header.session_time
        if self._last_time is not None and t <= self._last_time:
            return False  # a pause, a duplicate, or out of order
        if self._rewind is not None:
            if t - self._last_time > self.REWIND_MAX_STEP_S:
                return False  # stale datagram from the timeline the flashback undid
            if t >= self._rewind:
                self._rewind = None  # the new timeline has caught up with the old one
        lap = parse_all_lap_data(lap_bytes)[self.player_index]
        tel = parse_player_telemetry(tel_bytes, self.player_index)
        if not (math.isfinite(lap.lap_distance) and math.isfinite(lap.total_distance)):
            return False
        row = {'session_time': t, 'frame': header.frame_identifier,
               'lap': lap.current_lap_num, 'lap_distance': lap.lap_distance,
               'total_distance': lap.total_distance, 'lap_time': lap.current_lap_time_ms / 1000,
               'last_lap_time': lap.last_lap_time_ms / 1000 if lap.last_lap_time_ms else None,
               'sector': lap.sector, 'position': lap.car_position, 'pit_status': lap.pit_status,
               'num_pit_stops': lap.num_pit_stops, 'lap_invalid': lap.current_lap_invalid,
               'driver_status': lap.driver_status,
               'speed': tel.speed, 'throttle': tel.throttle, 'brake': tel.brake, 'steer': tel.steer,
               'clutch': tel.clutch, 'gear': tel.gear, 'rpm': tel.engine_rpm, 'drs': tel.drs,
               'engine_temp': tel.engine_temperature,
               **_wheels('brake_temp', tel.brakes_temperature),
               **_wheels('tyre_surface_temp', tel.tyres_surface_temperature),
               **_wheels('tyre_inner_temp', tel.tyres_inner_temperature),
               **_wheels('tyre_pressure', tel.tyres_pressure),
               **_wheels('surface', tel.surface_type),
               **self._motion, **self._status, **self._damage, **self._session_values}
        if self._writer is None:
            self._open(self.clock() if wall_time is None else wall_time)
        self._buffer.append(_clean(row))
        self._last_time = t
        self._flush_laps(keep=self.KEEP_LAPS)
        return True

    def _flush_laps(self, keep):
        """Write whole laps that are older than the last `keep` laps, one row group each."""
        while self._buffer:
            laps_held = sorted({r['lap'] for r in self._buffer})
            if len(laps_held) <= keep:
                return
            first_lap = self._buffer[0]['lap']
            n = 0
            while n < len(self._buffer) and self._buffer[n]['lap'] == first_lap:
                n += 1
            rows, self._buffer = self._buffer[:n], self._buffer[n:]
            self._writer.write_table(self.pa.Table.from_pylist(rows, schema=self.schema))
            self._session_rows += len(rows)
            self._row_groups += 1
            self.rows_written += len(rows)


def build_from_recording(recording, out_dir, compression='zstd'):
    """Rebuild channel files from a raw UDP recording (.f1rec[.gz]). Returns the file paths."""
    from udp_recording import RecordingReader
    with RecordingReader(recording) as reader:
        start = reader.start_unix_ns / 1e9 if getattr(reader, 'start_unix_ns', None) else time.time()
        with ChannelLogger(out_dir, compression=compression) as logger:
            for record in reader:
                logger.handle(record.data, wall_time=start + record.t)
        return logger.files


# ──────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────

def find_session_file(path='channels'):
    """Turn what the user typed into one channel file, or raise ValueError saying what is there.

    A folder means its newest .parquet file; a name without '.parquet' gets it added.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob('*.parquet'), key=lambda f: f.stat().st_mtime)
        if not files:
            raise ValueError(f'No .parquet files in {path}. Log a session first with '
                             f'--log-channels {path}, and stop the engineer with Ctrl+C so the file is saved.')
        return files[-1]
    if not path.exists() and path.suffix != '.parquet' and path.with_suffix('.parquet').exists():
        return path.with_suffix('.parquet')
    if not path.exists() and not path.suffix:
        raise ValueError(f'There is no folder called {path} here, so nothing has been logged to it '
                         f'yet. Start the engineer with --log-channels {path}, drive, then stop it '
                         'with Ctrl+C.')
    if not path.exists():
        folder = path.parent if str(path.parent) else Path('.')
        nearby = sorted(folder.glob('*.parquet'), key=lambda f: f.stat().st_mtime)[-5:] \
            if folder.is_dir() else []
        hint = ('Files in that folder: ' + ', '.join(f.name for f in reversed(nearby))
                + f'. Or give just the folder ({folder}) to use the newest one.') if nearby else \
            (f'The folder {folder} has no .parquet files.' if folder.is_dir()
             else f'The folder {folder} does not exist.')
        raise ValueError(f'No file at {path}. {hint}')
    return path


def _check_finished(path):
    with open(path, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 4))
        tail = f.read(4)
    if tail != b'PAR1':
        raise ValueError(f'{path} was never finished, so it cannot be read. This happens when the '
                         'engineer is closed with the window X or crashes instead of stopping with '
                         'Ctrl+C. If you also recorded the session (--record), rebuild it with: '
                         'python tools/channels.py build <recording>')


def read_session(path, columns=None):
    """Load a channel file as ({column: numpy array}, metadata).

    Rows are in time order, and any rows a flashback overwrote after they had been written
    are dropped (a row is kept only if every later row is later in session time). Nulls
    become NaN. session_time and total_distance are float64, everything else float32.
    """
    _, pq = _require_pyarrow()
    path = find_session_file(path)
    _check_finished(path)
    pf = pq.ParquetFile(path)
    raw_meta = pf.metadata.metadata or {}
    meta = json.loads(raw_meta.get(b'f1_session', b'{}'))
    names = list(columns) if columns else [c.name for c in CHANNELS if c.name in pf.schema_arrow.names]
    if 'session_time' not in names:
        names.insert(0, 'session_time')
    table = pf.read(columns=names)
    t = table.column('session_time').to_numpy()
    suffix_min = np.minimum.accumulate(t[::-1])[::-1]       # min of t[i:]
    keep = t < np.append(suffix_min[1:], np.inf)             # earlier than every later row
    cols = {}
    for name in names:
        dtype = np.float64 if name in ('session_time', 'total_distance') else np.float32
        arr = table.column(name).to_numpy(zero_copy_only=False)
        cols[name] = np.asarray(arr, dtype=np.float64).astype(dtype)[keep]
    return cols, meta


def lap_summary(cols):
    """One dict per lap present: time range, lap time, sample count, and whether it is clean.

    `complete` means the samples cover the lap from near the line to near the end;
    `clean` means complete, valid (no track-limits flag) and not an in- or out-lap.
    """
    laps = cols['lap']
    out = []
    track_end = np.nanmax(cols['lap_distance']) if len(laps) else 0.0
    for lap in np.unique(laps[np.isfinite(laps)]).astype(int):
        idx = np.flatnonzero(laps == lap)
        dist = cols['lap_distance'][idx]
        lap_time = None
        nxt = np.flatnonzero(laps == lap + 1)
        if len(nxt) and 'last_lap_time' in cols and np.isfinite(cols['last_lap_time'][nxt[0]]):
            lap_time = float(cols['last_lap_time'][nxt[0]])
        complete = bool(len(dist) and np.nanmin(dist) < 50 and np.nanmax(dist) > track_end - 50
                        and len(nxt) > 0)
        pitted = bool('pit_status' in cols and np.nanmax(cols['pit_status'][idx]) > 0)
        invalid = bool('lap_invalid' in cols and np.nanmax(cols['lap_invalid'][idx]) > 0)
        out.append({'lap': int(lap), 'start_time': float(cols['session_time'][idx[0]]),
                    'end_time': float(cols['session_time'][idx[-1]]), 'samples': int(len(idx)),
                    'lap_time': lap_time, 'complete': complete, 'invalid': invalid, 'pit': pitted,
                    'clean': complete and not invalid and not pitted})
    return out


def align_laps(cols, laps, channels, step_m=1.0):
    """Resample the given laps onto a common lap-distance grid so they can be overlaid.

    Returns (grid, {lap: {channel: array}}). The grid runs from 0 to the longest lap
    distance seen in those laps; points a lap did not cover are NaN.
    """
    picked = {}
    top = 0.0
    for lap in laps:
        idx = np.flatnonzero(cols['lap'] == lap)
        d = cols['lap_distance'][idx].astype(np.float64)
        ok = np.isfinite(d) & (d >= 0)
        idx, d = idx[ok], d[ok]
        # Lap distance only goes forward; drop any sample that does not (rewinds, spins).
        forward = d > np.maximum.accumulate(np.concatenate([[-np.inf], d[:-1]]))
        picked[lap] = (idx[forward], d[forward])
        if len(d):
            top = max(top, float(d.max()))
    grid = np.arange(0.0, top + step_m / 2, step_m)
    result = {}
    for lap, (idx, d) in picked.items():
        result[lap] = {}
        for name in channels:
            y = cols[name][idx].astype(np.float64)
            m = np.isfinite(y)
            if m.sum() < 2:
                result[lap][name] = np.full(grid.shape, np.nan)
                continue
            kind = CHANNELS_BY_NAME[name].kind if name in CHANNELS_BY_NAME else 'linear'
            result[lap][name] = resample(d[m], y[m], grid, kind)
    return grid, result


def resample(x, y, grid, kind='linear'):
    """Resample y(x) onto grid (x increasing). Outside x's range the result is NaN."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if kind == 'step':
        i = np.clip(np.searchsorted(x, grid, side='right') - 1, 0, len(x) - 1)
        out = y[i]
    elif kind == 'angle':
        out = np.interp(grid, x, np.unwrap(y))
        out = (out + np.pi) % (2 * np.pi) - np.pi
    else:
        out = np.interp(grid, x, y)
    out = out.astype(np.float64)
    out[(grid < x[0]) | (grid > x[-1])] = np.nan
    return out
