import json
import sqlite3
import uuid
from pathlib import Path


class TelemetryHistory:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                capture_id TEXT PRIMARY KEY, game_uid TEXT NOT NULL,
                track_id INTEGER NOT NULL, track_length INTEGER NOT NULL,
                formula INTEGER NOT NULL, weather_group TEXT NOT NULL,
                session_type INTEGER NOT NULL,
                started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS laps (
                id INTEGER PRIMARY KEY, capture_id TEXT NOT NULL,
                lap_num INTEGER NOT NULL, lap_time_ms INTEGER NOT NULL,
                eligible INTEGER NOT NULL, recording TEXT NOT NULL,
                FOREIGN KEY(capture_id) REFERENCES sessions(capture_id)
            );
            CREATE INDEX IF NOT EXISTS track_context
                ON sessions(track_id, track_length, formula, weather_group);
            CREATE INDEX IF NOT EXISTS lap_reference
                ON laps(eligible, lap_time_ms);
            CREATE TABLE IF NOT EXISTS lap_stats (
                id INTEGER PRIMARY KEY, capture_id TEXT NOT NULL,
                lap_num INTEGER NOT NULL, lap_time_ms INTEGER NOT NULL,
                eligible INTEGER NOT NULL, compound INTEGER, tyre_age_laps INTEGER,
                wear_rl REAL, wear_rr REAL, wear_fl REAL, wear_fr REAL,
                fuel_start REAL, fuel_end REAL,
                s1_ms INTEGER, s2_ms INTEGER, s3_ms INTEGER,
                s1_delta_ms INTEGER, s2_delta_ms INTEGER, s3_delta_ms INTEGER,
                UNIQUE(capture_id, lap_num),
                FOREIGN KEY(capture_id) REFERENCES sessions(capture_id)
            );
            CREATE INDEX IF NOT EXISTS lap_stats_compound
                ON lap_stats(compound, tyre_age_laps);
        ''')
        # speed_trap_kph was added after the initial lap_stats schema; ALTER a pre-existing
        # database that predates it rather than forcing users to delete their history file.
        existing_columns = {row[1] for row in
                            self.connection.execute('PRAGMA table_info(lap_stats)').fetchall()}
        if 'speed_trap_kph' not in existing_columns:
            self.connection.execute('ALTER TABLE lap_stats ADD COLUMN speed_trap_kph REAL')
            self.connection.commit()

    @staticmethod
    def weather_group(weather: int) -> str:
        return 'dry' if weather < 3 else 'intermediate' if weather == 3 else 'wet'

    def start_session(self, session, game_uid: int | None) -> str:
        capture_id = uuid.uuid4().hex
        with self.connection:
            self.connection.execute(
                'INSERT INTO sessions(capture_id,game_uid,track_id,track_length,formula,weather_group,session_type) '
                'VALUES (?,?,?,?,?,?,?)',
                (capture_id, str(game_uid), session.track_id, session.track_length,
                 session.formula, self.weather_group(session.weather), session.session_type))
        return capture_id

    def record_lap(self, capture_id: str, lap_num: int, lap_time_ms: int,
                   eligible: bool, recording: dict):
        with self.connection:
            self.connection.execute(
                'INSERT INTO laps(capture_id,lap_num,lap_time_ms,eligible,recording) VALUES (?,?,?,?,?)',
                (capture_id, lap_num, lap_time_ms, int(eligible), json.dumps(recording, allow_nan=False)))

    def best_reference(self, session) -> dict | None:
        row = self.connection.execute('''
            SELECT l.recording FROM laps l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND s.weather_group=?
              AND l.eligible=1 AND l.lap_time_ms>0
            ORDER BY l.lap_time_ms, l.id DESC LIMIT 1
        ''', (session.track_id, session.track_length, session.formula,
              self.weather_group(session.weather))).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.connection.close()

    def personal_sectors(self, session) -> dict[int, int]:
        rows = self.connection.execute('''
            SELECT json_extract(l.recording, '$.sectors')
            FROM laps l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND s.weather_group=?
              AND l.eligible=1
        ''', (session.track_id, session.track_length, session.formula,
              self.weather_group(session.weather)))
        best = {}
        for (sectors_json,) in rows:
            for sector, milliseconds in json.loads(sectors_json or '{}').items():
                sector = int(sector)
                if milliseconds > 0:
                    best[sector] = min(best.get(sector, milliseconds), milliseconds)
        return best

    # ──────────────────────────────────────────
    # Learning engine: per-lap strategy data
    # ──────────────────────────────────────────

    def record_lap_stats(self, capture_id: str, stats: dict):
        """Persist one lap's tyre wear, fuel use and sector deltas (idempotent per lap)."""
        wear = stats.get('wear') or (None,) * 4
        sectors = stats.get('sectors') or {}
        deltas = stats.get('sector_deltas') or {}
        with self.connection:
            self.connection.execute(
                'INSERT OR REPLACE INTO lap_stats(capture_id,lap_num,lap_time_ms,eligible,compound,'
                'tyre_age_laps,wear_rl,wear_rr,wear_fl,wear_fr,fuel_start,fuel_end,'
                's1_ms,s2_ms,s3_ms,s1_delta_ms,s2_delta_ms,s3_delta_ms,speed_trap_kph) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (capture_id, stats['lap_num'], stats['lap_time_ms'], int(stats.get('eligible', False)),
                 stats.get('compound'), stats.get('tyre_age_laps'),
                 wear[0], wear[1], wear[2], wear[3],
                 stats.get('fuel_start'), stats.get('fuel_end'),
                 sectors.get(1), sectors.get(2), sectors.get(3),
                 deltas.get(1), deltas.get(2), deltas.get(3),
                 stats.get('speed_trap_kph')))

    def race_report(self, capture_id: str, limit: int = 100) -> list[dict]:
        """This session's completed laps for the HUD race-report table: lap number, lap
        time, each sector, speed trap and tyre compound — newest-first lap order is left
        to the caller; this returns laps in the order they were driven."""
        rows = self.connection.execute('''
            SELECT lap_num, lap_time_ms, eligible, compound, s1_ms, s2_ms, s3_ms, speed_trap_kph
            FROM lap_stats WHERE capture_id=? ORDER BY lap_num ASC LIMIT ?
        ''', (capture_id, limit)).fetchall()
        return [
            {'lap_num': lap_num, 'lap_time_ms': lap_time_ms, 'eligible': bool(eligible),
             'compound': compound, 's1_ms': s1, 's2_ms': s2, 's3_ms': s3,
             'speed_trap_kph': speed_trap_kph}
            for lap_num, lap_time_ms, eligible, compound, s1, s2, s3, speed_trap_kph in rows
        ]

    def tyre_wear_samples(self, session, compound: int, exclude_capture: str | None = None,
                          limit: int = 400) -> list[tuple[int, float]]:
        """(tyre age in laps, worst-tyre wear %) from earlier sessions on this track and compound."""
        rows = self.connection.execute('''
            SELECT l.tyre_age_laps, MAX(l.wear_rl, l.wear_rr, l.wear_fl, l.wear_fr)
            FROM lap_stats l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND l.compound=?
              AND l.tyre_age_laps IS NOT NULL AND l.wear_rl IS NOT NULL
              AND l.capture_id != ?
            ORDER BY l.id DESC LIMIT ?
        ''', (session.track_id, session.track_length, session.formula, compound,
              exclude_capture or '', limit)).fetchall()
        return [(int(age), float(wear)) for age, wear in rows]

    def fuel_use_samples(self, session, exclude_capture: str | None = None,
                         limit: int = 200) -> list[float]:
        """Fuel burned per lap (kg) on completed laps at this track in earlier sessions."""
        rows = self.connection.execute('''
            SELECT l.fuel_start - l.fuel_end
            FROM lap_stats l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND l.eligible=1
              AND l.fuel_start IS NOT NULL AND l.fuel_end IS NOT NULL
              AND l.fuel_start > l.fuel_end AND l.capture_id != ?
            ORDER BY l.id DESC LIMIT ?
        ''', (session.track_id, session.track_length, session.formula,
              exclude_capture or '', limit)).fetchall()
        return [float(kg) for (kg,) in rows]
