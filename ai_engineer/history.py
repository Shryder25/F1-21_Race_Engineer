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
        ''')

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
