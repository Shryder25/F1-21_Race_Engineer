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
            -- Every car's laps and tyre stints from the Session History packet (id 11).
            CREATE TABLE IF NOT EXISTS car_laps (
                id INTEGER PRIMARY KEY, capture_id TEXT NOT NULL,
                car_index INTEGER NOT NULL, lap_num INTEGER NOT NULL,
                lap_time_ms INTEGER NOT NULL, s1_ms INTEGER, s2_ms INTEGER, s3_ms INTEGER,
                valid INTEGER NOT NULL, valid_flags INTEGER NOT NULL,
                stint_num INTEGER, actual_compound INTEGER, visual_compound INTEGER,
                tyre_age_laps INTEGER, in_lap INTEGER NOT NULL DEFAULT 0,
                out_lap INTEGER NOT NULL DEFAULT 0,
                UNIQUE(capture_id, car_index, lap_num),
                FOREIGN KEY(capture_id) REFERENCES sessions(capture_id)
            );
            CREATE TABLE IF NOT EXISTS car_stints (
                id INTEGER PRIMARY KEY, capture_id TEXT NOT NULL,
                car_index INTEGER NOT NULL, stint_num INTEGER NOT NULL,
                actual_compound INTEGER NOT NULL, visual_compound INTEGER NOT NULL,
                start_lap INTEGER NOT NULL, end_lap INTEGER NOT NULL, laps INTEGER NOT NULL,
                is_open INTEGER NOT NULL, driver TEXT,
                UNIQUE(capture_id, car_index, stint_num),
                FOREIGN KEY(capture_id) REFERENCES sessions(capture_id)
            );
            CREATE INDEX IF NOT EXISTS car_stints_compound
                ON car_stints(actual_compound, laps);
            -- Final Classification packet (id 8), one row per classified car.
            CREATE TABLE IF NOT EXISTS final_classification (
                id INTEGER PRIMARY KEY, capture_id TEXT NOT NULL, car_index INTEGER NOT NULL,
                position INTEGER, num_laps INTEGER, grid_position INTEGER, points INTEGER,
                num_pit_stops INTEGER, result_status INTEGER, best_lap_time_ms INTEGER,
                total_race_time_s REAL, penalties_time_s INTEGER, num_penalties INTEGER,
                tyre_stints_actual TEXT, tyre_stints_visual TEXT, driver TEXT,
                UNIQUE(capture_id, car_index),
                FOREIGN KEY(capture_id) REFERENCES sessions(capture_id)
            );
            -- Car Setups packet (id 5): the latest visible setup per car, as JSON.
            CREATE TABLE IF NOT EXISTS car_setups (
                id INTEGER PRIMARY KEY, capture_id TEXT NOT NULL, car_index INTEGER NOT NULL,
                setup TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(capture_id, car_index),
                FOREIGN KEY(capture_id) REFERENCES sessions(capture_id)
            );
        ''')
        # speed_trap_kph was added after the initial lap_stats schema; ALTER a pre-existing
        # database that predates it rather than forcing users to delete their history file.
        existing_columns = {
            row[1] for row in self.connection.execute('PRAGMA table_info(lap_stats)').fetchall()
        }
        if 'speed_trap_kph' not in existing_columns:
            self.connection.execute('ALTER TABLE lap_stats ADD COLUMN speed_trap_kph REAL')
            self.connection.commit()
        # player_car_index tells the player's car_laps rows apart from the rivals'; older
        # databases get the column empty.
        session_columns = {
            row[1] for row in self.connection.execute('PRAGMA table_info(sessions)').fetchall()
        }
        if 'player_car_index' not in session_columns:
            self.connection.execute('ALTER TABLE sessions ADD COLUMN player_car_index INTEGER')
            self.connection.commit()

    @staticmethod
    def weather_group(weather: int) -> str:
        return 'dry' if weather < 3 else 'intermediate' if weather == 3 else 'wet'

    def start_session(self, session, game_uid: int | None, player_car_index: int | None = None) -> str:
        capture_id = uuid.uuid4().hex
        with self.connection:
            self.connection.execute(
                'INSERT INTO sessions(capture_id,game_uid,track_id,track_length,formula,weather_group,'
                'session_type,player_car_index) VALUES (?,?,?,?,?,?,?,?)',
                (
                    capture_id,
                    str(game_uid),
                    session.track_id,
                    session.track_length,
                    session.formula,
                    self.weather_group(session.weather),
                    session.session_type,
                    player_car_index,
                ),
            )
        return capture_id

    def record_lap(self, capture_id: str, lap_num: int, lap_time_ms: int, eligible: bool, recording: dict):
        with self.connection:
            self.connection.execute(
                'INSERT INTO laps(capture_id,lap_num,lap_time_ms,eligible,recording) VALUES (?,?,?,?,?)',
                (capture_id, lap_num, lap_time_ms, int(eligible), json.dumps(recording, allow_nan=False)),
            )

    def best_reference(self, session) -> dict | None:
        row = self.connection.execute(
            '''
            SELECT l.recording FROM laps l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND s.weather_group=?
              AND l.eligible=1 AND l.lap_time_ms>0
            ORDER BY l.lap_time_ms, l.id DESC LIMIT 1
        ''',
            (session.track_id, session.track_length, session.formula, self.weather_group(session.weather)),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def close(self):
        self.connection.close()

    def personal_sectors(self, session) -> dict[int, int]:
        rows = self.connection.execute(
            '''
            SELECT json_extract(l.recording, '$.sectors')
            FROM laps l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND s.weather_group=?
              AND l.eligible=1
        ''',
            (session.track_id, session.track_length, session.formula, self.weather_group(session.weather)),
        )
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
                (
                    capture_id,
                    stats['lap_num'],
                    stats['lap_time_ms'],
                    int(stats.get('eligible', False)),
                    stats.get('compound'),
                    stats.get('tyre_age_laps'),
                    wear[0],
                    wear[1],
                    wear[2],
                    wear[3],
                    stats.get('fuel_start'),
                    stats.get('fuel_end'),
                    sectors.get(1),
                    sectors.get(2),
                    sectors.get(3),
                    deltas.get(1),
                    deltas.get(2),
                    deltas.get(3),
                    stats.get('speed_trap_kph'),
                ),
            )

    def race_report(self, capture_id: str, limit: int = 100) -> list[dict]:
        """This session's completed laps for the HUD race-report table: lap number, lap
        time, each sector, speed trap and tyre compound — newest-first lap order is left
        to the caller; this returns laps in the order they were driven."""
        rows = self.connection.execute(
            '''
            SELECT lap_num, lap_time_ms, eligible, compound, s1_ms, s2_ms, s3_ms, speed_trap_kph
            FROM lap_stats WHERE capture_id=? ORDER BY lap_num ASC LIMIT ?
        ''',
            (capture_id, limit),
        ).fetchall()
        return [
            {
                'lap_num': lap_num,
                'lap_time_ms': lap_time_ms,
                'eligible': bool(eligible),
                'compound': compound,
                's1_ms': s1,
                's2_ms': s2,
                's3_ms': s3,
                'speed_trap_kph': speed_trap_kph,
            }
            for lap_num, lap_time_ms, eligible, compound, s1, s2, s3, speed_trap_kph in rows
        ]

    def tyre_wear_samples(
        self, session, compound: int, exclude_capture: str | None = None, limit: int = 400
    ) -> list[tuple[int, float]]:
        """(tyre age in laps, worst-tyre wear %) from earlier sessions on this track and compound."""
        rows = self.connection.execute(
            '''
            SELECT l.tyre_age_laps, MAX(l.wear_rl, l.wear_rr, l.wear_fl, l.wear_fr)
            FROM lap_stats l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND l.compound=?
              AND l.tyre_age_laps IS NOT NULL AND l.wear_rl IS NOT NULL
              AND l.capture_id != ?
            ORDER BY l.id DESC LIMIT ?
        ''',
            (session.track_id, session.track_length, session.formula, compound, exclude_capture or '', limit),
        ).fetchall()
        return [(int(age), float(wear)) for age, wear in rows]

    def fuel_use_samples(self, session, exclude_capture: str | None = None, limit: int = 200) -> list[float]:
        """Fuel burned per lap (kg) on completed laps at this track in earlier sessions."""
        rows = self.connection.execute(
            '''
            SELECT l.fuel_start - l.fuel_end
            FROM lap_stats l JOIN sessions s USING(capture_id)
            WHERE s.track_id=? AND s.track_length=? AND s.formula=? AND l.eligible=1
              AND l.fuel_start IS NOT NULL AND l.fuel_end IS NOT NULL
              AND l.fuel_start > l.fuel_end AND l.capture_id != ?
            ORDER BY l.id DESC LIMIT ?
        ''',
            (session.track_id, session.track_length, session.formula, exclude_capture or '', limit),
        ).fetchall()
        return [float(kg) for (kg,) in rows]

    # ──────────────────────────────────────────
    # Every car: laps, tyre stints, classification, setups (session_history.py)
    # ──────────────────────────────────────────

    def record_car_history(
        self,
        capture_id: str,
        car_index: int,
        laps: list[dict],
        stints: list[dict],
        num_laps: int | None = None,
        driver: str | None = None,
    ):
        """Upsert one car's lap and stint rows (session_history.lap_rows / stint_rows).

        num_laps is the car's last completed lap; laps past it (and stints past the last one)
        belong to a timeline a flashback threw away, so they are deleted.
        """
        with self.connection:
            if num_laps is not None:
                self.connection.execute(
                    'DELETE FROM car_laps WHERE capture_id=? AND car_index=? AND lap_num>?',
                    (capture_id, car_index, num_laps),
                )
            self.connection.execute(
                'DELETE FROM car_stints WHERE capture_id=? AND car_index=? AND stint_num>=?',
                (capture_id, car_index, len(stints)),
            )
            self.connection.executemany(
                'INSERT OR REPLACE INTO car_laps(capture_id,car_index,lap_num,lap_time_ms,s1_ms,s2_ms,s3_ms,'
                'valid,valid_flags,stint_num,actual_compound,visual_compound,tyre_age_laps,in_lap,out_lap) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                [
                    (
                        capture_id,
                        car_index,
                        r['lap_num'],
                        r['lap_time_ms'],
                        r['s1_ms'],
                        r['s2_ms'],
                        r['s3_ms'],
                        int(r['valid']),
                        r['valid_flags'],
                        r['stint_num'],
                        r['actual_compound'],
                        r['visual_compound'],
                        r['tyre_age_laps'],
                        int(r['in_lap']),
                        int(r['out_lap']),
                    )
                    for r in laps
                ],
            )
            self.connection.executemany(
                'INSERT OR REPLACE INTO car_stints(capture_id,car_index,stint_num,actual_compound,'
                'visual_compound,start_lap,end_lap,laps,is_open,driver) VALUES (?,?,?,?,?,?,?,?,?,?)',
                [
                    (
                        capture_id,
                        car_index,
                        r['stint_num'],
                        r['actual_compound'],
                        r['visual_compound'],
                        r['start_lap'],
                        r['end_lap'],
                        r['laps'],
                        int(r['is_open']),
                        driver,
                    )
                    for r in stints
                ],
            )

    def record_final_classification(self, capture_id: str, rows: list[dict]):
        with self.connection:
            self.connection.executemany(
                'INSERT OR REPLACE INTO final_classification(capture_id,car_index,position,num_laps,'
                'grid_position,points,num_pit_stops,result_status,best_lap_time_ms,total_race_time_s,'
                'penalties_time_s,num_penalties,tyre_stints_actual,tyre_stints_visual,driver) '
                'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                [
                    (
                        capture_id,
                        r['car_index'],
                        r['position'],
                        r['num_laps'],
                        r['grid_position'],
                        r['points'],
                        r['num_pit_stops'],
                        r['result_status'],
                        r['best_lap_time_ms'],
                        r['total_race_time_s'],
                        r['penalties_time_s'],
                        r['num_penalties'],
                        json.dumps(r['tyre_stints_actual']),
                        json.dumps(r['tyre_stints_visual']),
                        r.get('driver'),
                    )
                    for r in rows
                ],
            )

    def record_car_setup(self, capture_id: str, car_index: int, setup: dict):
        with self.connection:
            self.connection.execute(
                'INSERT OR REPLACE INTO car_setups(capture_id,car_index,setup) VALUES (?,?,?)',
                (capture_id, car_index, json.dumps(setup, allow_nan=False)),
            )

    def car_laps(self, capture_id: str, car_index: int | None = None) -> list[dict]:
        where, args = (
            ('AND car_index=?', (capture_id, car_index)) if car_index is not None else ('', (capture_id,))
        )
        cursor = self.connection.execute(
            f'SELECT car_index,lap_num,lap_time_ms,s1_ms,s2_ms,s3_ms,valid,stint_num,actual_compound,'
            f'visual_compound,tyre_age_laps,in_lap,out_lap FROM car_laps WHERE capture_id=? {where} '
            f'ORDER BY car_index, lap_num',
            args,
        )
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=False)) for row in cursor.fetchall()]

    def car_stints(self, capture_id: str, car_index: int | None = None) -> list[dict]:
        where, args = (
            ('AND car_index=?', (capture_id, car_index)) if car_index is not None else ('', (capture_id,))
        )
        cursor = self.connection.execute(
            f'SELECT car_index,stint_num,actual_compound,visual_compound,start_lap,end_lap,laps,is_open,driver '
            f'FROM car_stints WHERE capture_id=? {where} ORDER BY car_index, stint_num',
            args,
        )
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=False)) for row in cursor.fetchall()]

    def final_classification(self, capture_id: str) -> list[dict]:
        cursor = self.connection.execute(
            'SELECT car_index,position,num_laps,grid_position,points,num_pit_stops,result_status,'
            'best_lap_time_ms,total_race_time_s,penalties_time_s,num_penalties,tyre_stints_actual,'
            'tyre_stints_visual,driver FROM final_classification WHERE capture_id=? ORDER BY position',
            (capture_id,),
        )
        names = [d[0] for d in cursor.description]
        out = []
        for row in cursor.fetchall():
            d = dict(zip(names, row, strict=False))
            d['tyre_stints_actual'] = json.loads(d['tyre_stints_actual'] or '[]')
            d['tyre_stints_visual'] = json.loads(d['tyre_stints_visual'] or '[]')
            out.append(d)
        return out

    def car_setup(self, capture_id: str, car_index: int) -> dict | None:
        row = self.connection.execute(
            'SELECT setup FROM car_setups WHERE capture_id=? AND car_index=?', (capture_id, car_index)
        ).fetchone()
        return json.loads(row[0]) if row else None
