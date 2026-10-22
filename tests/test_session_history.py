"""Session History (11), Car Setups (5) and Final Classification (8): parsing, every car's
stint and lap logging to the history database, and reading them back from recordings."""

import pathlib
import sqlite3
import struct
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT / 'tools', ROOT / 'tests' / 'golden', ROOT / 'ai_engineer'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fitted_models as fm  # noqa: E402
import recording  # noqa: E402
import session_history as sh  # noqa: E402
import synth  # noqa: E402
import udp_recording  # noqa: E402
from analysis import AnalysisEngine  # noqa: E402
from history import TelemetryHistory  # noqa: E402
from packets import (  # noqa: E402
    CAR_SETUP_FORMAT,
    CAR_SETUPS_PACKET_SIZE,
    FINAL_CLASSIFICATION_FORMAT,
    FINAL_CLASSIFICATION_PACKET_SIZE,
    LAP_HISTORY_FORMAT,
    SESSION_HISTORY_PACKET_SIZE,
    TYRE_STINT_HISTORY_FORMAT,
    parse_car_setups,
    parse_final_classification,
    parse_session_history,
)
from synth import Car, Scenario, header  # noqa: E402
from telemetry_stream import PACKET_SIZES, TelemetryStream  # noqa: E402
from test_analysis import SilentVoice  # noqa: E402

SOFT, MEDIUM, HARD = (17, 16), (18, 17), (19, 18)  # (actual, visual)


def history_packet(car, laps, stints, best=0):
    """laps: [(lap ms, s1, s2, s3, flags)]; stints: [(end lap, actual, visual)]."""
    body = struct.pack('<7B', car, len(laps), len(stints), best, 0, 0, 0)
    for n in range(100):
        body += struct.pack(LAP_HISTORY_FORMAT, *(laps[n] if n < len(laps) else (0, 0, 0, 0, 0)))
    for n in range(8):
        body += struct.pack(TYRE_STINT_HISTORY_FORMAT, *(stints[n] if n < len(stints) else (0, 0, 0)))
    return header(11, 100.0, 50) + body


def lap(ms, valid=True):
    return (ms, ms // 3, ms // 3, ms - 2 * (ms // 3), 0x0F if valid else 0)


def race_scenario(**overrides):
    cars = [
        Car('Player', 0.2, 'soft', grid=3, pits={4: 'medium'}, race_number=44),
        Car('VERSTAPPEN', -0.3, 'medium', grid=1, pits={5: 'hard'}, race_number=33),
        Car('BOTTAS', 0.1, 'medium', grid=2, race_number=77),
        Car('Nyck DE VRIES', 0.4, 'hard', grid=4, pits={3: 'soft', 6: 'medium'}, race_number=17),
    ]
    options = dict(
        name='history_race',
        track_id=4,
        track_length=4655,
        session_type=10,
        total_laps=8,
        cars=cars,
        fuel_kg=20.0,
        history_packets=True,
    )
    options.update(overrides)
    return Scenario(**options)


class PacketParsingTests(unittest.TestCase):
    def test_packet_sizes_match_the_f1_2021_spec(self):
        self.assertEqual(SESSION_HISTORY_PACKET_SIZE, 1155)
        self.assertEqual(CAR_SETUPS_PACKET_SIZE, 1102)
        self.assertEqual(FINAL_CLASSIFICATION_PACKET_SIZE, 839)
        self.assertEqual(PACKET_SIZES[11], 1155)
        self.assertEqual(PACKET_SIZES[5], 1102)
        self.assertEqual(PACKET_SIZES[8], 839)

    def test_session_history_laps_and_stints(self):
        data = history_packet(
            5,
            [lap(80000), lap(79000), lap(85000, valid=False), (0, 26000, 0, 0, 0)],
            [(2, *SOFT), (255, *MEDIUM)],
            best=2,
        )
        self.assertEqual(len(data), 1155)
        h = parse_session_history(data)
        self.assertEqual((h.car_index, h.num_laps, h.best_lap_num), (5, 4, 2))
        self.assertEqual([lap.lap_time_ms for lap in h.completed_laps()], [80000, 79000, 85000])
        self.assertTrue(h.laps[0].valid)
        self.assertFalse(h.laps[2].valid)
        self.assertEqual(h.laps[3].sector_ms, (26000, 0, 0))
        first, second = h.stints
        self.assertEqual((first.start_lap, first.end_lap, first.laps, first.is_open), (1, 2, 2, False))
        self.assertEqual((second.start_lap, second.end_lap, second.laps, second.is_open), (3, 4, 2, True))
        self.assertEqual((first.compound_name, second.compound_name), ('Soft', 'Medium'))

    def test_empty_stint_slots_are_skipped(self):
        h = parse_session_history(history_packet(0, [], [(0, 0, 0)]))
        self.assertEqual(h.stints, [])
        self.assertEqual(h.laps, [])

    def test_car_setups(self):
        body = struct.pack(
            CAR_SETUP_FORMAT,
            6,
            8,
            60,
            55,
            -3.0,
            -1.5,
            0.05,
            0.2,
            5,
            4,
            6,
            5,
            3,
            6,
            100,
            56,
            22.0,
            22.5,
            23.0,
            23.5,
            0,
            35.5,
        )
        data = header(5, 1.0, 2) + body * 2 + bytes(20 * 49)
        setups = parse_car_setups(data)
        self.assertEqual(len(setups), 22)
        s = setups[0]
        self.assertEqual((s.front_wing, s.rear_wing, s.brake_bias), (6, 8, 56))
        self.assertAlmostEqual(s.front_camber, -3.0)
        self.assertEqual(s.tyre_pressures, (22.0, 22.5, 23.0, 23.5))
        self.assertAlmostEqual(s.fuel_load, 35.5)
        self.assertFalse(s.is_hidden)
        self.assertTrue(setups[5].is_hidden)
        self.assertEqual(s.as_dict()['tyre_pressures'], [22.0, 22.5, 23.0, 23.5])

    def test_final_classification(self):
        car = struct.pack(
            FINAL_CLASSIFICATION_FORMAT,
            2,
            30,
            5,
            18,
            1,
            3,
            79123,
            2456.5,
            5,
            1,
            2,
            17,
            18,
            0,
            0,
            0,
            0,
            0,
            0,
            16,
            17,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        data = header(8, 1.0, 2) + struct.pack('<B', 2) + car + car + bytes(20 * 37)
        self.assertEqual(len(data), 839)
        rows = parse_final_classification(data)
        self.assertEqual(len(rows), 2)
        c = rows[1]
        self.assertEqual((c.car_index, c.position, c.num_laps, c.grid_position, c.points), (1, 2, 30, 5, 18))
        self.assertEqual((c.num_pit_stops, c.result_status, c.best_lap_time_ms), (1, 3, 79123))
        self.assertAlmostEqual(c.total_race_time_s, 2456.5)
        self.assertEqual((c.tyre_stints_actual, c.tyre_stints_visual), ((17, 18), (16, 17)))
        self.assertEqual(sh.classification_row(c)['tyre_stints'], ['Soft', 'Medium'])


class SessionLogTests(unittest.TestCase):
    def test_lap_rows_carry_stint_compound_age_and_in_out_laps(self):
        h = parse_session_history(
            history_packet(
                3,
                [lap(80000), lap(80100), lap(95000), lap(99000), lap(80500), (0, 0, 0, 0, 0)],
                [(3, *SOFT), (255, *HARD)],
            )
        )
        rows = sh.lap_rows(h)
        self.assertEqual([r['lap_num'] for r in rows], [1, 2, 3, 4, 5])
        self.assertEqual([r['stint_num'] for r in rows], [0, 0, 0, 1, 1])
        self.assertEqual([r['tyre_age_laps'] for r in rows], [0, 1, 2, 0, 1])
        self.assertEqual([r['in_lap'] for r in rows], [False, False, True, False, False])
        self.assertEqual([r['out_lap'] for r in rows], [False, False, False, True, False])
        self.assertEqual(rows[3]['actual_compound'], HARD[0])
        self.assertEqual(sh.stint_rows(h)[1]['laps'], 3)

    def test_update_returns_only_new_laps_and_handles_a_flashback(self):
        log = sh.SessionLog()
        first = parse_session_history(history_packet(1, [lap(80000), (0, 0, 0, 0, 0)], [(255, *SOFT)]))
        laps, stints, last = log.update_history(first)
        self.assertEqual(([r['lap_num'] for r in laps], last), ([1], 1))
        self.assertIsNone(log.update_history(first))  # nothing changed
        more = parse_session_history(
            history_packet(1, [lap(80000), lap(80200), lap(80300), (0, 0, 0, 0, 0)], [(255, *SOFT)])
        )
        laps, _, last = log.update_history(more)
        self.assertEqual(([r['lap_num'] for r in laps], last), ([2, 3], 3))
        rewound = parse_session_history(history_packet(1, [lap(80000), (0, 0, 0, 0, 0)], [(255, *SOFT)]))
        laps, _, last = log.update_history(rewound)
        self.assertEqual(([r['lap_num'] for r in laps], last), ([1], 1))

    def test_pit_laps_and_hidden_setups(self):
        log = sh.SessionLog()
        log.update_history(
            parse_session_history(
                history_packet(2, [lap(80000)] * 7, [(2, *SOFT), (5, *MEDIUM), (255, *HARD)])
            )
        )
        self.assertEqual(log.pit_laps(2), [2, 5])
        self.assertEqual(log.pit_laps(9), [])
        setups = parse_car_setups(header(5, 1.0, 2) + bytes(22 * 49))
        self.assertEqual(log.update_setups(setups), {})


class HistoryDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = TelemetryHistory(pathlib.Path(self.tmp.name) / 'h.sqlite3')
        self.capture = self.db.start_session(synth_session(), 123, player_car_index=0)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_car_history_upserts_and_flashback_deletes(self):
        h = parse_session_history(
            history_packet(
                4,
                [lap(80000), lap(81000), lap(90000), lap(82000), (0, 0, 0, 0, 0)],
                [(3, *SOFT), (255, *MEDIUM)],
            )
        )
        self.db.record_car_history(self.capture, 4, sh.lap_rows(h), sh.stint_rows(h), 4, driver='Vettel')
        self.db.record_car_history(self.capture, 4, sh.lap_rows(h), sh.stint_rows(h), 4, driver='Vettel')
        laps = self.db.car_laps(self.capture, 4)
        self.assertEqual([r['lap_num'] for r in laps], [1, 2, 3, 4])
        self.assertEqual(laps[2]['in_lap'], 1)
        stints = self.db.car_stints(self.capture)
        self.assertEqual(
            [
                (s['stint_num'], s['start_lap'], s['end_lap'], s['laps'], s['is_open'], s['driver'])
                for s in stints
            ],
            [(0, 1, 3, 3, 0, 'Vettel'), (1, 4, 5, 2, 1, 'Vettel')],
        )
        rewound = parse_session_history(
            history_packet(4, [lap(80000), lap(81000), (0, 0, 0, 0, 0)], [(255, *SOFT)])
        )
        self.db.record_car_history(self.capture, 4, sh.lap_rows(rewound), sh.stint_rows(rewound), 2)
        self.assertEqual([r['lap_num'] for r in self.db.car_laps(self.capture, 4)], [1, 2])
        self.assertEqual(len(self.db.car_stints(self.capture, 4)), 1)

    def test_classification_and_setup_round_trip(self):
        car = struct.pack(
            FINAL_CLASSIFICATION_FORMAT,
            1,
            30,
            2,
            25,
            1,
            3,
            79000,
            2400.0,
            0,
            0,
            2,
            17,
            18,
            0,
            0,
            0,
            0,
            0,
            0,
            16,
            17,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        rows = parse_final_classification(header(8, 1.0, 2) + b'\x01' + car + bytes(21 * 37))
        self.db.record_final_classification(self.capture, [sh.classification_row(c) for c in rows])
        stored = self.db.final_classification(self.capture)
        self.assertEqual(stored[0]['position'], 1)
        self.assertEqual(stored[0]['tyre_stints_actual'], [17, 18])
        self.db.record_car_setup(self.capture, 0, {'front_wing': 7})
        self.assertEqual(self.db.car_setup(self.capture, 0), {'front_wing': 7})
        self.assertIsNone(self.db.car_setup(self.capture, 1))

    def test_old_database_gains_player_car_index(self):
        path = pathlib.Path(self.tmp.name) / 'old.sqlite3'
        con = sqlite3.connect(path)
        con.execute(
            'CREATE TABLE sessions (capture_id TEXT PRIMARY KEY, game_uid TEXT NOT NULL, '
            'track_id INTEGER NOT NULL, track_length INTEGER NOT NULL, formula INTEGER NOT NULL, '
            'weather_group TEXT NOT NULL, session_type INTEGER NOT NULL, '
            'started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)'
        )
        con.commit()
        con.close()
        db = TelemetryHistory(path)
        try:
            columns = {row[1] for row in db.connection.execute('PRAGMA table_info(sessions)')}
            self.assertIn('player_car_index', columns)
            db.start_session(synth_session(), 1)
        finally:
            db.close()


def synth_session():
    from packets import SessionData

    return SessionData(0, 30, 20, 8, 4655, 10, 4, 0)


class RecordingTests(unittest.TestCase):
    """Synthetic race with Session History, written in the .f1rec recorder format."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.scenario = race_scenario()
        cls.path = pathlib.Path(cls.tmp.name) / 'history_race.f1rec.gz'
        with recording.ReproducibleWriter(cls.path, {'source': 'synthetic', 'scenario': 'history_race'}) as w:
            for t, data in synth.simulate(cls.scenario):
                w.write(data, round(t * 1e9))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def expected_pit_laps(self, index):
        return sorted(self.scenario.cars[index].pits)

    def test_recording_counts_the_new_packet_types(self):
        counts = udp_recording.summarize(self.path)['by_packet_id']
        self.assertGreater(counts.get(11, 0), 20)
        self.assertGreater(counts.get(5, 0), 0)
        self.assertEqual(counts.get(8), 1)

    def test_session_logs_from_recording(self):
        (log,) = sh.session_logs_from_recording(self.path)
        self.assertEqual(sorted(log.histories), [0, 1, 2, 3])
        for index in range(4):
            self.assertEqual(log.pit_laps(index), self.expected_pit_laps(index))
        stints = [s for s in log.all_stint_rows() if s['car_index'] == 3]
        self.assertEqual(
            [(s['compound'], s['start_lap'], s['end_lap']) for s in stints],
            [('Hard', 1, 3), ('Soft', 4, 6), ('Medium', 7, 8)],
        )
        self.assertEqual(stints[0]['driver'], 'De Vries')
        self.assertEqual(len(log.classification), 4)
        self.assertEqual(max(c.num_laps for c in log.classification), 8)
        for c in log.classification:  # a lapped car finishes a lap down
            self.assertEqual(c.num_laps, len(log.histories[c.car_index].completed_laps()))
        self.assertEqual(sorted(log.setups), [0, 1, 2, 3])
        laps = [r for r in log.all_lap_rows() if r['car_index'] == 1]
        self.assertEqual(len(laps), 8)
        self.assertTrue(all(r['lap_time_ms'] > 60000 for r in laps))

    def test_live_engine_logs_every_car_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = TelemetryHistory(pathlib.Path(tmp) / 'h.sqlite3')
            try:
                engine = AnalysisEngine(SilentVoice(), history=db)
                stream = TelemetryStream(engine)
                udp_recording.replay_into(stream, self.path)
                capture = engine.capture_id
                self.assertIsNotNone(capture)
                player = db.connection.execute(
                    'SELECT player_car_index FROM sessions WHERE capture_id=?', (capture,)
                ).fetchone()[0]
                self.assertEqual(player, 0)
                stints = db.car_stints(capture)
                for index in range(4):
                    car = [s for s in stints if s['car_index'] == index]
                    ends = [s['end_lap'] for s in car if not s['is_open']]
                    self.assertEqual(ends, self.expected_pit_laps(index), index)
                laps = db.car_laps(capture)
                classification = db.final_classification(capture)
                self.assertEqual(len(classification), 4)
                self.assertEqual(
                    {(r['car_index'], r['lap_num']) for r in laps},
                    {(c['car_index'], n) for c in classification for n in range(1, c['num_laps'] + 1)},
                )
                self.assertIsNotNone(db.car_setup(capture, 0))
                self.assertEqual(engine.player_setup.front_wing, 5)

                # The fitting tools read the same rows back for every other car.
                records, _ = fm.laps_from_history(db, rivals=True)
                rivals = [r for r in records if r.car_index is not None]
                self.assertEqual({r.car_index for r in rivals}, {1, 2, 3})
                self.assertFalse(any(r.time_ok for r in rivals if r.lap == 1))
                self.assertEqual(fm.laps_from_history(db)[0], [r for r in records if r.car_index is None])
            finally:
                db.close()

    def test_fit_records_include_rivals_only_when_asked(self):
        player_only = fm.laps_from_recording(self.path)
        everyone = fm.laps_from_recording(self.path, rivals=True)
        self.assertTrue(all(r.car_index is None for r in player_only))
        rivals = [r for r in everyone if r.car_index is not None]
        self.assertEqual({r.car_index for r in rivals}, {1, 2, 3})
        self.assertEqual(len(everyone) - len(rivals), len(player_only))
        verstappen = sorted((r for r in rivals if r.car_index == 1), key=lambda r: r.lap)
        self.assertEqual([r.compound for r in verstappen], ['C3'] * 5 + ['C2'] * 3)
        self.assertEqual([r.tyre_age for r in verstappen], [0, 1, 2, 3, 4, 0, 1, 2])
        clean = [r.lap for r in verstappen if r.corrected_time_s is not None]
        self.assertNotIn(1, clean)  # standing start
        self.assertNotIn(5, clean)  # in-lap
        self.assertNotIn(6, clean)  # out-lap
        self.assertIn(3, clean)
        self.assertTrue(all(r.wear_increment is None and r.fuel_used is None for r in rivals))

    def test_fit_models_tool_uses_rival_laps_for_fall_off(self):
        import fit_models

        records, sources, _ = fit_models.collect([self.path])
        self.assertGreater(sources[0]['rival_laps'], 0)
        store = fm.ModelStore.fit(records, sources)
        rival_stints = {
            r.stint for r in records if r.car_index is not None and r.corrected_time_s is not None
        }
        self.assertGreater(len(rival_stints), 2)
        self.assertTrue(store.deg)
        report = fit_models.render_report(store, [], records)
        self.assertIn('laps by other cars', report)
        player_records, player_sources, _ = fit_models.collect([self.path], rivals=False)
        self.assertNotIn('rival_laps', {k for k, v in player_sources[0].items() if v})

    def test_backtest_lists_every_cars_stints_and_cross_checks_pit_laps(self):
        import backtest

        report = backtest.backtest([self.path])
        (race,) = report['races']
        check = race['stints']
        self.assertEqual(check['disagree'], 0, check['cars'])
        self.assertEqual(check['agree'], 4)
        self.assertEqual(len(check['classification']), 4)
        text = backtest.render_markdown(report)
        self.assertIn("Every car's stints", text)
        self.assertIn('Hard 1-3, Soft 4-6, Medium 7-8', text)


class StreamValidationTests(unittest.TestCase):
    def test_malformed_session_history_is_rejected(self):
        engine = AnalysisEngine(SilentVoice())
        stream = TelemetryStream(engine)
        bad = history_packet(30, [lap(80000)], [(255, *SOFT)])  # car index out of range
        self.assertFalse(stream.handle(bad, now=1.0))
        self.assertIn('session history', stream.last_error)
        short = history_packet(1, [], [])[:-1]
        self.assertFalse(stream.handle(short, now=1.0))
        self.assertIsNotNone(stream.last_error)


if __name__ == '__main__':
    unittest.main()
