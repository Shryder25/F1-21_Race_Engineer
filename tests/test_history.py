import json
import pathlib
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from analysis import AnalysisEngine
from packets import SessionData, parse_player_lap_data, parse_player_telemetry
from simulate_telemetry import build_lap_packet, build_telemetry_packet
from test_analysis import SilentVoice


class HistoryTests(unittest.TestCase):
    def setUp(self):
        from history import TelemetryHistory
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = pathlib.Path(self.temp.name) / 'history.sqlite3'
        self.history = TelemetryHistory(self.path)
        self.addCleanup(self.history.close)
        self.engine = AnalysisEngine(SilentVoice(), history=self.history)
        self.session = SessionData(0, 30, 20, 20, 1000, 10, 11, 0)
        self.engine.update_session(self.session, session_uid=100)
        self.telemetry = parse_player_telemetry(build_telemetry_packet(1, 1, 100, 1, 0, 4, 95), 0)
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 3), 0)

    def record(self, invalid=False):
        for distance in range(0, 1000, 20):
            self.engine.update_telemetry(self.telemetry, replace(
                self.lap, lap_distance=distance, current_lap_time_ms=100 + distance * 50,
                current_lap_invalid=int(invalid)))
        self.engine.update_telemetry(self.telemetry, replace(
            self.lap, current_lap_num=2, lap_distance=10,
            current_lap_time_ms=500, last_lap_time_ms=50000))

    def test_track_switch_and_return_loads_only_matching_reference(self):
        self.record()
        self.engine.update_session(replace(self.session, track_id=7), session_uid=200)
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)
        self.assertEqual(self.engine.delta.current_lap_speeds, [])
        self.assertEqual(self.engine.corner_diagnostics, [])
        self.engine.update_session(self.session, session_uid=300)
        self.assertEqual(self.engine.delta.best_lap_time_ms, 50000)
        self.assertEqual(self.engine.delta.current_lap_num, 0)

    def test_new_race_resets_live_state_but_keeps_track_reference(self):
        self.record()
        first_capture = self.engine.capture_id
        self.engine.update_session(self.session, session_uid=101)
        self.assertNotEqual(self.engine.capture_id, first_capture)
        self.assertEqual(self.engine.delta.best_lap_time_ms, 50000)
        self.assertEqual(self.engine.delta.current_lap_num, 0)
        rows = self.history.connection.execute('SELECT game_uid FROM sessions').fetchall()
        self.assertEqual({row[0] for row in rows}, {'100', '101'})

    def test_history_survives_reopening_app(self):
        self.record()
        from history import TelemetryHistory
        other = TelemetryHistory(self.path)
        self.addCleanup(other.close)
        engine = AnalysisEngine(SilentVoice(), history=other)
        engine.update_session(self.session, session_uid=999)
        self.assertEqual(engine.delta.best_lap_time_ms, 50000)

    def test_invalid_recording_is_archived_but_never_loaded_as_reference(self):
        self.record(invalid=True)
        row = self.history.connection.execute('SELECT eligible, recording FROM laps').fetchone()
        self.assertEqual(row[0], 0)
        self.assertTrue(json.loads(row[1])['samples'])
        self.engine.update_session(self.session, session_uid=101)
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_formula_and_weather_references_are_separate(self):
        self.record()
        for session in [replace(self.session, formula=2), replace(self.session, weather=3)]:
            self.engine.update_session(session, session_uid=200)
            self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_session_change_clears_previous_track_hud_values(self):
        import web_hud
        web_hud.update_web_state({'speed': 280, 'pit_window': [10, 12],
                                  'lap_time_ms': 80000, 'last_update_monotonic': 1})
        self.engine.update_session(replace(self.session, track_id=7), session_uid=200)
        state = web_hud.get_web_state(now=2)
        self.assertEqual(state['speed'], 0)
        self.assertEqual(state['lap_time_ms'], 0)
        self.assertIsNone(state['pit_window'])
        self.assertIsNone(state['data_age_seconds'])

    def test_personal_sector_records_reload_and_drive_next_session_comparison(self):
        from history import TelemetryHistory
        for distance in range(0, 1000, 20):
            self.engine.update_telemetry(self.telemetry, replace(
                self.lap, lap_distance=distance, current_lap_time_ms=100 + distance * 50,
                sector=0 if distance < 300 else 1 if distance < 600 else 2,
                sector1_time_ms=15000, sector2_time_ms=15000))
        self.engine.update_telemetry(self.telemetry, replace(
            self.lap, current_lap_num=2, lap_distance=0, current_lap_time_ms=100, last_lap_time_ms=50000))
        reopened = TelemetryHistory(self.path)
        self.addCleanup(reopened.close)
        voice = SilentVoice()
        engine = AnalysisEngine(voice, history=reopened)
        engine.update_session(self.session, session_uid=300)
        self.assertEqual(engine.best_sector_times, {1: 15000, 2: 15000, 3: 20000})
        for distance in range(0, 320, 20):
            engine.update_telemetry(self.telemetry, replace(
                self.lap, lap_distance=distance, current_lap_time_ms=100 + distance * 50,
                sector=int(distance >= 300), sector1_time_ms=14800))
        self.assertTrue(any('faster than your personal best sector' in message[0]
                            for message in voice.messages))


if __name__ == '__main__':
    unittest.main()
