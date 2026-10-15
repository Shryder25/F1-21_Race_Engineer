import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from analysis import AnalysisEngine
from packets import parse_player_lap_data, parse_player_telemetry, SessionData
from simulate_telemetry import build_lap_packet, build_telemetry_packet


class SilentVoice:
    def __init__(self):
        self.messages = []

    def say(self, message, priority, **kwargs):
        self.messages.append((message, priority, kwargs))


class LapValidityTests(unittest.TestCase):
    def setUp(self):
        self.engine = AnalysisEngine(SilentVoice())
        self.engine.update_session(SessionData(0, 30, 20, 20, 1000, 10, 11, 0))
        self.telemetry = parse_player_telemetry(build_telemetry_packet(1, 1, 100, 1, 0, 4, 95), 0)
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 3), 0)

    def drive_lap(self, start=0, invalid_at=None, pit_at=None, skip_at=None):
        for distance in range(start, 1000, 20):
            if skip_at and skip_at[0] <= distance <= skip_at[1]:
                continue
            self.engine.update_telemetry(self.telemetry, replace(
                self.lap, lap_distance=distance, current_lap_time_ms=100 + distance * 50,
                current_lap_invalid=int(distance == invalid_at), pit_status=int(distance == pit_at)))

    def finish(self, lap_num=2):
        self.engine.update_telemetry(self.telemetry, replace(
            self.lap, lap_distance=10, current_lap_time_ms=500,
            current_lap_num=lap_num, last_lap_time_ms=50000))

    def test_complete_valid_lap_sets_baseline(self):
        self.drive_lap()
        self.finish()
        self.assertEqual(self.engine.delta.best_lap_time_ms, 50000)

    def test_invalid_flag_is_latched_until_lap_end(self):
        self.drive_lap(invalid_at=400)
        self.finish()
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_mid_lap_connection_is_not_a_reference(self):
        self.drive_lap(start=500)
        self.finish()
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_pit_lap_is_not_a_reference(self):
        self.drive_lap(pit_at=400)
        self.finish()
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_missing_large_section_is_not_a_reference(self):
        self.drive_lap(skip_at=(200, 600))
        self.finish()
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_nonsequential_lap_number_is_not_completion(self):
        self.drive_lap()
        self.finish(lap_num=4)
        self.assertEqual(self.engine.delta.best_lap_time_ms, 0)

    def test_live_delta_interpolates_matching_distance_and_clears_when_unavailable(self):
        self.engine.delta.best_lap_times = {0.0: 0, 100.0: 1000}
        self.engine.delta.best_lap_time_ms = 50000
        self.engine.update_telemetry(self.telemetry, replace(
            self.lap, lap_distance=50, current_lap_time_ms=600))
        self.assertEqual(self.engine.delta.last_reported_delta_ms, 100)
        self.engine.update_telemetry(self.telemetry, replace(
            self.lap, lap_distance=150, current_lap_time_ms=1600))
        self.assertIsNone(self.engine.delta.last_reported_delta_ms)


if __name__ == '__main__':
    unittest.main()
