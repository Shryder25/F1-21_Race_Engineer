import struct
import unittest
import test_analysis
from dataclasses import replace
from test_analysis import SilentVoice
from packets import SessionData, parse_session_data, parse_player_lap_data
from simulate_telemetry import build_lap_packet


class RaceRadioTests(unittest.TestCase):
    def setUp(self):
        from race_radio import RaceRadio
        self.voice = SilentVoice()
        self.radio = RaceRadio(self.voice.say)
        self.session = SessionData(0, 30, 20, 53, 5793, 10, 11, 0,
                                   pit_stop_window_ideal_lap=10, pit_stop_window_latest_lap=12)
        self.radio.reset(self.session)
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)

    def test_pit_window_is_prioritized_and_not_repeated_each_frame(self):
        from voice_engine import Priority
        lap = replace(self.lap, current_lap_num=10)
        self.radio.update_strategy(lap)
        self.radio.update_strategy(lap)
        self.assertEqual(len(self.voice.messages), 1)
        self.assertEqual(self.voice.messages[0][1], Priority.STRATEGY)
        self.assertIn('pit window', self.voice.messages[0][0].lower())

    def test_unknown_window_and_already_served_window_are_silent(self):
        self.radio.reset(replace(self.session, pit_stop_window_ideal_lap=0, pit_stop_window_latest_lap=0))
        self.radio.update_strategy(replace(self.lap, current_lap_num=10))
        self.assertEqual(self.voice.messages, [])
        self.radio.reset(self.session)
        self.radio.update_strategy(replace(self.lap, current_lap_num=9))
        count = len(self.voice.messages)
        self.radio.update_strategy(replace(self.lap, current_lap_num=10, num_pit_stops=1))
        self.assertEqual(len(self.voice.messages), count)

    def test_gaps_use_recorded_crossings_and_require_history(self):
        def field(offset):
            return [replace(self.lap, car_position=position, total_distance=distance + offset,
                            result_status=2) for position, distance in [(1, 20), (2, 10), (3, 0)]]
        self.radio.update_field(field(0), 1, 0, now=0)
        self.assertIsNone(self.radio.gap_ahead_s)
        self.radio.update_field(field(10), 1, 1, now=1)
        self.assertAlmostEqual(self.radio.gap_ahead_s, 1)
        self.assertAlmostEqual(self.radio.gap_behind_s, 1)
        self.assertTrue(any('ahead' in message[0] for message in self.voice.messages))
        self.radio.reset(self.session)
        self.assertIsNone(self.radio.gap_ahead_s)

    def test_sector_and_clean_race_pace_have_timing_priority(self):
        from voice_engine import Priority
        self.radio.report_sector(1, 1, 28000, 28500)
        self.assertIn('faster', self.voice.messages[-1][0])
        self.assertEqual(self.voice.messages[-1][1], Priority.TIMING)
        for lap_num, lap_time in enumerate([90000, 92000, 91000], 1):
            self.radio.completed_lap(lap_time, lap_num, {}, {})
        self.assertEqual(self.radio.race_pace_ms, 91000)
        self.assertEqual(self.voice.messages[-1][1], Priority.RACE)

    def test_session_strategy_fields_follow_2021_wire_offsets(self):
        packet = bytearray(625)
        struct.pack_into('<HBBBBQfIBB', packet, 0, 2021, 1, 18, 1, 1, 123, 0, 1, 0, 255)
        struct.pack_into('<BbbBHBbB', packet, 24, 0, 30, 20, 53, 5793, 10, 11, 0)
        packet[613:616] = bytes([10, 12, 5])
        packet[38] = 1
        session = parse_session_data(packet)
        self.assertEqual((session.pit_stop_window_ideal_lap, session.pit_stop_window_latest_lap,
                          session.pit_stop_rejoin_position, session.game_paused), (10, 12, 5, 1))

    def test_analysis_uses_only_clean_complete_laps_for_race_pace(self):
        fixture = test_analysis.LapValidityTests()
        fixture.setUp()
        for lap_num in range(1, 5):
            for distance in range(0, 1000, 20):
                lap = replace(fixture.lap, current_lap_num=lap_num, lap_distance=distance,
                              current_lap_time_ms=100 + distance * 50,
                              current_lap_invalid=int(lap_num == 3 and distance == 400))
                fixture.engine.update_telemetry(fixture.telemetry, lap)
            fixture.engine.update_telemetry(fixture.telemetry, replace(
                fixture.lap, current_lap_num=lap_num + 1, lap_distance=0,
                current_lap_time_ms=100, last_lap_time_ms=50000))
            if lap_num < 4:
                self.assertIsNone(fixture.engine.radio.race_pace_ms)
        self.assertEqual(fixture.engine.radio.race_pace_ms, 50000)
