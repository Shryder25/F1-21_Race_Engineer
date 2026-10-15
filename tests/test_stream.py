import pathlib
import struct
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from analysis import AnalysisEngine
from simulate_telemetry import build_session_packet, build_lap_packet, build_telemetry_packet
from test_analysis import SilentVoice
import web_hud


def header_change(packet, uid=None, packet_format=None, player=None):
    data = bytearray(packet)
    if uid is not None:
        struct.pack_into('<Q', data, 6, uid)
    if packet_format is not None:
        struct.pack_into('<H', data, 0, packet_format)
    if player is not None:
        data[22] = player
    return bytes(data)


class StreamTests(unittest.TestCase):
    def setUp(self):
        from telemetry_stream import TelemetryStream
        self.engine = AnalysisEngine(SilentVoice())
        self.stream = TelemetryStream(self.engine)
        self.session = build_session_packet(0, 1, 11)
        self.stream.handle(self.session, now=0)

    def pair(self, frame=2, now=1, uid=None):
        telem = header_change(build_telemetry_packet(frame / 25, frame, 280, 1, 0, 8, 95), uid=uid)
        lap = header_change(build_lap_packet(frame / 25, frame, 1000 + frame, 20000 + frame), uid=uid)
        self.stream.handle(telem, now=now)
        return self.stream.handle(lap, now=now)

    def test_only_matching_frames_update_analysis(self):
        telem = build_telemetry_packet(0.08, 2, 280, 1, 0, 8, 95)
        lap = build_lap_packet(0.12, 3, 1030, 21000)
        self.assertFalse(self.stream.handle(telem, now=1))
        self.assertFalse(self.stream.handle(lap, now=1))
        self.assertIsNone(self.stream.latest_telemetry)
        self.assertTrue(self.stream.handle(build_telemetry_packet(0.12, 3, 290, 1, 0, 8, 95), now=1.1))
        self.assertEqual(self.stream.latest_telemetry.speed, 290)

    def test_malformed_wrong_format_and_bad_player_are_ignored(self):
        good = build_telemetry_packet(0.08, 2, 280, 1, 0, 8, 95)
        for packet in [good[:40], header_change(good, packet_format=2022), header_change(good, player=22)]:
            self.assertFalse(self.stream.handle(packet, now=1))
        self.assertIsNone(self.stream.latest_telemetry)

    def test_nonfinite_lap_is_ignored(self):
        lap = build_lap_packet(0.08, 2, float('nan'), 21000)
        self.stream.handle(build_telemetry_packet(0.08, 2, 280, 1, 0, 8, 95), now=1)
        self.assertFalse(self.stream.handle(lap, now=1))

    def test_older_frames_cannot_overwrite_current_state(self):
        self.assertTrue(self.pair(frame=4))
        self.assertFalse(self.pair(frame=3))
        self.assertEqual(self.stream.latest_lap.lap_distance, 1004)

    def test_silent_telemetry_marks_hud_stale_and_invalidates_recording(self):
        self.pair()
        self.stream.check_timeout(now=4)
        self.assertFalse(web_hud.get_web_state(now=4)['connected'])
        self.assertTrue(self.engine.delta.current_lap_invalid)

    def test_other_session_data_requires_session_packet_and_clears_pending(self):
        self.pair()
        self.assertFalse(self.pair(frame=3, uid=999))
        self.stream.handle(header_change(self.session, uid=999), now=2)
        self.assertIsNone(self.stream.latest_telemetry)
        self.assertEqual(self.engine.delta.current_lap_num, 0)
        self.assertFalse(self.pair(frame=4))
        self.assertTrue(self.pair(frame=4, uid=999))

    def test_fresh_snapshot_expires_even_without_listener_tick(self):
        self.pair()
        self.assertTrue(web_hud.get_web_state(now=1.5)['connected'])
        self.assertFalse(web_hud.get_web_state(now=4)['connected'])

    def test_telemetry_recovers_after_timeout(self):
        self.pair()
        self.stream.check_timeout(now=4)
        self.assertTrue(self.pair(frame=3, now=4.1))
        self.assertTrue(web_hud.get_web_state(now=4.2)['connected'])

    def test_flashback_allows_lower_frame_numbers_but_rejects_current_reference_lap(self):
        self.pair(frame=100, now=1)
        event = bytearray(build_session_packet(4, 101, 11)[:24])
        event[5] = 3
        event.extend(b'FLBK' + struct.pack('<If', 2, 0.08))
        self.stream.handle(event, now=1.1)
        self.assertTrue(self.pair(frame=3, now=1.2))
        self.assertTrue(self.engine.delta.current_lap_invalid)

    def test_paused_game_does_not_feed_race_callouts(self):
        paused = bytearray(build_session_packet(0.08, 2, 11))
        paused[38] = 1
        self.stream.handle(paused, now=1)
        self.assertFalse(self.pair(frame=3, now=1.1))
        self.assertFalse(web_hud.get_web_state(now=1.2)['connected'])

    def test_other_car_packet_does_not_poison_player_frame_order(self):
        packet = header_change(build_telemetry_packet(0.08, 2, 280, 1, 0, 8, 95), player=1)
        self.assertFalse(self.stream.handle(packet, now=1))
        self.assertTrue(self.pair(frame=2, now=1.1))

    def test_delayed_pre_flashback_packets_cannot_poison_restored_timeline(self):
        self.pair(frame=100, now=1)
        event = bytearray(build_session_packet(4, 101, 11)[:24])
        event[5] = 3
        event.extend(b'FLBK' + struct.pack('<If', 2, 0.08))
        self.stream.handle(event, now=1.1)
        self.assertFalse(self.pair(frame=100, now=1.15))
        self.assertTrue(self.pair(frame=3, now=1.2))
        self.assertFalse(self.pair(frame=100, now=1.25))
        self.assertTrue(self.pair(frame=4, now=1.3))
