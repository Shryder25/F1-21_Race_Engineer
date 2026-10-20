"""Track map: the whole lap is kept, the circuit outline is stored per track and reused,
the pit lane is recorded, and a flashback cuts the trail back instead of drawing forward."""

import json
import pathlib
import struct
import sys
import tempfile
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))

from analysis import AnalysisEngine
from packets import parse_player_lap_data, parse_player_telemetry, parse_player_motion, SessionData
from simulate_telemetry import build_lap_packet, build_telemetry_packet, build_session_packet
from test_analysis import SilentVoice
from test_track_map_tyre_and_report import build_motion_packet
import track_maps
import web_hud

TRACK_LENGTH = 1000
SESSION = SessionData(0, 30, 20, 20, TRACK_LENGTH, 10, 11, 0)


def position(distance):
    """A square-ish loop: x, z for a lap distance."""
    return float(distance), float(distance) * 0.5


class Driver:
    def __init__(self, engine):
        self.engine = engine
        self.telemetry = parse_player_telemetry(build_telemetry_packet(1, 1, 200, 1, 0, 6, 95), 0)
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 3), 0)

    def at(self, distance, lap_num=1, pit_status=0, xz=None):
        x, z = xz if xz is not None else position(distance)
        lap = replace(self.lap, lap_distance=distance, current_lap_num=lap_num,
                      current_lap_time_ms=100 + int(distance * 50), pit_status=pit_status)
        self.engine.update_telemetry(self.telemetry, lap)
        self.engine.update_motion(parse_player_motion(build_motion_packet(1.0, 1, x, z), 0))

    def lap_from(self, start, end, lap_num=1, step=5):
        for distance in range(start, end, step):
            self.at(distance, lap_num)


class OutlineBuildingTests(unittest.TestCase):
    def test_full_lap_becomes_outline(self):
        dists = list(range(0, 1000, 5))
        outline = track_maps.build_outline(dists, [list(position(d)) for d in dists], 1000)
        self.assertIsNotNone(outline)
        self.assertEqual(outline[0], [0.0, 0.0])
        self.assertLessEqual(len(outline), 1000 / track_maps.OUTLINE_BIN_M + 1)

    def test_partial_lap_or_lap_with_a_hole_is_rejected(self):
        dists = list(range(0, 600, 5))
        self.assertIsNone(track_maps.build_outline(dists, [list(position(d)) for d in dists], 1000))
        dists = list(range(0, 400, 5)) + list(range(500, 1000, 5))
        self.assertIsNone(track_maps.build_outline(dists, [list(position(d)) for d in dists], 1000))

    def test_store_round_trips_through_file_and_checks_track_length(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'track_maps.json'
            store = track_maps.TrackMapStore(path)
            store.save(11, 5800, outline=[[0, 0], [1, 1]], pit_lane=[[2, 2], [3, 3]])
            again = track_maps.TrackMapStore(path)
            self.assertEqual(again.get(11, 5810)['outline'], [[0, 0], [1, 1]])
            self.assertEqual(again.get(11, 5810)['pit_lane'], [[2, 2], [3, 3]])
            self.assertEqual(again.get(11, 4000)['outline'], [])       # different layout
            again.forget(11)
            self.assertEqual(json.loads(path.read_text()), {})


class TrackMapRecordingTests(unittest.TestCase):
    def setUp(self):
        self.store = track_maps.TrackMapStore()
        self.engine = AnalysisEngine(SilentVoice(), track_store=self.store)
        self.engine.update_session(SESSION)
        self.driver = Driver(self.engine)

    def test_trail_keeps_the_whole_lap_instead_of_the_last_400_points(self):
        # 3 m steps for 2.9 km of a long track: well over the old 400-point cut-off.
        long_session = replace(SESSION, track_length=6000)
        self.engine.update_session(long_session)
        for distance in range(0, 2900, 3):
            self.driver.at(distance)
        trail = web_hud.get_web_state(now=0)['track_map']
        self.assertGreater(len(trail), 400)
        self.assertEqual(trail[0], list(position(0)))

    def test_first_clean_lap_becomes_the_stored_outline(self):
        self.driver.lap_from(0, 1000)
        self.driver.at(2, lap_num=2)
        state = web_hud.get_web_state(now=0)
        self.assertGreaterEqual(len(state['track_outline']), 90)
        self.assertEqual(state['track_map_source'], 'stored')
        self.assertEqual(self.store.get(11, TRACK_LENGTH)['outline'], state['track_outline'])

    def test_stored_outline_is_shown_straight_away_in_a_new_session(self):
        outline = [list(position(d)) for d in range(0, 1000, 8)]
        self.store.save(11, TRACK_LENGTH, outline=outline, pit_lane=[[1, 1], [2, 2]])
        engine = AnalysisEngine(SilentVoice(), track_store=self.store)
        engine.update_session(SESSION, session_uid=99)
        state = web_hud.get_web_state(now=0)
        self.assertEqual(state['track_outline'], outline)
        self.assertEqual(state['pit_lane'], [[1, 1], [2, 2]])
        self.assertEqual(state['track_map_source'], 'stored')

    def test_pit_lane_is_recorded_separately_and_kept(self):
        self.driver.lap_from(0, 900)
        for i in range(30):                      # drive down the pit lane, off the racing line
            self.driver.at(900 + i * 3, pit_status=1, xz=(900.0 + i * 5, -50.0))
        self.driver.at(990)
        pit_lane = web_hud.get_web_state(now=0)['pit_lane']
        self.assertGreaterEqual(len(pit_lane), track_maps.MIN_PIT_POINTS)
        self.assertTrue(all(p[1] == -50.0 for p in pit_lane))
        self.assertNotIn([900.0, -50.0], self.engine.current_lap_map_points)
        self.assertEqual(self.store.get(11, TRACK_LENGTH)['pit_lane'], pit_lane)

    def test_lap_through_the_pits_is_not_used_as_the_outline(self):
        self.driver.lap_from(0, 500)
        for i in range(30):
            self.driver.at(500 + i * 3, pit_status=2, xz=(500.0 + i * 5, -50.0))
        self.driver.lap_from(600, 1000)
        self.driver.at(2, lap_num=2)
        self.assertEqual(self.engine.track_outline, [])
        self.driver.lap_from(5, 1000, lap_num=2)     # a clean lap maps the circuit
        self.driver.at(2, lap_num=3)
        self.assertTrue(self.engine.track_outline)
        self.assertNotIn([500.0, -50.0], self.engine.track_outline)

    def test_flashback_cuts_the_trail_back_to_the_rewound_position(self):
        self.driver.lap_from(0, 600)
        self.engine.on_flashback()
        # Old-timeline motion arriving before the restored lap packet is not recorded.
        self.engine.update_motion(parse_player_motion(build_motion_packet(1.0, 1, 999.0, 999.0), 0))
        self.driver.at(300)
        dists = self.engine.current_lap_map_dists
        self.assertLessEqual(max(dists), 300)
        self.assertNotIn([999.0, 999.0], self.engine.current_lap_map_points)
        self.assertEqual(web_hud.get_web_state(now=0)['flashback_count'], 1)
        # Driving on from the rewound point continues the same lap and can still map it.
        self.driver.lap_from(305, 1000)
        self.driver.at(2, lap_num=2)
        self.assertTrue(self.engine.track_outline)

    def test_rewind_without_an_event_still_trims_the_trail(self):
        self.driver.lap_from(0, 500)
        self.driver.at(200)
        self.assertLessEqual(max(self.engine.current_lap_map_dists), 200)

    def test_reset_map_forgets_the_stored_outline(self):
        self.driver.lap_from(0, 1000)
        self.driver.at(2, lap_num=2)
        self.assertTrue(self.store.get(11, TRACK_LENGTH)['outline'])
        self.engine.reset_map()
        self.assertEqual(self.store.get(11, TRACK_LENGTH)['outline'], [])
        self.assertEqual(web_hud.get_web_state(now=0)['track_outline'], [])
        self.assertEqual(web_hud.get_web_state(now=0)['track_map_source'], 'recording')


class FlashbackStreamTests(unittest.TestCase):
    def test_flashback_event_tells_the_map_to_rewind(self):
        from telemetry_stream import TelemetryStream
        engine = AnalysisEngine(SilentVoice())
        stream = TelemetryStream(engine)
        stream.handle(build_session_packet(0, 1, 11), now=0)
        event = bytearray(build_session_packet(4, 101, 11)[:24])
        event[5] = 3
        event.extend(b'FLBK' + struct.pack('<If', 2, 0.08))
        stream.handle(bytes(event), now=1)
        self.assertTrue(engine.map_rewind_pending)
        self.assertEqual(engine.flashback_count, 1)


if __name__ == '__main__':
    unittest.main()
