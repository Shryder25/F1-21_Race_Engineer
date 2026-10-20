import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import session_modes
import web_hud
from analysis import AnalysisEngine
from engineer_brain import EngineerBrain, rule_based_answer
from packets import TRACK_NAMES, SessionData, parse_session_data
from simulate_telemetry import build_session_packet, build_lap_packet, build_telemetry_packet
from packets import parse_player_lap_data, parse_player_telemetry
from test_analysis import SilentVoice
from track_data import TRACK_DATABASE, corner_zones, get_braking_points, get_track_setup, resolve_track_id


class SessionModeTests(unittest.TestCase):
    def test_every_game_session_type_maps_to_a_mode(self):
        expected = {**{i: 'practice' for i in (1, 2, 3, 4)}, **{i: 'qualifying' for i in (5, 6, 7, 8, 9)},
                    **{i: 'race' for i in (10, 11, 12)}, 13: 'time_trial', 0: 'unknown', 99: 'unknown'}
        for session_type, mode in expected.items():
            self.assertEqual(session_modes.session_mode(session_type), mode, session_type)
        self.assertEqual(session_modes.mode_label(13), 'Time Trial')

    def test_what_each_mode_reports(self):
        self.assertTrue(session_modes.reports_gaps('race'))
        for mode in ('practice', 'qualifying', 'time_trial'):
            self.assertFalse(session_modes.reports_gaps(mode), mode)
        self.assertTrue(session_modes.has_pit_strategy('practice'))
        self.assertFalse(session_modes.has_pit_strategy('qualifying'))
        self.assertFalse(session_modes.has_pit_strategy('time_trial'))
        self.assertIn('time trial', session_modes.welcome_line('time_trial', 'Monza'))


class TrackAwarenessTests(unittest.TestCase):
    def test_every_database_circuit_gets_corner_zones_that_do_not_overlap(self):
        for track_id, info in TRACK_DATABASE.items():
            zones = corner_zones(track_id, info['length_m'])
            self.assertEqual(bool(zones), bool(get_braking_points(track_id)), track_id)
            self.assertGreaterEqual(len(zones), 4, TRACK_NAMES.get(track_id))
            for zone in zones:
                self.assertLess(zone['start'], zone['ref_brake'])
                self.assertLessEqual(zone['end'], info['length_m'])
            for a, b in zip(zones, zones[1:]):
                self.assertLess(a['end'], b['start'], (track_id, a, b))

    def test_short_layouts_use_their_parent_setup(self):
        self.assertEqual(resolve_track_id(22), 7)
        self.assertEqual(get_track_setup(22)['track_name'], TRACK_DATABASE[7]['name'])
        self.assertEqual(corner_zones(22), [])

    def test_analysis_follows_the_game_to_any_circuit_and_session(self):
        engine = AnalysisEngine(SilentVoice())
        seen = {}
        for track_id, session_type in ((11, 13), (5, 7), (3, 10), (7, 1)):
            session = parse_session_data(build_session_packet(0, 1, track_id, session_type,
                                                              TRACK_DATABASE[track_id]['length_m']))
            engine.update_session(session, 1000 + track_id, 0)
            state = web_hud.get_web_state()
            seen[track_id] = (engine.track_name, engine.session_mode, state['track_id'], state['session_mode'],
                              [c['name'] for c in engine.corners][:1])
        self.assertEqual(seen[11][:4], ('Monza', 'time_trial', 11, 'time_trial'))
        self.assertEqual(seen[5][:4], ('Monaco', 'qualifying', 5, 'qualifying'))
        self.assertEqual(seen[3][:4], ('Bahrain', 'race', 3, 'race'))
        self.assertEqual(seen[7][:4], ('Silverstone', 'practice', 7, 'practice'))
        for track_id in (11, 5, 3):               # not just Silverstone
            self.assertTrue(seen[track_id][4], track_id)
        self.assertEqual(web_hud.get_web_state()['session_label'], 'Free Practice')

    def test_corner_analysis_works_on_a_non_silverstone_track(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(parse_session_data(build_session_packet(0, 1, 11, 13, 5793)), 5, 0)
        zone = engine.corners[0]                   # Monza turn 1 chicane
        lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 1), 0)
        telemetry = parse_player_telemetry(build_telemetry_packet(1, 1, 200, 1, 0, 6, 95), 0)
        samples = [(zone['start'] - 20, 1000, 280, 0), (zone['start'] + 5, 1100, 270, 0),
                   (zone['ref_brake'] - 5, 1400, 260, 0), (zone['ref_brake'] + 5, 1450, 250, 1),
                   (zone['ref_brake'] + 60, 1700, 90, 1), (zone['end'] - 5, 2200, 150, 0),
                   (zone['end'] + 20, 2300, 200, 0)]
        for distance, elapsed, speed, brake in samples:
            engine.update_telemetry(replace(telemetry, speed=speed, brake=brake),
                                    replace(lap, lap_distance=distance, current_lap_time_ms=elapsed))
        diagnostics = engine.corner_diagnostics
        self.assertEqual(len(diagnostics), 1)
        self.assertEqual(diagnostics[0]['name'], zone['name'])
        self.assertIn('benchmark', diagnostics[0]['advice'])


class SessionAwareAnswerTests(unittest.TestCase):
    def test_commands_that_do_not_exist_in_the_session_are_answered_directly(self):
        class MustNotBeCalled:
            def chat(self, *args, **kwargs):
                raise AssertionError('AI should not be asked')

        for mode, command, expected in (
                ('time_trial', 'gap_ahead', 'No other cars in time trial.'),
                ('qualifying', 'gap_behind', 'only tracked in the race'),
                ('qualifying', 'pit_window', 'No pit stops in qualifying.'),
                ('time_trial', 'pit_window', 'No pit stops in time trial.')):
            brain = EngineerBrain(lambda m=mode: {'session_mode': m, 'gap_ahead_s': 1.0}, MustNotBeCalled())
            self.assertIn(expected, brain.answer(command))
        practice = EngineerBrain(lambda: {'session_mode': 'practice', 'pit_prediction': {
            'ideal_lap': 12, 'latest_lap': 16, 'finish_without_stop': False}}, None)
        self.assertIn('lap 12', practice.answer('pit_window'))          # tyre planning still works
        race = EngineerBrain(lambda: {'session_mode': 'race', 'gap_ahead_s': 2.43}, None)
        self.assertEqual(race.answer('gap_ahead'), 'Gap ahead is 2.4.')

    def test_pit_prediction_is_off_outside_race_and_practice(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(parse_session_data(build_session_packet(0, 1, 11, 13)), 9, 0)
        engine.latest_status = engine.latest_lap = object()
        self.assertIsNone(engine.pit_prediction())


class HudPlaceholderTests(unittest.TestCase):
    def test_hud_has_no_hard_coded_circuit_or_session(self):
        root = pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'
        html = (root / 'templates' / 'index.html').read_text(encoding='utf-8').lower()
        js = (root / 'static' / 'app.js').read_text(encoding='utf-8')
        self.assertNotIn('silverstone', html)
        self.assertIn('data-modes="race"', html)
        self.assertIn('let currentTrackId = null', js)
        # Check the module's own source for the default, not the live dashboard_state singleton:
        # other tests in the same process legitimately mutate that shared global via update_web_state.
        web_hud_src = (root / 'web_hud.py').read_text(encoding='utf-8')
        self.assertIn('"track_id": -1', web_hud_src)


if __name__ == '__main__':
    unittest.main()
