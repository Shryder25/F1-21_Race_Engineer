"""Quiet mode's end-of-lap coaching, chatty mode's race-info calls, and no braking-point calls
unless asked for."""

import json
import pathlib
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import lap_coaching
import race_briefing
from ai_client import AIClient
from analysis import PROACTIVE_KEYS
from physics_models import evaluate_opponent_threat
from pipeline.config import EngineConfig


def corner(start, end, sector, brake=None, apex=None):
    return {
        'start_time_ms': start,
        'end_time_ms': end,
        'sector': sector,
        'brake_point': brake,
        'min_speed': apex,
    }


class LapCoachingTests(unittest.TestCase):
    def test_names_the_sector_and_corner_and_why(self):
        best = {'Turn 12': corner(70000, 74000, 3, brake=2400, apex=110)}
        now = {'Turn 12': corner(70000, 74300, 3, brake=2375, apex=109), 'Turn 3': corner(20000, 22000, 1)}
        line = lap_coaching.coaching_line(
            {1: 23000, 2: 20000, 3: 22400}, {1: 23000, 2: 20000, 3: 22000}, now, best
        )
        self.assertEqual(
            line,
            'You lost four tenths in sector three. Turn 12 cost you three tenths, '
            'you braked twenty-five metres early. Fix that next lap.',
        )

    def test_slow_apex(self):
        best = {'Stowe': corner(0, 3000, 3, brake=1000, apex=180)}
        now = {'Stowe': corner(0, 3200, 3, brake=1000, apex=170)}
        line = lap_coaching.coaching_line({3: 30300}, {3: 30000}, now, best)
        self.assertIn('Stowe cost you two tenths, ten k slower at the apex', line)

    def test_only_corners_in_the_slow_sector(self):
        best = {'T1': corner(0, 2000, 1), 'T9': corner(0, 2000, 3)}
        now = {'T1': corner(0, 2500, 1), 'T9': corner(0, 2100, 3)}
        line = lap_coaching.coaching_line({1: 20000, 3: 30300}, {1: 20000, 3: 30000}, now, best)
        self.assertIn('T9', line)
        self.assertNotIn('T1', line)

    def test_nothing_to_say_when_close_to_best(self):
        self.assertIsNone(lap_coaching.coaching_line({1: 20050, 2: 20000}, {1: 20000, 2: 20000}, {}, {}))
        self.assertIsNone(lap_coaching.coaching_line({1: 20500}, {}, {}, {}))  # first lap: no reference

    def test_best_corners_keep_the_quickest_run(self):
        best = {}
        lap_coaching.update_best_corners({'T1': corner(0, 2000, 1)}, best)
        lap_coaching.update_best_corners({'T1': corner(0, 2100, 1), 'T2': corner(0, None, 1)}, best)
        self.assertEqual(best['T1']['end_time_ms'], 2000)
        self.assertNotIn('T2', best)


def prediction(position, name, compound, age, stop_lap=None, next_compound=None):
    return SimpleNamespace(
        position=position,
        name=name,
        compound=compound,
        tyre_age=age,
        will_stop=stop_lap is not None,
        stop_lap=stop_lap,
        next_compound=next_compound,
    )


class RaceBriefingTests(unittest.TestCase):
    def test_threat_behind(self):
        threat = evaluate_opponent_threat(3, 6, -0.8, -0.3, 'Hamilton')
        self.assertEqual(
            race_briefing.threat_line([threat]),
            'Hamilton behind is the threat, eight tenths back, three tenths a lap quicker, on you in three laps.',
        )

    def test_pit_lap_pace_is_not_quoted(self):
        threat = evaluate_opponent_threat(3, 6, -0.8, -8.1, 'Perez')
        self.assertEqual(
            race_briefing.threat_line([threat]), 'Perez behind is the threat, eight tenths back.'
        )

    def test_no_threat(self):
        far = evaluate_opponent_threat(3, 6, -9.0, 0.5, 'Hamilton')
        self.assertIsNone(race_briefing.threat_line([far]))

    def test_rival_tyres_and_stops(self):
        predictions = [
            prediction(4, 'Verstappen', 'Medium', 12, stop_lap=20, next_compound='Hard'),
            prediction(6, 'Hamilton', 'hard', 5),
            prediction(1, 'Bottas', 'soft', 3),
        ]
        self.assertEqual(
            race_briefing.rival_tyres_line(predictions, 5, 18),
            'Verstappen ahead on mediums, twelve laps old, likely to box lap twenty for hards. '
            'Hamilton behind on hards, five laps old.',
        )

    def test_safety_car_chance(self):
        self.assertEqual(
            race_briefing.safety_car_line({'p_safety_car': 0.42}),
            'Safety Car chance for the rest of the race about forty percent.',
        )
        self.assertIn('flexible', race_briefing.safety_car_line({'p_safety_car': 0.6}))
        self.assertIsNone(race_briefing.safety_car_line(None))

    def test_rotates_topics_once_per_lap(self):
        briefing = race_briefing.RaceBriefing()
        builders = {'threats': lambda: 'T', 'rival_tyres': lambda: None, 'safety_car': lambda: 'S'}
        self.assertEqual(briefing.line(3, builders), 'T')
        self.assertIsNone(briefing.line(3, builders))  # once per lap
        self.assertEqual(briefing.line(4, builders), 'S')  # rival tyres had nothing: skipped
        self.assertEqual(briefing.line(5, builders), 'T')


class ModeDefaultsTests(unittest.TestCase):
    def test_quiet_mode_gets_pace_not_tyre_temps(self):
        for key in ('lap_time', 'lap_coaching', 'race_pace'):
            self.assertIn(key, PROACTIVE_KEYS)
        self.assertNotIn('tyre_temperature', PROACTIVE_KEYS)
        self.assertNotIn('race_briefing', PROACTIVE_KEYS)  # chatty only

    def test_braking_calls_off_by_default(self):
        self.assertFalse(EngineConfig().radio.braking_calls)
        self.assertTrue(EngineConfig().radio.race_briefing)


class FakePs(BaseHTTPRequestHandler):
    vram = 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        data = json.dumps(
            {
                'models': [
                    {'name': 'qwen2.5:3b', 'model': 'qwen2.5:3b', 'size': 2000, 'size_vram': FakePs.vram}
                ]
            }
        ).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ModelPlacementTests(unittest.TestCase):
    def test_reports_how_much_is_on_the_graphics_card(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), FakePs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            client = AIClient(f'http://127.0.0.1:{server.server_port}/v1', 'qwen2.5:3b')
            client.ollama_root = f'http://127.0.0.1:{server.server_port}'
            FakePs.vram = 2000
            self.assertEqual(client.model_placement(), '100% on the graphics card')
            FakePs.vram = 800
            self.assertEqual(client.model_placement(), '40% on the graphics card, 60% on the processor')
            FakePs.vram = 0
            self.assertEqual(client.model_placement(), 'all on the processor (CPU)')
            client.model = 'llama3.2:3b'
            self.assertEqual(client.model_placement(), 'not loaded')
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
