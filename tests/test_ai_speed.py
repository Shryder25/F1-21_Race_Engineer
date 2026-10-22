"""Faster AI answers and less load on the game: streaming, instant answers, focused prompts,
matching Ollama load options, and the extra car data the AI can now answer from."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

from test_analysis import SilentVoice  # noqa: F401  (puts ai_engineer/ on the import path)
import ai_launcher
from ai_client import AIClient, ollama_options, voice_from_partial
from analysis import AnalysisEngine
from engineer_brain import EngineerBrain, build_prompt, format_context, question_topics


class StreamingOllama(BaseHTTPRequestHandler):
    """Streams a JSON radio reply a few characters at a time, slowly after the voice line."""
    bodies = []
    finished = threading.Event()

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        StreamingOllama.bodies.append((self.path, body))
        reply = '{"voice": "Battery is at sixty percent, use overtake on the straight.", "advice": "' + 'x' * 40 + '"}'
        pieces = [reply[i:i + 6] for i in range(0, len(reply), 6)]
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson')
        self.end_headers()
        try:
            for piece in pieces:
                line = {'message': {'content': piece}, 'done': False}
                self.wfile.write(json.dumps(line).encode() + b'\n')
                self.wfile.flush()
                if '"advice"' in ''.join(pieces[:pieces.index(piece) + 1]):
                    time.sleep(0.2)          # the on-screen note is slow to write
            self.wfile.write(json.dumps({'message': {'content': ''}, 'done': True}).encode() + b'\n')
            StreamingOllama.finished.set()
        except (BrokenPipeError, ConnectionResetError):
            pass


class StreamingTests(unittest.TestCase):
    def setUp(self):
        StreamingOllama.bodies = []
        StreamingOllama.finished = threading.Event()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), StreamingOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        root = f'http://127.0.0.1:{self.server.server_port}'
        self.client = AIClient(root + '/v1', 'deepseek-r1:8b', timeout=5,
                               load_options=ollama_options(4096, 20, 4))
        self.client.ollama_root = root

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_voice_line_returns_before_the_rest_of_the_reply(self):
        started = time.monotonic()
        voice = self.client.radio_voice('s', 'u')
        self.assertEqual(voice, 'Battery is at sixty percent, use overtake on the straight.')
        self.assertLess(time.monotonic() - started, 1.0)     # the slow note would take ~2 s
        self.assertFalse(StreamingOllama.finished.is_set())
        path, body = StreamingOllama.bodies[0]
        self.assertEqual(path, '/api/chat')
        self.assertTrue(body['stream'])
        self.assertIs(body['think'], False)
        self.assertEqual(body['format'], 'json')
        self.assertEqual(body['options']['num_ctx'], 4096)
        self.assertEqual(body['options']['num_gpu'], 20)
        self.assertEqual(body['options']['num_thread'], 4)

    def test_brain_uses_the_stream_for_free_questions(self):
        brain = EngineerBrain(lambda: {'ers': {'battery_pct': 60, 'deploy_mode': 'medium',
                                               'used_this_lap_pct': 10}}, self.client)
        self.assertIn('Battery', brain.answer_question('how much battery have I got left?'))
        self.assertEqual(brain.last_source, 'ai')

    def test_not_a_local_ollama_means_no_stream(self):
        self.client.ollama_root = None
        self.assertIsNone(self.client.radio_voice('s', 'u'))


class HelperTests(unittest.TestCase):
    def test_voice_from_partial(self):
        self.assertIsNone(voice_from_partial('{"voice": "Box this'))
        self.assertEqual(voice_from_partial('{"voice": "Box, box.", "adv'), 'Box, box.')
        self.assertEqual(voice_from_partial('{"voice": "Say \\"go\\" now"'), 'Say "go" now')

    def test_ollama_options(self):
        self.assertEqual(ollama_options(), {'num_ctx': 4096})
        self.assertEqual(ollama_options(8192, 0, 2), {'num_ctx': 8192, 'num_gpu': 0, 'num_thread': 2})
        self.assertEqual(ollama_options(None), {})

    def test_launcher_preloads_with_the_same_options_as_questions(self):
        sent = []

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return b'{}'

        launcher = ai_launcher.LocalAILauncher('http://localhost:11434/v1', 'm',
                                               load_options={'num_ctx': 4096, 'num_gpu': 0})
        launcher._request = lambda path, body=None, timeout=5.0: sent.append(body) or Response()
        self.assertTrue(launcher._load())
        self.assertEqual(sent[0]['options'], {'num_ctx': 4096, 'num_gpu': 0})

    def test_ollama_is_started_below_normal_priority(self):
        seen = {}

        def fake_popen(cmd, **kwargs):
            seen.update(kwargs)
            raise OSError('stop here')

        launcher = ai_launcher.LocalAILauncher('http://localhost:11434/v1', 'm', on_status=lambda t: None,
                                               executable_finder=lambda: '/opt/ollama', popen=fake_popen)
        launcher._start_server()
        if ai_launcher.sys.platform == 'win32':
            self.assertTrue(seen['creationflags'] & 0x4000)
        else:
            self.assertIn('preexec_fn', seen)
        seen.clear()
        launcher.low_priority = False
        launcher._start_server()
        self.assertNotIn('preexec_fn', seen)


class RulesFirstTests(unittest.TestCase):
    class CountingClient:
        supports_fast = False

        def __init__(self):
            self.calls = 0

        def chat_json(self, system, user, **kw):
            self.calls += 1
            return {'voice': 'From the AI.', 'advice': ''}

    def setUp(self):
        self.client = self.CountingClient()
        self.ctx = {'gap_ahead_s': 1.4, 'car_ahead_name': 'Hamilton', 'session_mode': 'race'}
        self.brain = EngineerBrain(lambda: self.ctx, self.client, rules_first=True)

    def test_fixed_commands_answer_instantly_without_the_ai(self):
        self.assertEqual(self.brain.answer('gap_ahead'), 'Gap to Hamilton ahead is 1.4.')
        self.assertEqual(self.brain.last_source, 'rules')
        self.assertEqual(self.client.calls, 0)

    def test_short_matching_question_is_instant_long_one_goes_to_ai(self):
        self.assertIn('Hamilton', self.brain.answer_question('gap ahead?', 'gap_ahead'))
        self.assertEqual(self.client.calls, 0)
        self.assertEqual(self.brain.answer_question(
            'can I get close enough to the car ahead to use DRS next lap', 'gap_ahead'), 'From the AI.')
        self.assertEqual(self.client.calls, 1)

    def test_default_still_asks_the_ai(self):
        brain = EngineerBrain(lambda: self.ctx, self.client)
        self.assertEqual(brain.answer('gap_ahead'), 'From the AI.')


CTX = {
    'track': 'Monza', 'session_mode': 'race', 'lap': 10, 'total_laps': 20, 'position': 4,
    'gap_ahead_s': 1.2, 'car_ahead_name': 'Hamilton', 'last_lap_ms': 83_400,
    'weather': 'clear', 'track_temp_c': 33,
    'tyre_surface_c': {'RL': 90, 'RR': 91, 'FL': 95, 'FR': 96},
    'ers': {'battery_pct': 62, 'deploy_mode': 'medium', 'used_this_lap_pct': 20},
    'brake_temps_c': {'RL': 500, 'RR': 500, 'FL': 610, 'FR': 600},
    'damage_pct': {'front left wing': 12, 'front right wing': 0, 'rear wing': 0, 'floor': 0,
                   'gearbox wear': 3, 'engine wear': 5, 'tyre damage': 0},
    'penalties': {'time_penalty_s': 5, 'warnings': 2, 'unserved_drive_through': 0,
                  'unserved_stop_go': 0, 'pit_stops_done': 1, 'grid_position': 6},
    'recent_race_events': ['Yellow flag in sector 2.'],
    'standings': [{'position': 1, 'name': 'Verstappen', 'gap_s': 0.0, 'last_lap_ms': 83_000,
                   'best_lap_ms': 82_500, 'compound': 'M', 'tyre_age': 8, 'stops': 1},
                  {'position': 4, 'name': 'Driver', 'is_player': True, 'gap_s': 6.3, 'interval_s': 1.2}],
}


class PromptTests(unittest.TestCase):
    def test_new_car_data_reaches_the_prompt(self):
        text = format_context(CTX)
        self.assertIn('ERS battery: 62% charge, deploy mode medium', text)
        self.assertIn('front left wing 12', text)
        self.assertNotIn('rear wing 0', text)
        self.assertIn('Brake temps C', text)
        self.assertIn('5s time penalty, 2 track-limit', text)
        self.assertIn('Yellow flag in sector 2.', text)
        self.assertIn('P1 Verstappen +0.0 last 1:23.000 best 1:22.500 M8 1 stop(s)', text)
        self.assertIn('P4 Driver (YOU)', text)

    def test_topics(self):
        self.assertEqual(question_topics('how is my battery'), {'car'})
        self.assertEqual(question_topics('tyre_temps'), {'tyres'})
        self.assertIn('rules', question_topics('do I have a penalty'))
        self.assertIsNone(question_topics('talk to me'))

    def test_focused_prompt_keeps_core_and_drops_the_rest(self):
        _, user = build_prompt('how is my battery', CTX)
        self.assertIn('ERS battery', user)
        self.assertIn('Gap to car ahead', user)          # the core always goes
        self.assertIn('Brake temps', user)               # 'car' covers brakes too
        self.assertNotIn('Running order', user)
        self.assertNotIn('Tyre surface temps', user)
        _, full = build_prompt('how is my battery', CTX, focused=False)
        self.assertIn('Running order', full)
        self.assertGreater(len(full), len(user))

    def test_a_drivers_name_brings_the_running_order(self):
        _, user = build_prompt('how far is Verstappen up the road', CTX)
        self.assertIn('Running order', user)


class EngineContextTests(unittest.TestCase):
    def test_build_context_has_car_health(self):
        status = SimpleNamespace(ers_store_energy=2_000_000.0, ers_deploy_mode=3,
                                 ers_deployed_this_lap=400_000.0)
        damage = SimpleNamespace(tyres_damage=(0, 0, 5, 0), front_left_wing=10, front_right_wing=0,
                                 rear_wing=0, floor=0, gearbox=2, engine=4)
        ctx = AnalysisEngine._car_context(None, None, damage, status)
        self.assertEqual(AnalysisEngine(SilentVoice()).build_context()['standings'], None)
        self.assertEqual(ctx['ers'], {'battery_pct': 50, 'deploy_mode': 'overtake', 'used_this_lap_pct': 10})
        self.assertEqual(ctx['damage_pct']['front left wing'], 10)
        self.assertEqual(ctx['damage_pct']['tyre damage'], 5)


if __name__ == '__main__':
    unittest.main()
