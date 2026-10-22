import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ai_client import AIClient, AITimeout
from engineer_brain import EngineerBrain
from test_analysis import SilentVoice  # noqa: F401  (puts ai_engineer/ on the import path)


class FakeServer(BaseHTTPRequestHandler):
    bodies = []
    reject_think = False
    delay = 0.0

    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeServer.bodies.append((self.path, body))
        time.sleep(FakeServer.delay)
        if self.path == '/api/chat' and FakeServer.reject_think and 'think' in body:
            self.send_response(400)
            self.end_headers()
            return
        voice = '{"voice": "Gap ahead one point four, you are catching.", "advice": ""}'
        if self.path == '/api/chat':
            payload = {'message': {'content': voice}}
        else:
            payload = {'choices': [{'message': {'content': voice}}]}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AILatencyTests(unittest.TestCase):
    def setUp(self):
        FakeServer.bodies, FakeServer.reject_think, FakeServer.delay = [], False, 0.0
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeServer)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        root = f'http://127.0.0.1:{self.server.server_port}'
        self.client = AIClient(root + '/v1', 'deepseek-r1:8b', timeout=2)
        self.client.ollama_root = root          # pretend it's the local Ollama on 11434

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_radio_answers_switch_thinking_off_with_a_small_budget(self):
        brain = EngineerBrain(lambda: {'gap_ahead_s': 1.4}, self.client)
        self.assertEqual(brain.answer_question('Can I catch the car ahead?'),
                         'Gap ahead one point four, you are catching.')
        path, body = FakeServer.bodies[0]
        self.assertEqual(path, '/api/chat')
        self.assertIs(body['think'], False)
        self.assertEqual(body['format'], 'json')
        self.assertLessEqual(body['options']['num_predict'], 300)
        self.assertEqual(len(FakeServer.bodies), 1)

    def test_older_ollama_without_think_switch_still_works(self):
        FakeServer.reject_think = True
        result = self.client.chat_json('s', 'u', fast=True)
        self.assertIn('voice', result)
        self.assertEqual([('think' in b) for _, b in FakeServer.bodies], [True, False])

    def test_timeout_is_not_retried(self):
        FakeServer.delay = 1.5
        self.client.timeout = 0.5
        brain = EngineerBrain(lambda: {'gap_ahead_s': 1.4, 'session_mode': 'race'}, self.client)
        started = time.monotonic()
        answer = brain.answer('gap_ahead')
        self.assertLess(time.monotonic() - started, 1.2)    # one wait, not three
        self.assertEqual(brain.last_source, 'fallback')
        self.assertIn('Gap ahead', answer)
        self.assertEqual(len(FakeServer.bodies), 1)

    def test_timeout_raises_ai_timeout(self):
        FakeServer.delay = 1.0
        self.client.timeout = 0.3
        self.client.ollama_root = None
        with self.assertRaises(AITimeout):
            self.client.chat_json('s', 'u')
        self.assertEqual(len(FakeServer.bodies), 1)


if __name__ == '__main__':
    unittest.main()
