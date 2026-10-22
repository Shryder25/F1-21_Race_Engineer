"""The AI is connected but its answers were thrown away: every reply shape a local model can
send must still reach the driver, and when nothing usable comes back the log says why."""

import json
import pathlib
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from ai_check import run_ai_test
from ai_client import AIClient, voice_from_reply
from engineer_brain import AI_FAILED_REPLY, AI_TIMEOUT_REPLY, EngineerBrain

ANSWER = 'Push for two laps, then bring the rears back in.'


class FakeOllama(BaseHTTPRequestHandler):
    """Behaves like one of the ways a real local model has been seen to answer."""

    mode = 'voice'
    bodies: list = []

    def log_message(self, *args):
        pass

    def _send(self, status, payload, stream=False):
        lines = payload if stream else [payload]
        data = b''.join(json.dumps(line).encode() + b'\n' for line in lines)
        self.send_response(status)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _native(self, content='', thinking='', done_reason='stop', body=None):
        if body and body.get('stream'):
            return [
                {'message': {'content': content, 'thinking': thinking}, 'done': False},
                {'message': {'content': ''}, 'done': True, 'done_reason': done_reason},
            ]
        return {'message': {'content': content, 'thinking': thinking}, 'done_reason': done_reason}

    def do_GET(self):
        if self.path == '/api/version':
            self._send(200, {'version': '0.12.6'})
        else:
            self._send(200, {'models': [{'name': 'deepseek-r1:8b'}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeOllama.bodies.append((self.path, body))
        mode = FakeOllama.mode
        native = self.path == '/api/chat'
        budget = (body.get('options') or {}).get('num_predict') or body.get('max_tokens') or 0
        if mode == 'missing_model':
            data = json.dumps({'error': "model 'deepseek-r1:8b' not found"}).encode()
            self.send_response(404)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if mode == 'slow':
            time.sleep(1.5)
        if mode == 'other_key':  # small models drift from the requested format
            text = json.dumps({'answer': ANSWER})
        elif mode == 'thinks_anyway':  # ignores think:false; needs room to finish thinking
            if budget < 900:
                if native:
                    self._send(
                        200, self._native('', 'Let me look at the tyres... ' * 20, 'length', body), True
                    )
                    return
                text = '<think>Let me look at the tyres...'
            else:
                text = '<think>Rears are at 30% wear.</think>' + json.dumps({'voice': ANSWER})
        elif mode == 'json_in_thinking':  # some Ollama builds file the JSON under "thinking"
            if native:
                payload = self._native('', json.dumps({'voice': ANSWER, 'advice': ''}), 'stop', body)
                self._send(200, payload, bool(body.get('stream')))
                return
            text = ''
        else:
            text = json.dumps({'voice': ANSWER, 'advice': ''})
        if native:
            self._send(200, self._native(text, body=body), bool(body.get('stream')))
        else:
            self._send(200, {'choices': [{'message': {'content': text}}]})


class ReplyShapeTests(unittest.TestCase):
    def setUp(self):
        FakeOllama.bodies = []
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        root = f'http://127.0.0.1:{self.server.server_port}'
        self.client = AIClient(root + '/v1', 'deepseek-r1:8b', timeout=1)
        self.client.ollama_root = root
        self.brain = EngineerBrain(lambda: {}, self.client)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def ask(self, mode):
        FakeOllama.mode = mode
        with self.assertLogs('f1', level='INFO') if mode in ('missing_model', 'slow') else _quiet():
            return self.brain.answer_question('Should I push now or look after the tyres?')

    def test_normal_reply(self):
        self.assertEqual(self.ask('voice'), ANSWER)
        self.assertEqual(self.brain.last_source, 'ai')

    def test_reply_under_another_key_is_used(self):
        self.assertEqual(self.ask('other_key'), ANSWER)

    def test_json_filed_under_thinking_is_used(self):
        self.assertEqual(self.ask('json_in_thinking'), ANSWER)

    def test_model_that_thinks_anyway_gets_room_to_finish(self):
        self.assertEqual(self.ask('thinks_anyway'), ANSWER)
        self.assertEqual(FakeOllama.bodies[-1][1]['max_tokens'], 900)

    def test_missing_model_says_so_in_the_log_and_on_the_radio(self):
        self.assertEqual(self.ask('missing_model'), AI_FAILED_REPLY)
        self.assertIn("model 'deepseek-r1:8b' not found", self.brain.last_failure)

    def test_timeout_has_its_own_radio_reply(self):
        self.assertEqual(self.ask('slow'), AI_TIMEOUT_REPLY)
        self.assertIn('timed out', self.brain.last_failure)

    def test_ai_test_command_reports_ok_and_failure(self):
        lines: list[str] = []
        FakeOllama.mode = 'voice'
        self.assertEqual(run_ai_test(self.client, autostart=False, out=lines.append), 0)
        self.assertIn('Ollama 0.12.6', '\n'.join(lines))
        self.assertIn(ANSWER, lines[-1])
        lines.clear()
        FakeOllama.mode = 'missing_model'
        with self.assertLogs('f1', level='WARNING'):
            self.assertEqual(run_ai_test(self.client, autostart=False, out=lines.append), 1)
        self.assertIn('not found', '\n'.join(lines))
        lines.clear()
        self.client.model = 'qwen2.5:3b'
        self.assertEqual(run_ai_test(self.client, autostart=False, out=lines.append), 1)
        self.assertIn('qwen2.5:3b is not installed', lines[-1])


class _quiet:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class VoiceFromReplyTests(unittest.TestCase):
    def test_shapes(self):
        self.assertEqual(voice_from_reply('{"voice": "Box this lap.", "advice": "x"}'), 'Box this lap.')
        self.assertEqual(voice_from_reply('{"Voice": "Box this lap."}'), 'Box this lap.')
        self.assertEqual(voice_from_reply('{"response": "Box this lap."}'), 'Box this lap.')
        self.assertEqual(voice_from_reply('Sure: {"voice": "Box this lap."}'), 'Box this lap.')
        self.assertEqual(voice_from_reply('<think>hmm</think>Box this lap.'), 'Box this lap.')
        self.assertEqual(voice_from_reply('{"voice": "Box this lap.", "advice": "cut o'), 'Box this lap.')
        self.assertIsNone(voice_from_reply('<think>still thinking'))
        self.assertIsNone(voice_from_reply('{"voice": ""}'))
        self.assertIsNone(voice_from_reply('{"voice": "cut o'))


if __name__ == '__main__':
    unittest.main()
