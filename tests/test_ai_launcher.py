import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from test_analysis import SilentVoice  # noqa: F401  (puts ai_engineer/ on the import path)
import ai_launcher
from ai_launcher import LocalAILauncher, model_installed, ollama_root


class FakeOllama(BaseHTTPRequestHandler):
    models = []
    calls = []

    def log_message(self, *args):
        pass

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        FakeOllama.calls.append(('GET', self.path, None))
        self._json({'models': [{'name': n} for n in FakeOllama.models]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeOllama.calls.append(('POST', self.path, body))
        if self.path == '/api/pull':
            lines = [{'status': 'pulling', 'total': 100, 'completed': c} for c in (10, 50, 100)]
            data = b''.join(json.dumps(line).encode() + b'\n' for line in lines)
            self.send_response(200)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            FakeOllama.models.append(body['model'])
        else:
            self._json({'done': True})


def serve():
    server = ThreadingHTTPServer(('127.0.0.1', 0), FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class HelperTests(unittest.TestCase):
    def test_only_local_ollama_is_managed(self):
        self.assertEqual(ollama_root('http://localhost:11434/v1'), 'http://localhost:11434')
        self.assertEqual(ollama_root('http://127.0.0.1:11434/v1/'), 'http://127.0.0.1:11434')
        self.assertIsNone(ollama_root('https://api.groq.com/openai/v1'))
        self.assertIsNone(ollama_root('http://localhost:8080/v1'))
        self.assertFalse(LocalAILauncher('https://api.deepseek.com/v1', 'deepseek-chat').start())

    def test_model_names(self):
        self.assertTrue(model_installed(['deepseek-r1:8b'], 'deepseek-r1:8b'))
        self.assertTrue(model_installed(['deepseek-r1:latest'], 'deepseek-r1'))
        self.assertFalse(model_installed(['deepseek-r1:1.5b'], 'deepseek-r1:8b'))


class LauncherFlowTests(unittest.TestCase):
    def setUp(self):
        FakeOllama.models, FakeOllama.calls = [], []
        self.statuses = []

    def launcher(self, server, **kwargs):
        launcher = LocalAILauncher('http://localhost:11434/v1', 'deepseek-r1:8b',
                                   on_status=self.statuses.append, **kwargs)
        launcher.root = f'http://127.0.0.1:{server.server_port}' if server else 'http://127.0.0.1:9'
        return launcher

    def test_downloads_missing_model_then_preloads_it(self):
        server = serve()
        ready = threading.Event()
        launcher = self.launcher(server, on_ready=ready.set)
        try:
            launcher.start()
            self.assertTrue(ready.wait(5))
            paths = [(m, p) for m, p, _ in FakeOllama.calls]
            self.assertIn(('POST', '/api/pull'), paths)
            load = [b for m, p, b in FakeOllama.calls if p == '/api/generate'][0]
            self.assertEqual((load['model'], load['prompt']), ('deepseek-r1:8b', ''))
            self.assertTrue(any('100%' in s for s in self.statuses))
        finally:
            launcher.shutdown()
            server.shutdown()
            server.server_close()

    def test_installed_model_is_not_downloaded_again(self):
        FakeOllama.models = ['deepseek-r1:8b']
        server = serve()
        launcher = self.launcher(server)
        try:
            launcher.start()
            self.assertTrue(launcher.ready.wait(5))
            self.assertNotIn('/api/pull', [p for _, p, _ in FakeOllama.calls])
        finally:
            launcher.shutdown()
            server.shutdown()
            server.server_close()

    def test_starts_ollama_when_it_is_not_running_and_stops_it_on_exit(self):
        FakeOllama.models = ['deepseek-r1:8b']
        holder = {}

        class FakeProcess:
            terminated = False

            def terminate(self):
                FakeProcess.terminated = True

        def fake_popen(cmd, **kwargs):
            holder['cmd'] = cmd
            holder['server'] = serve()
            launcher.root = f"http://127.0.0.1:{holder['server'].server_port}"
            return FakeProcess()

        launcher = self.launcher(None, executable_finder=lambda: '/opt/ollama', popen=fake_popen)
        try:
            launcher.start()
            self.assertTrue(launcher.ready.wait(5))
            self.assertEqual(holder['cmd'], ['/opt/ollama', 'serve'])
        finally:
            launcher.shutdown()
            if holder.get('server'):
                holder['server'].shutdown()
                holder['server'].server_close()
        self.assertTrue(FakeProcess.terminated)

    def test_missing_ollama_is_reported_and_engineer_keeps_running_on_rules(self):
        launcher = self.launcher(None, executable_finder=lambda: None)
        launcher.start()
        launcher._thread.join(5)
        self.assertFalse(launcher.ready.is_set())
        self.assertIn('not installed', self.statuses[-1])

    def test_keeps_model_loaded_between_questions(self):
        FakeOllama.models = ['deepseek-r1:8b']
        server = serve()
        launcher = self.launcher(server, ping_every_s=0.05)
        try:
            launcher.start()
            self.assertTrue(launcher.ready.wait(5))
            deadline = threading.Event()
            deadline.wait(0.3)
            loads = [p for _, p, _ in FakeOllama.calls if p == '/api/generate']
            self.assertGreaterEqual(len(loads), 3)
        finally:
            launcher.shutdown()
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
