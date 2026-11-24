"""Cloud mode: the PC relay (tools/cloud_relay.py) -> /relay -> RelaySource, auth, view-only HUD."""

import io
import json
import os
import pathlib
import socket
import sys
import time
import unittest
import urllib.error
import urllib.request
import zlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT / 'tools', ROOT / 'ai_engineer', ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cloud_server  # noqa: E402

import cloud_link  # noqa: E402
import cloud_relay  # noqa: E402
import web_hud  # noqa: E402

SECRET = 'relay-secret-0123456789'


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class BatchFormatTests(unittest.TestCase):
    def test_relay_and_server_agree_on_the_format(self):
        datagrams = [b'\x00' * 1464, os.urandom(1347), b'', b'x']
        self.assertEqual(cloud_link.decode_batch(cloud_relay.encode_batch(datagrams)), datagrams)

    def test_rejects_truncated_and_oversized_batches(self):
        with self.assertRaises(ValueError):
            cloud_link.decode_batch(zlib.compress(b'\x00\x10abc'))
        with self.assertRaises(ValueError):
            cloud_link.decode_batch(zlib.compress(b'\x00' * (cloud_link.MAX_BATCH_BYTES + 10)))
        with self.assertRaises(zlib.error):
            cloud_link.decode_batch(b'not zlib')

    def test_masked_frame_round_trip(self):
        payload = os.urandom(70000)
        frame = cloud_relay.client_frame(payload)
        self.assertEqual(web_hud.ws_read_frame(io.BytesIO(frame), 1 << 20), (0x2, payload))
        self.assertEqual(web_hud.ws_read_frame(io.BytesIO(cloud_relay.client_frame(b'', 0x9))), (0x9, b''))

    def test_relay_url(self):
        self.assertEqual(cloud_relay.relay_url('my-app.fly.dev'), (True, 'my-app.fly.dev', 443, '/relay'))
        self.assertEqual(
            cloud_relay.relay_url('http://localhost:5000/'), (False, 'localhost', 5000, '/relay')
        )
        self.assertEqual(cloud_relay.relay_url('wss://h/relay'), (True, 'h', 443, '/relay'))
        with self.assertRaises(ValueError):
            cloud_relay.relay_url('ftp://h')


class RelaySourceTests(unittest.TestCase):
    def test_reads_like_a_socket(self):
        source = cloud_link.RelaySource()
        source.settimeout(0.01)
        with self.assertRaises(TimeoutError):
            source.recvfrom(65535)
        source.feed([b'a', b'b'])
        self.assertEqual(source.recvfrom(65535)[0], b'a')
        self.assertEqual(source.recvfrom(65535)[0], b'b')

    def test_drops_the_oldest_when_the_engine_falls_behind(self):
        source = cloud_link.RelaySource(limit=3)
        source.feed([b'1', b'2', b'3', b'4', b'5'])
        self.assertEqual(source.dropped, 2)
        self.assertEqual(source.recvfrom(10)[0], b'3')


class CloudServerTests(unittest.TestCase):
    """A real HUD server with the relay endpoint, driven by the real relay client code."""

    @classmethod
    def setUpClass(cls):
        cls.port = free_port()
        cls.source = cloud_link.setup({'RELAY_SECRET': SECRET, 'CONTROL_SECRET': 'owner-token'})
        cls.source.settimeout(5.0)
        web_hud.set_command_handler(lambda text: {'heard': text})
        cls.server = web_hud.start_web_hud_server(cls.port)
        cls.base = f'http://127.0.0.1:{cls.port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        web_hud.set_relay_handler(None)
        web_hud.set_control_secret(None)
        web_hud.set_command_handler(None)

    def post(self, path, headers=None):
        request = urllib.request.Request(
            self.base + path, data=b'{"text": "gap ahead"}', headers=headers or {}
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def test_relayed_packets_reach_the_engine(self):
        ws = cloud_relay.connect(self.base, SECRET)
        try:
            ws.sendall(cloud_relay.client_frame(cloud_relay.encode_batch([b'one', b'two'])))
            ws.sendall(cloud_relay.client_frame(b'', 0x9))  # ping is answered, not fed in
            self.assertEqual(self.source.recvfrom(65535)[0], b'one')
            self.assertEqual(self.source.recvfrom(65535)[0], b'two')
            self.assertTrue(web_hud.get_web_state()['relay']['connected'])
        finally:
            ws.close()
        deadline = time.monotonic() + 5
        while web_hud.get_web_state()['relay']['connected'] and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(web_hud.get_web_state()['relay']['connected'])

    def test_wrong_secret_is_refused(self):
        with self.assertRaises(cloud_relay.AuthError):
            cloud_relay.connect(self.base, 'wrong-secret-wrong-secret')

    def test_public_hud_is_view_only_without_the_owner_token(self):
        self.assertEqual(self.post('/api/command'), 403)
        self.assertEqual(self.post('/api/reset_map'), 403)
        self.assertEqual(self.post('/api/command', {'Cookie': 'f1_control=guess'}), 403)
        self.assertEqual(self.post('/api/command', {'Cookie': 'a=b; f1_control=owner-token'}), 200)
        self.assertEqual(self.post('/api/command', {'X-Control-Token': 'owner-token'}), 200)
        with urllib.request.urlopen(self.base + '/api/telemetry', timeout=5) as response:
            self.assertIn('track_name', json.loads(response.read()))

    def test_control_link_sets_the_cookie(self):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        opener = urllib.request.build_opener(NoRedirect)
        with self.assertRaises(urllib.error.HTTPError) as bad:
            opener.open(self.base + '/control?token=nope', timeout=5)
        self.assertEqual(bad.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as good:
            opener.open(self.base + '/control?token=owner-token', timeout=5)
        self.assertEqual(good.exception.code, 303)
        self.assertIn('f1_control=owner-token', good.exception.headers['Set-Cookie'])
        self.assertIn('HttpOnly', good.exception.headers['Set-Cookie'])

    def test_health_check(self):
        with urllib.request.urlopen(self.base + '/healthz', timeout=5) as response:
            self.assertEqual(response.read(), b'ok')


class CloudSettingsTests(unittest.TestCase):
    def test_setup_needs_a_real_secret(self):
        for env in ({}, {'RELAY_SECRET': 'short'}):
            with self.assertRaises(ValueError):
                cloud_link.setup(env)
        web_hud.set_relay_handler(None)
        web_hud.set_control_secret(None)

    def test_local_mode_is_untouched(self):
        self.assertFalse(cloud_link.enabled({}))
        self.assertTrue(cloud_link.enabled({'CLOUD_MODE': '1'}))
        self.assertIsNone(web_hud._control_secret)  # anyone on the LAN may still control
        self.assertTrue(web_hud.control_allowed({}))

    def test_headless_flags(self):
        argv = cloud_server.cloud_argv(['--chatty'], {'DATA_DIR': '/d'})
        self.assertIn('--no-voice', argv)
        self.assertIn('--no-ai', argv)  # no API configured: rules only
        self.assertEqual(argv[argv.index('--history-dir') + 1], str(pathlib.Path('/d/telemetry-history')))
        self.assertEqual(argv[-1], '--chatty')
        self.assertNotIn(
            '--no-ai', cloud_server.cloud_argv([], {'F1_AI_BASE_URL': 'https://api.deepseek.com'})
        )
        self.assertEqual(cloud_link.hud_port(5000, {'PORT': '8080'}), 8080)
        self.assertEqual(cloud_link.hud_port(5000, {}), 5000)


if __name__ == '__main__':
    unittest.main()
