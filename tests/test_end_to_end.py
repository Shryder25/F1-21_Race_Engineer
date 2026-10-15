"""Exercise the actual UDP listener and HTTP endpoint in a child process."""
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from simulate_telemetry import build_session_packet, build_lap_packet, build_telemetry_packet


class ListenerTests(unittest.TestCase):
    def test_udp_to_http_and_disconnect(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as reservation:
            reservation.bind(('127.0.0.1', 0))
            udp_port = reservation.getsockname()[1]
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            http_port = reservation.getsockname()[1]
        with tempfile.TemporaryDirectory() as history_dir:
            script = f'''
import sys
sys.path.insert(0, {str(root / 'ai_engineer')!r})
import main as app
app.webbrowser.open = lambda *args, **kwargs: False
app.print_live_dashboard = lambda *args, **kwargs: None
original_server = app.start_web_hud_server
app.start_web_hud_server = lambda port: original_server({http_port})
sys.argv = ['main.py', '--no-voice', '--port', {str(udp_port)!r}, '--history-dir', {history_dir!r}]
app.main()
'''
            process = subprocess.Popen([sys.executable, '-B', '-u', '-c', script],
                                       cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            try:
                def read():
                    with urlopen(f'http://127.0.0.1:{http_port}/api/telemetry', timeout=1) as response:
                        return json.load(response)
                deadline = time.monotonic() + 10
                while True:
                    if process.poll() is not None:
                        self.fail(process.stderr.read().decode())
                    try:
                        read()
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise
                        time.sleep(0.05)
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
                    target = ('127.0.0.1', udp_port)
                    sender.sendto(build_session_packet(0, 1, 11), target)
                    sender.sendto(b'invalid datagram', target)
                    for frame in range(2, 6):
                        sender.sendto(build_telemetry_packet(frame / 25, frame, 280, 1, 0, 8, 95), target)
                        sender.sendto(build_lap_packet(frame / 25, frame, 1000 + frame * 10, 20000 + frame * 100), target)
                        time.sleep(0.05)
                state = read()
                self.assertTrue(state['connected'])
                self.assertEqual((state['track_name'], state['speed']), ('Monza', 280))
                time.sleep(2.2)
                self.assertFalse(read()['connected'])
            finally:
                process.terminate()
                process.wait(timeout=5)
                process.stderr.close()
