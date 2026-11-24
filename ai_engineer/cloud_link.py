"""
Cloud mode: telemetry arrives from the game PC over a WebSocket instead of local UDP.
====================================================================================
tools/cloud_relay.py runs next to F1 2021, reads its UDP packets and sends them in small
batches to ``wss://<cloud-host>/relay``. This module is the cloud end of that link:

* ``RelayEndpoint`` serves ``GET /relay`` on the HUD's own HTTP server (one port, which is
  what Railway and Fly.io give an app). It checks the shared secret (``RELAY_SECRET``),
  upgrades to a WebSocket and unpacks each batch.
* ``RelaySource`` stands in for the UDP socket in main.py's read loop: same ``recvfrom`` /
  ``settimeout`` / ``close`` as a socket (and as udp_recording.ReplaySource), so the whole
  telemetry -> analysis -> HUD pipeline runs unchanged.

Wire format (one binary WebSocket message = one batch):
    zlib( [u16 big-endian length][datagram bytes] [u16 length][datagram] ... )

Scaling note: everything lives in this one process (the HUD state is web_hud's in-memory
dict). That is right for a single instance on Railway / Fly.io. Running several web workers
would need the state in a shared store such as Redis: web_hud.update_web_state() is the one
place that writes it and web_state_message() the one place that reads it, so those two are
where Redis would slot in, with the relay pinned to one worker that runs the analysis.
"""

from __future__ import annotations

import collections
import hmac
import os
import struct
import threading
import time
import zlib
from typing import Any

from log_setup import get_logger
from web_hud import update_web_state, ws_accept_key, ws_frame, ws_read_frame

log = get_logger('cloud')

ENV_FLAG = 'CLOUD_MODE'
RELAY_PATH = '/relay'
MAX_FRAME = 1 << 20  # one compressed batch (the relay sends ~50 ms of packets, a few KB)
MAX_BATCH_BYTES = 4 << 20  # decompressed: refuse a zip bomb
QUEUE_LIMIT = 8192  # datagrams waiting for the engine (~10 s at full rate); oldest dropped
IDLE_TIMEOUT_S = 60.0  # the relay pings every 15 s; silence this long = dead link


def enabled(environ=None) -> bool:
    return (environ if environ is not None else os.environ).get(ENV_FLAG, '').strip() == '1'


def encode_batch(datagrams) -> bytes:
    """Pack datagrams into one relay message (tools/cloud_relay.py has the same code)."""
    return zlib.compress(b''.join(struct.pack('!H', len(d)) + d for d in datagrams), 1)


def decode_batch(payload: bytes) -> list[bytes]:
    inflater = zlib.decompressobj()
    raw = inflater.decompress(payload, MAX_BATCH_BYTES)
    if inflater.unconsumed_tail or not inflater.eof:
        raise ValueError('batch too large or truncated')
    datagrams = []
    offset = 0
    while offset < len(raw):
        if offset + 2 > len(raw):
            raise ValueError('truncated length')
        (length,) = struct.unpack_from('!H', raw, offset)
        offset += 2
        if offset + length > len(raw):
            raise ValueError('truncated datagram')
        datagrams.append(raw[offset : offset + length])
        offset += length
    return datagrams


class RelaySource:
    """A queue of relayed datagrams that reads like a UDP socket."""

    def __init__(self, limit: int = QUEUE_LIMIT):
        self._queue: collections.deque[bytes] = collections.deque(maxlen=limit)
        self._ready = threading.Condition()
        self._timeout: float | None = 1.0
        self.dropped = 0
        self.received = 0
        self.peer = ('relay', 0)

    def settimeout(self, timeout: float | None) -> None:
        self._timeout = timeout

    def feed(self, datagrams: list[bytes]) -> None:
        with self._ready:
            limit = self._queue.maxlen or 0
            self.dropped += max(0, len(self._queue) + len(datagrams) - limit)
            self._queue.extend(datagrams)
            self.received += len(datagrams)
            self._ready.notify()

    def recvfrom(self, bufsize: int) -> tuple[bytes, tuple[str, int]]:
        with self._ready:
            if not self._ready.wait_for(lambda: bool(self._queue), self._timeout):
                raise TimeoutError('no relayed telemetry')
            return self._queue.popleft()[:bufsize], self.peer

    def close(self) -> None:
        with self._ready:
            self._queue.clear()


class RelayEndpoint:
    """Serves GET /relay: auth, WebSocket upgrade, then batches into a RelaySource.

    One relay feeds the engine at a time. A new connection (the relay reconnecting after a
    network blip, before the server noticed the old one died) takes over straight away.
    """

    def __init__(self, source: RelaySource, secret: str):
        if not secret:
            raise ValueError('RELAY_SECRET must be set')
        self.source = source
        self._secret = secret.encode('utf-8')
        self._lock = threading.Lock()
        self._generation = 0
        self.connected = False

    def authorized(self, headers) -> bool:
        auth = headers.get('Authorization') or ''
        scheme, _, token = auth.partition(' ')
        return scheme.lower() == 'bearer' and hmac.compare_digest(token.strip().encode('utf-8'), self._secret)

    def _status(self, connected: bool, address: Any = None) -> None:
        self.connected = connected
        update_web_state({'relay': {'connected': connected, 'since': time.time()}})
        if connected:
            log.info(f'Relay connected from {address}')
        else:
            log.info('Relay disconnected')

    def __call__(self, request) -> None:
        if not self.authorized(request.headers):
            log.warning(f'Relay rejected (bad secret) from {request.client_address[0]}')
            request.send_response(401)
            request.end_headers()
            return
        key = request.headers.get('Sec-WebSocket-Key')
        if not key or 'websocket' not in request.headers.get('Upgrade', '').lower():
            request.send_response(400)
            request.end_headers()
            return
        request.send_response(101, 'Switching Protocols')
        request.send_header('Upgrade', 'websocket')
        request.send_header('Connection', 'Upgrade')
        request.send_header('Sec-WebSocket-Accept', ws_accept_key(key.strip()))
        request.end_headers()
        request.wfile.flush()
        request.close_connection = True
        try:
            request.connection.settimeout(IDLE_TIMEOUT_S)
        except OSError:
            pass
        with self._lock:
            self._generation += 1
            mine = self._generation
        self._status(True, request.client_address[0])
        try:
            while True:
                frame = ws_read_frame(request.rfile, MAX_FRAME)
                if frame is None or frame[0] == 0x8:
                    break
                opcode, payload = frame
                if opcode == 0x9:
                    request.wfile.write(ws_frame(payload, 0xA))
                    request.wfile.flush()
                elif opcode == 0x2:
                    if mine != self._generation:
                        break  # a newer relay connection has taken over
                    try:
                        self.source.feed(decode_batch(payload))
                    except (ValueError, zlib.error) as error:
                        log.debug(f'Bad relay batch: {error}')
        except (OSError, ValueError, struct.error):
            pass
        finally:
            with self._lock:
                current = mine == self._generation
            if current:
                self._status(False)
            try:
                request.wfile.write(ws_frame(b'', 0x8))
                request.wfile.flush()
            except OSError:
                pass


def setup(environ=None) -> RelaySource:
    """Wire the HUD server for cloud mode; returns the socket stand-in for main.py.

    RELAY_SECRET (required): shared with tools/cloud_relay.py.
    CONTROL_SECRET (optional): unlocks the HUD controls in your browser via
    https://<host>/control?token=<CONTROL_SECRET>. Unset = the public HUD is view-only.
    """
    import web_hud

    env = environ if environ is not None else os.environ
    secret = env.get('RELAY_SECRET', '').strip()
    if len(secret) < 16:
        raise ValueError('RELAY_SECRET must be set to a random string of at least 16 characters')
    source = RelaySource()
    web_hud.set_relay_handler(RelayEndpoint(source, secret))
    web_hud.set_control_secret(env.get('CONTROL_SECRET', '').strip())
    update_web_state({'relay': {'connected': False, 'since': time.time()}})
    return source


def hud_port(default: int, environ=None) -> int:
    """The cloud host tells the app which port to serve on with $PORT."""
    env = environ if environ is not None else os.environ
    try:
        return int(env.get('PORT', default))
    except ValueError:
        return default
