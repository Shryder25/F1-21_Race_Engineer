"""
Telemetry relay: F1 2021 on this PC -> the engineer running in the cloud.
=========================================================================
Reads the game's UDP telemetry and streams it over one outbound WebSocket to
cloud_server.py, so no port forwarding is needed. Reconnects on its own if the
connection drops. Standard library only: no pip install.

    python tools/cloud_relay.py --url https://your-app.up.railway.app --secret <RELAY_SECRET>

or set RELAY_URL and RELAY_SECRET in the environment and run it with no arguments.

Keep the local engineer too (voice in your headset, HUD on this PC) with --also-forward:
    python tools/cloud_relay.py --also-forward 127.0.0.1:20778
    python main.py --port 20778

Game: Settings -> Telemetry -> UDP Telemetry ON, IP 127.0.0.1, port 20777, format 2021.
A UDP send rate of 20 or 30 Hz uses about half the upload of 60 Hz.
"""

import argparse
import base64
import hashlib
import os
import select
import socket
import ssl
import struct
import sys
import time
import urllib.parse
import zlib

WS_GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
BATCH_S = 0.05  # send what arrived in the last 50 ms as one message
BATCH_BYTES = 32 * 1024
PING_S = 15.0  # keeps proxies (Railway, Fly.io) from closing an idle link


class AuthError(Exception):
    pass


def encode_batch(datagrams) -> bytes:
    """Same format as ai_engineer/cloud_link.py: zlib([u16 length][datagram]...)."""
    return zlib.compress(b''.join(struct.pack('!H', len(d)) + d for d in datagrams), 1)


def relay_url(url: str) -> tuple[bool, str, int, str]:
    """(tls, host, port, path) for the server's /relay endpoint from its base URL."""
    if '://' not in url:
        url = 'https://' + url
    parts = urllib.parse.urlsplit(url)
    tls = parts.scheme in ('https', 'wss')
    if parts.scheme not in ('http', 'https', 'ws', 'wss') or not parts.hostname:
        raise ValueError(f'not a server URL: {url}')
    path = parts.path.rstrip('/') or ''
    if not path.endswith('/relay'):
        path += '/relay'
    return tls, parts.hostname, parts.port or (443 if tls else 80), path


def client_frame(payload: bytes, opcode: int = 0x2) -> bytes:
    """One masked client frame (RFC 6455 requires clients to mask)."""
    n = len(payload)
    if n < 126:
        head = struct.pack('!BB', 0x80 | opcode, 0x80 | n)
    elif n < 1 << 16:
        head = struct.pack('!BBH', 0x80 | opcode, 0x80 | 126, n)
    else:
        head = struct.pack('!BBQ', 0x80 | opcode, 0x80 | 127, n)
    mask = os.urandom(4)
    if n:
        key = (mask * (n // 4 + 1))[:n]
        payload = (int.from_bytes(payload, 'big') ^ int.from_bytes(key, 'big')).to_bytes(n, 'big')
    return head + mask + payload


def connect(url: str, secret: str, timeout: float = 10.0) -> socket.socket:
    tls, host, port, path = relay_url(url)
    sock = socket.create_connection((host, port), timeout=timeout)
    if tls:
        sock = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
    key = base64.b64encode(os.urandom(16)).decode('ascii')
    host_header = host if port in (80, 443) else f'{host}:{port}'
    sock.sendall(
        (
            f'GET {path} HTTP/1.1\r\nHost: {host_header}\r\nUpgrade: websocket\r\n'
            f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n'
            f'Authorization: Bearer {secret}\r\nUser-Agent: f1-race-engineer-relay\r\n\r\n'
        ).encode('ascii')
    )
    response = b''
    while b'\r\n\r\n' not in response:
        chunk = sock.recv(4096)
        if not chunk:
            break
        response += chunk
        if len(response) > 16384:
            break
    status_line, _, headers = response.decode('latin-1').partition('\r\n')
    status = status_line.split(' ')[1] if ' ' in status_line else '?'
    if status == '401':
        sock.close()
        raise AuthError('the server rejected the secret (check RELAY_SECRET on both sides)')
    expected = base64.b64encode(hashlib.sha1((key + WS_GUID).encode('ascii')).digest()).decode('ascii')
    if status != '101' or expected not in headers:
        sock.close()
        raise ConnectionError(f'no WebSocket upgrade: {status_line or "empty reply"}')
    sock.settimeout(timeout)  # a send stuck this long = dead link, reconnect
    return sock


def drain_replies(sock: socket.socket) -> None:
    """Read (and ignore) the server's pongs without blocking; raise when it hung up.

    Done in the sending thread on purpose: one TLS connection must not be read and
    written from two threads at once.
    """
    while select.select([sock], [], [], 0)[0]:
        try:
            if not sock.recv(4096):
                raise ConnectionError('server closed the connection')
        except ssl.SSLWantReadError:
            return  # only part of a TLS record so far


def run(args) -> None:
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    udp.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    udp.bind(('0.0.0.0', args.port))
    udp.settimeout(BATCH_S)
    forward = None
    if args.also_forward:
        host, _, port = args.also_forward.rpartition(':')
        forward = (host or '127.0.0.1', int(port))
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if forward else None
    print(f'Relay: listening for F1 2021 on UDP {args.port}, sending to {args.url}')
    if forward:
        print(f'Relay: also forwarding every packet to {forward[0]}:{forward[1]}')

    ws = None
    backoff = 1.0
    next_try = 0.0
    batch: list[bytes] = []
    batch_bytes = 0
    batch_started = last_ping = time.monotonic()
    sent_packets = sent_bytes = 0
    last_report = time.monotonic()
    while True:
        now = time.monotonic()
        if ws is None and now >= next_try:
            try:
                ws = connect(args.url, args.secret)
                print('Relay: connected to the cloud engineer')
                backoff, last_ping = 1.0, now
            except AuthError as error:
                print(f'Relay: {error}. Retrying in 60 s.')
                next_try = now + 60
            except (OSError, ValueError) as error:
                print(f'Relay: cannot connect ({error}). Retrying in {backoff:.0f} s.')
                next_try = now + backoff
                backoff = min(backoff * 2, 30.0)

        try:
            data, _ = udp.recvfrom(65535)
            if out and forward:
                out.sendto(data, forward)
            if ws is not None:
                if not batch:
                    batch_started = time.monotonic()
                batch.append(data)
                batch_bytes += len(data)
        except TimeoutError:
            pass
        except ConnectionResetError:  # Windows reports ICMP errors from --also-forward here
            pass

        now = time.monotonic()
        if ws is None:
            batch, batch_bytes = [], 0  # offline: drop, the engine wants live data only
            continue
        try:
            drain_replies(ws)
            if batch and (now - batch_started >= BATCH_S or batch_bytes >= BATCH_BYTES):
                message = encode_batch(batch)
                ws.sendall(client_frame(message))
                sent_packets += len(batch)
                sent_bytes += len(message)
                batch, batch_bytes = [], 0
            if now - last_ping >= PING_S:
                ws.sendall(client_frame(b'', 0x9))
                last_ping = now
        except OSError as error:
            print(f'Relay: connection lost ({error}). Reconnecting...')
            try:
                ws.close()
            except OSError:
                pass
            ws, next_try = None, now + 1.0
            batch, batch_bytes = [], 0

        if now - last_report >= 10.0:
            span = now - last_report
            if sent_packets:
                print(
                    f'Relay: {sent_packets / span:.0f} packets/s, {sent_bytes / span / 1024:.0f} KB/s upload'
                )
            sent_packets = sent_bytes = 0
            last_report = now


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='Stream F1 2021 telemetry to the cloud engineer.')
    parser.add_argument('--url', default=os.environ.get('RELAY_URL'), help='cloud server URL ($RELAY_URL)')
    parser.add_argument(
        '--secret', default=os.environ.get('RELAY_SECRET'), help='shared secret ($RELAY_SECRET)'
    )
    parser.add_argument('--port', type=int, default=20777, help='UDP port the game sends to (default 20777)')
    parser.add_argument(
        '--also-forward',
        metavar='HOST:PORT',
        help='also send every packet here, e.g. 127.0.0.1:20778 for a local `main.py --port 20778`',
    )
    args = parser.parse_args(argv)
    if not args.url or not args.secret:
        parser.error('give --url and --secret (or set RELAY_URL and RELAY_SECRET)')
    try:
        relay_url(args.url)
    except ValueError as error:
        parser.error(str(error))
    try:
        run(args)
    except KeyboardInterrupt:
        print('Relay stopped.')
    except OSError as error:
        print(f'Relay: cannot listen on UDP {args.port}: {error}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
