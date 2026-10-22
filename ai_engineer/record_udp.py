"""
F1 2021 UDP Recorder
====================
Captures every datagram the game sends, with its arrival time, to a recording file
(format: docs/udp-recording-format.md). Nothing is parsed or filtered, so the file is a
faithful copy of the session for replay and backtesting.

Usage:
    python ai_engineer/record_udp.py                      # -> recordings/<date-time>.f1rec.gz
    python ai_engineer/record_udp.py --out race.f1rec --port 20777 --duration 600

To record while the engineer is running instead, use `python main.py --record race.f1rec.gz`
(the game sends to one port, so only one program can listen at a time).
"""

import argparse
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from udp_recording import RecordingWriter, record_socket

DEFAULT_DIR = Path(__file__).resolve().parent.parent / 'recordings'


def default_path() -> Path:
    return DEFAULT_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.f1rec.gz"


def main(argv=None):
    parser = argparse.ArgumentParser(description='Record raw F1 2021 UDP telemetry to a file')
    parser.add_argument('--port', type=int, default=20777, help='UDP port to listen on (default: 20777)')
    parser.add_argument('--host', default='0.0.0.0', help='Address to bind (default: 0.0.0.0)')
    parser.add_argument(
        '--out',
        type=Path,
        default=None,
        help='Output file (default: recordings/<date-time>.f1rec.gz; a name ending .gz is gzip-compressed)',
    )
    parser.add_argument(
        '--duration', type=float, default=None, help='Stop after this many seconds (default: until Ctrl+C)'
    )
    parser.add_argument('--note', default=None, help='Free text stored in the file metadata')
    parser.add_argument('--force', action='store_true', help='Overwrite the output file if it exists')
    args = parser.parse_args(argv)

    out = args.out or default_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((args.host, args.port))
    except OSError as e:
        print(f'[ERROR] Cannot bind to port {args.port}: {e}')
        print('  Is the engineer (or another recorder) already listening? Use main.py --record instead.')
        sock.close()
        return 1

    metadata = {'port': args.port, 'started_at': datetime.now().astimezone().isoformat()}
    if args.note:
        metadata['note'] = args.note
    try:
        writer = RecordingWriter(out, metadata=metadata, overwrite=args.force)
    except FileExistsError:
        print(f'[ERROR] {out} already exists (use --force to overwrite)')
        sock.close()
        return 1

    def progress(w):
        sys.stdout.write(f'\r  [REC] {w.count:>8} datagrams  {w.bytes / 1e6:8.1f} MB')
        sys.stdout.flush()

    print(f'[REC] Listening on {args.host}:{args.port}, writing {out}  (Ctrl+C to stop)')
    started = time.monotonic()
    with writer:
        try:
            record_socket(sock, writer, duration=args.duration, on_progress=progress)
        finally:
            sock.close()
    print(
        f'\n[REC] Saved {writer.count} datagrams ({writer.bytes / 1e6:.1f} MB) '
        f'over {time.monotonic() - started:.0f} s to {out}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
