"""
F1 2021 UDP Replay
==================
Plays a recording made by record_udp.py (or main.py --record) back to the engineer as UDP
datagrams, exactly as the game would send them.

Usage:
    python ai_engineer/replay_udp.py race.f1rec.gz                 # real time (1x)
    python ai_engineer/replay_udp.py race.f1rec.gz --speed 10      # 10x
    python ai_engineer/replay_udp.py race.f1rec.gz --speed max     # as fast as possible
    python ai_engineer/replay_udp.py race.f1rec.gz --info          # describe the file only

At --speed max a receiver can drop datagrams if it falls behind. For a lossless fast replay
run the engineer straight from the file instead: python main.py --replay race.f1rec.gz --replay-speed max
"""

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from udp_recording import RecordingError, parse_speed, replay_to_udp, summarize


def speed_arg(text):
    try:
        return parse_speed(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def print_info(path):
    info = summarize(path)
    started = datetime.fromtimestamp(info['start_unix_ns'] / 1e9, tz=UTC).astimezone()
    print(
        f"File:        {info['path']}  (format v{info['format_version']}, "
        f"{'gzip' if info['compressed'] else 'uncompressed'})"
    )
    print(f"Recorded:    {started:%Y-%m-%d %H:%M:%S %Z}")
    print(f"Datagrams:   {info['datagrams']}  ({info['payload_bytes'] / 1e6:.1f} MB)")
    print(f"Duration:    {info['duration_s']:.1f} s")
    print(
        'Packet ids:  '
        + ', '.join(
            f'{k}: {v}'
            for k, v in sorted(info['by_packet_id'].items(), key=lambda kv: (kv[0] is None, kv[0] or 0))
        )
    )
    print(f"Metadata:    {info['metadata']}")
    if info['truncated']:
        print(
            'Note:        the file ends part-way through a record (recorder was stopped abruptly); '
            'everything before that is intact.'
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description='Replay a recorded F1 2021 UDP session')
    parser.add_argument('recording', type=Path, help='Recording file (.f1rec or .f1rec.gz)')
    parser.add_argument('--host', default='127.0.0.1', help='Where to send (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=20777, help='UDP port to send to (default: 20777)')
    parser.add_argument(
        '--speed',
        type=speed_arg,
        default=1.0,
        help='1 = real time (default), 10 = ten times faster, max = as fast as possible',
    )
    parser.add_argument('--info', action='store_true', help='Describe the recording and exit')
    args = parser.parse_args(argv)

    try:
        if args.info:
            print_info(args.recording)
            return 0
        label = 'max speed' if args.speed is None else f'{args.speed:g}x'
        print(f'[REPLAY] {args.recording} -> {args.host}:{args.port} at {label}  (Ctrl+C to stop)')
        started = time.monotonic()

        def progress(count, record):
            if count % 500 == 0:
                sys.stdout.write(f'\r  [REPLAY] {count:>8} datagrams  t={record.t:8.1f} s')
                sys.stdout.flush()

        count = replay_to_udp(args.recording, args.host, args.port, args.speed, on_progress=progress)
        print(f'\n[REPLAY] Sent {count} datagrams in {time.monotonic() - started:.1f} s')
        return 0
    except (RecordingError, OSError) as e:
        print(f'[ERROR] {e}')
        return 1
    except KeyboardInterrupt:
        print('\n[REPLAY] Stopped')
        return 0


if __name__ == '__main__':
    sys.exit(main())
