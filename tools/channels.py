"""Work with the per-sample channel store (Parquet) and export it to MoTeC i2.

Usage:
    python tools/channels.py build recordings/race.f1rec.gz [--out channels/]
        Rebuild channel files from a raw UDP recording (one .parquet per session).
    python tools/channels.py laps channels/SESSION.parquet
        List the laps in a file: lap time, samples, and whether each is a clean lap.
    python tools/channels.py compare channels/SESSION.parquet 3 5 [--channels speed throttle brake]
                                     [--step 5] [--csv out.csv]
        Overlay laps on a common lap-distance grid; prints a short table or writes a CSV.
    python tools/channels.py motec channels/SESSION.parquet [--out race.ld] [--rate 60]
                                   [--format ld|csv]
        Export to MoTeC i2: SESSION.ld + SESSION.ldx (laps), or a MoTeC-style CSV.

Live logging: `python main.py --log-channels channels/` (format: docs/channel-store.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'ai_engineer') not in sys.path:
    sys.path.insert(0, str(ROOT / 'ai_engineer'))

import channel_store  # noqa: E402
import motec  # noqa: E402


def cmd_build(args):
    out = args.out or Path('channels')
    for recording in args.recordings:
        files = channel_store.build_from_recording(recording, out)
        if not files:
            print(f'{recording}: no player telemetry found')
        for f in files:
            print(f'{recording} -> {f}')
    return 0


def cmd_laps(args):
    cols, meta = channel_store.read_session(args.file)
    print(
        f"{meta.get('track', '?')} {meta.get('session', '')}  driver {meta.get('driver', '?')}  "
        f"{len(cols['session_time'])} samples"
    )
    print(f"{'lap':>4} {'time':>10} {'samples':>8}  notes")
    for lap in channel_store.lap_summary(cols):
        time_text = motec.format_lap_time(lap['lap_time']) if lap['lap_time'] else '-'
        notes = [
            n
            for n, on in (
                ('clean', lap['clean']),
                ('invalid', lap['invalid']),
                ('pit', lap['pit']),
                ('partial', not lap['complete']),
            )
            if on
        ]
        print(f"{lap['lap']:>4} {time_text:>10} {lap['samples']:>8}  {', '.join(notes)}")
    return 0


def cmd_compare(args):
    cols, _ = channel_store.read_session(args.file)
    unknown = [c for c in args.channels if c not in cols]
    if unknown:
        print(f'Unknown channel(s): {", ".join(unknown)}. Known: {", ".join(sorted(cols))}')
        return 2
    grid, laps = channel_store.align_laps(cols, args.laps, args.channels, step_m=args.step)
    headers = ['lap_distance'] + [f'{c}_lap{lap}' for lap in args.laps for c in args.channels]
    table = np.column_stack([grid] + [laps[lap][c] for lap in args.laps for c in args.channels])
    if args.csv:
        np.savetxt(args.csv, table, delimiter=',', header=','.join(headers), comments='', fmt='%.6g')
        print(f'Wrote {len(grid)} points to {args.csv}')
        return 0
    print(' '.join(f'{h:>16}' for h in headers))
    every = max(1, len(grid) // 40)
    for row in table[::every]:
        print(' '.join(f'{v:>16.2f}' for v in row))
    return 0


def cmd_motec(args):
    result = motec.export_session(args.file, args.out, rate_hz=args.rate, fmt=args.format)
    print(
        f"Wrote {result['path']} ({result['channels']} channels, {result['samples']} samples "
        f"at {result['rate_hz']} Hz, {result['laps']} laps)"
    )
    if result.get('ldx'):
        print(f"Wrote {result['ldx']} (lap markers; keep it next to the .ld)")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build', help='Rebuild channel files from UDP recordings')
    p.add_argument('recordings', nargs='+', type=Path)
    p.add_argument('--out', type=Path, default=None, help='Output folder (default channels/)')
    p.set_defaults(func=cmd_build)
    p = sub.add_parser('laps', help='List the laps in a channel file')
    p.add_argument('file', type=Path)
    p.set_defaults(func=cmd_laps)
    p = sub.add_parser('compare', help='Overlay laps by lap distance')
    p.add_argument('file', type=Path)
    p.add_argument('laps', nargs='+', type=int)
    p.add_argument('--channels', nargs='+', default=['speed', 'throttle', 'brake', 'gear'])
    p.add_argument('--step', type=float, default=5.0, help='Grid spacing in metres (default 5)')
    p.add_argument('--csv', type=Path, default=None)
    p.set_defaults(func=cmd_compare)
    p = sub.add_parser('motec', help='Export to MoTeC i2 (.ld + .ldx) or MoTeC CSV')
    p.add_argument('file', type=Path)
    p.add_argument('--out', type=Path, default=None)
    p.add_argument('--rate', type=int, default=None, help='Sample rate in Hz (default: as sent)')
    p.add_argument('--format', choices=('ld', 'csv'), default='ld')
    p.set_defaults(func=cmd_motec)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f'[ERROR] {error}')
        return 1


if __name__ == '__main__':
    sys.exit(main())
