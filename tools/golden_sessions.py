"""Manage the golden-session regression store (tests/golden/sessions/).

    python tools/golden_sessions.py list
    python tools/golden_sessions.py check [NAME ...]          # same comparison the tests run
    python tools/golden_sessions.py bless [NAME ...]          # accept current behaviour as expected
    python tools/golden_sessions.py generate [NAME ...]       # rebuild synthetic recordings
    python tools/golden_sessions.py add real_race --recording my_race.f1rec.gz [--every 3]
    python tools/golden_sessions.py show NAME                 # print the radio transcript

"bless" rewrites expected.json from what the engineer does now. Only bless after you
have read the diff that `check` printed and agreed the new behaviour is right; the git
diff of expected.json is the review record.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import struct
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parents[1] / 'tests' / 'golden'
sys.path.insert(0, str(GOLDEN))

import harness      # noqa: E402
import recording    # noqa: E402

# Lap data, car telemetry and motion arrive every frame; these are thinned by --every.
PER_FRAME_PACKETS = (0, 2, 6)


def _names(manifest, requested, sources=None):
    sessions = manifest['sessions']
    names = requested or list(sessions)
    unknown = [n for n in names if n not in sessions]
    if unknown:
        sys.exit(f'Unknown session(s): {", ".join(unknown)}. Known: {", ".join(sessions)}')
    return [n for n in names if sources is None or sessions[n]['source'] in sources]


def cmd_list(args):
    manifest = harness.load_manifest()
    for name, info in manifest['sessions'].items():
        rec, exp = harness.session_paths(name)
        state = ('ready' if rec.exists() and exp.exists()
                 else 'NEEDS A RECORDING' if info['source'] == 'needs-recording' else 'missing files')
        size = f'{rec.stat().st_size // 1024} KB' if rec.exists() else ''
        print(f'{name:32} {info["kind"]:12} {info["source"]:16} {state:18} {size}')


def cmd_check(args):
    manifest = harness.load_manifest()
    failed = 0
    for name in _names(manifest, args.names):
        rec, exp = harness.session_paths(name)
        if not rec.exists():
            print(f'-- {name}: skipped (no recording yet)')
            continue
        problems = harness.compare(harness.load_expected(name), harness.replay(rec),
                                   manifest.get('tolerances'))
        if problems:
            failed += 1
            print(f'FAIL {name}')
            for problem in problems:
                print('  ' + problem.replace('\n', '\n  '))
        else:
            print(f'ok   {name}')
    sys.exit(1 if failed else 0)


def cmd_bless(args):
    manifest = harness.load_manifest()
    for name in _names(manifest, args.names):
        rec, _ = harness.session_paths(name)
        if not rec.exists():
            print(f'-- {name}: skipped (no recording yet)')
            continue
        transcript = harness.replay(rec)
        harness.save_expected(name, transcript)
        print(f'blessed {name}: {len(transcript["calls"])} calls, {len(transcript["laps"])} lap checkpoints')


def cmd_generate(args):
    import scenarios
    import synth
    manifest = harness.load_manifest()
    for name in _names(manifest, args.names, sources=('synthetic',)):
        scenario = scenarios.SCENARIOS[manifest['sessions'][name]['scenario']]
        rec, _ = harness.session_paths(name)
        rec.parent.mkdir(parents=True, exist_ok=True)
        datagrams = synth.simulate(scenario)
        metadata = {'source': 'synthetic', 'scenario': scenario.name, 'generator': 'tests/golden/synth.py',
                    'game': 'F1 2021 (simulated)', 'udp_format': 2021, 'description': scenario.description}
        with recording.ReproducibleWriter(rec, metadata) as writer:
            for t, data in datagrams:
                writer.write(data, int(round(t * 1e9)))
        print(f'generated {name}: {writer.count} datagrams, {rec.stat().st_size // 1024} KB')
        if args.bless:
            cmd_bless(argparse.Namespace(names=[name]))


def thin(source: Path, target: Path, every: int):
    """Copy a recording keeping one frame in `every` of the per-frame packets, gzipped."""
    metadata = recording.read_metadata(source)
    metadata['thinned_every_n_frames'] = every
    kept = 0
    with recording.ReproducibleWriter(target, metadata) as writer:
        for record in recording.read_records(source):
            data = record.data
            if every > 1 and len(data) >= 24 and data[5] in PER_FRAME_PACKETS:
                frame = struct.unpack_from('<I', data, 18)[0]
                if frame % every:
                    continue
            writer.write(data, record.t_ns)
            kept += 1
    return kept


def cmd_add(args):
    manifest = harness.load_manifest()
    info = manifest['sessions'].get(args.name)
    if info is None:
        sys.exit(f'Unknown slot {args.name}. Slots: {", ".join(manifest["sessions"])}')
    source = Path(args.recording)
    recording.read_metadata(source)                 # fails loudly if it isn't a recording
    rec, _ = harness.session_paths(args.name)
    rec.parent.mkdir(parents=True, exist_ok=True)
    if args.every > 1:
        kept = thin(source, rec, args.every)
        print(f'thinned to every {args.every} frames: {kept} datagrams')
    else:
        with open(source, 'rb') as probe:
            gzipped = probe.read(2) == b'\x1f\x8b'
        if gzipped:
            shutil.copyfile(source, rec)
        else:
            with open(source, 'rb') as src, gzip.open(rec, 'wb', compresslevel=9) as dst:
                shutil.copyfileobj(src, dst)
    size_mb = rec.stat().st_size / 1e6
    print(f'stored {rec} ({size_mb:.1f} MB)')
    if size_mb > 20:
        print('WARNING: over 20 MB. Re-run with --every 3 (or record at a 20 Hz UDP send rate).')
    info['source'] = 'real'
    if args.note:
        info['note'] = args.note
    harness.save_manifest(manifest)
    cmd_bless(argparse.Namespace(names=[args.name]))
    print('Now read tests/golden/sessions/%s/expected.json (or run `show`) and check every '
          'call is one a real engineer should make, before committing.' % args.name)


def cmd_show(args):
    rec, exp = harness.session_paths(args.name)
    transcript = harness.load_expected(args.name) if exp.exists() and not args.replay else harness.replay(rec)
    for call in transcript['calls']:
        if args.all or call['key'] != 'braking':
            flag = '*' if call['proactive'] else ' '
            print(f"{call['t']:8.2f} {flag} [{call['key']}] {call['text']}")
    print('\n(* = spoken even in the default on-demand mode; braking callouts hidden unless --all)')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list').set_defaults(func=cmd_list)
    for name, func in (('check', cmd_check), ('bless', cmd_bless)):
        p = sub.add_parser(name)
        p.add_argument('names', nargs='*')
        p.set_defaults(func=func)
    p = sub.add_parser('generate')
    p.add_argument('names', nargs='*')
    p.add_argument('--bless', action='store_true', help='also re-bless expected.json')
    p.set_defaults(func=cmd_generate)
    p = sub.add_parser('add', help='store a real recording in a slot and bless it')
    p.add_argument('name')
    p.add_argument('--recording', required=True)
    p.add_argument('--every', type=int, default=1,
                   help='keep one frame in N of lap/telemetry/motion packets (shrinks 60 Hz captures)')
    p.add_argument('--note', default='', help='what happened in the session, for reviewers')
    p.set_defaults(func=cmd_add)
    p = sub.add_parser('show')
    p.add_argument('name')
    p.add_argument('--all', action='store_true', help='include braking-point callouts')
    p.add_argument('--replay', action='store_true', help='replay now instead of reading expected.json')
    p.set_defaults(func=cmd_show)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()
