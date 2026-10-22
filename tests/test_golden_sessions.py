"""Golden-session regression tests.

Every session in tests/golden/sessions/manifest.json is replayed through the real engineer
(offline, simulated clock) and its radio calls, voice-command answers and key numbers
are compared with the checked-in expected.json. A failure means the engineer now behaves
differently on the same input. If the change is intended, review the printed diff, then
run `python tools/golden_sessions.py bless NAME` and commit the updated expected.json.

Slots marked "needs-recording" are skipped until a real F1 2021 session is recorded into
them (see tests/golden/README.md).
"""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / 'golden'))

import harness  # noqa: E402
import recording  # noqa: E402

MANIFEST = harness.load_manifest()


class GoldenSessionTests(unittest.TestCase):
    """One test per stored session (methods are added below)."""

    maxDiff = None


def _make_test(name, info):
    def test(self):
        rec, exp = harness.session_paths(name)
        if info['source'] == 'needs-recording' and not rec.exists():
            self.skipTest(f'{name}: waiting for a real F1 2021 recording (tests/golden/README.md)')
        self.assertTrue(rec.exists(), f'{rec} is missing')
        self.assertTrue(exp.exists(), f'{exp} is missing; run tools/golden_sessions.py bless {name}')
        problems = harness.compare(harness.load_expected(name), harness.replay(rec),
                                   MANIFEST.get('tolerances'))
        if problems:
            self.fail(f'Golden session "{name}" changed behaviour ({info["description"]}):\n'
                      + '\n'.join(problems)
                      + f'\nIf this is intended: python tools/golden_sessions.py bless {name}')
    test.__name__ = f'test_{name}'
    test.__doc__ = f'{info["kind"]}: {info["description"]}'
    return test


for _name, _info in MANIFEST['sessions'].items():
    setattr(GoldenSessionTests, f'test_{_name}', _make_test(_name, _info))


class StoreTests(unittest.TestCase):
    def test_every_kind_has_a_session(self):
        kinds = {info['kind'] for info in MANIFEST['sessions'].values()
                 if info['source'] != 'needs-recording'}
        self.assertTrue({'race', 'qualifying', 'wet', 'safety_car'} <= kinds, kinds)

    def test_recordings_carry_metadata(self):
        for name in MANIFEST['sessions']:
            rec, _ = harness.session_paths(name)
            if rec.exists():
                self.assertIsInstance(recording.read_metadata(rec), dict, name)


class HarnessTests(unittest.TestCase):
    def test_reader_round_trip_and_crc(self):
        records = [(0, b'abc'), (1_500_000, b'defg')]
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'r.f1rec'
            path.write_bytes(recording.dumps_for_test(records, {'k': 1}))
            self.assertEqual(recording.read_metadata(path), {'k': 1})
            got = list(recording.read_records(path))
            self.assertEqual([(r.t_ns, r.data) for r in got], records)
            self.assertAlmostEqual(got[1].t, 0.0015)
            # A truncated tail is ignored, a corrupted payload is an error.
            path.write_bytes(recording.dumps_for_test(records)[:-2])
            self.assertEqual(len(list(recording.read_records(path))), 1)
            corrupt = bytearray(recording.dumps_for_test(records))
            corrupt[-1] ^= 0xFF
            path.write_bytes(bytes(corrupt))
            with self.assertRaises(recording.RecordingError):
                list(recording.read_records(path))

    def test_compare_reports_changed_call_and_number(self):
        name = next(n for n, i in MANIFEST['sessions'].items() if i['source'] == 'synthetic')
        expected = harness.load_expected(name)
        actual = json.loads(json.dumps(expected))
        self.assertEqual(harness.compare(expected, actual), [])
        actual['calls'][1]['text'] = 'Something else entirely.'
        actual['final']['position'] = (actual['final'].get('position') or 0) + 1
        problems = '\n'.join(harness.compare(expected, actual))
        self.assertIn('Radio calls changed', problems)
        self.assertIn('Something else entirely.', problems)
        self.assertIn('Final position', problems)

    def test_small_float_noise_is_tolerated(self):
        expected = {'final': {'gap_ahead_s': 1.234}}
        self.assertEqual(harness.compare(expected, {'final': {'gap_ahead_s': 1.24}}), [])
        self.assertTrue(harness.compare(expected, {'final': {'gap_ahead_s': 1.3}}))


if __name__ == '__main__':
    unittest.main()
