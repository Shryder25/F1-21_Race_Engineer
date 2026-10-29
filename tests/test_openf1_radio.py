"""Real-radio dataset builder (tools/openf1_radio.py): situation rebuild, speaker split, tags, build."""

import contextlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
import urllib.error
from datetime import UTC, datetime, timedelta

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / 'tools') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tools'))

import openf1_radio as of1  # noqa: E402

START = datetime(2024, 7, 7, 14, 0, tzinfo=UTC)
LAP_S = 90.0


def iso(seconds):
    return (START + timedelta(seconds=seconds)).isoformat()


def race_rows():
    """A 30-lap race: HAM (44) P2 chasing VER (1), NOR (4) P3 behind; NOR pits on lap 20."""
    laps, intervals, positions = [], [], []
    for lap in range(1, 31):
        t = (lap - 1) * LAP_S
        for number in (1, 44, 4):
            laps.append(
                {'driver_number': number, 'lap_number': lap, 'date_start': iso(t), 'lap_duration': LAP_S}
            )
        # HAM closes on VER by 0.3s a lap from 8.0s; NOR stays 1.4s behind HAM.
        intervals.append({'driver_number': 1, 'date': iso(t + 1), 'interval': None})
        intervals.append({'driver_number': 44, 'date': iso(t + 1), 'interval': round(8.0 - 0.3 * lap, 2)})
        intervals.append({'driver_number': 4, 'date': iso(t + 1), 'interval': 1.4})
    for number, position in ((1, 1), (44, 2), (4, 3)):
        positions.append({'driver_number': number, 'date': iso(0), 'position': position})
    return {
        'drivers': [
            {'driver_number': 1, 'name_acronym': 'VER'},
            {'driver_number': 44, 'name_acronym': 'HAM'},
            {'driver_number': 4, 'name_acronym': 'NOR'},
        ],
        'laps': laps,
        'stints': [
            {
                'driver_number': 44,
                'stint_number': 1,
                'lap_start': 1,
                'lap_end': None,
                'compound': 'MEDIUM',
                'tyre_age_at_start': 2,
            },
            {
                'driver_number': 4,
                'stint_number': 1,
                'lap_start': 1,
                'lap_end': 19,
                'compound': 'SOFT',
                'tyre_age_at_start': 0,
            },
            {
                'driver_number': 4,
                'stint_number': 2,
                'lap_start': 20,
                'lap_end': None,
                'compound': 'HARD',
                'tyre_age_at_start': 0,
            },
        ],
        'pit': [{'driver_number': 4, 'lap_number': 20, 'date': iso(19 * LAP_S + 30)}],
        'intervals': intervals,
        'position': positions,
        'race_control': [
            {'date': iso(5 * LAP_S), 'category': 'SafetyCar', 'message': 'SAFETY CAR DEPLOYED', 'flag': None},
            {
                'date': iso(7 * LAP_S),
                'category': 'SafetyCar',
                'message': 'SAFETY CAR IN THIS LAP',
                'flag': None,
            },
            {
                'date': iso(8 * LAP_S),
                'category': 'Flag',
                'message': 'GREEN LIGHT - PIT EXIT OPEN',
                'flag': 'GREEN',
                'scope': 'Track',
            },
        ],
        'weather': [{'date': iso(0), 'rainfall': 0, 'track_temperature': 41.2}],
        'team_radio': [
            {'driver_number': 44, 'date': iso(19 * LAP_S + 60), 'recording_url': 'https://x/ham1.mp3'},
            {'driver_number': 44, 'date': iso(25 * LAP_S + 20), 'recording_url': 'https://x/ham2.mp3'},
            {'driver_number': 4, 'date': iso(3 * LAP_S), 'recording_url': 'https://x/nor1.mp3'},
        ],
    }


SESSION = {'session_key': 9001, 'year': 2024, 'circuit_short_name': 'Silverstone', 'date_start': iso(0)}


class SituationTest(unittest.TestCase):
    def setUp(self):
        self.race = of1.RaceData.from_rows(SESSION, race_rows())

    def test_rebuilds_the_pit_wall_view_after_a_rival_stop(self):
        text = of1.situation(self.race, 44, START + timedelta(seconds=19 * LAP_S + 60))
        self.assertIn('Lap 20/30.', text)
        self.assertIn('P2.', text)
        self.assertIn('Medium, 21 laps old.', text)
        self.assertIn('0 stops so far.', text)
        self.assertIn('Ahead VER +2.0s, closing 0.3s in the last lap.', text)
        self.assertIn('Behind NOR -1.4s, steady.', text)
        self.assertIn('NOR just pitted.', text)
        self.assertIn('Last lap 90.000s.', text)
        self.assertIn('Track 41C.', text)

    def test_tyre_age_follows_the_new_stint(self):
        text = of1.situation(self.race, 4, START + timedelta(seconds=24 * LAP_S + 10))
        self.assertIn('Hard, 5 laps old.', text)
        self.assertIn('1 stop so far.', text)
        self.assertNotIn('just pitted', text)

    def test_safety_car_shows_until_the_green(self):
        during = of1.situation(self.race, 4, START + timedelta(seconds=6 * LAP_S))
        self.assertIn('Safety car deployed.', during)
        ending = of1.situation(self.race, 4, START + timedelta(seconds=7 * LAP_S + 10))
        self.assertIn('Safety car in this lap.', ending)
        after = of1.situation(self.race, 4, START + timedelta(seconds=9 * LAP_S))
        self.assertNotIn('Safety car', after)

    def test_leader_has_no_car_ahead(self):
        text = of1.situation(self.race, 1, START + timedelta(seconds=10 * LAP_S + 5))
        self.assertIn('P1.', text)
        self.assertNotIn('Ahead', text)
        self.assertIn('Behind HAM', text)


class EngineerLineTest(unittest.TestCase):
    def test_keeps_engineer_segments_and_drops_driver_ones(self):
        segments = [
            {'start': 0.0, 'end': 2.0, 'text': "I've got no grip, my tyres are gone."},
            {'start': 2.5, 'end': 5.0, 'text': 'Copy. Box this lap, box box. We go hards.'},
        ]
        text, score = of1.engineer_text(segments)
        self.assertEqual(text, 'Copy. Box this lap, box box. We go hards.')
        self.assertGreater(score, 0.9)

    def test_tags(self):
        self.assertEqual(of1.tags_for('Box this lap, Norris has pitted, we go hards.')[:1], ['pit_call'])
        self.assertIn('gap', of1.tags_for('Gap to Verstappen is 1.2 seconds, DRS next lap.'))
        self.assertIn('flags', of1.tags_for('Safety car, safety car. Stay off the kerbs.'))
        self.assertIn('weather', of1.tags_for('Rain expected in five minutes.'))

    def test_filters_acks_and_monologues(self):
        self.assertFalse(of1.keep_line('Copy that.'))
        self.assertFalse(of1.keep_line('ok'))
        self.assertFalse(of1.keep_line('word ' * 40))
        self.assertTrue(of1.keep_line('Push now, gap is 1.5.'))


class BuildTest(unittest.TestCase):
    def test_build_writes_train_and_held_out_test_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            second = dict(SESSION, session_key=9002, date_start=iso(86400 * 7))
            (out / 'sessions.json').write_text(json.dumps([SESSION, second]))
            transcripts = {
                '44_20240707142930000.mp3': [
                    {'start': 0, 'end': 3, 'text': 'Norris has boxed. Box box, box for hards.'}
                ],
                '44_20240707143750000.mp3': [{'start': 0, 'end': 1, 'text': 'Copy.'}],
            }
            for key in (9001, 9002):
                folder = out / str(key)
                folder.mkdir()
                for endpoint, rows in race_rows().items():
                    (folder / f'{endpoint}.json').write_text(json.dumps(rows))
                (folder / 'transcripts.json').write_text(json.dumps(transcripts))
            train, test = of1.build(out, holdout=1)
            self.assertEqual((train, test), (1, 1))
            example = json.loads((out / 'examples_train.jsonl').read_text().splitlines()[0])
            self.assertEqual(example['session_key'], 9001)
            self.assertEqual(example['driver'], 'HAM')
            self.assertEqual(example['lap'], 20)
            self.assertIn('NOR just pitted.', example['situation'])
            self.assertEqual(example['tags'][0], 'pit_call')
            held_out = json.loads((out / 'examples_test.jsonl').read_text().splitlines()[0])
            self.assertEqual(held_out['session_key'], 9002)


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FetchTest(unittest.TestCase):
    def test_fetch_caches_every_endpoint_and_clip(self):
        calls = []

        def opener(url, timeout):
            calls.append(url)
            if '/sessions?' in url:
                return FakeResponse(json.dumps([SESSION]).encode())
            if '/team_radio?' in url:
                return FakeResponse(json.dumps(race_rows()['team_radio'][:1]).encode())
            if url.endswith('.mp3'):
                return FakeResponse(b'ID3audio')
            if '/weather?' in url:
                raise urllib.error.HTTPError(url, 404, 'none', None, None)
            return FakeResponse(b'[]')

        with tempfile.TemporaryDirectory() as tmp:
            out = pathlib.Path(tmp)
            fetcher = of1.Fetcher(gap_s=0, opener=opener)
            self.assertEqual(of1.fetch(out, [2024], fetcher), 1)
            folder = out / '9001'
            self.assertEqual(sorted(p.stem for p in folder.glob('*.json')), sorted(of1.ENDPOINTS))
            self.assertEqual(json.loads((folder / 'weather.json').read_text()), [])
            self.assertEqual([p.read_bytes() for p in (folder / 'radio').glob('*.mp3')], [b'ID3audio'])
            first_run = len(calls)
            of1.fetch(out, [2024], fetcher)  # everything cached: only the season list is asked again
            self.assertEqual(len(calls), first_run + 1)

    def test_error_pages_are_not_saved_as_clips_and_old_ones_are_fetched_again(self):
        bodies = {'c0': b'<?xml version="1.0"?><Error><Code>AccessDenied</Code></Error>', 'c1': b'ID3good'}

        def opener(url, timeout):
            if '/sessions?' in url:
                return FakeResponse(json.dumps([SESSION]).encode())
            if '/team_radio?' in url:
                clips = [
                    {'driver_number': 44, 'date': iso(i), 'recording_url': f'https://x/c{i}.mp3'}
                    for i in (0, 1)
                ]
                return FakeResponse(json.dumps(clips).encode())
            if url.endswith('.mp3'):
                return FakeResponse(bodies[url.rsplit('/', 1)[1][:-4]])
            return FakeResponse(b'[]')

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            out = pathlib.Path(tmp)
            radio = out / '9001' / 'radio'
            radio.mkdir(parents=True)
            stale = radio / of1.clip_filename({'driver_number': 44, 'date': iso(1)})
            stale.write_bytes(b'<html>Forbidden</html>')  # saved by an older version
            of1.fetch(out, [2024], of1.Fetcher(gap_s=0, opener=opener))
            self.assertEqual({p.name: p.read_bytes() for p in radio.glob('*.mp3')}, {stale.name: b'ID3good'})
            self.assertIn('server sent no audio', log.getvalue())
            self.assertIn('AccessDenied', log.getvalue())

    def test_clip_urls_with_accents_are_percent_encoded(self):
        url = 'https://livetiming.formula1.com/static/2023/2023-11-05_São_Paulo_Grand_Prix/Race/TeamRadio/MAXVER01_1_x.mp3'
        encoded = of1.ascii_url(url)
        self.assertTrue(encoded.isascii())
        self.assertIn('S%C3%A3o_Paulo', encoded)
        self.assertEqual(of1.ascii_url(encoded), encoded)  # already-encoded URLs are left alone
        self.assertEqual(
            of1.ascii_url('https://api.openf1.org/v1/laps?session_key=9221'),
            'https://api.openf1.org/v1/laps?session_key=9221',
        )
        seen = []

        def opener(u, timeout):
            seen.append(u)
            return FakeResponse(b'ID3')

        of1.Fetcher(gap_s=0, opener=opener).get_bytes(url)
        self.assertEqual(seen, [encoded])

    def test_looks_like_audio(self):
        self.assertTrue(of1.looks_like_audio(b'ID3\x04\x00'))
        self.assertTrue(of1.looks_like_audio(bytes([0xFF, 0xFB, 0x90])))
        self.assertTrue(of1.looks_like_audio(b'\x00\x00\x00\x20ftypM4A '))
        self.assertFalse(of1.looks_like_audio(b'<html><body>403</body></html>'))
        self.assertFalse(of1.looks_like_audio(b''))

    def test_refused_clips_are_skipped_and_audio_stops_after_a_run_of_refusals(self):
        clips = [
            {'driver_number': 44, 'date': iso(i), 'recording_url': f'https://x/c{i}.mp3'}
            for i in range(of1.MAX_REFUSED_IN_A_ROW + 10)
        ]
        audio_calls = []

        def opener(url, timeout):
            if '/sessions?' in url:
                return FakeResponse(json.dumps([SESSION]).encode())
            if '/team_radio?' in url:
                return FakeResponse(json.dumps(clips).encode())
            if url.endswith('.mp3'):
                audio_calls.append(url)
                if url.endswith('/c0.mp3'):
                    return FakeResponse(b'ID3audio')
                raise urllib.error.HTTPError(url, 403, 'Forbidden', None, None)
            return FakeResponse(b'[]')

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            out = pathlib.Path(tmp)
            self.assertEqual(of1.fetch(out, [2024], of1.Fetcher(gap_s=0, opener=opener)), len(clips))
            saved = list((out / '9001' / 'radio').glob('*.mp3'))
            self.assertEqual([p.read_bytes() for p in saved], [b'ID3audio'])
            # One good clip, then refusals until the limit; the rest aren't asked for.
            self.assertEqual(len(audio_calls), 1 + of1.MAX_REFUSED_IN_A_ROW)
            self.assertIn('timing data only', log.getvalue())
            self.assertTrue((out / '9001' / 'laps.json').exists())


class TranscribeTest(unittest.TestCase):
    @unittest.skipUnless(shutil.which('ffmpeg'), 'needs ffmpeg')
    def test_clips_are_decoded_with_ffmpeg_and_handed_to_whisper_as_samples(self):
        seen = []

        class FakeModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, audio, **kwargs):
                seen.append(audio)
                return [types.SimpleNamespace(start=0.0, end=1.0, text=' Box this lap. ')], None

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as log:
            radio = pathlib.Path(tmp) / '9001' / 'radio'
            radio.mkdir(parents=True)
            good = radio / '44_1.mp3'
            subprocess.run(
                [
                    'ffmpeg',
                    '-loglevel',
                    'error',
                    '-f',
                    'lavfi',
                    '-i',
                    'sine=frequency=440:duration=1',
                    str(good),
                ],
                check=True,
            )
            (radio / '44_2.mp3').write_bytes(b'not audio')
            fake = types.ModuleType('faster_whisper')
            fake.WhisperModel = FakeModel
            sys.modules['faster_whisper'], saved = fake, sys.modules.get('faster_whisper')
            try:
                self.assertEqual(of1.transcribe(pathlib.Path(tmp)), 1)
            finally:
                if saved is None:
                    del sys.modules['faster_whisper']
                else:
                    sys.modules['faster_whisper'] = saved
            self.assertEqual(len(seen), 1)
            self.assertEqual(str(seen[0].dtype), 'float32')
            self.assertAlmostEqual(len(seen[0]) / of1.SAMPLE_RATE, 1.0, delta=0.1)
            transcripts = json.loads((radio.parent / 'transcripts.json').read_text())
            self.assertEqual(transcripts, {'44_1.mp3': [{'start': 0.0, 'end': 1.0, 'text': 'Box this lap.'}]})
            self.assertIn('1 unreadable clips skipped', log.getvalue())
            self.assertIn('not audio, skipped 44_2.mp3', log.getvalue())

    @unittest.skipUnless(shutil.which('ffmpeg'), 'needs ffmpeg to make the test clip')
    def test_pyav_decoder_matches_ffmpeg_and_is_used_when_ffmpeg_is_missing(self):
        try:
            import av  # noqa: F401
        except ImportError:
            self.skipTest('needs PyAV')
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmp:
            radio = pathlib.Path(tmp) / '9001' / 'radio'
            radio.mkdir(parents=True)
            clip = radio / '44_1.mp3'
            subprocess.run(
                [
                    'ffmpeg',
                    '-loglevel',
                    'error',
                    '-f',
                    'lavfi',
                    '-i',
                    'sine=frequency=440:duration=1',
                    str(clip),
                ],
                check=True,
            )
            by_pyav = of1.decode_audio_pyav(clip)
            by_ffmpeg = of1.decode_audio(clip)
            self.assertEqual(str(by_pyav.dtype), 'float32')
            self.assertAlmostEqual(len(by_pyav) / of1.SAMPLE_RATE, 1.0, delta=0.1)
            self.assertLess(abs(len(by_pyav) - len(by_ffmpeg)), of1.SAMPLE_RATE // 20)
            self.assertAlmostEqual(float(abs(by_pyav).max()), float(abs(by_ffmpeg).max()), delta=0.05)

            seen = []

            class FakeModel:
                def __init__(self, *args, **kwargs):
                    pass

                def transcribe(self, audio, **kwargs):
                    seen.append(audio)
                    return [], None

            fake = types.ModuleType('faster_whisper')
            fake.WhisperModel = FakeModel
            with (
                mock.patch.dict(sys.modules, {'faster_whisper': fake}),
                mock.patch.object(of1.shutil, 'which', return_value=None),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(of1.transcribe(pathlib.Path(tmp)), 1)
            self.assertEqual(
                str(seen[0].dtype), 'float32'
            )  # samples, not a file path for PyAV inside Whisper

    def test_whisper_falls_back_to_the_cpu_when_cuda_cannot_start(self):
        made = []

        def fake_model(name, device, compute_type):
            made.append((device, compute_type))
            if device != 'cpu':
                raise RuntimeError(
                    'CUDA failed with error CUDA driver version is insufficient for CUDA runtime version'
                )
            return 'model'

        with contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(of1.load_whisper(fake_model, 'small.en', 'cuda'), 'model')
        self.assertEqual(made, [('cuda', 'auto'), ('cpu', 'int8')])
        self.assertIn('on the CPU', log.getvalue())

    def test_other_whisper_errors_are_not_hidden(self):
        def broken(name, device, compute_type):
            raise RuntimeError('model files missing')

        with self.assertRaises(RuntimeError):
            of1.load_whisper(broken, 'small.en', 'cuda')


if __name__ == '__main__':
    unittest.main()
