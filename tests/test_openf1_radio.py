"""Real-radio dataset builder (tools/openf1_radio.py): situation rebuild, speaker split, tags, build."""

import io
import json
import pathlib
import sys
import tempfile
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


if __name__ == '__main__':
    unittest.main()
