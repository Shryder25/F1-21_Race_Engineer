"""Backtest report (tools/backtest.py): ground truth, error maths and an end-to-end replay."""

import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT / 'tools', ROOT / 'tests' / 'golden', ROOT / 'ai_engineer'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import backtest  # noqa: E402
import recording  # noqa: E402
from packets import SessionData  # noqa: E402
from synth import Car, Scenario  # noqa: E402


def snapshot(lap, **overrides):
    snap = {'lap': lap, 't': lap * 80.0, 'stops_done': 0, 'pit_status': 0, 'compound': 'Soft',
            'tyre_age': lap - 1, 'worst_wear': 3.0 * (lap - 1), 'safety_car': False,
            'truth_gaps': {}, 'stop_counts': {0: 0, 5: 0}, 'pit_status_all': {},
            'pit': {}, 'wear_pred': {}, 'race_pace_ms': None, 'gaps': []}
    snap.update(overrides)
    return snap


def segment(snapshots, stops=(), lap_times=None, safety_car_laps=(), total_laps=10):
    truth = backtest.GroundTruth(1)
    truth.session = SessionData(0, 30, 20, total_laps, 4318, 10, 17, 0)
    truth.player_index = 0
    truth.stops[0] = list(stops)
    truth.lap_times_ms = dict(lap_times or {})
    truth.safety_car_laps = set(safety_car_laps)
    seg = backtest.Segment(1, truth)
    seg.snapshots = {s['lap']: s for s in snapshots}
    return seg


def errors(result, metric, predictor):
    return [s['error'] for s in result['samples'] if s['metric'] == metric and s['predictor'] == predictor]


class SummaryTests(unittest.TestCase):
    def test_summary_statistics(self):
        stats = backtest.summarize([1.0, -3.0, 2.0])
        self.assertEqual(stats['n'], 3)
        self.assertAlmostEqual(stats['mae'], 2.0)
        self.assertAlmostEqual(stats['median_ae'], 2.0)
        self.assertAlmostEqual(stats['bias'], 0.0)
        self.assertAlmostEqual(stats['max_ae'], 3.0)
        self.assertAlmostEqual(stats['rmse'], (14 / 3) ** 0.5, places=3)
        self.assertEqual(backtest.summarize([]), {'n': 0})


class GroundTruthTests(unittest.TestCase):
    def test_gap_is_the_time_since_the_other_car_passed_the_same_point(self):
        truth = backtest.GroundTruth(1)
        truth.player_index = 0
        truth.trajectories[1] = [(0.0, 0.0), (100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]
        truth.trajectories[0] = [(0.0, 0.5), (100.0, 1.5), (200.0, 2.5)]
        truth.trajectories[2] = [(0.0, 1.0), (50.0, 2.0), (150.0, 3.0)]
        truth.positions = {0: 250.0, 1: 300.0, 2: 150.0}
        truth.session_time = 3.0
        self.assertAlmostEqual(truth.gap(1), 0.5)       # car 1 passed 250 m at t=2.5
        truth.trajectories[0].append((250.0, 3.0))
        self.assertAlmostEqual(truth.gap(2), -1.0)      # the player passed 150 m at t=2.0
        self.assertIsNone(truth.gap(0))


class EvaluateTests(unittest.TestCase):
    def test_pit_lap_errors_and_outcomes(self):
        snaps = [snapshot(1, pit={'strategy': 6, 'ml_ideal': 'none'}),
                 snapshot(3, pit={'strategy': 4, 'game_window': 5}),
                 snapshot(5, pit={'strategy': 5}),
                 snapshot(7, stops_done=1, pit={'strategy': 'none', 'game_window': 5}),
                 snapshot(8, stops_done=1, pit={'ml_ideal': 12})]
        result = backtest.evaluate(segment(snaps, stops=[5]))
        self.assertEqual(errors(result, 'pit_lap', 'strategy'), [1, -1, 0])
        self.assertEqual(errors(result, 'pit_lap', 'game_window'), [0])
        outcomes = result['pit_outcomes']
        self.assertEqual(outcomes['ml_ideal']['missed_stop'], 1)
        self.assertEqual(outcomes['ml_ideal']['stop_that_never_came'], 1)
        self.assertEqual(outcomes['strategy']['correct_no_stop'], 1)
        self.assertEqual(outcomes['game_window']['stop_that_never_came'], 1)
        horizons = [s['horizon'] for s in result['samples'] if s['predictor'] == 'strategy']
        self.assertEqual(horizons, ['4-7 laps before', '0-3 laps before', '0-3 laps before'])

    def test_wear_compared_within_the_same_stint_only(self):
        snaps = [snapshot(n) for n in range(1, 5)]
        snaps[1]['wear_pred'] = {1: 6.5, 3: 12.0}          # lap 2: actual lap 3 = 6.0
        snaps.append(snapshot(5, stops_done=1, worst_wear=1.0))
        result = backtest.evaluate(segment(snaps, stops=[4]))
        self.assertEqual(errors(result, 'tyre_wear', 'ml_curve'), [0.5])   # +3 crosses the stop
        baseline = [s for s in result['samples'] if s['predictor'] == 'last_lap_rate']
        self.assertTrue(baseline)
        self.assertTrue(all(abs(s['error']) < 1e-9 for s in baseline))

    def test_lap_time_skips_lap_one_pit_laps_and_safety_car(self):
        times = {n: 80_000 + n * 100 for n in range(1, 10)}
        snaps = [snapshot(n, race_pace_ms=80_500) for n in range(1, 10)]
        result = backtest.evaluate(segment(snaps, stops=[4], lap_times=times, safety_car_laps=[7]))
        laps = sorted({s['lap'] for s in result['samples'] if s['metric'] == 'lap_time'})
        self.assertEqual(laps, [2, 3, 6, 8, 9])
        last_lap = {s['lap']: s for s in result['samples'] if s['predictor'] == 'last_lap'}
        self.assertEqual(sorted(last_lap), [3, 9])          # needs a clean lap before it too
        self.assertAlmostEqual(last_lap[3]['error'], -0.1)

    def test_gap_forecast_uses_pace_delta_and_skips_pit_windows(self):
        ahead = {'side': 'ahead', 'car': 5, 'engineer_gap': 2.0, 'rate_s_per_lap': -0.5, 'name': None}
        snaps = [snapshot(1, truth_gaps={5: 2.1}, gaps=[ahead]),
                 snapshot(2, truth_gaps={5: 1.4}),
                 snapshot(3, truth_gaps={5: 1.0}),
                 snapshot(4, truth_gaps={5: 0.6}, stop_counts={0: 0, 5: 1})]
        result = backtest.evaluate(segment(snaps))
        self.assertEqual(errors(result, 'gap_now', 'engineer'), [-0.1])
        forecast = {s['horizon']: s['error'] for s in result['samples'] if s['predictor'] == 'pace_delta'}
        self.assertEqual(forecast, {'+1 lap': 0.1})          # +3 laps crosses car 5's stop
        hold = [s['error'] for s in result['samples'] if s['predictor'] == 'hold']
        self.assertEqual(hold, [0.6])

    def test_gap_behind_is_compared_with_the_same_sign(self):
        behind = {'side': 'behind', 'car': 5, 'engineer_gap': 1.5, 'rate_s_per_lap': 0.2, 'name': None}
        snaps = [snapshot(1, truth_gaps={5: -1.5}, gaps=[behind]), snapshot(2, truth_gaps={5: -1.8})]
        result = backtest.evaluate(segment(snaps))
        self.assertEqual(errors(result, 'gap_now', 'engineer'), [0.0])
        self.assertEqual(errors(result, 'gap', 'pace_delta'), [-0.1])


def small_race():
    cars = [Car('Player', pace=0.3, compound='soft', grid=3, pits={4: 'hard'}, race_number=99)]
    for index, name in enumerate(['HAMILTON', 'VERSTAPPEN', 'BOTTAS', 'PEREZ', 'NORRIS'], start=1):
        cars.append(Car(name, pace=0.15 * index, compound='medium' if index % 2 else 'soft',
                        grid=index + (index >= 3), pits={3 + index % 3: 'hard'}, race_number=index))
    return Scenario(name='backtest_test_race', track_id=17, track_length=4318, session_type=10,
                    total_laps=7, cars=cars, fuel_kg=7 * 1.7 + 2, pit_window=(3, 5),
                    description='Short test race')


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.path = pathlib.Path(cls.tmp.name) / 'race.f1rec.gz'
        backtest.write_scenario(small_race(), cls.path)
        cls.report = backtest.backtest([cls.path, pathlib.Path(cls.tmp.name) / 'missing.f1rec'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_replay_finds_the_race_and_its_ground_truth(self):
        self.assertEqual(len(self.report['races']), 1)
        race = self.report['races'][0]
        self.assertEqual(race['name'], 'backtest_test_race')
        self.assertTrue(race['simulated'])
        self.assertEqual(race['player_stops'], [4])
        self.assertEqual(race['total_laps'], 7)
        self.assertEqual(race['laps_snapshotted'], 7)
        self.assertTrue(self.report['skipped'][0]['path'].endswith('missing.f1rec'))

    def test_every_kind_of_prediction_is_scored(self):
        metrics = {(s['metric'], s['predictor']) for s in self.report['races'][0]['samples']}
        for wanted in (('pit_lap', 'strategy'), ('pit_lap', 'game_window'), ('tyre_wear', 'last_lap_rate'),
                       ('lap_time', 'race_pace'), ('lap_time', 'last_lap'), ('gap_now', 'engineer'),
                       ('gap', 'hold')):
            self.assertIn(wanted, metrics)

    def test_live_gap_matches_the_ground_truth_on_clean_data(self):
        rows = [r for r in self.report['aggregate']['summary']
                if r['metric'] == 'gap_now' and r['horizon'] == 'all']
        self.assertEqual(len(rows), 1)
        self.assertLess(rows[0]['mae'], 0.05)

    def test_report_says_the_numbers_are_simulated(self):
        text = backtest.render_markdown(self.report)
        self.assertIn('simulated races only', text)
        self.assertIn('### backtest_test_race (simulated)', text)
        self.assertIn('| Pit lap (laps) |', text)
        self.assertIn('Skipped files', text)

    def test_a_recording_without_the_synthetic_marker_counts_as_real(self):
        self.assertFalse(backtest.is_simulated({'port': 20777, 'recorder': 'ai_engineer/udp_recording.py'}))
        self.assertTrue(backtest.is_simulated(recording.read_metadata(self.path)))
        mixed = dict(self.report, real_races=1)
        self.assertIn('Simulated races are included', backtest.render_markdown(mixed))


if __name__ == '__main__':
    unittest.main()
