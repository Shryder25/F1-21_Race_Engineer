import tempfile
import unittest
from pathlib import Path

import auto_refit
import driver_profile
import fitted_models as fm
from engineer_brain import format_context
from history import TelemetryHistory
from packets import SessionData
from test_analysis import SilentVoice  # noqa: F401  (puts ai_engineer/ on the import path)

SILVERSTONE = SessionData(0, 30, 20, 20, 5891, 10, 7, 0)


def lap_recording(lap_ms, s2_extra=0, stowe_extra=0):
    return {'lap_time_ms': lap_ms, 'times': {}, 'speeds': [], 'samples': [],
            'sectors': {'1': 29000, '2': 33000 + s2_extra, '3': lap_ms - 62000 - s2_extra},
            'corners': {'Stowe': {'start_time_ms': 50000, 'end_time_ms': 53000 + stowe_extra,
                                  'min_speed': 160, 'brake_point': 4000},
                        'Copse': {'start_time_ms': 10000, 'end_time_ms': 12000,
                                  'min_speed': 250, 'brake_point': 1000}}}


class LearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.history = TelemetryHistory(Path(self.tmp.name) / 'telemetry.sqlite3')

    def tearDown(self):
        self.history.close()
        self.tmp.cleanup()

    def drive_session(self, uid, laps, compound=17):
        capture = self.history.start_session(SILVERSTONE, uid, 0)
        for n, (ms, s2, stowe) in enumerate(laps, 1):
            self.history.record_lap(capture, n, ms, True, lap_recording(ms, s2, stowe))
            self.history.record_lap_stats(capture, {
                'lap_num': n, 'lap_time_ms': ms, 'eligible': True, 'compound': compound,
                'tyre_age_laps': n, 'wear': (2.0 * n, 2.1 * n, 2.4 * n, 2.5 * n),
                'fuel_start': 50 - 1.5 * (n - 1), 'fuel_end': 50 - 1.5 * n})
        return capture

    def test_profile_needs_a_few_laps(self):
        self.drive_session(1, [(90000, 0, 0), (90200, 100, 0)])
        self.assertIsNone(driver_profile.build_profile(self.history, SILVERSTONE))

    def test_profile_finds_where_the_driver_loses_time(self):
        self.drive_session(1, [(90000, 0, 0), (90400, 300, 150), (90350, 300, 200)])
        self.drive_session(2, [(90300, 250, 120), (99000, 0, 0)])     # 99 s lap = traffic, ignored
        profile = driver_profile.build_profile(self.history, SILVERSTONE)
        self.assertEqual((profile['laps'], profile['sessions'], profile['best_lap_ms']), (4, 2, 90000))
        self.assertEqual(profile['weakest_sector']['sector'], 2)
        self.assertEqual(profile['weakest_corners'][0]['corner'], 'Stowe')
        self.assertNotIn('Copse', [c['corner'] for c in profile['weakest_corners']])
        line = driver_profile.describe(profile)
        self.assertIn('most time left in sector 2', line)
        self.assertIn("Driver's own history at this track: 4 clean laps",
                      format_context({'driver_profile': line}))

    def test_refit_learns_from_new_laps_only(self):
        self.drive_session(1, [(90000 + 100 * n, 0, 0) for n in range(8)])
        path = Path(self.tmp.name) / 'fitted_models.json'
        store = auto_refit.refit(self.history, path, log=lambda _: None)
        self.assertIsNotNone(store)
        self.assertTrue(path.exists())
        self.assertTrue(store.wear)
        self.assertIsNone(auto_refit.refit(self.history, path, log=lambda _: None))  # nothing new
        self.drive_session(2, [(90000 + 100 * n, 0, 0) for n in range(4)])
        again = auto_refit.refit(self.history, path, log=lambda _: None)
        self.assertIsNotNone(again)
        self.assertEqual(fm.ModelStore.load(path).sources[-1]['laps'], 12)

    def test_refit_with_no_laps_writes_nothing(self):
        path = Path(self.tmp.name) / 'fitted_models.json'
        self.assertIsNone(auto_refit.refit(self.history, path, log=lambda _: None))
        self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
