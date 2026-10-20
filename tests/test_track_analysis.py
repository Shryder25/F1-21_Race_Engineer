import json
import pathlib
import sys
import unittest
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ai_engineer'))

import track_analysis
from ai_client import AIUnavailable
from race_calibration import fuel_mass, stint_degradation, summarize_races, theil_sen_slope
from track_analysis import (SETUP_RANGES, TrackAnalyst, clamp_setup, merge_ai_analysis,
                            model_analysis, wet_setup)
from track_data import TRACK_DATABASE, get_track_setup
from track_profiles import BASELINE_PROFILES, get_profile

ORDER = {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3}


class TyreDegradationTests(unittest.TestCase):
    def test_every_track_has_a_profile(self):
        self.assertEqual(set(BASELINE_PROFILES), set(TRACK_DATABASE))

    def test_degradation_differs_by_track_instead_of_always_high(self):
        labels = {tid: model_analysis(tid, 'dry')['tyre_degradation'] for tid in TRACK_DATABASE}
        self.assertGreaterEqual(len(set(labels.values())), 3)
        self.assertLess(list(labels.values()).count('High'), len(labels) // 2)
        self.assertEqual(labels[5], 'Low')                     # Monaco: smooth, slow
        self.assertEqual(labels[20], 'Low')                    # Baku
        for abrasive in (3, 7, 13):                            # Bahrain, Silverstone, Suzuka
            self.assertGreaterEqual(ORDER[labels[abrasive]], ORDER['High'])

    def test_wet_analysis_uses_wet_tyres_and_its_own_degradation(self):
        dry, wet = model_analysis(7, 'dry'), model_analysis(7, 'wet')
        self.assertEqual(set(dry['degradation']['rates_s_per_lap']), {'SOFT', 'MEDIUM', 'HARD'})
        self.assertEqual(set(wet['degradation']['rates_s_per_lap']), {'INTERMEDIATE', 'WET'})
        self.assertLess(ORDER[wet['tyre_degradation']], ORDER[dry['tyre_degradation']])

    def test_softer_compounds_degrade_faster(self):
        for tid in TRACK_DATABASE:
            rates = model_analysis(tid)['degradation']['rates_s_per_lap']
            self.assertGreater(rates['SOFT'], rates['MEDIUM'])
            self.assertGreater(rates['MEDIUM'], rates['HARD'])


class StrategyTests(unittest.TestCase):
    def test_strategies_are_legal_and_cover_the_race(self):
        for tid in TRACK_DATABASE:
            for weather in ('dry', 'wet'):
                result = model_analysis(tid, weather)
                best = result['strategy']['best']
                laps = sum(s['laps'] for s in best['stints'])
                self.assertEqual(laps, result['strategy']['race_laps'])
                life = model_analysis(tid, weather)['degradation']['life_laps']
                for stint in best['stints']:
                    self.assertLessEqual(stint['laps'], life[stint['compound']])
                if weather == 'dry':
                    self.assertGreaterEqual(len({s['compound'] for s in best['stints']}), 2)

    def test_hard_to_pass_tracks_favour_fewer_stops(self):
        self.assertEqual(model_analysis(5)['strategy']['best']['stops'], 1)      # Monaco
        self.assertGreaterEqual(model_analysis(3)['strategy']['best']['stops'], 2)  # Bahrain

    def test_race_distance_scales_the_plan(self):
        full = model_analysis(7, race_fraction=1.0)['strategy']['race_laps']
        half = model_analysis(7, race_fraction=0.5)['strategy']['race_laps']
        self.assertEqual(half, round(full * 0.5))


class SetupTests(unittest.TestCase):
    def test_wet_setup_is_numerically_different_for_every_track(self):
        for tid in TRACK_DATABASE:
            dry, wet = model_analysis(tid, 'dry')['setup'], model_analysis(tid, 'wet')['setup']
            self.assertNotEqual({k: dry[k] for k in SETUP_RANGES}, {k: wet[k] for k in SETUP_RANGES})
            self.assertGreaterEqual(wet['aerodynamics']['rear_wing'], dry['aerodynamics']['rear_wing'])
            self.assertLess(wet['transmission']['diff_on_throttle'], dry['transmission']['diff_on_throttle'])
            self.assertGreater(wet['suspension']['front_ride_height'], dry['suspension']['front_ride_height'])
            self.assertLess(wet['brakes']['brake_pressure'], dry['brakes']['brake_pressure'])

    def test_legacy_get_track_setup_also_changes_in_the_wet(self):
        self.assertNotEqual(get_track_setup(11, 'dry')['setup']['aerodynamics'],
                            get_track_setup(11, 'wet')['setup']['aerodynamics'])

    def test_handling_preference_changes_the_setup(self):
        base = model_analysis(7, handling='balanced')['setup']
        loose = model_analysis(7, handling='oversteer_fix')['setup']
        self.assertGreater(loose['aerodynamics']['rear_wing'], base['aerodynamics']['rear_wing'])

    def test_values_are_clamped_to_game_ranges(self):
        wild = wet_setup(clamp_setup(TRACK_DATABASE[5]['setup']))   # Monaco already at max wing
        self.assertEqual(wild['aerodynamics']['front_wing'], 11)
        clamped = clamp_setup({'aerodynamics': {'front_wing': 40}, 'brakes': {'brake_bias': 10}})
        self.assertEqual(clamped['aerodynamics']['front_wing'], 11)
        self.assertEqual(clamped['brakes']['brake_bias'], 50)


class DeepSeekAnalysisTests(unittest.TestCase):
    REPLY = {
        'tyre_degradation': 'very high',
        'degradation_reasoning': 'Copse and Maggotts load the front-left.',
        'limiting_tyre': 'front-left',
        'strategy': {'stops': 1, 'plan': 'Medium 20, Hard 32', 'reasoning': 'Track position.'},
        'setup': {'aerodynamics': {'front_wing': 99, 'rear_wing': '6'}, 'brakes': {'brake_bias': 'abc'}},
        'setup_notes': {'aerodynamics': 'Trim for Hangar straight.', 'nonsense': 'x'},
        'key_corners': ['Copse', 'Stowe'],
        'briefing': 'Look after the front-left.',
    }

    def test_reply_is_validated_and_merged_over_the_model(self):
        model = model_analysis(7)
        merged = merge_ai_analysis(model, self.REPLY)
        self.assertEqual(merged['analysis_source'], 'deepseek')
        self.assertEqual(merged['tyre_degradation'], 'Very High')
        self.assertEqual(merged['setup']['aerodynamics'], {'front_wing': 11, 'rear_wing': 6})
        self.assertEqual(merged['setup']['brakes']['brake_bias'], model['setup']['brakes']['brake_bias'])
        self.assertEqual(merged['setup_notes'], {'aerodynamics': 'Trim for Hangar straight.'})
        self.assertEqual(merged['strategy']['ai']['stops'], 1)
        self.assertEqual(merged['engineer_advice'], 'Look after the front-left.')
        self.assertEqual(model['analysis_source'], 'model')            # model untouched

    def test_invalid_label_keeps_the_model_verdict(self):
        model = model_analysis(5)
        merged = merge_ai_analysis(model, {'tyre_degradation': 'extreme'})
        self.assertEqual(merged['tyre_degradation'], model['tyre_degradation'])

    def test_prompt_carries_track_and_condition_specific_data(self):
        system, user = track_analysis.build_analysis_prompt(model_analysis(10, 'wet'))
        self.assertIn('JSON', system)
        self.assertIn('Spa', user)
        self.assertIn('WET', user)
        self.assertIn('INTERMEDIATE', user)
        self.assertIn('Pit-lane loss', user)

    def test_analyst_caches_and_falls_back(self):
        calls = []

        class Client:
            timeout = 15.0

            def chat_json(self, system, user, **kw):
                calls.append(user)
                return DeepSeekAnalysisTests.REPLY

        client = Client()
        analyst = TrackAnalyst(client)
        self.assertEqual(analyst.ai_client.timeout, 90.0)
        self.assertEqual(client.timeout, 15.0)                      # caller's client untouched
        first = analyst.analyze(7, 'dry')
        second = analyst.analyze(7, 'dry')
        self.assertEqual(first['analysis_source'], 'deepseek')
        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertIs(analyst.cached(7, 'dry'), first)
        self.assertIsNone(analyst.cached(7, 'wet'))
        analyst.analyze(7, 'dry', force=True)
        self.assertEqual(len(calls), 2)

        class Down:
            timeout = 15.0
            attempts = 0

            def chat_json(self, system, user, **kw):
                Down.attempts += 1
                raise AIUnavailable('connection refused')

        analyst = TrackAnalyst(Down())
        result = analyst.analyze(3, 'wet')
        self.assertEqual(result['analysis_source'], 'model')
        self.assertIn('connection refused', result['ai_error'])
        analyst.analyze(3, 'wet')                                   # backs off, no second call
        self.assertEqual(Down.attempts, 1)

        offline = TrackAnalyst(None).analyze(3)
        self.assertIn('not configured', offline['ai_error'])


class RealRaceCalibrationTests(unittest.TestCase):
    def test_theil_sen_ignores_an_outlier(self):
        ys = [1.0 + 0.1 * x for x in range(10)]
        ys[4] += 5.0
        self.assertAlmostEqual(theil_sen_slope(range(10), ys), 0.1, places=6)

    @staticmethod
    def _stint(compound, start_lap, laps, deg, race_laps=52, base=90.0):
        rows = []
        for age in range(1, laps + 1):
            lap = start_lap + age - 1
            rows.append((lap, age, base + deg * age + 0.032 * fuel_mass(lap, race_laps)))
        return {'compound': compound, 'laps': rows, 'length': laps}

    def test_stint_degradation_is_fuel_corrected(self):
        slope = stint_degradation(self._stint('MEDIUM', 2, 20, 0.07), 52)
        self.assertAlmostEqual(slope, 0.07, places=3)

    def test_summary_feeds_the_profile(self):
        races = [{'name': f'20{y} British Grand Prix', 'race_laps': 52, 'rain': y == 21,
                  'safety_car': True, 'pit_losses': [20.5, 21.0, 19.8], 'stops': [1, 1, 2],
                  'stints': [self._stint('MEDIUM', 2, 20, 0.08), self._stint('HARD', 22, 30, 0.04),
                             self._stint('SOFT', 2, 12, 0.13)]}
                 for y in (19, 21, 22, 23)]
        summary = summarize_races(races)
        self.assertAlmostEqual(summary['deg_s_per_lap']['MEDIUM'], 0.08, places=3)
        self.assertEqual(summary['pit_loss_s'], 20.5)
        self.assertEqual(summary['typical_stops'], 1)
        self.assertEqual(summary['rain_probability'], 0.25)
        self.assertEqual(summary['sc_probability'], 1.0)
        profile = get_profile(7, {'tracks': {'7': summary}})
        self.assertEqual(profile['source'], 'fastf1')
        self.assertEqual(profile['pit_loss_s'], 20.5)
        self.assertEqual(get_profile(7, {})['source'], 'pirelli_baseline')


class FastF1ConversionTests(unittest.TestCase):
    def test_race_from_session_converts_fastf1_laps(self):
        try:
            import pandas as pd
        except ImportError:
            self.skipTest('pandas not installed')
        sys.path.insert(0, str(ROOT / 'tools'))
        from build_real_race_calibration import race_from_session

        rows = []
        for driver in ('VER', 'HAM'):
            for lap in range(1, 31):
                stint = 1 if lap <= 15 else 2
                age = lap if stint == 1 else lap - 15
                pit_in = pd.Timedelta(seconds=100) if lap == 15 else pd.NaT
                pit_out = pd.Timedelta(seconds=120) if lap == 16 else pd.NaT
                extra = 12.0 if lap in (15, 16) else 0.0
                rows.append({'Driver': driver, 'LapNumber': lap, 'Stint': stint,
                             'Compound': 'MEDIUM' if stint == 1 else 'HARD', 'TyreLife': float(age),
                             'LapTime': pd.Timedelta(seconds=90 + 0.05 * age + extra),
                             'PitInTime': pit_in, 'PitOutTime': pit_out,
                             'TrackStatus': '4' if lap == 10 else '1'})

        class Session:
            laps = pd.DataFrame(rows)
            weather_data = pd.DataFrame({'Rainfall': [False, False]})

        race = race_from_session(Session(), '2021 Test Grand Prix')
        self.assertEqual(race['race_laps'], 30)
        self.assertTrue(race['safety_car'])
        self.assertFalse(race['rain'])
        self.assertEqual(race['stops'], [1, 1])
        self.assertEqual(len(race['stints']), 4)
        self.assertTrue(all(10 <= loss <= 45 for loss in race['pit_losses']))
        self.assertNotIn(10, [lap for lap, _a, _t in race['stints'][0]['laps']])   # SC lap dropped


class HudEndpointTests(unittest.TestCase):
    def setUp(self):
        import web_hud
        self.web_hud = web_hud
        self.server = web_hud.start_web_hud_server(port=0)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(web_hud.set_setup_ai_client, None)
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=5) as response:
            return json.loads(response.read())

    def post(self, path, body):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method='POST',
                                         headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read())

    def test_setup_endpoint_returns_track_and_weather_specific_analysis(self):
        self.web_hud.set_setup_ai_client(None)
        dry = self.get('/api/setup?track_id=5&weather=dry')
        wet = self.get('/api/setup?track_id=5&weather=wet&race_fraction=0.5')
        bahrain = self.get('/api/setup?track_id=3&weather=dry')
        self.assertEqual(dry['tyre_degradation'], 'Low')
        self.assertNotEqual(dry['tyre_degradation'], bahrain['tyre_degradation'])
        self.assertNotEqual(dry['setup']['suspension'], wet['setup']['suspension'])
        self.assertEqual(wet['strategy']['race_laps'], 39)
        self.assertFalse(dry['ai_available'])

    def test_analyze_endpoint_uses_deepseek(self):
        class Client:
            timeout = 15.0

            def chat_json(self, system, user, **kw):
                return {'tyre_degradation': 'High', 'briefing': 'DeepSeek says hi.'}

        self.web_hud.set_setup_ai_client(Client())
        result = self.post('/api/track/analyze', {'track_id': 13, 'weather': 'wet'})
        self.assertEqual(result['analysis_source'], 'deepseek')
        self.assertEqual(result['engineer_advice'], 'DeepSeek says hi.')
        cached = self.get('/api/setup?track_id=13&weather=wet')
        self.assertEqual(cached['analysis_source'], 'deepseek')
        self.assertTrue(cached['ai_available'])


if __name__ == '__main__':
    unittest.main()
