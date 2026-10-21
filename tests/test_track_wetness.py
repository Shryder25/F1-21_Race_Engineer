"""Tests for the track wetness model (track_wetness.py), the weather forecast parser, and how
the simulator and the Plan B logic use wetness: soaking and drying physics, the tyre crossovers,
the lap-time readings, rivals switching tyre type, forecast rain, and the drying-track calls.
"""

import pathlib
import struct
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import race_sim as sim
import strategy_replan as rp
import track_wetness as tw
from analysis import AnalysisEngine
from engineer_brain import format_context, rule_based_answer
from packets import (HEADER_SIZE, WEATHER_SAMPLE_FORMAT, WeatherSample, parse_session_data)
from simulate_telemetry import build_session_packet
from test_analysis import SilentVoice
from test_race_sim import RACE, SILVERSTONE, field, lap, status

INTER, WET_TYRE = 7, 8


def sample(offset, weather, pct, session_type=10):
    return WeatherSample(session_type, offset, weather, 30, 2, 22, 2, pct)


class PhysicsTests(unittest.TestCase):
    def test_crossovers_where_f1_sees_them(self):
        self.assertAlmostEqual(tw.crossover('slick', 'inter'), 0.2, places=2)
        self.assertAlmostEqual(tw.crossover('inter', 'wet'), 0.7, places=2)
        self.assertEqual(tw.best_family(0.05), 'slick')
        self.assertEqual(tw.best_family(0.4), 'inter')
        self.assertEqual(tw.best_family(0.9), 'wet')
        self.assertAlmostEqual(float(tw.penalty('inter', 0.0, 45.0)), 2.0)    # scales with lap length

    def test_rain_soaks_and_dries(self):
        w = tw.step(0.0, 10.0, 4, 30)
        self.assertGreater(w, 0.75)                                  # heavy rain soaks in minutes
        self.assertLessEqual(w, tw.WET_TARGET[4])
        self.assertAlmostEqual(tw.step(0.5, 10.0, 1, 30), 0.5 - 0.3, places=6)
        self.assertLess(tw.step(0.5, 10.0, 0, 30), tw.step(0.5, 10.0, 2, 30))      # sun dries faster than cloud
        self.assertLess(tw.step(0.5, 10.0, 1, 45), tw.step(0.5, 10.0, 1, 20))      # hot track dries faster
        self.assertEqual(tw.step(0.1, 60.0, 1, 30), 0.0)
        self.assertLess(tw.step(0.9, 10.0, 3, 30), 0.9)                  # lighter rain drains a soaked track
        self.assertGreaterEqual(tw.step(0.9, 100.0, 3, 30), tw.WET_TARGET[3])

    def test_family(self):
        self.assertEqual([tw.family(c) for c in ('Soft', 'C3', 'Inter', 'Intermediate', 'Wet', None)],
                         ['slick', 'slick', 'inter', 'inter', 'wet', None])


class EstimatorTests(unittest.TestCase):
    def test_starts_from_the_weather_and_follows_it(self):
        t = tw.TrackWetness()
        t.update_weather(4, 25, 0.0)
        self.assertEqual(t.wetness, tw.WET_TARGET[4])
        t.update_weather(1, 25, 60.0)                    # rain stops
        t.update_weather(1, 25, 660.0)                   # ten minutes later
        self.assertLess(t.wetness, 0.6)
        self.assertGreater(t.sd, tw.START_SD[True] - 1e-9)
        dry = tw.TrackWetness()
        dry.update_weather(0, 40, 0.0)
        self.assertEqual(dry.wetness, 0.0)
        self.assertIsNone(dry.radio('slick'))

    def test_trajectory_uses_the_forecast(self):
        t = tw.TrackWetness()
        t.update_weather(1, 30, 0.0, (sample(0, 1, 0), sample(5, 4, 80)))
        traj = t.trajectory(10, 90.0)
        self.assertEqual(traj[0], 0.0)
        self.assertGreater(traj[-1], 0.5)                # heavy rain due in five minutes
        self.assertEqual(t.weather_at(2), 1)
        self.assertEqual(t.weather_at(6), 4)
        t2 = tw.TrackWetness()
        t2.update_weather(1, 30, 0.0, (sample(0, 1, 0), sample(5, 2, 20)))
        self.assertEqual(max(t2.trajectory(10, 90.0)), 0.0)    # 20% chance: plan for dry

    def test_laps_until_slicks_on_a_drying_track(self):
        t = tw.TrackWetness()
        t.update_weather(3, 30, 0.0)
        t.wetness = 0.4
        t.update_weather(1, 30, 1.0)
        k = t.laps_until('slick', 'inter')
        self.assertTrue(4 <= k <= 7, k)                 # 0.4 -> 0.2 at 0.03/min, 1.5-min laps
        s = t.summary('inter')
        self.assertEqual((s['switch_to'], s['switch_in_laps']), ('slick', k))
        self.assertIn(f'Slicks quicker in about {k} laps.', t.radio('inter'))

    def _feed(self, t, laps, now):
        for car, (fam, lap_s) in enumerate(laps):
            t.add_lap(car, fam, lap_s, now)
        t.assimilate(now)

    def test_slicks_quicker_than_inters_pins_down_a_drying_track(self):
        t = tw.TrackWetness()
        t.update_weather(3, 30, 0.0)
        t.wetness, t.sd = 0.35, 0.1                     # physics says still past the crossover
        t.update_weather(1, 30, 1.0)
        self._feed(t, [('slick', 91.0), ('slick', 91.2), ('slick', 90.9),
                       ('inter', 93.0), ('inter', 93.1), ('inter', 92.8)], 100.0)
        self.assertLess(t.wetness, 0.2)                 # two seconds quicker on slicks: past it
        self.assertEqual(t.basis, 'slick vs inter lap times')
        self.assertEqual(tw.best_family(t.wetness), 'slick')

    def test_lap_times_cannot_make_a_track_wetter_without_rain(self):
        t = tw.TrackWetness()
        t.update_weather(0, 38, 0.0)
        for lap_s in (90.0, 90.1, 89.9, 90.2):
            for car in range(4):
                t.add_lap(car, 'slick', lap_s, 1.0)
        self._feed(t, [('slick', 92.5)] * 6, 200.0)     # heavy-fuel or traffic laps
        self.assertEqual(t.wetness, 0.0)
        damp = tw.TrackWetness()
        damp.update_weather(3, 30, 0.0)
        damp.wetness = 0.25
        damp.update_weather(1, 30, 1.0)
        self._feed(damp, [('slick', 100.0), ('slick', 100.0), ('inter', 90.0), ('inter', 90.0)], 50.0)
        self.assertLessEqual(damp.wetness, 0.25 + 1e-9)  # says "wetter", physics says no rain: capped

    def test_each_car_against_its_own_dry_laps(self):
        t = tw.TrackWetness()
        t.update_weather(0, 30, 0.0)
        for car, base in enumerate((89.0, 90.0, 91.0, 92.0)):
            for _ in range(3):
                t.add_lap(car, 'slick', base, 1.0)
        t.update_weather(4, 30, 2.0)
        t.wetness, t.sd = 0.5, 0.2
        # Everyone on inters now, 8.5 s off their own dry pace -> about half wet.
        self._feed(t, [('inter', b + 8.5) for b in (89.0, 90.0, 91.0, 92.0)], 300.0)
        self.assertEqual(t.basis, "inter lap times vs each car's dry laps")
        self.assertAlmostEqual(t.wetness, 0.5, delta=0.08)

    def test_observe_field_uses_clean_laps_only(self):
        t = tw.TrackWetness()
        t.update_weather(3, 30, 0.0)
        statuses = [status(5, INTER), status(5, INTER)]
        t.observe_field([lap(1, 2), lap(2, 2)], statuses, 0.0)
        t.observe_field([lap(1, 3, 95000), lap(2, 3, 96000, stops=1)], statuses, 95.0)   # car 2 pitted
        t.observe_field([lap(1, 4, 95000), lap(2, 4, 99000)], statuses, 190.0)           # car 2 out-lap
        cars = [r[1] for r in t.records]
        # Lap 3 was joined mid-lap (no reading), car 2's in-lap and out-lap are left out.
        self.assertEqual(cars, [0])
        self.assertEqual(t.records[0][2], 'inter')


class ForecastPacketTests(unittest.TestCase):
    def test_parses_this_sessions_samples(self):
        data = bytearray(build_session_packet(1, 1))
        data[HEADER_SIZE + 126] = 3
        for i, (st, off, w, pct) in enumerate([(10, 10, 4, 90), (10, 0, 2, 10), (12, 0, 0, 0)]):
            struct.pack_into(WEATHER_SAMPLE_FORMAT, data, HEADER_SIZE + 127 + 8 * i, st, off, w, 30, 2, 22, 2, pct)
        forecast = parse_session_data(bytes(data)).weather_forecast
        self.assertEqual([(s.offset_min, s.weather, s.rain_percentage) for s in forecast], [(0, 2, 10), (10, 4, 90)])
        self.assertEqual(parse_session_data(build_session_packet(1, 1)).weather_forecast, ())
        self.assertEqual(tw.rain_probability(sample(0, 4, 20)), 0.75)
        self.assertEqual(tw.rain_probability(sample(0, 2, 40)), 0.4)


def drying_state(**kw):
    cars = field(10, player=4, compound='INTERMEDIATE', age=8)
    base = dict(wetness=0.4, wetness_sd=0.05, weather=1, track_temp=30, lap_s=90.0)
    base.update(kw)
    return sim.RaceState(SILVERSTONE, 40, 16, cars, **base)


class SimulatorConditionsTests(unittest.TestCase):
    def test_best_switch_to_slicks_is_at_the_crossover(self):
        stay = rp._candidate(['INTERMEDIATE'], [])
        switches = [rp._candidate(['INTERMEDIATE', 'MEDIUM'], [k]) for k in (16, 18, 20, 21, 24, 28)]
        result = sim.simulate(drying_state(), [stay, *switches], reference=0, runs=200, seed=1,
                              profile={'sc_probability': 0})
        best = result.recommended
        self.assertIn(best.pit_laps[0], (20, 21))       # 0.4 -> 0.2 takes about 4.5 laps
        by = {tuple(o.pit_laps): o for o in result.outcomes}
        self.assertLess(by[(16,)].vs_reference['time_gain_s'], by[(20,)].vs_reference['time_gain_s'])
        self.assertLess(by[(28,)].vs_reference['time_gain_s'], by[(20,)].vs_reference['time_gain_s'])
        self.assertGreater(by[(20,)].vs_reference['time_gain_s'], 60)   # inters on a dry track are slow

    def test_rivals_change_to_slicks_as_it_dries(self):
        s = sim.RaceSimulator(drying_state(), runs=60, seed=2, profile={'sc_probability': 0})
        self.assertLess(s.W[:, -1].mean(), 0.05)
        self.assertTrue((s.W[:, 1:] <= s.W[:, :-1] + 1e-9).all())      # no rain: only dries
        self.assertFalse(s.player_reacts.any())                          # the plan decides for us
        s._simulate(rp._candidate(['INTERMEDIATE'], []))
        on_slicks = sim.FAM_IDX[s.last_compounds] == 0
        rivals = [i for i in range(s.n) if i != s.player_slot]
        self.assertGreater(on_slicks[:, rivals].mean(), 0.95)           # the field goes to slicks
        self.assertFalse(on_slicks[:, s.player_slot].any())               # we stayed out, as planned

    def test_forecast_rain_drives_the_runs(self):
        dry = drying_state(wetness=0.0, forecast=(sample(0, 1, 0), sample(10, 4, 100)))
        s = sim.RaceSimulator(dry, runs=100, seed=3, profile={'sc_probability': 0, 'rain_probability': 0})
        self.assertEqual(s.p_rain_input, 1.0)
        self.assertTrue((s.rain_at == 7).all())          # 10 min at 1.5-min laps -> lap index 7
        self.assertTrue(s.player_reacts[:, 7:].all() and not s.player_reacts[:, :7].any())
        self.assertGreater(s.W[:, -1].mean(), 0.7)
        none = sim.RaceSimulator(replace(dry, forecast=(sample(0, 1, 0), sample(10, 2, 0))), runs=100, seed=3,
                                 profile={'sc_probability': 0, 'rain_probability': 1.0})
        self.assertEqual(none.p_rain_input, 0.0)         # the game's forecast beats the track average

    def test_shower_without_forecast_ends(self):
        state = drying_state(wetness=0.5, weather=3, raining=True)
        s = sim.RaceSimulator(state, runs=300, seed=4, profile={'sc_probability': 0})
        stopped = (s.weather_code[:, -1] < 3).mean()
        self.assertGreater(stopped, 0.6)                 # 25 laps x 1.5 min vs a 20-min shower


class ReplanConditionsTests(unittest.TestCase):
    def test_switch_candidates_around_the_crossover(self):
        traj = [0.4, 0.35, 0.3, 0.25, 0.21, 0.18, 0.15, 0.1]
        cands = rp.condition_switches(16, 'Inter', traj, 39)
        self.assertEqual(sorted(c.pit_laps[0] for c in cands), [20, 21, 23])
        self.assertTrue(all(c.compounds[0] == 'INTERMEDIATE' and c.compounds[1] in ('MEDIUM', 'HARD') for c in cands))
        now = rp.condition_switches(16, 'Medium', [0.5] * 5, 39)
        self.assertIn(('INTERMEDIATE', 16), [(c.compounds[1], c.pit_laps[0]) for c in now])
        self.assertTrue(any(c.label.endswith('box now') for c in now))
        self.assertEqual(rp.condition_switches(16, 'Medium', [0.0] * 5, 39), [])
        self.assertEqual(rp.dry_compound_for(5), 'SOFT')
        self.assertEqual(rp.dry_compound_for(60), 'HARD')

    def test_crossover_and_first_car_on_slicks_are_events(self):
        r = rp.Replanner()
        r.observe(20, 0, 1, best_tyre='inter', families={1: 'inter', 2: 'inter'})
        self.assertEqual(r.observe(20, 0, 1, best_tyre='inter', families={1: 'slick', 2: 'inter'},
                                   name_of=lambda i: f'D{i}'), ['tyre_gamble'])
        self.assertEqual(r.gamble, ('D1', 'slick'))
        self.assertEqual(r.observe(20, 0, 1, best_tyre='slick'), ['crossover'])
        self.assertEqual(r.events, 2)
        self.assertEqual(r.observe(20, 0, 1, best_tyre='slick', families={0: 'slick'}, player_index=0), [])

    def test_drying_call_wording(self):
        cfg = rp.ReplanConfig()
        a = rp._candidate(['INTERMEDIATE'], [])
        b = rp._candidate(['INTERMEDIATE', 'MEDIUM'], [20], ', box now')
        self.assertEqual(rp.plan_b_message(b, a, 20, 25.0, 2.0, ['crossover'], None, cfg),
                         "Track's ready for slicks. Plan B, Plan B. We go one stop. Box this lap for mediums. "
                         "Box, box. Worth 2 places.")
        later = rp._candidate(['INTERMEDIATE', 'MEDIUM'], [24])
        self.assertTrue(rp.plan_b_message(later, a, 20, 9.0, 0.0, ['crossover'], None, cfg)
                        .startswith("Track's drying. Plan B"))
        self.assertTrue(rp.plan_b_message(b, a, 20, 9.0, 0.0, ['tyre_gamble'], None, cfg, ('Norris', 'slick'))
                        .startswith('Norris has gone to slicks. Plan B'))


class EngineWiringTests(unittest.TestCase):
    def setUp(self):
        self.engine = AnalysisEngine(SilentVoice(), race_simulation='sync', sim_runs=40)

    def test_dry_day_says_nothing_about_wetness(self):
        e = self.engine
        e.update_session(RACE, 1, 0)
        e.update_car_status(status(12), [status(12 + i) for i in range(6)])
        e.latest_lap = lap(3, 20)
        self.assertIsNone(e.conditions_summary())
        self.assertIsNone(e.build_context()['track_conditions'])

    def test_rain_feeds_summary_answer_context_and_sim(self):
        e = self.engine
        wet = replace(RACE, weather=3)                 # light rain: inters weather
        e.update_session(wet, 1, 0)
        laps = [lap(i + 1, 20) for i in range(6)]
        e.update_car_status(status(12), [status(12 + i) for i in range(6)])     # all on slicks in heavy rain
        e.update_field(laps, 2, 1000.0)
        e.latest_lap = laps[2]
        c = e.conditions_summary()
        self.assertEqual(c['best_tyre'], 'inter')
        self.assertEqual((c['switch_to'], c['switch_in_laps']), ('inter', 0))
        ctx = e.build_context()
        self.assertIn('Inters are the quicker tyre now.', rule_based_answer('weather', ctx))
        self.assertIn('Track wetness', format_context(ctx))
        state = e.race_sim_state()
        self.assertEqual(state.wetness, tw.WET_TARGET[3])
        self.assertEqual(state.weather, 3)
        e.update_race_simulation()
        labels = [o.label for o in e.sim_runner.latest.outcomes]
        self.assertTrue(any('box now' in l and o.compounds[1] == 'INTERMEDIATE'
                            for l, o in zip(labels, e.sim_runner.latest.outcomes)), labels)

    def test_reset_on_new_session(self):
        e = self.engine
        e.update_session(replace(RACE, weather=4), 1, 0)
        self.assertIsNotNone(e.wetness.wetness)
        e.update_session(replace(RACE, track_id=11), 2, 0)
        self.assertEqual(e.wetness.wetness, 0.0)


if __name__ == '__main__':
    unittest.main()
