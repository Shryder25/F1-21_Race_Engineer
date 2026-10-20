"""Tests for the Monte Carlo race simulator (race_sim.py).

The sandbox has no recorded F1 2021 races, so these check the simulator's mechanics on
made-up fields: the odds add up, the Safety Car rate follows the track profile, overtaking
is harder where it should be, a cheap stop under the Safety Car pays, a clearly worse plan
loses, and the live wiring (field tracker, analysis engine, AI context, voice) works.
Real recordings would let the SC, overtaking and pit-loss constants be fitted instead.
"""

import pathlib
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import race_sim as sim
import race_strategy as rs
from analysis import AnalysisEngine
from engineer_brain import format_context, rule_based_answer
from packets import CarStatusData, SessionData, parse_player_lap_data
from simulate_telemetry import build_lap_packet
from test_analysis import SilentVoice
from voice_commands import match_command

MEDIUM = 17
USED = ('MEDIUM', 'HARD')      # two dry compounds already run: no forced stop
SILVERSTONE = 7


def field(n=10, player=4, compound='MEDIUM', age=10, gap_step=1.5, pace_step=0.1):
    return [sim.SimCar(i, i + 1, i * gap_step, compound, age, 90.0 + pace_step * i, 5, 60.0, (),
                       f'D{i}', i == player) for i in range(n)]


def candidates(laps_to_go, compound='MEDIUM', age=10, current_lap=11, track=SILVERSTONE):
    plan = rs.plan_race(laps_to_go, compound, age, current_lap, rs.pit_loss_s(track))
    return sim.candidates_from_plan(plan)


class OutcomeShapeTests(unittest.TestCase):
    def setUp(self):
        self.state = sim.RaceState(SILVERSTONE, 52, 11, field())
        self.cands = candidates(42)
        self.result = sim.simulate(self.state, self.cands, runs=120, seed=3)

    def test_probabilities_add_up(self):
        self.assertAlmostEqual(sum(o.p_best for o in self.result.outcomes), 1.0, places=2)
        for o in self.result.outcomes:
            self.assertAlmostEqual(sum(o.position_probs), 1.0, places=2)
            self.assertEqual(len(o.position_probs), 10)
            lo, mid, hi = o.position_range
            self.assertTrue(1 <= lo <= mid <= hi <= 10)
            self.assertLessEqual(o.p_win, o.p_podium)
            self.assertLessEqual(o.p_podium, o.p_points)
        self.assertEqual(len(self.result.field), 10)
        self.assertAlmostEqual(sum(f['p_win'] for f in self.result.field), 1.0, places=2)

    def test_one_outcome_per_plan_best_first(self):
        self.assertEqual(len(self.result.outcomes), len(self.cands))
        self.assertEqual([o.p_best for o in self.result.outcomes],
                         sorted((o.p_best for o in self.result.outcomes), reverse=True))
        self.assertIs(self.result.recommended, self.result.outcomes[0])

    def test_same_seed_same_answer(self):
        again = sim.simulate(self.state, self.cands, runs=120, seed=3)
        self.assertEqual(again.as_dict(), self.result.as_dict())

    def test_radio_line_quotes_the_odds(self):
        top = self.result.outcomes[0]
        self.assertIn(f'{round(top.p_best * 100)}% of the time', self.result.radio)
        self.assertIn(f'P{top.position_range[1]}', self.result.radio)
        self.assertTrue(self.result.radio.startswith(top.label.split(' (')[0]))

    def test_summary_is_json_friendly(self):
        import json
        summary = self.result.summary()
        json.dumps(summary)
        json.dumps(self.result.as_dict())
        self.assertEqual(summary['recommended'], self.result.outcomes[0].label)
        self.assertEqual(summary['runs'], 120)

    def test_labels(self):
        self.assertEqual(sim.plan_label(2, ['MEDIUM', 'HARD', 'SOFT']), 'Two-stop (M-H-S)')
        self.assertEqual(sim.plan_label(0, ['HARD']), 'No-stop (H)')

    def test_needs_a_player_and_a_plan(self):
        with self.assertRaises(ValueError):
            sim.RaceSimulator(sim.RaceState(SILVERSTONE, 52, 11, field(player=-1)))
        with self.assertRaises(ValueError):
            sim.simulate(self.state, [], runs=10)


class SafetyCarAndRainTests(unittest.TestCase):
    def neutral(self, profile, laps=50, current=1, runs=4000, safety_car=0):
        state = sim.RaceState(SILVERSTONE, laps, current, field(3, player=0), safety_car=safety_car)
        return sim.RaceSimulator(state, runs=runs, seed=11, profile=profile)

    def test_safety_car_rate_matches_track_profile_over_a_full_race(self):
        for p in (0.3, 0.6, 0.9):
            s = self.neutral({'sc_probability': p, 'vsc_probability': 0.0})
            observed = (s.status == 1).any(axis=1).mean()
            self.assertAlmostEqual(observed, p, delta=0.05, msg=p)
            self.assertFalse((s.status == 2).any())

    def test_vsc_defaults_to_a_share_of_the_sc_chance(self):
        s = self.neutral({'sc_probability': 0.5})
        observed = (s.status == 2).any(axis=1).mean()
        # VSCs can't start during a Safety Car, so a little under the nominal rate.
        self.assertAlmostEqual(observed, sim.VSC_SHARE_OF_SC * 0.5, delta=0.07)

    def test_fewer_laps_left_fewer_safety_cars(self):
        full = (self.neutral({'sc_probability': 0.6}).status == 1).any(axis=1).mean()
        late = (self.neutral({'sc_probability': 0.6}, current=45).status == 1).any(axis=1).mean()
        self.assertLess(late, full / 3)

    def test_lap_one_is_the_likeliest_lap(self):
        s = self.neutral({'sc_probability': 0.6, 'vsc_probability': 0.0})
        starts = (s.status[:, 0] == 1).mean()
        later = (s.status[:, 20] == 1).mean() / 4      # an SC covers ~4 laps
        self.assertGreater(starts, 2 * later)

    def test_safety_car_out_now_starts_every_run_neutralised(self):
        s = self.neutral({'sc_probability': 0.0, 'vsc_probability': 0.0}, current=20, safety_car=1)
        self.assertTrue((s.status[:, 0] == 1).all())
        self.assertFalse((s.status[:, 5:] != 0).any())

    def test_rain_chance_scales_with_laps_left(self):
        state = sim.RaceState(SILVERSTONE, 50, 26, field(3, player=0))
        s = sim.RaceSimulator(state, runs=4000, seed=2, profile={'rain_probability': 0.4, 'sc_probability': 0})
        self.assertAlmostEqual((s.rain_at >= 0).mean(), 0.4 * 25 / 50, delta=0.03)
        dry = sim.RaceSimulator(replace(state, raining=True), runs=500, seed=2, profile={'rain_probability': 1.0})
        self.assertFalse((dry.rain_at >= 0).any())       # already wet: no new shower
        off = sim.RaceSimulator(state, runs=500, seed=2, profile={'rain_probability': 1.0}, rain=False)
        self.assertFalse((off.rain_at >= 0).any())

    def test_rain_sends_everyone_to_inters(self):
        state = sim.RaceState(SILVERSTONE, 30, 11, field(4, player=0))
        s = sim.RaceSimulator(state, runs=50, seed=5, profile={'rain_probability': 1.0, 'sc_probability': 0})
        # With rain certain, a plan that stays on slicks to the flag must lose to the field's
        # forced stop only by its own stop: everyone pays it, so finishing order stays sane.
        result = s.evaluate(candidates(20, current_lap=11))
        self.assertGreater(result.p_rain, 0.5)
        self.assertAlmostEqual(sum(o.p_best for o in result.outcomes), 1.0, places=2)

    def test_cheap_stop_under_safety_car_pays(self):
        """Boxing now under a Safety Car beats boxing now under green, by roughly the saving."""
        cars = field(6, player=2, age=18)
        one = [sim.Candidate('One-stop (M-H)', 1, ['MEDIUM', 'HARD'], [20])]
        green = sim.RaceSimulator(sim.RaceState(SILVERSTONE, 40, 20, cars), runs=200, seed=9,
                                  profile={'sc_probability': 0.0, 'overtaking': 3}).evaluate(one)
        sc = sim.RaceSimulator(sim.RaceState(SILVERSTONE, 40, 20, cars, safety_car=1), runs=200, seed=9,
                               profile={'sc_probability': 0.0, 'overtaking': 3}).evaluate(one)
        self.assertLess(sc.outcomes[0].mean_position, green.outcomes[0].mean_position)

    def test_safety_car_brings_planned_stops_forward(self):
        """Under a fresh SC, a stop planned within SC_PIT_WINDOW_LAPS is taken straight away."""
        cars = field(2, player=0)
        state = sim.RaceState(SILVERSTONE, 40, 20, cars, safety_car=1)
        s = sim.RaceSimulator(state, runs=1, seed=1, profile={'sc_probability': 0.0})
        s.status[:] = 0
        s.status[0, 0] = 0
        s.status[0, 1] = 1                                # SC starts on lap 21
        s.pit_loss[:] = 20.0
        s.noise[:] = 0.0
        later = sim.Candidate('One-stop', 1, ['MEDIUM', 'HARD'], [25])
        never = sim.Candidate('No-stop', 0, ['MEDIUM'], [])
        t_later = s._simulate(later)[0, 0]
        t_never = s._simulate(never)[0, 0]
        # The stop was taken under the SC (55% of 20 s), not at green-flag cost on lap 25.
        self.assertLess(t_later - t_never, 20.0 * 0.8 + 5.0)


class PitLossTests(unittest.TestCase):
    def test_pit_loss_is_a_distribution_around_the_track_figure(self):
        state = sim.RaceState(12, 61, 10, field(5, player=0))      # Singapore, long pit lane
        s = sim.RaceSimulator(state, runs=400, seed=4)
        loss = s.pit_loss.ravel()
        self.assertAlmostEqual(float(loss.mean()), rs.pit_loss_s(12) + sim.SLOW_STOP_PROB * sim.SLOW_STOP_MEAN_S,
                               delta=0.15)
        self.assertGreater(float(loss.std()), sim.PIT_LOSS_SD_S * 0.9)
        self.assertGreater(float(loss.max()), rs.pit_loss_s(12) + 5)      # the odd slow stop
        result = s.evaluate(candidates(52, current_lap=10, track=12))
        lo, mean, hi = result.pit_loss_s
        self.assertLess(lo, mean)
        self.assertLess(mean, hi)


class TrafficAndOvertakingTests(unittest.TestCase):
    def pass_rate(self, overtaking, advantage=0.8, laps=10):
        """Player 1 s behind a car `advantage` s/lap slower: how often do we finish ahead?"""
        cars = [sim.SimCar(0, 1, 0.0, 'HARD', 5, 90.0 + advantage, 5, 30.0, USED, 'Slow'),
                sim.SimCar(1, 2, 1.0, 'HARD', 5, 90.0, 5, 30.0, USED, 'Us', True)]
        state = sim.RaceState(SILVERSTONE, 30, 30 - laps + 1, cars)
        no_stop = [sim.Candidate('No-stop (H)', 0, ['HARD'], [])]
        s = sim.RaceSimulator(state, runs=400, seed=7, profile={'sc_probability': 0.0, 'overtaking': overtaking})
        return s.evaluate(no_stop).outcomes[0].p_win

    def test_overtaking_is_harder_on_harder_tracks(self):
        easy, hard = self.pass_rate(1), self.pass_rate(5)
        self.assertGreater(easy, 0.8)
        self.assertLess(hard, 0.5)
        self.assertGreater(easy, hard + 0.3)

    def test_bigger_pace_gap_passes_more_often(self):
        self.assertGreater(self.pass_rate(4, advantage=2.0), self.pass_rate(4, advantage=0.6))

    def test_being_held_up_costs_time(self):
        """Stuck behind a slower car at Monaco, we cross the line just behind it, lap after lap."""
        cars = [sim.SimCar(0, 1, 0.0, 'HARD', 5, 91.0, 5, 30.0, USED, 'Slow'),
                sim.SimCar(1, 2, 0.5, 'HARD', 5, 90.0, 5, 30.0, USED, 'Us', True)]
        state = sim.RaceState(5, 78, 70, cars)
        s = sim.RaceSimulator(state, runs=200, seed=1, profile={'sc_probability': 0.0, 'overtaking': 5})
        s.noise[:] = 0.0
        s.form[:] = 0.0
        s.u_pass[:] = 0.999                                    # no pass ever succeeds
        s.z_slope[:] = 0.0                                     # identical tyres: no cliff surprises
        s.z_life[:] = 0.0
        s.rival_laps[:] = sim.NEVER                            # and the slow car never boxes
        T = s._simulate(sim.Candidate('No-stop (H)', 0, ['HARD'], []))
        self.assertTrue((T[:, 1] > T[:, 0]).all())
        self.assertTrue(((T[:, 1] - T[:, 0]) <= sim.MIN_GAP_S + 1e-6).all())

    def test_car_on_its_in_lap_does_not_block(self):
        """A slower car ahead boxes: the car behind is not held up by it and comes out ahead."""
        cars = [sim.SimCar(0, 1, 0.0, 'HARD', 5, 91.0, 5, 30.0, USED, 'Slow'),
                sim.SimCar(1, 2, 0.5, 'HARD', 5, 90.0, 5, 30.0, USED, 'Us', True)]
        state = sim.RaceState(5, 78, 70, cars)
        s = sim.RaceSimulator(state, runs=20, seed=1, profile={'sc_probability': 0.0, 'overtaking': 5})
        s.u_pass[:] = 0.999
        s.rival_laps[:] = sim.NEVER
        s.rival_laps[:, 0, 0] = 70
        s.rival_comps[:, 0, 0] = sim._CI['MEDIUM']
        T = s._simulate(sim.Candidate('No-stop (H)', 0, ['HARD'], []))
        self.assertTrue((T[:, 1] < T[:, 0]).all())


class StrategyChoiceTests(unittest.TestCase):
    def test_clearly_better_plan_wins_most_runs(self):
        """Old softs with 30 laps to go: staying out falls off the cliff, stopping wins."""
        cars = field(6, player=2, compound='SOFT', age=15)
        state = sim.RaceState(SILVERSTONE, 50, 21, cars)
        stay = sim.Candidate('No-stop (S)', 0, ['SOFT'], [])
        stop = sim.Candidate('One-stop (S-H)', 1, ['SOFT', 'HARD'], [21])
        result = sim.simulate(state, [stay, stop], runs=200, seed=5)
        self.assertEqual(result.recommended.label, 'One-stop (S-H)')
        self.assertGreater(result.recommended.p_best, 0.85)

    def test_common_random_numbers_identical_plans_tie(self):
        state = sim.RaceState(SILVERSTONE, 52, 11, field())
        plan = candidates(42)[0]
        twin = sim.Candidate('Twin', plan.stops, plan.compounds, plan.pit_laps)
        result = sim.simulate(state, [plan, twin], runs=50, seed=1)
        self.assertEqual([o.p_best for o in result.outcomes], [0.5, 0.5])
        self.assertEqual(result.outcomes[0].position_probs, result.outcomes[1].position_probs)

    def test_player_calibration_changes_the_answer(self):
        """If our tyres fall off three times faster than the model, more stops look better."""
        state = sim.RaceState(SILVERSTONE, 52, 11, field())
        cands = candidates(42)
        normal = sim.simulate(state, cands, runs=150, seed=2)
        fragile = sim.simulate(state, cands, runs=150, seed=2, player_deg_scale=3.0, player_life_scale=0.6)
        stops = lambda r: sum(o.p_best * o.stops for o in r.outcomes)
        self.assertGreater(stops(fragile), stops(normal))

    def test_fitted_models_feed_the_tyre_inputs(self):
        import fitted_models as fm
        store = fm.ModelStore()
        prior = store.deg_model(SILVERSTONE, 'C3')
        store.deg[f'{SILVERSTONE}/C3'] = replace(prior, mean=0.2, var=1e-6, tau2=1e-6, basis='track')
        s = sim.RaceSimulator(sim.RaceState(SILVERSTONE, 52, 11, field()), models=store, runs=5, seed=1)
        self.assertAlmostEqual(s.slope_mu[sim._CI['MEDIUM']], 0.2)
        self.assertEqual(s.deg_basis['MEDIUM'], 'track')
        self.assertEqual(s.deg_basis['SOFT'], 'prior')

    def test_rival_predictions_drive_rival_stops(self):
        cars = field(4, player=0)
        cars[1].prediction = SimpleNamespace(stop_lap_mean=30.0, stop_lap_sd=0.5, will_stop=True,
                                             next_compound_probs={'HARD': 1.0})
        cars[2].prediction = SimpleNamespace(stop_lap_mean=30.0, stop_lap_sd=0.5, will_stop=False,
                                             next_compound_probs={})
        s = sim.RaceSimulator(sim.RaceState(SILVERSTONE, 52, 11, cars), runs=300, seed=1)
        first = s.rival_laps[:, 1, 0]
        self.assertAlmostEqual(float(first.mean()), 30.0, delta=0.3)
        self.assertTrue((s.rival_comps[:, 1, 0] == sim._CI['HARD']).all())
        self.assertTrue((s.rival_laps[:, 2, :] == sim.NEVER).all())
        self.assertTrue((s.rival_laps[:, 0, :] == sim.NEVER).all())     # the player has no rival plan
        jittered = s.rival_laps[:, 3, 0]
        self.assertGreater(float(jittered.std()), 0.5)                    # modelled plan + jitter

    def test_compound_allocation_from_one_known_set(self):
        alloc = sim.compound_allocation([('Medium', 'C2')])
        self.assertEqual((alloc['SOFT'], alloc['MEDIUM'], alloc['HARD']), ('C3', 'C2', 'C1'))
        self.assertEqual(sim.compound_allocation([])['MEDIUM'], 'C3')
        self.assertEqual(sim.compound_allocation([('Soft', 'C5')])['HARD'], 'C3')


def lap(position, lap_num, last_ms=90000, pit_status=0, stops=0):
    base = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)
    return replace(base, car_position=position, current_lap_num=lap_num, last_lap_time_ms=last_ms,
                   pit_status=pit_status, num_pit_stops=stops, result_status=2)


def status(age=5, compound=MEDIUM, fuel=50.0):
    return CarStatusData(fuel, 110.0, fuel / 1.6, compound, compound, age, 0, 2_000_000.0, 1, 0.0)


class FieldTrackerTests(unittest.TestCase):
    def test_gaps_and_clean_pace_skip_in_and_out_laps(self):
        tracker = sim.FieldTracker()
        t = 0.0
        tracker.update([lap(1, 1), lap(2, 1)], t, [status(), status()])
        for n in range(2, 8):
            t += 90.0
            pit = n == 5                                  # car 1 boxes on lap 5
            tracker.update([lap(1, n, 90000), lap(2, n - 1, 91000, pit_status=1 if pit else 0,
                                                   stops=1 if n >= 5 else 0)], t, [status(n), status(n)])
            tracker.update([lap(1, n, 90000), lap(2, n, 91000 if n not in (5, 6) else 112000,
                                                   stops=1 if n >= 5 else 0)], t + 1.5, [status(n), status(n)])
        gaps = tracker.gap_to_leader([lap(1, 7), lap(2, 7)])
        self.assertAlmostEqual(gaps[0], 0.0)
        self.assertAlmostEqual(gaps[1], 1.5)
        pace0, n0 = tracker.pace(0, 0.0)
        self.assertEqual(n0, sim.CLEAN_LAPS_KEPT if n0 >= sim.CLEAN_LAPS_KEPT else n0)
        pace1, n1 = tracker.pace(1, 0.0)
        # Lap 1 (standing start) is never used; car 1's in-lap and out-lap are left out.
        self.assertLess(n1, n0)
        self.assertLess(pace1 + 50 * 0.03, 92.0)

    def test_flashback_resets(self):
        tracker = sim.FieldTracker()
        tracker.update([lap(1, 3)], 300.0)
        tracker.update([lap(1, 4)], 390.0)
        tracker.update([lap(1, 2)], 100.0)
        self.assertEqual(tracker.crossings, {})


RACE = SessionData(0, 30, 20, 52, 5891, 10, SILVERSTONE, 0)


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.engine = AnalysisEngine(SilentVoice(), race_simulation='sync', sim_runs=60)
        self.engine.update_session(RACE, 1, 0)
        laps = [lap(i + 1, 20) for i in range(6)]
        self.engine.update_car_status(status(12), [status(12 + i) for i in range(6)])
        self.engine.update_field(laps, 2, 1000.0)
        self.engine.latest_lap = laps[2]

    def test_simulation_feeds_strategy_summary_context_and_voice(self):
        self.engine.update_race_simulation()
        summary = self.engine.strategy_summary()
        sim_summary = summary['simulation']
        self.assertIsNotNone(sim_summary)
        self.assertEqual(sim_summary['runs'], 60)
        self.assertEqual(len(sim_summary['options']), len(summary['alternatives']) + 1)
        self.assertIn('% of the time', summary['radio'])
        text = format_context(self.engine.build_context())
        self.assertIn('Race simulation (60 Monte Carlo races from lap 20)', text)
        answer = rule_based_answer('strategy', {'session_mode': 'race', 'strategy': summary})
        self.assertIn('comes out best', answer)

    def test_runs_once_per_lap(self):
        self.engine.update_race_simulation()
        first = self.engine.sim_runner.latest
        self.engine.update_race_simulation()
        self.assertIs(self.engine.sim_runner.latest, first)

    def test_off_by_default_and_cleared_on_new_session(self):
        plain = AnalysisEngine(SilentVoice())
        self.assertFalse(plain.sim_runner.enabled)
        self.engine.update_race_simulation()
        self.engine.update_session(replace(RACE, track_id=11), 2, 0)
        self.assertIsNone(self.engine.sim_runner.latest)
        self.assertEqual(self.engine.field_tracker.crossings, {})

    def test_stale_result_is_not_quoted(self):
        self.engine.update_race_simulation()
        self.engine.latest_lap = replace(self.engine.latest_lap, current_lap_num=25)
        self.assertIsNone(self.engine.race_simulation())

    def test_async_runner_delivers_a_result(self):
        runner = sim.SimRunner('async', runs=30)
        state = sim.RaceState(SILVERSTONE, 52, 11, field())
        self.assertTrue(runner.submit('k', state, candidates(42), seed=1))
        for _ in range(200):
            if runner.latest is not None:
                break
            import time
            time.sleep(0.02)
        self.assertIsNotNone(runner.latest)
        self.assertFalse(runner.submit('k', state, candidates(42)))       # same key: no re-run

    def test_runner_survives_a_bad_state(self):
        runner = sim.SimRunner('sync')
        bad = sim.RaceState(SILVERSTONE, 52, 11, field(player=-1))
        runner.submit('x', bad, candidates(42))
        self.assertIsNone(runner.latest)
        self.assertIn('ValueError', runner.error)

    def test_voice_phrases(self):
        for phrase in ('what are our chances', 'what are the odds', 'chances of a podium'):
            self.assertEqual(match_command(phrase), 'strategy', phrase)
        self.assertEqual(match_command('is it going to rain'), 'weather')


if __name__ == '__main__':
    unittest.main()
