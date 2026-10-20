"""Rival strategy models: next stop lap and compound for every rival, and how strategy uses them."""

import pathlib
import random
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import fitted_models                                        # noqa: E402
import race_strategy as rs                                  # noqa: E402
import rival_strategy as rv                                 # noqa: E402
from analysis import AnalysisEngine                         # noqa: E402
from engineer_brain import format_context, rule_based_answer   # noqa: E402
from packets import CarDamageData, CarStatusData, SessionData, parse_player_lap_data   # noqa: E402
from simulate_telemetry import build_lap_packet             # noqa: E402
from test_analysis import SilentVoice                       # noqa: E402
from voice_commands import match_command                    # noqa: E402

SOFT, MEDIUM, HARD, INTER = 16, 17, 18, 7
ACTUAL = {SOFT: 17, MEDIUM: 18, HARD: 19, INTER: 7}           # C4, C3, C2, inter
SILVERSTONE = dict(track_id=7, track_length=5891)
RACE = SessionData(0, 30, 20, 40, 5891, 10, 7, 0)             # Silverstone, 40-lap race
BASE_LAP = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)


def status(compound=MEDIUM, age=0):
    return CarStatusData(50.0, 110.0, 30.0, ACTUAL[compound], compound, age, 0, 2_000_000.0, 1, 0.0)


def damage(wear):
    return CarDamageData((wear, wear, wear * 0.9, wear * 0.95), (0, 0, 0, 0), 0, 0, 0, 0, 0, 0)


def state(compound='MEDIUM', age=0, lap=None, **kw):
    lap = lap if lap is not None else age + 1
    actual = {'SOFT': 'C4', 'MEDIUM': 'C3', 'HARD': 'C2', 'INTERMEDIATE': 'INTER'}.get(compound)
    return rv.RivalState(car_index=kw.pop('car_index', 1), compound=compound, tyre_age=age, current_lap=lap,
                         actual_compound=actual, **kw)


def ctx(total=40, **kw):
    return rv.RaceContext(total_laps=total, **{**SILVERSTONE, **kw})


def wear_stint(rate, laps, start_age=0):
    """Tyre ages and per-lap wear increments for a stint wearing ``rate`` % a lap."""
    return [start_age + i for i in range(laps)], [rate] * laps


class FieldStatsTests(unittest.TestCase):
    def test_priors_until_cars_stop(self):
        stats = rv.FieldStats()
        self.assertEqual(stats.stop_wear_pct()[0], rv.STOP_WEAR_PRIOR_PCT)
        self.assertEqual(stats.life_factor()[0], 1.0)

    def test_learns_stop_wear_life_and_tyre_choice_from_the_field(self):
        stats = rv.FieldStats()
        for _ in range(6):
            stats.record_stop(45.0, 1.3, 'MEDIUM', 'HARD')
        wear, sd = stats.stop_wear_pct()
        self.assertLess(wear, 52)
        self.assertLess(sd, rv.STOP_WEAR_PRIOR_SD)
        self.assertGreater(stats.life_factor()[0], 1.15)
        self.assertEqual(stats.next_compound_counts('MEDIUM'), {'HARD': 6})

    def test_nonsense_observations_are_ignored_or_clamped(self):
        stats = rv.FieldStats()
        stats.record_stop(2.0, 9.0, 'INTERMEDIATE', 'SOFT')
        self.assertEqual(stats.stop_wear, [])
        self.assertEqual(stats.life_ratio, [2.0])
        self.assertEqual(stats.transitions, {})


class PredictionTests(unittest.TestCase):
    def test_fresh_set_predicts_a_stop_and_a_legal_next_tyre(self):
        p = rv.predict_rival(state('MEDIUM', 0, 1), ctx())
        self.assertTrue(p.will_stop)
        self.assertTrue(p.must_stop)
        low, mid, high = p.stop_lap_range
        self.assertTrue(1 <= low <= mid <= high <= 40)
        self.assertAlmostEqual(sum(p.next_compound_probs.values()), 1.0, places=2)
        self.assertIn(p.next_compound, rs.DRY_COMPOUNDS)
        self.assertEqual(p.basis, 'age')
        self.assertIn('likely to box lap', p.summary)

    def test_older_tyres_box_sooner(self):
        fresh = rv.predict_rival(state('MEDIUM', 2, 10), ctx())
        old = rv.predict_rival(state('MEDIUM', 18, 10), ctx())
        self.assertLess(old.stop_lap, fresh.stop_lap)

    def test_observed_fast_wear_brings_the_stop_forward(self):
        fitted = fitted_models.ModelStore()
        ages, incs = wear_stint(4.5, 6)
        fast = rv.predict_rival(state('MEDIUM', 6, 7, wear_pct=27.0, wear_ages=ages, wear_increments=incs),
                                ctx(), fitted)
        ages, incs = wear_stint(1.5, 6)
        slow = rv.predict_rival(state('MEDIUM', 6, 7, wear_pct=9.0, wear_ages=ages, wear_increments=incs),
                                ctx(), fitted)
        self.assertEqual(fast.basis, 'wear')
        self.assertIn('wear', fast.sources)
        self.assertLess(fast.stop_lap, slow.stop_lap)
        # 27% now at 4.5%/lap reaches the 60% stop wear about 7-8 laps later.
        self.assertTrue(12 <= fast.stop_lap <= 16, fast.sources)

    def test_hidden_wear_falls_back_to_age(self):
        p = rv.predict_rival(state('MEDIUM', 6, 7, wear_pct=None), ctx(), fitted_models.ModelStore())
        self.assertEqual(p.basis, 'age')
        self.assertNotIn('wear', p.sources)

    def test_both_compounds_used_and_tyres_last_means_no_more_stops(self):
        p = rv.predict_rival(state('HARD', 2, 36, compounds_used=('MEDIUM', 'HARD'), stops_done=1), ctx())
        self.assertFalse(p.will_stop)
        self.assertIsNone(p.stop_lap)
        self.assertIn('going to the end', p.summary)

    def test_compound_rule_forces_a_stop_before_the_flag(self):
        p = rv.predict_rival(state('HARD', 30, 36), ctx())
        self.assertTrue(p.must_stop)
        self.assertTrue(p.will_stop)
        self.assertLessEqual(p.stop_lap, 39)
        self.assertNotEqual(p.next_compound, 'HARD')

    def test_overdue_car_is_expected_in_now(self):
        ages, incs = wear_stint(4.0, 5, 12)
        p = rv.predict_rival(state('SOFT', 17, 18, wear_pct=78.0, wear_ages=ages, wear_increments=incs),
                             ctx(), fitted_models.ModelStore())
        self.assertEqual(p.stop_lap, 18)
        self.assertLessEqual(p.stop_lap_sd, rv.OVERDUE_SD)

    def test_game_window_steers_the_first_stop(self):
        base = rv.predict_rival(state('MEDIUM', 0, 1), ctx())
        early = rv.predict_rival(state('MEDIUM', 0, 1), ctx(game_window=(8, 12), player_start_compound='MEDIUM'))
        self.assertIn('game_window', early.sources)
        self.assertLess(early.stop_lap_mean, base.stop_lap_mean)
        # Not after our own stop, and not for a rival on its second set.
        self.assertNotIn('game_window', rv.predict_rival(
            state('MEDIUM', 0, 1), ctx(game_window=(8, 12), player_start_compound='MEDIUM', player_stops=1)).sources)

    def test_field_life_factor_moves_the_plan(self):
        stats = rv.FieldStats(life_ratio=[1.5] * 8)
        longer = rv.predict_rival(state('MEDIUM', 0, 1), ctx(), field_stats=stats)
        base = rv.predict_rival(state('MEDIUM', 0, 1), ctx())
        self.assertGreater(longer.stop_lap_mean, base.stop_lap_mean)

    def test_field_tyre_choices_shift_the_next_compound(self):
        stats = rv.FieldStats(transitions={('MEDIUM', 'SOFT'): 8})
        p = rv.predict_rival(state('MEDIUM', 10, 11), ctx(), field_stats=stats)
        self.assertEqual(p.next_compound, 'SOFT')

    def test_heavy_rain_on_slicks_means_box_now_for_inters(self):
        p = rv.predict_rival(state('MEDIUM', 3, 10), ctx(weather=4))
        self.assertEqual(p.stop_lap, 10)
        self.assertIn('weather', p.sources)
        self.assertIn(p.next_compound, ('INTERMEDIATE', 'WET'))
        light = rv.predict_rival(state('MEDIUM', 3, 10), ctx(weather=3))
        self.assertEqual(light.next_compound, 'INTERMEDIATE')

    def test_inters_on_a_dry_track_come_off_for_slicks(self):
        p = rv.predict_rival(state('INTERMEDIATE', 8, 20, compounds_used=('MEDIUM', 'INTERMEDIATE')), ctx(weather=0))
        self.assertTrue(p.will_stop)
        self.assertLessEqual(p.stop_lap, 23)
        self.assertIn(p.next_compound, rs.DRY_COMPOUNDS)

    def test_flat_cost_curve_gives_a_wide_stop_window(self):
        short = rs.plan_race(7, 'MEDIUM', 0, 1)
        long = rs.plan_race(50, 'MEDIUM', 0, 1)
        _m, short_sd = rv.first_stop_distribution(short, 0, 1, 1.0)
        mean, long_sd = rv.first_stop_distribution(long, 0, 1, 1.0)
        self.assertGreater(short_sd / 7, long_sd / 50)
        self.assertAlmostEqual(mean, long.best.pit_laps[0], delta=3)

    def test_nothing_to_predict(self):
        self.assertIsNone(rv.predict_rival(state('MEDIUM', 3, 41), ctx()))
        self.assertIsNone(rv.predict_rival(state(None, 3, 5), ctx()))

    def test_sampling_stays_inside_the_race(self):
        rng = random.Random(4)
        p = rv.predict_rival(state('MEDIUM', 5, 6), ctx())
        laps = [rv.sample_stop_lap(p, rng) for _ in range(300)]
        self.assertTrue(all(lap is None or 6 <= lap <= 40 for lap in laps))
        self.assertAlmostEqual(sum(l for l in laps if l) / len([l for l in laps if l]), p.stop_lap_mean, delta=1.0)
        sets = {rv.sample_next_compound(p, rng) for _ in range(200)}
        self.assertTrue(sets <= set(p.next_compound_probs))
        done = rv.predict_rival(state('HARD', 2, 36, compounds_used=('MEDIUM', 'HARD')), ctx())
        self.assertIsNone(rv.sample_stop_lap(done, rng))

    def test_track_priors(self):
        self.assertLess(rv.track_life_scale(7, 5891), rv.track_life_scale(5, 3337))   # Silverstone vs Monaco
        self.assertEqual(rv.track_life_scale(None, None), 1.0)
        self.assertGreater(rv.position_cost_s(5), rv.position_cost_s(3))
        sequence = rv.typical_sequence(17, 4318, 71)
        self.assertGreaterEqual(len(sequence), 2)
        self.assertEqual(rv.typical_next(['MEDIUM', 'HARD'], 'MEDIUM', 0), 'HARD')
        self.assertIsNone(rv.typical_next(['MEDIUM', 'HARD'], 'HARD', 1))


class TrackerTests(unittest.TestCase):
    """RivalTracker: stints, wear snapshots and stops from per-car packets."""

    def setUp(self):
        self.tracker = rv.RivalTracker()
        self.cars = {1: dict(compound=MEDIUM, age=0, wear=0.0, pit=0, stops=0)}

    def frame(self, lap, distance):
        laps = [replace(BASE_LAP, car_position=0, result_status=0)] * 22
        statuses = [status()] * 22
        damages = [damage(0.0)] * 22
        for index, car in self.cars.items():
            laps[index] = replace(BASE_LAP, car_position=index + 1, result_status=2, current_lap_num=lap,
                                  lap_distance=distance, pit_status=car['pit'], num_pit_stops=car['stops'])
            statuses[index] = status(car['compound'], car['age'])
            damages[index] = damage(car['wear'])
        self.tracker.update_status(statuses)
        self.tracker.update_damage(damages)
        self.tracker.update_laps(laps, player_index=0)

    def drive_laps(self, first, last, rate):
        car = self.cars[1]
        for lap in range(first, last + 1):
            self.frame(lap, 100.0)
            self.frame(lap, 1500.0)
            car['age'] += 1
            car['wear'] += rate

    def test_wear_increments_and_state(self):
        self.drive_laps(1, 6, 3.0)
        s = self.tracker.state(1, 'HAMILTON', 5891)
        self.assertEqual(s.compound, 'MEDIUM')
        self.assertEqual(s.actual_compound, 'C3')
        self.assertEqual(len(s.wear_increments), 5)
        self.assertTrue(all(abs(i - 3.0) < 1e-6 for i in s.wear_increments))
        self.assertEqual(s.stops_done, 0)
        self.assertEqual(s.compounds_used, ('MEDIUM',))

    def test_stop_is_detected_and_teaches_the_field(self):
        self.drive_laps(1, 12, 4.0)
        prediction = self.tracker.predict(1, ctx(), fitted_models.ModelStore(), 'HAMILTON')
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.name, 'HAMILTON')
        car = self.cars[1]
        car['pit'] = 1
        self.frame(13, 5800.0)                         # pit entry at the end of lap 13
        self.assertIsNone(self.tracker.predict(1, ctx()))          # in the pit lane
        self.assertIs(self.tracker.last_prediction(1), prediction)
        car.update(compound=HARD, age=0, wear=0.0, stops=1)
        self.frame(14, 50.0)
        car['pit'] = 0
        self.frame(14, 600.0)
        self.assertEqual(len(self.tracker.cars[1].stints), 2)
        log = self.tracker.stops_log[-1]
        self.assertEqual((log['lap'], log['from'], log['to'], log['age']), (13, 'MEDIUM', 'HARD', 12))
        self.assertGreater(log['wear'], 40)
        self.assertEqual(self.tracker.field.stop_wear, [log['wear']])
        self.assertEqual(len(self.tracker.field.life_ratio), 1)
        self.assertEqual(self.tracker.field.next_compound_counts('MEDIUM'), {'HARD': 1})
        after = self.tracker.predict(1, ctx())
        self.assertEqual(after.stops_done, 1)
        self.assertFalse(after.must_stop)

    def test_player_and_restricted_cars(self):
        self.cars[0] = dict(compound=MEDIUM, age=0, wear=0.0, pit=0, stops=0)
        self.drive_laps(1, 4, 0.0)                     # rival wear stays at zero: restricted telemetry
        self.assertIsNone(self.tracker.state(1).wear_pct)
        preds = self.tracker.predict_all(ctx(), exclude=0)
        self.assertEqual([p.car_index for p in preds], [1])

    def test_bad_damage_values_are_ignored(self):
        self.tracker.update_damage([damage(float('nan'))] * 22)
        self.assertEqual(self.tracker.cars, {})

    def test_prediction_is_cached_until_something_changes(self):
        self.drive_laps(1, 3, 3.0)
        first = self.tracker.predict(1, ctx())
        self.assertIs(self.tracker.predict(1, ctx()), first)
        self.drive_laps(4, 4, 3.0)
        self.assertIsNot(self.tracker.predict(1, ctx()), first)


class StrategyHookTests(unittest.TestCase):
    def test_plan_reports_best_option_per_next_tyre(self):
        plan = rs.plan_race(30, 'MEDIUM', 5, 11)
        self.assertTrue(set(plan.best_by_next) <= set(rs.DRY_COMPOUNDS))
        self.assertIn(plan.best.compounds[1], plan.best_by_next)
        self.assertEqual(min(o.total_time_s for o in plan.best_by_next.values()), plan.best.total_time_s)

    def test_rival_staying_out_longer_makes_the_undercut_bigger(self):
        self.assertEqual(rs.cover_laps(None, 20), 1)
        self.assertEqual(rs.cover_laps(0, 20), 1)
        self.assertEqual(rs.cover_laps(9, 20), rs.MAX_COVER_LAPS)
        self.assertEqual(rs.cover_laps(9, 4), 2)
        kwargs = dict(gap_ahead_s=2.6, my_compound='MEDIUM', my_age=18, rival_compound='MEDIUM', rival_age=24,
                      laps_to_planned_stop=2, laps_remaining=20)
        self.assertIsNone(rs.undercut_opportunity(**kwargs))
        call = rs.undercut_opportunity(**kwargs, rival_stop_in_laps=4)
        self.assertIsNotNone(call)
        self.assertEqual(call.kind, 'undercut')

    def test_rival_stop_reaction_uses_the_predicted_tyre_without_naming_it(self):
        guess = rs.react_to_rival_stop('behind', -1.2, 'MEDIUM', 20, 'MEDIUM', None, 20, 1,
                                       assumed_new_compound='SOFT')
        default = rs.react_to_rival_stop('behind', -1.2, 'MEDIUM', 20, 'MEDIUM', None, 20, 1)
        self.assertNotEqual(guess.gain_s, default.gain_s)
        self.assertNotIn('soft', guess.message.lower())


class AnalysisRivalTests(unittest.TestCase):
    def setUp(self):
        self.voice = SilentVoice()
        self.engine = AnalysisEngine(self.voice)
        self.engine.update_session(RACE, 1, 0)
        self.engine.update_participants([])
        self.engine.radio.set_driver_names({1: 'Leclerc', 2: 'Hamilton'})
        self.lap = replace(BASE_LAP, current_lap_num=10, pit_status=0, num_pit_stops=0, car_position=3,
                           lap_distance=1500.0)
        self.voice.messages.clear()

    def feed(self, lap_num, rival_wear_rate=4.0, player_age=9):
        """Lap data, status and damage for the player (P3) and rivals ahead (P2) and behind (P4)."""
        laps = [replace(BASE_LAP, car_position=0, result_status=0)] * 22
        statuses, damages = [status()] * 22, [damage(0.0)] * 22
        for lap in range(1, lap_num + 1):
            for distance in (100.0, 1500.0):
                for index, position in ((0, 3), (1, 4), (2, 2)):
                    laps[index] = replace(BASE_LAP, car_position=position, result_status=2,
                                          current_lap_num=lap, lap_distance=distance)
                    age = lap - 1
                    statuses[index] = status(MEDIUM, age if index else min(age, player_age))
                    damages[index] = damage(rival_wear_rate * age)
                self.engine.update_car_status(statuses[0], statuses)
                self.engine.update_car_damage(damages[0], damages)
                self.engine.update_field(laps, 0)
        self.engine.latest_lap = replace(self.lap, current_lap_num=lap_num)

    def test_predictions_in_race_only(self):
        self.feed(6)
        predictions = self.engine.rival_predictions()
        self.assertEqual({p.car_index for p in predictions}, {1, 2})
        self.assertEqual(predictions[0].name, 'Hamilton')       # P2 first
        self.engine.update_session(replace(RACE, session_type=5), 2, 0)
        self.assertEqual(self.engine.rival_predictions(), [])

    def test_summary_context_and_answers(self):
        self.feed(6)
        self.engine.radio.car_ahead_index, self.engine.radio.gap_ahead_s = 2, 1.5
        self.engine.radio.car_behind_index, self.engine.radio.gap_behind_s = 1, 1.0
        summary = self.engine.strategy_summary()
        rivals = summary['rivals']
        self.assertEqual(rivals['ahead']['name'], 'Hamilton')
        self.assertEqual(rivals['behind']['name'], 'Leclerc')
        self.assertIn('Hamilton likely to box lap', summary['rival_radio'])
        ctx = self.engine.build_context()
        self.assertEqual(len(ctx['rival_predictions']), 2)
        self.assertIn('Rival stop predictions', format_context(ctx))
        answer = rule_based_answer('rivals', ctx)
        self.assertIn('Hamilton', answer)
        self.assertIn('Leclerc', answer)
        self.assertIn('No race strategy', rule_based_answer('rivals', {'session_mode': 'qualifying'}))
        self.assertEqual(rule_based_answer('rivals', {'session_mode': 'race'}), 'No read on the other cars yet.')

    def test_car_behind_about_to_box_triggers_a_pre_emptive_call_once(self):
        self.feed(14, rival_wear_rate=4.4, player_age=13)       # rival behind at ~57% wear: due now
        self.engine.rivals.field.transitions[('MEDIUM', 'SOFT')] = 5   # the field is going onto softs
        self.engine.on_demand = True
        self.engine.radio.car_behind_index, self.engine.radio.gap_behind_s = 1, 1.0
        prediction = self.engine.rival_prediction(1)
        self.assertEqual(prediction.basis, 'wear')
        self.assertLessEqual(prediction.stop_lap, 15)
        plan = self.engine.strategy_plan()
        plan.best.stops, plan.best.pit_laps = 1, [16]
        lap = replace(self.lap, current_lap_num=14)
        self.engine.update_strategy_calls(lap)
        self.engine.update_strategy_calls(lap)
        spoken = [m for m, _, _ in self.voice.messages]
        calls = [m for m in spoken if 'Leclerc behind' in m]
        self.assertEqual(len(calls), 1, spoken)
        self.assertIn('box this lap', calls[0])
        self.assertEqual(self.engine.last_strategy_call['kind'], 'pre_empt')

    def test_heads_up_when_car_ahead_is_due(self):
        self.feed(13, rival_wear_rate=4.4, player_age=3)
        self.engine.radio.car_ahead_index, self.engine.radio.gap_ahead_s = 2, 2.0
        lap = replace(self.lap, current_lap_num=13)
        self.engine.update_strategy_calls(lap)
        spoken = [m for m, _, _ in self.voice.messages]
        self.assertTrue(any(m.startswith('Hamilton ahead likely to box lap') for m in spoken), spoken)

    def test_no_rival_calls_on_lap_one(self):
        self.feed(1)
        self.engine.radio.car_behind_index, self.engine.radio.gap_behind_s = 1, 0.5
        self.engine.update_strategy_calls(replace(self.lap, current_lap_num=1))
        self.assertFalse(any('likely to box' in m for m, _, _ in self.voice.messages))

    def test_new_session_clears_rival_state(self):
        self.feed(4)
        self.engine.update_session(replace(RACE, track_id=11), 3, 0)
        self.assertEqual(self.engine.rivals.cars, {})


class RivalVoiceCommandTests(unittest.TestCase):
    def test_rival_questions_match(self):
        for phrase in ('when will Hamilton box', 'what are the others doing on tyres', 'rival strategy',
                       'is the car ahead pitting', 'when are they stopping', 'when is Leclerc going to pit'):
            self.assertEqual(match_command(phrase), 'rivals', phrase)
        for phrase, command in (('when do we box', 'pit_window'), ('when will I pit', 'pit_window'),
                                ('what is the strategy', 'strategy')):
            self.assertEqual(match_command(phrase), command, phrase)
        self.assertNotEqual(match_command('will it stop raining'), 'rivals')


if __name__ == '__main__':
    unittest.main()
