"""Tests for race strategy: stop-count/compound planning, undercut and overcut, rival pit reactions."""

import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import race_strategy as rs
from analysis import AnalysisEngine
from engineer_brain import format_context, rule_based_answer
from packets import CarStatusData, SessionData, parse_player_lap_data
from race_radio import RaceRadio
from simulate_telemetry import build_car_status_packet, build_lap_packet
from test_analysis import SilentVoice
from voice_commands import match_command

SOFT, MEDIUM, HARD = 16, 17, 18
RACE = SessionData(0, 30, 20, 50, 5891, 10, 7, 0)  # Silverstone, race, 50 laps


def car_status(compound=MEDIUM, age=0, fuel=60.0):
    return CarStatusData(fuel, 110.0, fuel / 1.6, compound, compound, age, 0, 2_000_000.0, 1, 0.0)


class CompoundAndTrackTests(unittest.TestCase):
    def test_game_compound_names_normalise(self):
        self.assertEqual(rs.normalise_compound('Soft'), 'SOFT')
        self.assertEqual(rs.normalise_compound('Intermediate'), 'INTERMEDIATE')
        self.assertEqual(rs.normalise_compound('C3'), 'MEDIUM')
        self.assertEqual(rs.normalise_compound('C5'), 'SOFT')
        self.assertIsNone(rs.normalise_compound('Unknown'))
        self.assertIsNone(rs.normalise_compound(None))

    def test_pit_loss_uses_track_and_short_layout_parent(self):
        self.assertEqual(rs.pit_loss_s(12), 28.0)  # Singapore: long pit lane
        self.assertEqual(rs.pit_loss_s(22), rs.pit_loss_s(7))  # Silverstone Short -> Silverstone
        self.assertEqual(rs.pit_loss_s(-1), rs.DEFAULT_PIT_LOSS_S)
        self.assertEqual(rs.pit_loss_s(99), rs.DEFAULT_PIT_LOSS_S)

    def test_calibration_follows_observed_data_and_is_clamped(self):
        self.assertEqual(rs.calibration('Medium', 10), (1.0, 1.0))
        deg, life = rs.calibration('Medium', 10, observed_deg_s_per_lap=0.09, observed_laps_to_limit=20)
        self.assertGreater(deg, 1.0)
        self.assertGreater(life, 1.0)
        self.assertEqual(rs.calibration('Medium', 10, 50.0, 500)[0], 3.0)
        self.assertEqual(rs.calibration('Medium', 10, 50.0, 500)[1], 2.0)
        self.assertEqual(rs.calibration('Wet', 10, 0.2, 5), (1.0, 1.0))


class PlanRaceTests(unittest.TestCase):
    def test_plan_covers_every_remaining_lap(self):
        plan = rs.plan_race(40, 'Medium', 3, current_lap=11, pit_loss=21.0)
        for option in [plan.best, *plan.alternatives]:
            self.assertEqual(sum(option.stint_laps), 40)
            self.assertEqual(len(option.compounds), option.stops + 1)
            self.assertEqual(len(option.pit_laps), option.stops)
            self.assertTrue(all(11 <= lap < 50 for lap in option.pit_laps))
            self.assertEqual(option.pit_laps, sorted(option.pit_laps))
        self.assertEqual(plan.best.compounds[0], 'MEDIUM')

    def test_two_dry_compound_rule_forbids_no_stop_on_one_compound(self):
        plan = rs.plan_race(8, 'Medium', 2, current_lap=43, pit_loss=21.0)
        self.assertGreaterEqual(plan.best.stops, 1)
        self.assertNotIn('MEDIUM', plan.best.compounds[1:])
        self.assertNotIn(0, [o.stops for o in plan.alternatives])

    def test_no_stop_once_both_compounds_used_and_few_laps_left(self):
        plan = rs.plan_race(6, 'Hard', 10, current_lap=45, pit_loss=21.0, compounds_used={'MEDIUM'})
        self.assertEqual(plan.best.stops, 0)
        self.assertEqual(plan.best.pit_laps, [])
        self.assertIn('No more stops', rs.describe_plan(plan))

    def test_wet_running_waives_the_compound_rule(self):
        plan = rs.plan_race(6, 'Medium', 2, current_lap=45, pit_loss=21.0, compounds_used={'Intermediate'})
        self.assertEqual(plan.best.stops, 0)

    def test_wet_tyres_plan_only_fresh_sets_of_the_same_tyre(self):
        short = rs.plan_race(5, 'Intermediate', 3, 30)
        self.assertEqual((short.best.stops, short.best.compounds), (0, ['INTERMEDIATE']))
        self.assertIn('Bring the intermediates home', rs.describe_plan(short))
        long = rs.plan_race(60, 'Wet', 10, 10)
        self.assertGreater(long.best.stops, 0)
        self.assertEqual(set(long.best.compounds), {'WET'})

    def test_finished_race_has_no_plan(self):
        self.assertIsNone(rs.plan_race(0, 'Medium', 3, 51))
        self.assertEqual(rs.describe_plan(None), 'No dry strategy to plan right now.')

    def test_worn_set_is_boxed_soon(self):
        life = rs.tyre_curve('SOFT').life_laps
        plan = rs.plan_race(30, 'Soft', int(life * rs.MAX_LIFE_MULTIPLE) - 1, current_lap=21)
        self.assertLessEqual(plan.best.pit_laps[0], 22)

    def test_stints_never_exceed_the_tyre_age_limit(self):
        plan = rs.plan_race(60, 'Soft', 0, current_lap=1, pit_loss=21.0)
        for option in [plan.best, *plan.alternatives]:
            for compound, laps in zip(option.compounds, option.stint_laps, strict=False):
                self.assertLessEqual(laps, rs.tyre_curve(compound).max_age)

    def test_longer_pit_loss_never_adds_stops(self):
        cheap = rs.plan_race(55, 'Medium', 0, 1, pit_loss=12.0)
        dear = rs.plan_race(55, 'Medium', 0, 1, pit_loss=35.0)
        self.assertLessEqual(dear.best.stops, cheap.best.stops)

    def test_heavier_degradation_never_removes_stops(self):
        mild = rs.plan_race(55, 'Medium', 0, 1, pit_loss=21.0, deg_scale=0.5)
        heavy = rs.plan_race(55, 'Medium', 0, 1, pit_loss=21.0, deg_scale=3.0)
        self.assertGreaterEqual(heavy.best.stops, mild.best.stops)

    def test_alternatives_are_slower_and_reported(self):
        plan = rs.plan_race(45, 'Medium', 0, 6, pit_loss=21.0)
        self.assertTrue(plan.alternatives)
        for option in plan.alternatives:
            self.assertGreaterEqual(option.total_time_s, plan.best.total_time_s)
            self.assertGreaterEqual(plan.margin_to(option.stops), 0)
        data = plan.as_dict()
        self.assertEqual(data['laps_remaining'], 45)
        self.assertTrue(all(o['slower_by_s'] >= 0 for o in data['alternatives']))
        self.assertIn('Box lap', rs.describe_plan(plan))

    def test_allocation_splits_laps_optimally(self):
        # Two identical new sets: the convex cost makes an even split optimal.
        curve = rs.tyre_curve('MEDIUM')
        lengths, _ = rs._allocate([curve, curve], [0.0, 0.0], 30)
        self.assertLessEqual(abs(lengths[0] - lengths[1]), 1)
        self.assertIsNone(rs._allocate([curve], [0.0], 500))  # beyond any tyre's life
        self.assertIsNone(rs._allocate([curve, curve], [0.0, 0.0], 1))


class UndercutOvercutTests(unittest.TestCase):
    def test_undercut_gain_grows_with_rival_tyre_age_and_degradation(self):
        young = rs.undercut_gain('Medium', 3, 'Hard')
        old = rs.undercut_gain('Medium', 25, 'Hard')
        self.assertLess(young, 0)  # nothing to gain against fresh tyres
        self.assertGreater(old, young)
        self.assertGreater(rs.undercut_gain('Medium', 25, 'Hard', deg_scale=3.0), old)

    def test_undercut_called_when_gain_beats_gap(self):
        call = rs.undercut_opportunity(
            0.8,
            'Medium',
            24,
            'Medium',
            26,
            laps_to_planned_stop=1,
            laps_remaining=25,
            deg_scale=2.5,
            rival_index=4,
        )
        self.assertIsNotNone(call)
        self.assertEqual(call.kind, 'undercut')
        self.assertEqual(call.car_index, 4)
        self.assertIn('Box this lap', call.message)
        self.assertGreater(call.gain_s, call.gap_s)

    def test_no_undercut_when_gap_too_big_stop_far_or_race_ending(self):
        args = dict(my_compound='Medium', my_age=24, rival_compound='Medium', rival_age=26, deg_scale=2.5)
        self.assertIsNone(rs.undercut_opportunity(5.0, laps_to_planned_stop=1, laps_remaining=25, **args))
        self.assertIsNone(rs.undercut_opportunity(0.8, laps_to_planned_stop=12, laps_remaining=25, **args))
        self.assertIsNone(rs.undercut_opportunity(0.8, laps_to_planned_stop=1, laps_remaining=2, **args))
        self.assertIsNone(rs.undercut_opportunity(None, laps_to_planned_stop=1, laps_remaining=25, **args))
        self.assertIsNone(rs.undercut_opportunity(0.8, laps_to_planned_stop=None, laps_remaining=25, **args))

    def test_rival_behind_boxing_close_is_covered(self):
        call = rs.react_to_rival_stop('behind', -0.9, 'Medium', 24, 'Medium', None, 25, 1, deg_scale=2.5)
        self.assertEqual(call.kind, 'cover')
        self.assertIn('Box this lap', call.message)

    def test_rival_behind_boxing_with_big_gap_means_stay_out(self):
        call = rs.react_to_rival_stop('behind', -4.5, 'Medium', 10, 'Medium', None, 25, 1)
        self.assertEqual(call.kind, 'stay_out')
        self.assertIn('Stay out', call.message)

    def test_rival_ahead_boxing_opens_overcut_on_fresh_tyres(self):
        call = rs.react_to_rival_stop('ahead', 0.1, 'Soft', 1, 'Medium', 'Hard', 30, 1)
        self.assertEqual(call.kind, 'overcut')
        self.assertIn('push', call.message)

    def test_rival_ahead_boxing_on_worn_tyres_keeps_plan(self):
        call = rs.react_to_rival_stop('ahead', 1.5, 'Medium', 25, 'Medium', 'Hard', 30, 1)
        self.assertEqual(call.kind, 'stay_out')
        self.assertIn('hards', call.message)

    def test_rival_call_names_the_driver_and_never_guesses_the_compound(self):
        call = rs.react_to_rival_stop(
            'ahead', 1.5, 'Medium', 25, 'Medium', None, 30, 1, rival_name='Hamilton'
        )
        self.assertEqual(call.kind, 'stay_out')
        self.assertTrue(call.message.startswith('Hamilton ahead has boxed.'), call.message)
        self.assertNotIn('hards', call.message)

    def test_no_stops_left_means_stay_out_to_the_flag(self):
        call = rs.react_to_rival_stop('behind', -0.5, 'Hard', 20, 'Medium', None, 10, 0)
        self.assertEqual(call.kind, 'stay_out')
        self.assertIn('flag', call.message)


class RivalPitDetectionTests(unittest.TestCase):
    def setUp(self):
        self.radio = RaceRadio(lambda *args, **kwargs: None)
        self.radio.reset(RACE)
        self.base = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)

    def drive(self, now, offset, pit_cars=()):
        laps = [replace(self.base, car_position=0, result_status=0)] * 22
        cars = {0: (3, 20), 1: (2, 30), 2: (4, 10)}
        for index, (position, distance) in cars.items():
            laps[index] = replace(
                self.base,
                car_position=position,
                total_distance=distance + offset,
                result_status=2,
                current_lap_num=12,
                pit_status=1 if index in pit_cars else 0,
            )
        self.radio.update_field(laps, player_index=0, session_time=now)

    def test_rival_entering_pits_queues_event_with_last_gap(self):
        for now, offset in ((0, 0), (10, 10), (11, 20)):
            self.drive(now, offset)
        self.assertEqual(self.radio.car_ahead_index, 1)
        self.assertEqual(self.radio.car_behind_index, 2)
        self.assertEqual(self.radio.pop_rival_pit_events(), [])
        self.drive(11.5, 25, pit_cars=(2,))
        events = self.radio.pop_rival_pit_events()
        self.assertEqual([e['car_index'] for e in events], [2])
        self.assertLess(events[0]['gap_s'], 0)  # behind the player
        self.assertEqual(events[0]['lap'], 12)
        self.assertEqual(self.radio.pop_rival_pit_events(), [])  # consumed
        self.drive(12, 30, pit_cars=(2,))  # still in the pit lane: no repeat
        self.assertEqual(self.radio.pop_rival_pit_events(), [])

    def test_car_first_seen_in_pits_or_without_gap_is_ignored(self):
        self.drive(0, 0, pit_cars=(1,))
        self.assertEqual(self.radio.pop_rival_pit_events(), [])

    def test_reset_clears_rival_tracking(self):
        for now, offset in ((0, 0), (10, 10), (11, 20)):
            self.drive(now, offset)
        self.radio.reset(RACE)
        self.assertEqual(self.radio.rival_gaps, {})
        self.assertIsNone(self.radio.car_ahead_index)


class AnalysisStrategyTests(unittest.TestCase):
    def setUp(self):
        self.voice = SilentVoice()
        self.engine = AnalysisEngine(self.voice)
        self.engine.update_session(RACE, 1, 0)
        self.lap = replace(
            parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0),
            current_lap_num=20,
            pit_status=0,
            num_pit_stops=0,
            car_position=3,
        )
        self.engine.latest_lap = self.lap
        rivals = [car_status(MEDIUM, 5)] * 22
        rivals[1] = car_status(MEDIUM, 26)
        rivals[2] = car_status(MEDIUM, 2)
        self.engine.update_car_status(car_status(MEDIUM, 22), rivals)
        self.voice.messages.clear()  # drop the session welcome line

    def spoken(self):
        return [m for m, _, _ in self.voice.messages]

    def test_plan_is_built_in_race_and_cached(self):
        plan = self.engine.strategy_plan()
        self.assertIsNotNone(plan)
        self.assertEqual(plan.laps_remaining, 31)
        self.assertIs(self.engine.strategy_plan(), plan)
        ctx = self.engine.build_context()
        self.assertEqual(ctx['strategy']['best']['compounds'][0], 'MEDIUM')
        self.assertIn('Race strategy model', format_context(ctx))

    def test_no_plan_outside_race(self):
        self.engine.update_session(replace(RACE, session_type=5), 2, 0)  # qualifying
        self.engine.latest_lap = self.lap
        self.engine.update_car_status(car_status(SOFT, 2))
        self.assertIsNone(self.engine.strategy_plan())
        self.assertIsNone(self.engine.build_context()['strategy'])

    def test_compounds_used_tracked_and_cleared_on_new_session(self):
        self.engine.update_car_status(car_status(HARD, 0))
        self.assertEqual(self.engine.compounds_used, {'MEDIUM', 'HARD'})
        self.engine.update_session(replace(RACE, track_id=11), 3, 0)
        self.assertEqual(self.engine.compounds_used, set())
        self.assertIsNone(self.engine.all_status)

    def test_rival_behind_pitting_triggers_cover_call_even_in_on_demand_mode(self):
        self.engine.on_demand = True
        self.engine._strategy_cache = None
        self.engine.strategy_plan()
        self.engine._strategy_cache = (
            self.engine._strategy_cache[0],
            self.engine._strategy_cache[1],
            (2.5, 1.0),
        )
        self.engine.radio.rival_pit_events = [
            {'car_index': 2, 'gap_s': -0.8, 'lap': 20, 'position': 4, 'session_time': 100.0}
        ]
        self.engine.update_strategy_calls(self.lap)
        self.assertTrue(any('undercut threat' in m for m in self.spoken()), self.spoken())
        self.assertEqual(self.engine.last_strategy_call['kind'], 'cover')
        self.assertEqual(self.engine.last_strategy_call['lap'], 20)

    def test_distant_rival_pitting_is_ignored(self):
        self.engine.radio.rival_pit_events = [
            {'car_index': 2, 'gap_s': -15.0, 'lap': 20, 'position': 9, 'session_time': 100.0}
        ]
        self.engine.update_strategy_calls(self.lap)
        self.assertEqual(self.spoken(), [])
        self.assertEqual(self.engine.radio.rival_pit_events, [])

    def test_undercut_called_once_per_lap_on_car_ahead(self):
        self.engine.strategy_plan()
        plan = self.engine._strategy_cache[1]
        plan.best.stops, plan.best.pit_laps = 1, [21]
        self.engine._strategy_cache = (self.engine._strategy_cache[0], plan, (2.5, 1.0))
        self.engine.radio.car_ahead_index, self.engine.radio.gap_ahead_s = 1, 0.7
        self.engine.update_strategy_calls(self.lap)
        self.engine.update_strategy_calls(self.lap)
        calls = [m for m in self.spoken() if 'Undercut is on' in m]
        self.assertEqual(len(calls), 1)
        summary = self.engine.strategy_summary()
        self.assertTrue(summary['undercut_on_car_ahead']['works'])
        self.assertEqual(summary['last_call']['kind'], 'undercut')
        self.assertIn('Undercut on car ahead', format_context(self.engine.build_context()))

    def test_no_calls_while_player_is_in_pit_lane(self):
        self.engine.radio.rival_pit_events = [
            {'car_index': 2, 'gap_s': -0.8, 'lap': 20, 'position': 4, 'session_time': 100.0}
        ]
        self.engine.update_strategy_calls(replace(self.lap, pit_status=1))
        self.assertEqual(self.spoken(), [])


class StrategyVoiceAndStreamTests(unittest.TestCase):
    def test_strategy_phrases_match(self):
        for phrase in (
            'Is the undercut on?',
            'can we overcut him',
            'what is the strategy',
            'how many stops',
            'which tyres next',
            "what's the plan",
        ):
            self.assertEqual(match_command(phrase), 'strategy', phrase)
        self.assertEqual(match_command('when do we box'), 'pit_window')
        self.assertEqual(match_command('tyre temperatures please'), 'tyre_temps')

    def test_rule_based_strategy_answer(self):
        plan = rs.plan_race(30, 'Medium', 10, 21)
        ctx = {
            'session_mode': 'race',
            'strategy': {
                **plan.as_dict(),
                'radio': rs.describe_plan(plan),
                'undercut_on_car_ahead': {
                    'works': True,
                    'gap_s': 0.5,
                    'gain_s': 1.4,
                    'rival_compound': 'Medium',
                    'rival_tyre_age': 20,
                },
            },
        }
        answer = rule_based_answer('strategy', ctx)
        self.assertIn('Plan is', answer)
        self.assertIn('Undercut on the car ahead is on', answer)
        self.assertEqual(rule_based_answer('strategy', {'session_mode': 'race'}), 'No strategy data yet.')
        self.assertIn('No race strategy', rule_based_answer('strategy', {'session_mode': 'qualifying'}))

    def test_stream_passes_every_cars_status_to_analysis(self):
        from simulate_telemetry import build_session_packet
        from telemetry_stream import TelemetryStream

        engine = AnalysisEngine(SilentVoice())
        stream = TelemetryStream(engine)
        stream.handle(build_session_packet(0, 1, 11), now=0)
        packet = bytearray(build_car_status_packet(0.1, 2, 50.0, SOFT, 3))
        stream.handle(bytes(packet), now=0.1)
        self.assertEqual(len(engine.all_status), 22)
        self.assertEqual(engine.latest_status.tyres_age_laps, 3)
        self.assertIn('SOFT', engine.compounds_used)


if __name__ == '__main__':
    unittest.main()
