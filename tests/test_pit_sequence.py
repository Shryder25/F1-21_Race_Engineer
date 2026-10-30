"""Tests for the pit stop call sequence: box -> confirm -> stay-out abort -> out-lap brief."""

import pathlib
import sys
import tempfile
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import pit_sequence as ps
from analysis import AnalysisEngine
from packets import parse_player_lap_data
from simulate_telemetry import build_lap_packet
from test_analysis import SilentVoice
from test_race_strategy import MEDIUM, RACE, car_status
from track_maps import TrackMapStore
from voice_engine import Priority

LENGTH = 5000.0


def lap(distance, lap_num=20, pit=0, stops=0, position=3, status=2):
    return SimpleNamespace(
        lap_distance=distance,
        current_lap_num=lap_num,
        pit_status=pit,
        num_pit_stops=stops,
        car_position=position,
        result_status=status,
    )


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, message, priority=Priority.NORMAL, **options):
        self.calls.append((options.get('key'), message, priority))
        return True

    def keys(self):
        return [key for key, _, _ in self.calls]

    def text(self, key):
        return next(message for k, message, _ in self.calls if k == key)


def new_sequence(stored=None):
    speak = Recorder()
    sequence = ps.PitSequence(speak)
    sequence.reset(LENGTH, racing=True, stored_entry_m=stored)
    return sequence, speak


def drive(sequence, frames, session_time=100.0, step=0.2, neighbours=None):
    """Feed player-only frames (the player's lap is also the whole field)."""
    for frame in frames:
        sequence.update(frame, [frame], 0, session_time, neighbours)
        session_time += step
    return session_time


class PitEntryTests(unittest.TestCase):
    def test_default_entry_is_just_before_the_line_until_one_is_seen(self):
        sequence, _ = new_sequence()
        self.assertEqual(sequence.entry_m, LENGTH - ps.DEFAULT_ENTRY_BEFORE_LINE_M)
        self.assertEqual(sequence.entry_source, 'default')

    def test_stored_entry_is_used_then_replaced_by_what_the_field_does(self):
        sequence, _ = new_sequence(stored=4500.0)
        self.assertEqual((sequence.entry_m, sequence.entry_source), (4500.0, 'stored'))
        rivals = [lap(4000.0), lap(4100.0)]
        sequence.update(None, rivals, 5, 0.0)
        rivals = [lap(4200.0, pit=1), lap(4300.0, pit=1)]  # both turn in
        sequence.update(None, rivals, 5, 0.1)
        self.assertEqual((sequence.entry_m, sequence.entry_source), (4250.0, 'observed'))

    def test_a_car_starting_in_the_pit_lane_is_not_an_entry(self):
        sequence, _ = new_sequence()
        sequence.update(None, [lap(100.0, pit=1)], 5, 0.0)
        self.assertEqual(sequence.entry_samples, [])

    def test_entry_after_the_line_wraps_round_the_lap(self):
        sequence, _ = new_sequence(stored=150.0)
        self.assertEqual(sequence.to_entry_m(4900.0), 250.0)

    def test_learned_entry_is_offered_for_saving_after_a_few_stops(self):
        sequence, _ = new_sequence()
        for distance in (4600.0, 4610.0, 4620.0):
            sequence.update(None, [lap(distance - 100)], 5, 0.0)
            sequence.update(None, [lap(distance, pit=1)], 5, 0.1)
            sequence.update(None, [lap(10.0)], 5, 0.2)
        self.assertEqual(sequence.take_entry_to_save(), 4610.0)
        self.assertIsNone(sequence.take_entry_to_save())  # nothing new since

    def test_track_store_keeps_the_pit_entry(self):
        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / 'maps.json'
            TrackMapStore(path).save(7, 5891.0, pit_entry_m=5600.0)
            self.assertEqual(TrackMapStore(path).get(7, 5891.0)['pit_entry_m'], 5600.0)
            self.assertIsNone(TrackMapStore(path).get(9, 5891.0)['pit_entry_m'])


class BoxAndConfirmTests(unittest.TestCase):
    def test_box_call_early_in_the_lap_is_confirmed_before_pit_entry(self):
        sequence, speak = new_sequence(stored=4600.0)
        sequence.box_called('window', lap(1000.0))
        self.assertTrue(sequence.pending)
        drive(sequence, [lap(d) for d in (1000.0, 3000.0, 3800.0)])
        self.assertEqual(speak.keys(), [])
        drive(sequence, [lap(3950.0), lap(4100.0)])  # inside 700 m of the entry
        self.assertEqual(speak.keys(), ['pit_confirm'])
        self.assertIn('Box, box', speak.text('pit_confirm'))
        drive(sequence, [lap(4300.0), lap(4500.0)])
        self.assertEqual(speak.keys(), ['pit_confirm'])  # once only
        drive(sequence, [lap(4610.0, pit=1)])
        self.assertEqual(sequence.state, ps.IN_PIT)

    def test_box_call_made_close_to_the_entry_needs_no_confirm(self):
        sequence, speak = new_sequence(stored=4600.0)
        sequence.box_called('cover', lap(3800.0))
        drive(sequence, [lap(3800.0), lap(4100.0), lap(4500.0), lap(4605.0, pit=1)])
        self.assertEqual(speak.keys(), [])
        self.assertEqual(sequence.state, ps.IN_PIT)

    def test_too_late_for_this_lap_means_the_next_entry(self):
        sequence, speak = new_sequence(stored=4600.0)
        sequence.box_called('wear', lap(4500.0))  # 100 m from the entry
        drive(sequence, [lap(4500.0), lap(4700.0), lap(50.0, 21), lap(2000.0, 21)])
        self.assertTrue(sequence.pending)  # not "missed": the call was for next time round
        drive(sequence, [lap(4000.0, 21)])
        self.assertEqual(speak.keys(), ['pit_confirm'])

    def test_driving_past_the_entry_drops_the_call(self):
        sequence, speak = new_sequence(stored=4600.0)
        sequence.box_called('window', lap(1000.0))
        drive(sequence, [lap(1000.0), lap(4000.0), lap(4500.0), lap(4700.0), lap(200.0, 21)])
        self.assertEqual(sequence.state, ps.IDLE)
        self.assertEqual(sequence.last_event['kind'], 'missed')

    def test_nothing_is_armed_outside_a_race(self):
        speak = Recorder()
        sequence = ps.PitSequence(speak)
        sequence.reset(LENGTH, racing=False)
        sequence.box_called('damage', lap(1000.0))
        self.assertFalse(sequence.pending)

    def test_flashback_to_an_earlier_lap_clears_the_call(self):
        sequence, _ = new_sequence(stored=4600.0)
        sequence.box_called('window', lap(1000.0, 20))
        drive(sequence, [lap(1000.0, 20), lap(3000.0, 19)])
        self.assertEqual(sequence.state, ps.IDLE)


class StayOutTests(unittest.TestCase):
    def test_strategy_box_call_can_be_taken_back_before_the_entry(self):
        sequence, speak = new_sequence(stored=4600.0)
        sequence.box_called('undercut', lap(1000.0), car_index=4)
        self.assertTrue(sequence.can_abort(lap(3000.0)))
        sequence.abort('Undercut is gone.', 'target_pitted')
        self.assertEqual(sequence.state, ps.ABORTED)
        key, message, priority = speak.calls[-1]
        self.assertEqual(key, 'pit_abort')
        self.assertTrue(message.startswith('Stay out, stay out.'))
        self.assertEqual(priority, Priority.CRITICAL)
        drive(sequence, [lap(4200.0)])
        self.assertNotIn('pit_confirm', speak.keys())

    def test_too_close_to_the_entry_line_to_abort(self):
        sequence, _ = new_sequence(stored=4600.0)
        sequence.box_called('window', lap(1000.0))
        self.assertFalse(sequence.can_abort(lap(4550.0)))

    def test_damage_and_wear_stops_are_never_aborted(self):
        for reason in ('damage', 'wear', 'cover', 'pre_empt'):
            sequence, _ = new_sequence(stored=4600.0)
            sequence.box_called(reason, lap(1000.0))
            self.assertFalse(sequence.can_abort(lap(2000.0)), reason)

    def test_a_must_stop_reason_replaces_an_abortable_one_on_the_same_lap(self):
        sequence, _ = new_sequence(stored=4600.0)
        sequence.box_called('window', lap(1000.0))
        sequence.box_called('damage', lap(1500.0))
        self.assertFalse(sequence.can_abort(lap(2000.0)))


class OutLapBriefTests(unittest.TestCase):
    def neighbours(self, gap_ahead=2.1, gap_behind=3.4):
        return lambda: [
            ps.Neighbour('behind', 5, 'Norris', gap_behind, 'mediums', 1, 1),
            ps.Neighbour('ahead', 4, 'Hamilton', gap_ahead, 'hards', 20, 0),
        ]

    def run_stop(self, neighbours, exit_position=5):
        sequence, speak = new_sequence(stored=4600.0)
        t = drive(sequence, [lap(4500.0, position=3), lap(4610.0, pit=1, position=3)])
        t = drive(sequence, [lap(300.0, 21, pit=1, stops=1, position=exit_position)], t)
        drive(
            sequence,
            [lap(500.0 + i * 10, 21, stops=1, position=exit_position) for i in range(80)],
            t,
            0.2,
            neighbours,
        )
        return sequence, speak

    def test_brief_names_places_lost_cars_around_and_tyre_ages(self):
        sequence, speak = self.run_stop(self.neighbours())
        self.assertEqual(speak.keys(), ['pit_brief'])
        brief = speak.text('pit_brief')
        self.assertTrue(brief.startswith('Lost two places in the stop, P5.'), brief)
        self.assertIn('Hamilton ahead, two point one, on twenty-lap-old hards, yet to stop.', brief)
        self.assertIn('Norris behind, three point four, on fresh mediums.', brief)
        self.assertLess(brief.index('Hamilton'), brief.index('Norris'))
        self.assertIn('go and get Hamilton', brief)
        self.assertEqual(sequence.state, ps.IDLE)

    def test_close_car_behind_gets_a_keep_them_behind(self):
        _, speak = self.run_stop(self.neighbours(gap_behind=0.8), exit_position=3)
        brief = speak.text('pit_brief')
        self.assertNotIn('in the stop', brief)  # same place as before the stop
        self.assertIn('keep Norris behind', brief)

    def test_brief_waits_for_gaps_but_not_for_ever(self):
        _, speak = self.run_stop(self.neighbours(gap_ahead=None))
        brief = speak.text('pit_brief')
        self.assertIn('Hamilton ahead, on twenty-lap-old hards', brief)

    def test_drive_through_gets_no_brief(self):
        sequence, speak = new_sequence(stored=4600.0)
        t = drive(sequence, [lap(4500.0), lap(4610.0, pit=1)])
        drive(sequence, [lap(500.0 + i * 10, 21) for i in range(80)], t, 0.2, self.neighbours())
        self.assertEqual(speak.keys(), [])


class EngineSequenceTests(unittest.TestCase):
    """The sequence wired into the engine: strategy calls arm it, rival stops abort it."""

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
            lap_distance=1000.0,
        )
        self.engine.latest_lap = self.lap
        rivals = [car_status(MEDIUM, 5)] * 22
        rivals[1] = car_status(MEDIUM, 26)
        rivals[2] = car_status(MEDIUM, 2)
        self.engine.update_car_status(car_status(MEDIUM, 22), rivals)
        self.voice.messages.clear()

    def spoken(self):
        return [(kwargs.get('key', '').split(':')[-1], m) for m, _, kwargs in self.voice.messages]

    def arm_undercut(self):
        self.engine.strategy_plan()
        plan = self.engine._strategy_cache[1]
        plan.best.stops, plan.best.pit_laps = 1, [21]
        self.engine._strategy_cache = (self.engine._strategy_cache[0], plan, (2.5, 1.0))
        self.engine.radio.car_ahead_index, self.engine.radio.gap_ahead_s = 1, 0.7
        self.engine.update_strategy_calls(self.lap)

    def test_undercut_call_arms_the_sequence(self):
        self.arm_undercut()
        self.assertTrue(self.engine.pit_sequence.pending)
        self.assertEqual(self.engine.pit_sequence.call.reason, 'undercut')
        self.assertEqual(self.engine.pit_sequence.call.car_index, 1)
        self.assertEqual(self.engine.strategy_summary()['pit_call']['state'], 'boxing')

    def test_undercut_target_boxing_first_means_stay_out(self):
        self.arm_undercut()
        self.engine.radio.rival_pit_events = [
            {'car_index': 1, 'gap_s': 0.7, 'lap': 20, 'position': 2, 'session_time': 100.0}
        ]
        self.engine.update_strategy_calls(self.lap)
        last_key, last = self.spoken()[-1]
        self.assertEqual(last_key, 'pit_abort')
        self.assertTrue(last.startswith('Stay out, stay out.'), last)
        self.assertEqual(self.engine.pit_sequence.state, ps.ABORTED)

    def test_rival_stop_lines_are_held_back_while_we_are_boxing(self):
        self.engine.pit_sequence.box_called('damage', self.lap)
        self.engine.radio.rival_pit_events = [
            {'car_index': 2, 'gap_s': -0.8, 'lap': 20, 'position': 4, 'session_time': 100.0}
        ]
        self.engine.update_strategy_calls(self.lap)
        self.assertEqual(self.spoken(), [])
        self.assertTrue(self.engine.pit_sequence.pending)

    def test_wing_damage_call_arms_and_new_session_resets(self):
        self.engine.radio.update_damage_alert(10.0, 60, now=1000.0)
        self.assertTrue(self.engine.pit_sequence.pending)
        self.assertEqual(self.engine.pit_sequence.call.reason, 'damage')
        self.engine.update_session(replace(RACE, track_id=11), 2, 0)
        self.assertEqual(self.engine.pit_sequence.state, ps.IDLE)

    def test_box_call_kept_off_air_does_not_arm(self):
        self.engine.on_demand = True
        self.engine.config.radio.proactive_keys = ()
        self.engine.radio.update_damage_alert(10.0, 60, now=1000.0)
        self.assertFalse(self.engine.pit_sequence.pending)

    def plan_b(self, pit_laps):
        import race_sim
        import strategy_replan

        new = race_sim.Candidate('plan', len(pit_laps), ['MEDIUM', 'HARD'][: len(pit_laps) + 1], pit_laps)
        old = race_sim.Candidate('old', 1, ['MEDIUM', 'HARD'], [20])
        message = 'Plan B, Plan B. ' + ('Box this lap. Box, box.' if pit_laps else 'No more stops.')
        call = strategy_replan.PlanBCall(message, new, old, 20, ['lap'], 5.0, 0.0, 0.9)
        self.engine.replanner.consider = lambda *args: call

    def test_plan_b_that_drops_the_stop_aborts_the_box_call(self):
        self.engine.pit_sequence.box_called('window', self.lap)
        self.plan_b([])
        self.engine._consider_plan_b()
        last_key, last = self.spoken()[-1]
        self.assertEqual(last_key, 'pit_abort')
        self.assertEqual(last, 'Stay out, stay out. Plan B, Plan B. No more stops.')

    def test_plan_b_box_this_lap_arms_the_sequence(self):
        self.plan_b([20])
        self.engine._consider_plan_b()
        self.assertEqual(self.spoken()[-1][0], 'plan_b')
        self.assertEqual(self.engine.pit_sequence.call.reason, 'plan_b')

    def test_safety_car_box_decision_arms_and_stay_out_cancels(self):
        self.engine.last_strategy_call = {'kind': 'sc_box'}
        self.engine.radio.update_safety_car(1, decide=lambda status: 'Box, box, cheap stop.')
        self.assertTrue(self.engine.pit_sequence.pending)
        self.assertEqual(self.engine.pit_sequence.call.reason, 'safety_car')
        self.assertFalse(self.engine.pit_sequence.can_abort(self.lap))

        self.engine.pit_sequence.reset(RACE.track_length, racing=True)
        self.engine.pit_sequence.box_called('window', self.lap)
        self.engine.last_strategy_call = {'kind': 'sc_stay_out'}
        self.engine.radio.update_safety_car(2, decide=lambda status: 'Stay out.')
        self.assertEqual(self.engine.pit_sequence.state, ps.ABORTED)
        self.assertNotIn('pit_abort', [key for key, _ in self.spoken()])  # the SC call said it

    def test_new_keys_break_quiet_mode_and_skip_the_budget(self):
        import radio_discipline
        from pipeline.config import DEFAULT_PROACTIVE_KEYS

        for key in ('pit_confirm', 'pit_abort', 'pit_brief'):
            self.assertIn(key, DEFAULT_PROACTIVE_KEYS)
            self.assertIn(key, radio_discipline.URGENT_TOPICS)


if __name__ == '__main__':
    unittest.main()
