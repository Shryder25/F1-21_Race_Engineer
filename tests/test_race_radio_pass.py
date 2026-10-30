"""Race radio pass: session-aware calls, race call budget, quiet windows, the Safety Car
decision in the same call, DRS calls as information, and pace stats that skip lap 1 and
in/out laps."""

import pathlib
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import lap_coaching
import race_strategy
import radio_discipline
from analysis import AnalysisEngine
from packets import SessionData, parse_player_lap_data
from race_radio import RaceRadio
from simulate_telemetry import build_lap_packet
from test_analysis import LapValidityTests, SilentVoice
from voice_engine import Priority

RACE, QUALI = 10, 5


def corner(seconds, brake_point=None):
    return {'start_time_ms': 0, 'end_time_ms': int(seconds * 1000), 'brake_point': brake_point}


class CallBudgetPerSessionTests(unittest.TestCase):
    def test_race_gets_the_tighter_limit(self):
        budget = radio_discipline.CallBudget(4, 2)
        self.assertEqual(budget.per_lap, 4)
        budget.set_race(True)
        self.assertEqual(budget.per_lap, 2)
        budget.set_race(False)
        self.assertEqual(budget.per_lap, 4)

    def test_race_limit_never_raises_the_base_and_no_limit_stays_unlimited(self):
        budget = radio_discipline.CallBudget(1, 2)
        budget.set_race(True)
        self.assertEqual(budget.per_lap, 1)
        unlimited = radio_discipline.CallBudget(0, 2)
        unlimited.set_race(True)
        self.assertIsNone(unlimited.per_lap)

    def test_engine_switches_budget_with_the_session(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(SessionData(0, 30, 20, 20, 1000, QUALI, 11, 0))
        self.assertEqual(engine.call_budget.per_lap, 4)
        engine.update_session(SessionData(0, 30, 20, 20, 1000, RACE, 11, 0))
        self.assertEqual(engine.call_budget.per_lap, 2)


class QuietWindowTests(unittest.TestCase):
    def test_lap_one_final_laps_and_drs_fight(self):
        quiet = radio_discipline.race_quiet_reason
        self.assertEqual(quiet(1, 50, None, None), 'lap_1')
        self.assertEqual(quiet(49, 50, None, None), 'final_laps')
        self.assertEqual(quiet(50, 50, None, None), 'final_laps')
        self.assertIsNone(quiet(48, 50, None, None))
        self.assertEqual(quiet(20, 50, 0.8, None), 'drs_fight')
        self.assertEqual(quiet(20, 50, None, 0.6), 'drs_fight')
        self.assertIsNone(quiet(20, 50, 1.4, 2.0))

    def test_the_drs_call_itself_is_not_held_by_the_fight(self):
        self.assertIsNone(radio_discipline.race_quiet_reason(20, 50, 0.8, None, 'drs_ahead'))

    def _engine(self, session_type, lap_num, total=20):
        voice = SilentVoice()
        engine = AnalysisEngine(voice)
        engine.update_session(SessionData(0, 30, 20, total, 1000, session_type, 11, 0))
        engine.latest_lap = replace(
            parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 3), 0), current_lap_num=lap_num
        )
        voice.messages.clear()
        return engine, voice

    def test_race_lap_one_holds_routine_calls_but_not_urgent_ones(self):
        engine, voice = self._engine(RACE, 1)
        engine.speak('Last three laps averaging...', Priority.RACE, key='race_pace')
        engine.speak('Safety Car, Safety Car.', Priority.CRITICAL, key='safety_car')
        self.assertEqual([m for m, _, _ in voice.messages], ['Safety Car, Safety Car.'])

    def test_final_two_laps_are_quiet_unless_critical(self):
        engine, voice = self._engine(RACE, 19, total=20)
        engine.speak('Gap ahead two point one.', Priority.RACE, key='race_gaps')
        engine.speak('Box, box.', Priority.CRITICAL, key='critical_wear')
        self.assertEqual([m for m, _, _ in voice.messages], ['Box, box.'])

    def test_drs_fight_holds_routine_calls(self):
        engine, voice = self._engine(RACE, 8)
        engine.radio.gap_behind_s = 0.7
        engine.speak('Gap ahead two point one.', Priority.RACE, key='race_gaps')
        self.assertEqual(voice.messages, [])
        engine.radio.gap_behind_s = 1.5
        engine.speak('Gap ahead two point one.', Priority.RACE, key='race_gaps')
        self.assertEqual(len(voice.messages), 1)

    def test_qualifying_has_no_quiet_windows(self):
        engine, voice = self._engine(QUALI, 1)
        engine.speak('One twenty-nine four.', Priority.TIMING, key='lap_time')
        self.assertEqual(len(voice.messages), 1)


class RaceLapCallTests(unittest.TestCase):
    """Race: no "X off your best" every lap. Qualifying keeps it."""

    def drive(self, session_type, laps):
        fixture = LapValidityTests()
        fixture.setUp()
        engine = fixture.engine
        engine.update_session(SessionData(0, 30, 20, 20, 1000, session_type, 11, 0))
        voice = engine.voice
        for lap_num, lap_ms in enumerate(laps, 1):
            for distance in range(0, 1000, 20):
                engine.update_telemetry(
                    fixture.telemetry,
                    replace(
                        fixture.lap,
                        current_lap_num=lap_num,
                        lap_distance=distance,
                        current_lap_time_ms=100 + distance * 50,
                    ),
                )
            engine.update_telemetry(
                fixture.telemetry,
                replace(
                    fixture.lap,
                    current_lap_num=lap_num + 1,
                    lap_distance=0,
                    current_lap_time_ms=100,
                    last_lap_time_ms=lap_ms,
                ),
            )
        return [m for m, _, _ in voice.messages]

    def test_race_never_says_off_your_best(self):
        messages = self.drive(RACE, [52000, 50000, 50400, 50600])
        self.assertFalse(any('off your best' in m for m in messages), messages)
        self.assertFalse(any('up on your best' in m or 'down on your best' in m for m in messages))

    def test_qualifying_keeps_lap_coaching(self):
        messages = self.drive(QUALI, [50000, 50400])
        self.assertTrue(any('off your best' in m for m in messages), messages)


class RaceCornerCoachingTests(unittest.TestCase):
    best = {'Turn 1': corner(5.0), 'Turn 4': corner(8.0), 'Turn 9': corner(6.0)}

    def lap(self, t4=8.0, drift=0.0):
        return {
            'Turn 1': corner(5.0 * (1 + drift)),
            'Turn 4': corner(t4 * (1 + drift), brake_point=380),
            'Turn 9': corner(6.0 * (1 + drift)),
        }

    def test_only_a_corner_lost_three_laps_running_gets_a_call(self):
        streaks = lap_coaching.CornerLossStreaks()
        best = dict(self.best, **{'Turn 4': corner(8.0, brake_point=400)})
        for lap_num in (2, 3):
            streaks.observe(lap_num, self.lap(t4=8.3), best)
            self.assertIsNone(streaks.line(lap_num, self.lap(t4=8.3), best))
        streaks.observe(4, self.lap(t4=8.3), best)
        line = streaks.line(4, self.lap(t4=8.3), best)
        self.assertIn('Turn 4 again, three laps running', line)
        self.assertIn('three tenths', line)
        self.assertIn('twenty metres early', line)
        # ...and not again on the very next lap
        streaks.observe(5, self.lap(t4=8.3), best)
        self.assertIsNone(streaks.line(5, self.lap(t4=8.3), best))

    def test_tyre_drift_across_every_corner_is_not_coaching(self):
        streaks = lap_coaching.CornerLossStreaks()
        for lap_num in range(2, 8):
            streaks.observe(lap_num, self.lap(drift=0.03), self.best)
            self.assertIsNone(streaks.line(lap_num, self.lap(drift=0.03), self.best))

    def test_a_skipped_lap_breaks_the_streak(self):
        streaks = lap_coaching.CornerLossStreaks()
        streaks.observe(2, self.lap(t4=8.3), self.best)
        streaks.observe(3, self.lap(t4=8.3), self.best)
        streaks.observe(5, self.lap(t4=8.3), self.best)  # lap 4 was a pit / SC / invalid lap
        self.assertIsNone(streaks.line(5, self.lap(t4=8.3), self.best))


class SafetyCarDecisionTests(unittest.TestCase):
    def test_old_tyres_box_under_the_safety_car(self):
        decision = race_strategy.safety_car_decision(30, 'MEDIUM', 22, 21, 21.0, 0.55, {'MEDIUM'})
        self.assertTrue(decision.box)
        self.assertTrue(decision.message.startswith('Box, box. Box this lap for'))
        self.assertIn('cheap stop under the Safety Car', decision.message)

    def test_fresh_tyres_with_no_stop_left_stay_out(self):
        decision = race_strategy.safety_car_decision(15, 'HARD', 3, 36, 21.0, 0.55, {'MEDIUM', 'HARD'})
        self.assertFalse(decision.box)
        self.assertTrue(decision.message.startswith('Stay out'))

    def test_too_late_to_stop(self):
        decision = race_strategy.safety_car_decision(2, 'MEDIUM', 25, 49, 21.0, 0.55, {'MEDIUM', 'HARD'})
        self.assertFalse(decision.box)

    def test_decision_is_in_the_same_call_as_the_safety_car(self):
        voice = SilentVoice()
        radio = RaceRadio(voice.say)
        radio.reset(SessionData(0, 30, 20, 50, 5000, RACE, 11, 0))
        radio.update_safety_car(
            1, lambda status: 'Box, box. Box this lap for hards, cheap stop under the Safety Car.'
        )
        self.assertEqual(len(voice.messages), 1)
        message, priority, kwargs = voice.messages[0]
        self.assertEqual(
            message,
            'Safety Car, Safety Car. Box, box. Box this lap for hards, cheap stop under the Safety Car.',
        )
        self.assertEqual(priority, Priority.CRITICAL)
        radio.update_safety_car(0)
        radio.update_safety_car(2, lambda status: 'Stay out, we keep track position.')
        self.assertEqual(voice.messages[-1][0], 'VSC, VSC. Stay out, we keep track position.')

    def test_no_decision_keeps_the_plain_call(self):
        voice = SilentVoice()
        radio = RaceRadio(voice.say)
        radio.reset(SessionData(0, 30, 20, 50, 5000, RACE, 11, 0))
        radio.update_safety_car(1, lambda status: None)
        self.assertIn('Keep the delta positive', voice.messages[0][0])


class DrsInformationTests(unittest.TestCase):
    def setUp(self):
        self.voice = SilentVoice()
        self.radio = RaceRadio(self.voice.say)
        self.radio.reset(SessionData(0, 30, 20, 53, 5793, RACE, 11, 0))
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)

    def field(self, t, ahead_m, behind_m):
        player = 1000 + 50 * t
        return [
            replace(self.lap, car_position=1, total_distance=player + ahead_m, result_status=2),
            replace(self.lap, car_position=2, total_distance=player, result_status=2),
            replace(self.lap, car_position=3, total_distance=player - behind_m, result_status=2),
        ]

    def test_drs_calls_give_facts_not_instructions(self):
        self.radio.set_driver_names({0: 'Hamilton', 2: 'Verstappen'})
        self.radio.tyre_note = lambda index: 'on hards, nine laps older than yours' if index == 0 else None
        for t in range(0, 10):
            self.radio.update_field(self.field(t, 40, 35), 1, float(t), now=float(t))
        calls = {kw['key']: m for m, _, kw in self.voice.messages}
        self.assertIn('Hamilton', calls['drs_ahead'])
        self.assertIn('They are on hards, nine laps older than yours.', calls['drs_ahead'])
        self.assertIn('Verstappen', calls['drs_behind'])
        for message in calls.values():
            for instruction in ('defend', 'cover', 'go get', 'braking zone', 'is yours'):
                self.assertNotIn(instruction, message.lower())


class PaceStatTests(unittest.TestCase):
    def test_race_pace_skips_the_standing_start(self):
        voice = SilentVoice()
        radio = RaceRadio(voice.say)
        radio.reset(SessionData(0, 30, 20, 50, 5000, RACE, 11, 0))
        for lap_num, lap_ms in ((1, 99000), (2, 91000), (3, 91000)):
            radio.completed_lap(lap_ms, lap_num, {}, {})
        self.assertIsNone(radio.race_pace_ms)
        radio.completed_lap(91000, 4, {}, {})
        self.assertEqual(radio.race_pace_ms, 91000)

    def test_rival_pace_skips_in_and_out_laps(self):
        voice = SilentVoice()
        radio = RaceRadio(voice.say)
        radio.reset(SessionData(0, 30, 20, 50, 5000, RACE, 11, 0))
        base = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)

        def frame(t, lap_num, rival_last, rival_pit=0):
            player = 1000 + 50 * t
            return [
                replace(
                    base,
                    car_position=2,
                    total_distance=player,
                    result_status=2,
                    current_lap_num=lap_num,
                    last_lap_time_ms=90000,
                ),
                replace(
                    base,
                    car_position=1,
                    total_distance=player + 100,
                    result_status=2,
                    current_lap_num=lap_num,
                    last_lap_time_ms=rival_last,
                    pit_status=rival_pit,
                ),
            ]

        radio.update_field(frame(0, 5, 90500), 0, 0.0, now=0.0)
        self.assertEqual(radio.last_lap_times_ms[1], 90500)
        radio.update_field(frame(1, 6, 90500, rival_pit=1), 0, 1.0, now=1.0)  # in the pit lane, lap 6
        radio.update_field(frame(2, 7, 112000), 0, 2.0, now=2.0)  # lap 6 was the in-lap
        self.assertEqual(radio.last_lap_times_ms[1], 90500)
        radio.update_field(frame(3, 8, 95000), 0, 3.0, now=3.0)  # lap 7 was the out-lap
        self.assertEqual(radio.last_lap_times_ms[1], 90500)
        radio.update_field(frame(4, 9, 89800), 0, 4.0, now=4.0)  # lap 8: a clean lap again
        self.assertEqual(radio.last_lap_times_ms[1], 89800)
        self.assertTrue(radio.near_pit_stop(1, 6) and radio.near_pit_stop(1, 7))
        self.assertFalse(radio.near_pit_stop(1, 8))


class TyreNoteTests(unittest.TestCase):
    def test_tyre_note_compares_with_the_player(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(SessionData(0, 30, 20, 50, 5000, RACE, 11, 0))
        engine.latest_status = SimpleNamespace(compound_name='Medium', tyres_age_laps=5)
        engine.all_status = [
            SimpleNamespace(compound_name='Hard', tyres_age_laps=14),
            SimpleNamespace(compound_name='Medium', tyres_age_laps=6),
            SimpleNamespace(compound_name='Soft', tyres_age_laps=4),
        ]
        self.assertEqual(engine.rival_tyre_note(0), 'on hards, nine laps older than yours')
        self.assertIsNone(engine.rival_tyre_note(1))  # same tyres, same age
        self.assertEqual(engine.rival_tyre_note(2), 'on four lap old softs')


if __name__ == '__main__':
    unittest.main()
