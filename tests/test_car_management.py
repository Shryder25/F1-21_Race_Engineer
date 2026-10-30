"""Car management radio calls: fuel, ERS, brakes, engine temperature, damage (car_management.py)."""

import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))

import car_management
import radio_discipline
from analysis import PROACTIVE_KEYS, AnalysisEngine
from packets import CarDamageData, CarStatusData, SessionData, parse_player_lap_data, parse_player_telemetry
from pipeline.config import CarConfig
from simulate_telemetry import build_lap_packet, build_telemetry_packet
from test_analysis import SilentVoice
from voice_engine import Priority

CORNERS = [{'name': 'Turn 4', 'start': 1000, 'end': 1200}]


def lap(num, distance=500.0, pit=0):
    return SimpleNamespace(current_lap_num=num, lap_distance=distance, pit_status=pit)


def telemetry(speed=200, brakes=(500, 500, 600, 600), engine=105):
    return SimpleNamespace(speed=speed, brakes_temperature=brakes, engine_temperature=engine)


def status(fuel_laps=0.5, mix=1, ers_pct=50.0, ers_mode=1):
    return SimpleNamespace(
        fuel_remaining_laps=fuel_laps,
        fuel_mix=mix,
        ers_store_energy=ers_pct / 100 * 4_000_000,
        ers_deploy_mode=ers_mode,
    )


def damage(tyres=(0, 0, 0, 0), engine=0, gearbox=0, drs_fault=0):
    return SimpleNamespace(tyres_damage=tyres, engine=engine, gearbox=gearbox, drs_fault=drs_fault)


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, message, priority=Priority.NORMAL, **options):
        self.calls.append((options.get('key'), message, priority))

    def keys(self):
        return [key for key, _, _ in self.calls]

    def messages(self, key):
        return [message for k, message, _ in self.calls if k == key]


class CarManagerTestCase(unittest.TestCase):
    def setUp(self):
        self.speak = Recorder()
        self.manager = car_management.CarManager(self.speak, CarConfig())
        self.now = 0.0

    def frame(self, lap_num, tel=None, st=None, dmg=None, *, race=True, total=20, distance=500.0, gap=None):
        self.now += 0.1
        self.manager.update(
            self.now,
            lap(lap_num, distance),
            tel or telemetry(),
            st,
            dmg,
            race=race,
            total_laps=total,
            corners=CORNERS,
            gap_ahead_s=gap,
        )

    def cross_line(self, to_lap, st, **kw):
        self.frame(to_lap - 1, st=st, **kw)
        self.frame(to_lap, st=st, **kw)


class FuelTests(CarManagerTestCase):
    def test_margin_reading_as_game_delta(self):
        # 19 laps to go and the game says -0.6: that is the margin itself.
        self.cross_line(2, status(fuel_laps=-0.6))
        self.assertEqual(self.manager.state.fuel_reading, 'delta')
        self.assertEqual(
            self.speak.messages('fuel_save'),
            ['Fuel is tight, we are nought point six laps short. Lean mix, lift and coast.'],
        )

    def test_margin_reading_as_laps_of_fuel(self):
        # 19 laps to go and the game says 19.1 laps of fuel: a 0.1 lap margin.
        self.cross_line(2, status(fuel_laps=19.1))
        self.assertEqual(self.manager.state.fuel_reading, 'absolute')
        self.assertAlmostEqual(self.manager.state.fuel_margin, 0.1)
        self.assertEqual(
            self.speak.messages('fuel_save'), ['Fuel is marginal. Lift slightly into the big stops.']
        )

    def test_save_call_mentions_lean_mix_already_set(self):
        self.cross_line(2, status(fuel_laps=-1.0, mix=0))
        self.assertIn('Stay lean', self.speak.messages('fuel_save')[0])

    def test_save_call_repeats_only_every_few_laps_and_is_urgent(self):
        for n in range(2, 9):
            self.cross_line(n, status(fuel_laps=-1.0))
        self.assertEqual(len(self.speak.messages('fuel_save')), 3)  # laps 2, 5, 8
        self.assertTrue(radio_discipline.is_urgent('fuel_save', Priority.STRATEGY))
        self.assertIn('fuel_save', PROACTIVE_KEYS)

    def test_good_fuel_is_said_once_and_is_chatty_only(self):
        for n in range(2, 6):
            self.cross_line(n, status(fuel_laps=1.5))
        self.assertEqual(
            self.speak.messages('fuel_status'), ['Fuel is good, one point five laps spare. You can push.']
        )
        self.assertNotIn('fuel_status', PROACTIVE_KEYS)

    def test_back_on_target_after_saving(self):
        self.cross_line(2, status(fuel_laps=-1.0))
        self.cross_line(3, status(fuel_laps=0.5))
        self.assertEqual(self.speak.messages('fuel_status'), ['Fuel is back on target, normal running.'])

    def test_no_fuel_calls_outside_a_race_or_in_the_pits(self):
        self.cross_line(2, status(fuel_laps=-1.0), race=False)
        self.manager.update(1.0, lap(2, pit=1), telemetry(), status(fuel_laps=-1.0), race=True, total_laps=20)
        self.manager.update(1.1, lap(3, pit=1), telemetry(), status(fuel_laps=-1.0), race=True, total_laps=20)
        self.assertEqual(self.speak.calls, [])

    def test_laps_words(self):
        self.assertEqual(car_management.laps_words(0.5), 'half a lap')
        self.assertEqual(car_management.laps_words(-2.0), 'two laps')
        self.assertEqual(car_management.laps_words(1.0), 'one lap')
        self.assertEqual(car_management.laps_words(1.96), 'two laps')


class ErsTests(CarManagerTestCase):
    def test_low_battery_in_overtake_mode(self):
        self.cross_line(3, status(ers_pct=8, ers_mode=3))
        self.assertEqual(
            self.speak.messages('ers'), ['Battery is low, eight percent. Off overtake, harvest this lap.']
        )

    def test_full_battery_all_lap(self):
        self.cross_line(3, status(ers_pct=99))
        self.assertEqual(len(self.speak.messages('ers')), 1)
        self.assertIn('full', self.speak.messages('ers')[0])

    def test_full_at_the_line_after_using_it_is_quiet(self):
        self.frame(2, st=status(ers_pct=40))
        self.frame(2, st=status(ers_pct=99))
        self.frame(3, st=status(ers_pct=99))
        self.assertEqual(self.speak.messages('ers'), [])

    def test_ers_calls_wait_when_the_lap_budget_is_tight(self):
        self.manager.chat_ok = lambda: False
        self.cross_line(3, status(ers_pct=5))
        self.assertEqual(self.speak.calls, [])


class BrakeTests(CarManagerTestCase):
    def lap_through_turn_4(self, lap_num, fl_temp):
        self.frame(lap_num, distance=900)
        self.frame(lap_num, telemetry(brakes=(500, 500, fl_temp, 600)), distance=1100)
        self.frame(lap_num, distance=1500)

    def test_brake_spiking_in_the_same_corner_two_laps_running(self):
        self.lap_through_turn_4(1, 1100)
        self.lap_through_turn_4(2, 1100)
        self.assertEqual(self.speak.messages('brake_temp'), [])
        self.lap_through_turn_4(3, 900)  # lap 2 ends: second hot lap in a row
        calls = self.speak.messages('brake_temp')
        self.assertEqual(len(calls), 1)
        self.assertIn('Front left brake is spiking into Turn 4', calls[0])
        self.assertIn('rearward', calls[0])
        self.assertNotIn('brake earlier', calls[0].lower())  # car info, not a braking tip

    def test_one_hot_lap_is_not_a_trend(self):
        self.lap_through_turn_4(1, 1100)
        self.lap_through_turn_4(2, 900)
        self.lap_through_turn_4(3, 1100)
        self.lap_through_turn_4(4, 900)
        self.assertEqual(self.speak.messages('brake_temp'), [])


class EngineTests(CarManagerTestCase):
    def run_engine(self, temp, seconds, gap=None):
        for _ in range(int(seconds / 0.1)):
            self.frame(5, telemetry(engine=temp), gap=gap)

    def test_short_spike_is_quiet(self):
        self.run_engine(125, 3)
        self.run_engine(105, 1)
        self.assertEqual(self.speak.calls, [])

    def test_hot_engine_in_the_dirty_air(self):
        self.run_engine(122, 6, gap=0.6)
        calls = self.speak.messages('engine_temp')
        self.assertEqual(len(calls), 1)
        self.assertIn('dirty air', calls[0])
        self.run_engine(122, 30, gap=0.6)
        self.assertEqual(len(self.speak.messages('engine_temp')), 1)  # cooldown

    def test_critical_engine_is_urgent(self):
        self.run_engine(132, 6)
        self.assertEqual(len(self.speak.messages('engine_temp_critical')), 1)
        self.assertIn('engine_temp_critical', PROACTIVE_KEYS)


class DamageTests(CarManagerTestCase):
    def test_puncture_is_called_once_per_set(self):
        self.frame(4, dmg=damage(tyres=(2, 2, 3, 3)))
        self.frame(4, dmg=damage(tyres=(2, 2, 3, 80)))
        self.frame(4, dmg=damage(tyres=(2, 2, 3, 100)))
        self.assertEqual(
            self.speak.messages('puncture'), ['Puncture, puncture, front right. Box this lap, box box.']
        )
        self.assertEqual(self.speak.calls[0][2], Priority.CRITICAL)
        # New tyres, then another puncture on the same corner: called again.
        self.frame(5, dmg=damage(tyres=(0, 0, 0, 0)))
        self.frame(5, dmg=damage(tyres=(0, 0, 0, 95)))
        self.assertEqual(len(self.speak.messages('puncture')), 2)

    def test_gradual_tyre_damage_is_not_a_puncture(self):
        for d in range(0, 60, 5):
            self.frame(4, dmg=damage(tyres=(d, d, d, d)))
        self.assertEqual(self.speak.messages('puncture'), [])

    def test_damage_present_at_start_is_only_a_baseline(self):
        self.frame(1, dmg=damage(engine=60, gearbox=55))
        self.assertEqual(self.speak.calls, [])

    def test_gearbox_damage_levels_and_jumps(self):
        self.frame(4, dmg=damage(gearbox=45))
        self.frame(4, dmg=damage(gearbox=52))
        self.frame(4, dmg=damage(gearbox=55))
        self.frame(4, dmg=damage(gearbox=80))
        calls = self.speak.messages('gearbox_damage')
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], 'Gearbox damage at fifty-two percent. Short shift and stay off the kerbs.')
        self.assertIn('critical, eighty percent', calls[1])

    def test_sudden_engine_damage_from_contact(self):
        self.frame(4, dmg=damage(engine=5))
        self.frame(4, dmg=damage(engine=25))
        self.assertEqual(
            self.speak.messages('engine_damage'),
            ['We are seeing engine damage, now twenty-five percent. Lean mix and short shift.'],
        )

    def test_drs_fault_once(self):
        self.frame(4, dmg=damage(drs_fault=1))
        self.frame(4, dmg=damage(drs_fault=1))
        self.assertEqual(self.speak.keys(), ['drs_fault'])


class EngineWiringTests(unittest.TestCase):
    """Through the full engine: quiet mode hears urgent car calls only; state resets per session."""

    def setUp(self):
        self.engine = AnalysisEngine(SilentVoice(), on_demand=True)
        self.engine.update_session(SessionData(0, 30, 20, 20, 5000, 10, 11, 0), session_uid=1)

    def drive_lap(self, lap_num, fuel_laps):
        st = CarStatusData(
            fuel_in_tank=20.0,
            fuel_capacity=110.0,
            fuel_remaining_laps=fuel_laps,
            actual_tyre_compound=17,
            visual_tyre_compound=17,
            tyres_age_laps=lap_num,
            fia_flags=0,
            ers_store_energy=4_000_000.0,
            ers_deploy_mode=1,
            ers_deployed_this_lap=0.0,
        )
        self.engine.update_car_status(st)
        tel = parse_player_telemetry(build_telemetry_packet(1, 1, 200, 1, 0, 6, 95), 0)
        lap_data = parse_player_lap_data(build_lap_packet(1, 1, 300.0, 30000, lap_num), 0)
        self.engine.update_telemetry(tel, lap_data)

    def said(self):
        return [str(kw.get('key', '')).split(':')[-1] for _, _, kw in self.engine.voice.messages]

    def test_quiet_mode_hears_fuel_save_but_not_good_fuel_or_ers(self):
        self.drive_lap(1, -1.0)
        self.drive_lap(2, -1.0)
        self.assertIn('fuel_save', self.said())
        self.assertNotIn('ers', self.said())

        quiet_good = AnalysisEngine(SilentVoice(), on_demand=True)
        quiet_good.update_session(SessionData(0, 30, 20, 20, 5000, 10, 11, 0), session_uid=2)
        self.engine = quiet_good
        self.drive_lap(1, 3.0)
        self.drive_lap(2, 3.0)
        self.assertNotIn('fuel_status', self.said())
        self.assertEqual(self.engine.car_manager.state.fuel_band, 'good')

    def test_new_session_resets_car_state(self):
        self.drive_lap(1, -1.0)
        self.drive_lap(2, -1.0)
        self.assertIsNotNone(self.engine.car_manager.state.fuel_reading)
        self.engine.update_session(SessionData(0, 30, 20, 20, 5000, 10, 11, 0), session_uid=99)
        self.assertIsNone(self.engine.car_manager.state.fuel_reading)

    def test_snapshot_reaches_the_ai_context(self):
        self.engine.update_car_damage(
            CarDamageData(
                tyres_wear=(5, 5, 5, 5),
                tyres_damage=(0, 0, 0, 0),
                front_left_wing=0,
                front_right_wing=0,
                rear_wing=0,
                floor=0,
                gearbox=12,
                engine=7,
            )
        )
        self.drive_lap(1, -1.0)
        self.drive_lap(2, -1.0)
        car = self.engine.build_context()['car_management']
        self.assertEqual(car['fuel_status'], 'save')
        self.assertEqual(car['gearbox_damage_pct'], 12)
        self.assertEqual(car['fuel_mix'], 'standard')


class RaceBudgetTests(unittest.TestCase):
    def test_chatty_car_calls_still_fit_the_two_call_race_budget(self):
        engine = AnalysisEngine(SilentVoice(), on_demand=False)
        engine.update_session(SessionData(0, 30, 20, 20, 5000, 10, 11, 0), session_uid=3)
        engine.call_budget.set_race(True)
        engine.call_budget.new_lap(5)
        self.assertEqual(engine.call_budget.remaining, 2)
        self.assertTrue(engine._car_chat_ok())
        engine.call_budget.spend('lap_time', Priority.LOW)
        self.assertFalse(engine._car_chat_ok())  # the last call is kept for race info


if __name__ == '__main__':
    unittest.main()
