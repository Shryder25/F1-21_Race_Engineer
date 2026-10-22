"""Participants packet decoding and driver names in gap, threat and pace radio calls."""

import pathlib
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import web_hud
from analysis import AnalysisEngine
from engineer_brain import format_context, rule_based_answer
from packets import (
    PARTICIPANTS_PACKET_SIZE,
    ParticipantData,
    SessionData,
    parse_participants,
    parse_player_lap_data,
)
from simulate_telemetry import build_lap_packet, build_participants_packet, build_session_packet
from test_analysis import SilentVoice


def participant(name):
    return ParticipantData(True, 0, 0, 44, 0, name, True)


class ParticipantsPacketTests(unittest.TestCase):
    def test_packet_size_matches_2021_spec(self):
        self.assertEqual(PARTICIPANTS_PACKET_SIZE, 1257)
        self.assertEqual(len(build_participants_packet(0, 1)), 1257)

    def test_parses_every_active_car_with_names_and_race_numbers(self):
        cars = parse_participants(build_participants_packet(0, 1))
        self.assertEqual(len(cars), 22)
        self.assertEqual(cars[1].name, 'HAMILTON')
        self.assertEqual(cars[1].race_number, 2)
        self.assertTrue(cars[1].ai_controlled)
        self.assertFalse(cars[0].ai_controlled)

    def test_only_active_cars_are_returned(self):
        cars = parse_participants(build_participants_packet(0, 1, ['Player', 'NORRIS', 'SAINZ']))
        self.assertEqual([c.name for c in cars], ['Player', 'NORRIS', 'SAINZ'])

    def test_name_stops_at_null_and_survives_bad_utf8(self):
        packet = bytearray(build_participants_packet(0, 1, ['Player', 'GASLY']))
        offset = 24 + 1 + 56 + 7
        packet[offset : offset + 48] = b'ALBON\x00garbage' + b'\xff' * 35
        self.assertEqual(parse_participants(bytes(packet))[1].name, 'ALBON')
        packet[offset : offset + 6] = b'\xffABC\x00\x00'
        self.assertTrue(parse_participants(bytes(packet))[1].name.endswith('ABC'))

    def test_radio_name_is_the_spoken_surname(self):
        cases = {
            'HAMILTON': 'Hamilton',
            'Lewis HAMILTON': 'Hamilton',
            'DE VRIES': 'De Vries',
            'Nyck DE VRIES': 'De Vries',
            'Shryder25': 'Shryder25',
            'Player': None,
            'Driver': None,
            '': None,
        }
        for raw, spoken in cases.items():
            self.assertEqual(participant(raw).radio_name, spoken, raw)


class StreamParticipantsTests(unittest.TestCase):
    def setUp(self):
        from telemetry_stream import TelemetryStream

        self.engine = AnalysisEngine(SilentVoice())
        self.stream = TelemetryStream(self.engine)
        self.stream.handle(build_session_packet(0, 1, 11), now=0)

    def test_participants_packet_feeds_radio_names(self):
        self.stream.handle(build_participants_packet(0.1, 2), now=0.1)
        self.assertEqual(self.engine.radio.driver_name(1), 'Hamilton')
        self.assertIsNone(self.engine.radio.driver_name(0))  # "Player" placeholder

    def test_wrong_size_or_bad_count_is_ignored(self):
        self.stream.handle(build_participants_packet(0.1, 2)[:-1], now=0.1)
        self.assertEqual(self.engine.radio.driver_names, {})
        packet = bytearray(build_participants_packet(0.1, 3))
        packet[24] = 40
        self.stream.handle(bytes(packet), now=0.1)
        self.assertEqual(self.engine.radio.driver_names, {})

    def test_names_survive_flashback_but_not_a_new_session(self):
        self.stream.handle(build_participants_packet(0.1, 2), now=0.1)
        self.engine.radio.reset(self.engine.session)  # what a flashback does
        self.assertEqual(self.engine.radio.driver_name(2), 'Verstappen')
        self.stream.handle(build_session_packet(0.2, 4, 7), now=0.2)  # different track
        self.assertEqual(self.engine.radio.driver_names, {})


class NamedRadioCallTests(unittest.TestCase):
    def setUp(self):
        from race_radio import RaceRadio

        self.voice = SilentVoice()
        self.radio = RaceRadio(self.voice.say)
        self.radio.reset(SessionData(0, 30, 20, 53, 5793, 10, 11, 0))
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)

    def field(self, offset, last_laps=(0, 0, 0)):
        return [
            replace(
                self.lap,
                car_position=position,
                total_distance=distance + offset,
                result_status=2,
                last_lap_time_ms=last,
            )
            for (position, distance), last in zip([(1, 20), (2, 10), (3, 0)], last_laps, strict=False)
        ]

    def drive(self, last_laps=(0, 0, 0)):
        self.radio.update_field(self.field(0, last_laps), 1, 0, now=0)
        self.radio.update_field(self.field(10, last_laps), 1, 1, now=1)

    def test_gap_call_names_the_cars_ahead_and_behind(self):
        self.radio.set_driver_names({0: 'Hamilton', 2: 'Verstappen'})
        self.drive()
        self.assertEqual(
            self.voice.messages[-1][0],
            'Gap to Hamilton ahead one point zero. Verstappen behind one point zero.',
        )

    def test_gap_call_without_names_is_unchanged(self):
        self.drive()
        self.assertEqual(self.voice.messages[-1][0], 'Gap ahead one point zero. Car behind one point zero.')

    def test_partial_names_only_name_the_known_car(self):
        self.radio.set_driver_names({2: 'Verstappen'})
        self.drive()
        self.assertEqual(
            self.voice.messages[-1][0], 'Gap ahead one point zero. Verstappen behind one point zero.'
        )

    def test_threats_carry_the_driver_name(self):
        self.radio.set_driver_names({0: 'Hamilton', 2: 'Verstappen'})
        self.drive()
        names = {t.car_index: t.driver_name for t in self.radio.opponent_threats}
        self.assertEqual(names, {0: 'Hamilton', 2: 'Verstappen'})

    def test_race_pace_call_compares_against_named_neighbours(self):
        self.radio.set_driver_names({0: 'Hamilton', 2: 'Verstappen'})
        self.drive(last_laps=(90800, 91000, 91000))
        for lap_num in range(1, 4):
            self.radio.completed_lap(91000, lap_num, {}, {})
        message = self.voice.messages[-1][0]
        self.assertIn('Last three laps averaging', message)
        self.assertIn('Hamilton ahead two tenths quicker.', message)
        self.assertIn('Verstappen behind on the same pace.', message)

    def test_pace_comparison_falls_back_to_car_without_names(self):
        self.drive(last_laps=(91500, 91000, 90500))
        self.assertEqual(
            self.radio.pace_comparison(91000),
            'Car ahead half a second slower. Car behind half a second quicker.',
        )


class NamedAnswersTests(unittest.TestCase):
    CONTEXT = {
        'session_mode': 'race',
        'gap_ahead_s': 1.24,
        'gap_behind_s': 0.81,
        'car_ahead_name': 'Hamilton',
        'car_behind_name': 'Verstappen',
        'opponent_threats': [
            {
                'threat_level': 'HIGH',
                'threat_score': 80,
                'gap_s': -0.8,
                'pace_delta_s': -0.3,
                'laps_to_catch': 2.7,
                'factors': ['gap 0.8s (DRS range)'],
                'driver_name': 'Verstappen',
            }
        ],
    }

    def test_rule_based_gap_and_threat_answers_use_names(self):
        self.assertEqual(rule_based_answer('gap_ahead', self.CONTEXT), 'Gap to Hamilton ahead is 1.2.')
        self.assertEqual(rule_based_answer('gap_behind', self.CONTEXT), 'Gap to Verstappen behind is 0.8.')
        self.assertEqual(
            rule_based_answer('threats', self.CONTEXT),
            'High threat from Verstappen behind, 0.8, gap 0.8s (DRS range).',
        )

    def test_unnamed_threat_answer_is_unchanged(self):
        ctx = dict(
            self.CONTEXT, opponent_threats=[dict(self.CONTEXT['opponent_threats'][0], driver_name=None)]
        )
        self.assertEqual(rule_based_answer('threats', ctx), 'High threat behind, 0.8, gap 0.8s (DRS range).')

    def test_ai_prompt_lists_driver_names(self):
        text = format_context(self.CONTEXT)
        self.assertIn('Car ahead driver: Hamilton', text)
        self.assertIn('Car behind driver: Verstappen', text)
        self.assertIn('Verstappen HIGH', text)


class HudNamesTests(unittest.TestCase):
    def test_hud_state_has_name_defaults(self):
        self.assertIn('car_ahead_name', web_hud.dashboard_state)
        self.assertIn('car_behind_name', web_hud.dashboard_state)


if __name__ == '__main__':
    unittest.main()
