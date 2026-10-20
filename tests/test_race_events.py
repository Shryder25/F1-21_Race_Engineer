"""Race events on the radio: decoded Event packets, driver names, and calls worked out
from the field (overtakes, rival pit stops, flags, lap one and finishing positions)."""

import pathlib
import struct
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from analysis import AnalysisEngine
from packets import (SessionData, CarStatusData, RaceEvent, parse_race_event, parse_participants,
                     parse_player_lap_data, PARTICIPANT_FORMAT, HEADER_SIZE)
from race_events import RaceControl
from simulate_telemetry import build_header, build_lap_packet, build_session_packet
from test_analysis import SilentVoice
from voice_engine import Priority

RACE = SessionData(0, 30, 20, 53, 5793, 10, 11, 0)
QUALI = SessionData(0, 30, 20, 53, 5793, 8, 11, 0)
NAMES = ['Player', 'HAMILTON', 'Max VERSTAPPEN', 'NORRIS', 'LECLERC', '']


def event_packet(code: bytes, body: bytes = b'', size: int = 40, session_time=1.0, frame=5) -> bytes:
    packet = build_header(3, session_time, frame) + code + body
    return packet + bytes(size - len(packet))


def participants_packet(names=NAMES) -> bytes:
    data = bytearray(build_header(4, 1.0, 5) + bytes([len(names)]))
    for index in range(22):
        name = names[index].encode() if index < len(names) else b''
        data += struct.pack(PARTICIPANT_FORMAT, 1, index, 255, index % 10, 0, 10 + index, 0, name, 0)
    return bytes(data)


def status(compound=17, flag=0):
    return CarStatusData(50, 110, 30, compound, compound, 1, flag, 2e6, 1, 0)


class Field:
    """A race of len(NAMES) cars; car i starts in P(i + 1). The player is car 2 (P3)."""

    def __init__(self, cars=6, player=2):
        template = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 3, 1), 0)
        self.player = player
        self.laps = [replace(template, car_position=i + 1, grid_position=i + 1, result_status=2)
                     if i < cars else replace(template, car_position=0, result_status=0)
                     for i in range(22)]

    def set(self, index, **changes):
        self.laps[index] = replace(self.laps[index], **changes)

    def swap(self, a, b):
        pa, pb = self.laps[a].car_position, self.laps[b].car_position
        self.set(a, car_position=pb)
        self.set(b, car_position=pa)


class PacketParsingTests(unittest.TestCase):
    def test_penalty_fastest_lap_and_retirement_bodies(self):
        penalty = parse_race_event(event_packet(b'PENA', bytes([4, 7, 3, 255, 5, 12, 0])))
        self.assertEqual((penalty.code, penalty.penalty_type, penalty.vehicle_index, penalty.time_s,
                          penalty.lap_num), ('PENA', 4, 3, 5, 12))
        self.assertEqual((penalty.penalty_name, penalty.infringement_name),
                         ('Time penalty', 'corner cutting'))
        fastest = parse_race_event(event_packet(b'FTLP', struct.pack('<Bf', 1, 81.234)))
        self.assertEqual(fastest.vehicle_index, 1)
        self.assertAlmostEqual(fastest.lap_time_s, 81.234, places=3)
        self.assertEqual(parse_race_event(event_packet(b'RTMT', bytes([4]))).vehicle_index, 4)
        self.assertEqual(parse_race_event(event_packet(b'CHQF')).code, 'CHQF')
        self.assertIsNone(parse_race_event(event_packet(b'BUTN', bytes(4))))

    def test_participants_give_spoken_surnames(self):
        participants = parse_participants(participants_packet())
        self.assertEqual(len(participants), len(NAMES))
        self.assertEqual([p.radio_name for p in participants],
                         [None, 'Hamilton', 'Verstappen', 'Norris', 'Leclerc', None])
        self.assertEqual(participants[3].race_number, 13)


class StreamRoutingTests(unittest.TestCase):
    def setUp(self):
        from telemetry_stream import TelemetryStream
        self.voice = SilentVoice()
        self.engine = AnalysisEngine(self.voice)
        self.stream = TelemetryStream(self.engine)
        self.stream.handle(build_session_packet(0, 1, 11, session_type=10), now=0)

    def spoken(self):
        return [message for message, _, _ in self.voice.messages]

    def test_race_events_of_either_padded_size_reach_race_control(self):
        self.stream.handle(event_packet(b'DRSE', size=36), now=1)
        self.stream.handle(event_packet(b'PENA', bytes([4, 7, 0, 255, 5, 3, 0]), size=40), now=1)
        self.assertIn('DRS enabled.', self.spoken())
        self.assertTrue(any('5 second time penalty' in m for m in self.spoken()))

    def test_two_events_in_one_frame_are_both_handled(self):
        self.stream.handle(event_packet(b'PENA', bytes([5, 25, 0, 255, 0, 3, 0]), frame=9), now=1)
        self.stream.handle(event_packet(b'DRSD', frame=9), now=1)
        self.assertEqual(len(self.engine.race_control.feed), 2)

    def test_bad_event_size_and_other_sessions_are_ignored(self):
        self.stream.handle(event_packet(b'DRSE', size=38), now=1)
        other_session = bytearray(event_packet(b'DRSE'))
        struct.pack_into('<Q', other_session, 6, 99)
        self.stream.handle(bytes(other_session), now=1)
        self.assertEqual(len(self.engine.race_control.feed), 0)

    def test_participants_packet_names_cars(self):
        self.stream.handle(participants_packet(), now=1)
        self.assertEqual(self.engine.race_control.name(2), 'Verstappen')
        self.assertEqual(self.engine.race_control.name(5), 'car 15')

    def test_race_events_feed_the_ai_context(self):
        self.stream.handle(event_packet(b'DRSE'), now=1)
        self.assertEqual(self.engine.build_context()['recent_race_events'], ['DRS enabled.'])


class RaceControlTests(unittest.TestCase):
    def setUp(self):
        self.voice = SilentVoice()
        self.control = RaceControl(self.voice.say)
        self.control.reset(RACE)
        self.control.update_participants(parse_participants(participants_packet()))
        self.field = Field()
        self.time = 100.0
        self.tick()

    def tick(self, seconds=0.5):
        self.time += seconds
        self.control.update_field(self.field.laps, self.field.player, self.time)

    def settle(self):
        for _ in range(5):
            self.tick()

    def said(self):
        return [message for message, _, _ in self.voice.messages]

    def event(self, code, **fields):
        self.control.handle_event(RaceEvent(code, **fields), self.field.player, self.time)

    # Penalties
    def test_own_time_penalty_is_critical(self):
        self.event('PENA', penalty_type=4, infringement_type=7, vehicle_index=2, time_s=5)
        message, priority, options = self.voice.messages[-1]
        self.assertEqual(message, '5 second time penalty, for corner cutting.')
        self.assertEqual((priority, options['key']), (Priority.CRITICAL, 'penalty'))

    def test_rival_penalty_only_for_nearby_cars_or_contact(self):
        self.event('PENA', penalty_type=4, infringement_type=17, vehicle_index=3, time_s=5)
        self.assertIn('Norris has a 5 second time penalty, for pit lane speeding.', self.said())
        self.field.set(5, car_position=12)
        self.tick()
        self.event('PENA', penalty_type=0, infringement_type=17, vehicle_index=5)
        self.assertEqual(len(self.said()), 1)
        self.event('PENA', penalty_type=0, infringement_type=3, vehicle_index=5, other_vehicle_index=2)
        self.assertIn('Car 15 has a drive through, for the contact with you.', self.said())

    def test_deleted_lap_is_spoken_in_qualifying_but_only_logged_in_the_race(self):
        self.event('PENA', penalty_type=10, infringement_type=26, vehicle_index=2)
        self.assertEqual(self.said(), [])
        self.assertEqual(self.control.recent()[0]['text'], 'Lap deleted, running wide.')
        self.control.reset(QUALI)
        self.event('PENA', penalty_type=10, infringement_type=26, vehicle_index=2)
        self.assertEqual(self.said(), ['Lap deleted, running wide.'])

    def test_penalty_served(self):
        self.event('DTSV', vehicle_index=2)
        self.event('SGSV', vehicle_index=4)
        self.assertEqual(self.said(), ['Penalty served. Back to racing.'])

    # Other event packets
    def test_fastest_lap_for_player_and_rival(self):
        self.event('FTLP', vehicle_index=2, lap_time_s=81.2345)
        self.event('FTLP', vehicle_index=1, lap_time_s=81.0)
        self.assertEqual(self.said(), ['Fastest lap, 1:21.234. Purple, mega.',
                                       'Hamilton sets the fastest lap, 1:21.000.'])

    def test_retirement_from_event_and_lap_data_is_called_once(self):
        self.event('RTMT', vehicle_index=1)
        self.field.set(1, result_status=7)
        self.tick()
        self.assertEqual(self.said(), ['Hamilton is out of the race. That is a place for us.'])
        self.field.set(4, result_status=4)
        self.tick()
        self.assertEqual(self.said()[-1], 'Leclerc is out of the race.')

    def test_race_winner_and_chequered_flag(self):
        self.event('CHQF')
        self.event('RCWN', vehicle_index=1)
        self.assertEqual(self.said(), ['Chequered flag is out. Bring it home.', 'Hamilton wins the race.'])

    def test_events_outside_the_race_stay_quiet(self):
        self.control.reset(QUALI)
        for code in ('DRSE', 'CHQF', 'TMPT'):
            self.event(code, vehicle_index=1)
        self.event('FTLP', vehicle_index=1, lap_time_s=80.0)
        self.assertEqual(self.said(), [])

    # Overtakes
    def test_overtake_is_called_once_position_settles(self):
        self.field.swap(2, 1)
        self.tick()
        self.assertEqual(self.said(), [])                   # not settled yet
        self.settle()
        self.assertEqual(self.said(), ['Nice move on Hamilton. P2.'])
        self.field.swap(2, 1)
        self.settle()
        self.assertEqual(self.said()[-1], 'Hamilton is through. P3.')

    def test_flicker_across_the_line_is_not_called(self):
        self.field.swap(2, 1)
        self.tick()
        self.field.swap(2, 1)
        self.settle()
        self.assertEqual(self.said(), [])

    def test_place_gained_through_a_rival_stop_is_not_an_overtake(self):
        self.field.set(1, pit_status=1)
        self.tick()
        self.field.swap(2, 1)
        self.settle()
        self.assertIn('P2.', self.said())
        self.assertFalse(any('Nice move' in m for m in self.said()))

    def test_lap_one_moves_are_summed_up_at_the_end_of_the_lap(self):
        self.control.reset(RACE)
        for i in range(22):
            self.field.set(i, current_lap_num=1)
        self.tick()
        self.field.swap(2, 1)
        self.field.swap(2, 0)
        self.settle()
        self.assertEqual(self.said(), [])
        for i in range(22):
            self.field.set(i, current_lap_num=2)
        self.tick()
        self.assertEqual(self.said(), ['End of lap one, P1. 2 places gained from the grid.'])

    def test_lead_change_between_rivals(self):
        self.field.swap(0, 1)
        self.settle()
        self.assertEqual(self.said(), ['Hamilton takes the lead from car 10.'])

    # Pit stops
    def test_nearby_rival_stop_reports_time_compound_and_rejoin_position(self):
        self.control.update_status([status()] * 22, 2)
        self.field.set(3, pit_status=1)
        self.tick()
        self.field.set(3, pit_status=2, pit_stop_timer_ms=2400, num_pit_stops=1)
        self.tick()
        self.control.update_status([status(compound=18)] * 22, 2)
        self.field.set(3, pit_status=0, pit_stop_timer_ms=0, car_position=6)
        self.field.set(4, car_position=4)
        self.field.set(5, car_position=5)
        self.tick()
        self.assertIn('Norris has pitted from P4, 2.4 second stop, onto the hards. Rejoins P6.', self.said())

    def test_drive_through_is_not_a_pit_stop(self):
        self.field.set(3, pit_status=1)
        self.tick()
        self.field.set(3, pit_status=0)
        self.tick()
        self.assertFalse(any('pitted' in m for m in self.said()))

    def test_own_stop_time(self):
        self.field.set(2, pit_status=1)
        self.tick()
        self.field.set(2, pit_status=2, pit_stop_timer_ms=2300, num_pit_stops=1)
        self.tick()
        self.field.set(2, pit_status=0, car_position=5)
        self.field.set(3, car_position=3)
        self.field.set(4, car_position=4)
        self.tick()
        self.assertIn('Good stop, 2.3 seconds. Rejoining P5.', self.said())

    # Flags and finish
    def test_flags_waved_at_the_player(self):
        for flag in (0, 3, 3, 1, 2, 4):
            self.control.update_status([status(flag=flag)] * 22, 2)
        self.assertEqual(self.said(), ['Yellow flag, yellow flag. No overtaking.',
                                       'Green flag, track is clear.',
                                       'Blue flags. Let the leaders through.',
                                       'Red flag, red flag. Slow down and bring it back to the pit lane.'])
        self.assertEqual(self.voice.messages[0][1], Priority.CRITICAL)

    def test_finish_position_against_the_grid(self):
        self.field.set(2, result_status=3, car_position=2, grid_position=5)
        self.tick()
        self.tick()
        self.assertEqual(self.said(), ["That's the flag. We finish P2. Up 3 from the grid, good job."])


class OnDemandModeTests(unittest.TestCase):
    def test_penalties_and_flags_break_radio_silence_but_rival_news_does_not(self):
        voice = SilentVoice()
        engine = AnalysisEngine(voice, on_demand=True)
        engine.update_session(RACE)
        voice.messages.clear()
        control = engine.race_control
        control.handle_event(RaceEvent('PENA', penalty_type=4, infringement_type=7, vehicle_index=0,
                                       time_s=5), 0, 1.0)
        control.handle_event(RaceEvent('FTLP', vehicle_index=1, lap_time_s=80.0), 0, 1.0)
        control.update_status([status(flag=0)] * 22, 0)
        control.update_status([status(flag=3)] * 22, 0)
        self.assertEqual([m for m, _, _ in voice.messages],
                         ['5 second time penalty, for corner cutting.',
                          'Yellow flag, yellow flag. No overtaking.'])
        self.assertEqual(len(control.feed), 3)


if __name__ == '__main__':
    unittest.main()
