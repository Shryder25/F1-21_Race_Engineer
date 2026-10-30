import unittest
from dataclasses import replace

import radio_phrases as say
from packets import SessionData, parse_player_lap_data
from simulate_telemetry import build_lap_packet
from test_analysis import SilentVoice  # also puts ai_engineer/ on the import path
from voice_engine import Priority


class SpokenNumberTests(unittest.TestCase):
    def test_lap_times_are_said_like_an_engineer(self):
        self.assertEqual(say.lap_time(92456), 'one thirty-two five')
        self.assertEqual(say.lap_time(65040), 'one oh five zero')
        self.assertEqual(say.lap_time(31420), 'thirty-one four')
        self.assertEqual(say.lap_time(119960), 'two oh zero zero')  # rounds up into the next minute

    def test_small_differences_are_tenths(self):
        self.assertEqual(say.time_amount(0.02), 'less than a tenth')
        self.assertEqual(say.time_amount(-0.1), 'a tenth')
        self.assertEqual(say.time_amount(0.31), 'three tenths')
        self.assertEqual(say.time_amount(0.5), 'half a second')
        self.assertEqual(say.time_amount(1.0), 'a second')
        self.assertEqual(say.time_amount(1.27), 'one point three seconds')

    def test_gaps(self):
        self.assertEqual(say.gap_value(0.8), 'eight tenths')
        self.assertEqual(say.gap_value(2.44), 'two point four')

    def test_number_words(self):
        self.assertEqual(say.number_words(78), 'seventy-eight')
        self.assertEqual(say.number_words(105), 'one oh five')
        self.assertEqual(say.number_words(110), 'one ten')

    def test_phrasebook_never_repeats_back_to_back(self):
        book = say.Phrasebook()
        picks = [book.pick('t', ('a', 'b')) for _ in range(4)]
        self.assertEqual(picks, ['a', 'b', 'a', 'b'])


class RadioTimingTests(unittest.TestCase):
    def setUp(self):
        from race_radio import RaceRadio

        self.voice = SilentVoice()
        self.radio = RaceRadio(self.voice.say)
        self.radio.reset(SessionData(0, 30, 20, 53, 5793, 10, 11, 0))
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)

    def _field(self, ahead_offset, t):
        # player P2 at distance 1000 + 50t; car ahead P1 leads by ahead_offset metres
        player = 1000 + 50 * t
        return [
            replace(self.lap, car_position=1, total_distance=player + ahead_offset, result_status=2),
            replace(self.lap, car_position=2, total_distance=player, result_status=2),
        ]

    def _keys(self):
        return [kw.get('key') for _, _, kw in self.voice.messages]

    def test_drs_range_called_once_with_hysteresis(self):
        # 50 m/s, so 40 m ahead = 0.8 s gap
        for t in range(0, 30):
            self.radio.update_field(self._field(40, t), 1, float(t), now=float(t))
        self.assertEqual(self._keys().count('drs_ahead'), 1)
        message = next(m for m, _, kw in self.voice.messages if kw.get('key') == 'drs_ahead')
        self.assertIn('eight tenths', message)

    def test_gap_update_is_once_a_lap_at_the_line_with_the_trend(self):
        for t in range(0, 60):
            self.radio.update_field(self._field(150, t), 1, float(t), now=float(t))
        self.assertEqual(self._keys().count('race_gaps'), 0)  # no timed gap calls mid-lap
        self.radio.call_lap_gaps()
        for t in range(60, 70):
            self.radio.update_field(self._field(130, t), 1, float(t), now=float(t))
        self.radio.call_lap_gaps()
        calls = [m for m, _, kw in self.voice.messages if kw.get('key') == 'race_gaps']
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], 'Gap ahead three point zero.')
        self.assertEqual(calls[1], "Gap ahead two point six, you're catching, four tenths that lap.")

    def test_race_pace_summary_every_three_laps(self):
        for lap_num in range(1, 8):  # lap 1 is a standing start and never counts
            self.radio.completed_lap(91000, lap_num, {}, {})
        pace_calls = [m for m, p, kw in self.voice.messages if kw.get('key') == 'race_pace']
        self.assertEqual(len(pace_calls), 2)
        self.assertIn('one thirty-one zero', pace_calls[0])
        self.assertEqual(self.voice.messages[-1][1], Priority.RACE)


if __name__ == '__main__':
    unittest.main()
