"""Bad calls from a real 18-lap wet race at Mexico, and what the engineer says now.

* Plan B to box on lap 15 of 18: no new stop this close to the flag.
* "Box for wets" while inters were lapping quicker: the field's inter pace caps the wetness.
* "Where is Verstappen" answered with the gap to the car ahead: answered from the timing tower.
* A thin pace update: now against your best and the cars either side.
"""

import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import strategy_replan as rp
import track_wetness as tw
from engineer_brain import EngineerBrain, driver_answer, pace_answer
from test_strategy_replan import cand, outcome

INTERS = cand(['INTERMEDIATE'], [])


def run(r, lap_num, challenger, gain=12.0, reasons=('lap',), key=1):
    r.pending = list(reasons)
    r.note_submitted(key)
    result = SimpleNamespace(
        lap=lap_num,
        runs=100,
        outcomes=[outcome(r.plan_a, ref=True), outcome(challenger, gain=gain, places=1.0, p=0.9)],
    )
    return r.consider(result, key, lap_num)


class LateStopTests(unittest.TestCase):
    def replanner(self, last_lap=17):
        r = rp.Replanner()
        r.adopt(INTERS, 1)
        r.last_lap = last_lap  # an 18-lap race
        return r

    def test_no_plan_b_box_with_four_laps_left(self):
        r = self.replanner()
        wets = rp._candidate(['INTERMEDIATE', 'WET'], [15], ', box now')
        self.assertIsNone(run(r, 15, wets, reasons=('crossover',)))
        self.assertEqual(r.plan_a.compounds, ['INTERMEDIATE'])

    def test_still_calls_it_with_more_laps_left(self):
        r = self.replanner()
        wets = rp._candidate(['INTERMEDIATE', 'WET'], [12], ', box now')
        self.assertIsNotNone(run(r, 12, wets, reasons=('crossover',)))

    def test_slicks_in_the_rain_still_box_late(self):
        r = rp.Replanner()
        r.adopt(cand(['MEDIUM'], []), 1)
        r.last_lap = 17
        inters = rp._candidate(['MEDIUM', 'INTERMEDIATE'], [16], ', box now')
        self.assertIsNotNone(run(r, 16, inters, reasons=('rain',)))

    def test_dropping_a_stop_late_is_fine(self):
        r = rp.Replanner()
        r.adopt(cand(['INTERMEDIATE', 'INTERMEDIATE'], [16]), 1)
        r.last_lap = 17
        self.assertIsNotNone(run(r, 15, INTERS, reasons=('safety_car',)))


class InterPaceCapTests(unittest.TestCase):
    def soaked(self):
        w = tw.TrackWetness()
        w.update_weather(4, 25.0, 0.0)  # heavy rain from the start: physics says 0.8
        self.assertGreater(w.wetness, 0.7)
        return w

    def feed(self, w, laps_by_car, start=100.0):
        t = start
        for lap in range(len(next(iter(laps_by_car.values())))):
            t += 95.0
            for car, laps in laps_by_car.items():
                w.add_lap(car, 'inter', laps[lap], t)
            w.assimilate(t)
        return t

    def test_inters_getting_quicker_keep_the_estimate_below_wets(self):
        w = self.soaked()
        self.feed(w, {car: [98.5, 98.0, 97.0, 95.5] for car in range(6)})
        self.assertTrue(w.inters_holding_pace())
        self.assertLess(w.wetness, tw.crossover('inter', 'wet'))
        self.assertEqual(tw.best_family(w.wetness), 'inter')
        self.assertTrue(all(x < 0.7 for x in w.trajectory(10)))
        self.assertNotEqual(w.summary('inter')['switch_to'], 'wet')

    def test_inters_falling_away_let_the_track_get_wetter(self):
        w = self.soaked()
        self.feed(w, {car: [95.0, 96.5, 99.0, 102.0] for car in range(6)})
        self.assertFalse(w.inters_holding_pace())
        self.assertIsNone(w.wet_cap)

    def test_cars_on_wets_override_the_cap(self):
        w = self.soaked()
        t = self.feed(w, {car: [98.0, 97.5, 97.0, 96.5] for car in range(6)})
        for car in (10, 11, 12):
            w.add_lap(car, 'wet', 94.0, t + 1)
        w.new_records = 99
        w.assimilate(t + 2)
        self.assertIsNone(w.wet_cap)


STANDINGS = [
    {
        'position': 1,
        'name': 'SAINZ',
        'gap_s': None,
        'is_player': True,
        'last_lap_ms': 93_300,
        'compound': 'I',
        'tyre_age': 7,
    },
    {'position': 2, 'name': 'HAMILTON', 'gap_s': 2.1, 'last_lap_ms': 93_800, 'compound': 'I', 'tyre_age': 9},
    {'position': 3, 'name': 'LECLERC', 'gap_s': 5.0, 'last_lap_ms': 94_100, 'compound': 'I', 'tyre_age': 9},
    {
        'position': 4,
        'name': 'VERSTAPPEN',
        'gap_s': 8.4,
        'last_lap_ms': 93_100,
        'compound': 'W',
        'tyre_age': 2,
    },
]
CTX = {
    'session_mode': 'race',
    'lap': 16,
    'total_laps': 18,
    'position': 1,
    'last_lap_ms': 93_300,
    'best_lap_ms': 93_300,
    'gap_behind_s': 2.1,
    'car_behind_name': 'HAMILTON',
    'standings': STANDINGS,
}


class DriverAnswerTests(unittest.TestCase):
    def test_where_is_verstappen(self):
        self.assertEqual(
            driver_answer('where is Verstappen', CTX),
            'Verstappen is P4, 8.4 behind you, on 2-lap wets. Last lap 1:33.100.',
        )

    def test_ahead_of_you(self):
        ctx = {**CTX, 'standings': [{**r, 'is_player': r['name'] == 'LECLERC'} for r in STANDINGS]}
        self.assertTrue(
            driver_answer("what's Hamilton's gap", ctx).startswith('Hamilton is P2, 2.9 ahead of you')
        )

    def test_other_questions_about_a_driver_go_elsewhere(self):
        self.assertIsNone(driver_answer('when will Verstappen box', CTX))
        self.assertIsNone(driver_answer('can I beat Verstappen', CTX))
        self.assertIsNone(driver_answer('where is the rain', CTX))

    def test_brain_answers_it_before_the_car_ahead_command(self):
        brain = EngineerBrain(lambda: dict(CTX), None, rules_first=True)
        answer = brain.answer_question('where is Verstappen', 'gap_ahead')
        self.assertTrue(answer.startswith('Verstappen is P4'))
        self.assertEqual(brain.last_source, 'rules')


class PaceAnswerTests(unittest.TestCase):
    def test_against_best_and_cars_either_side(self):
        ctx = {**CTX, 'position': 2, 'last_lap_ms': 93_800, 'best_lap_ms': 93_400}
        ctx['standings'] = [{**r, 'is_player': r['name'] == 'HAMILTON'} for r in STANDINGS]
        self.assertEqual(
            pace_answer(ctx),
            'Last lap 1:33.800. 0.4 off your best. 0.5 slower than Sainz ahead. 0.3 quicker than Leclerc behind.',
        )

    def test_slow_lap_is_not_compared(self):
        ctx = {**CTX, 'last_lap_ms': 115_000}
        self.assertEqual(pace_answer(ctx), 'Last lap 1:55.000.')

    def test_no_lap_yet(self):
        self.assertEqual(pace_answer({}), 'No lap time yet, give me one more lap.')


if __name__ == '__main__':
    unittest.main()
