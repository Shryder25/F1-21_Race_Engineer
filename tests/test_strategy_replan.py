"""Tests for live strategy re-planning (strategy_replan.py and its wiring in analysis.py).

Checks the paired Plan A comparison in the simulator, the re-plan triggers (lap, Safety
Car, VSC, rain, a nearby rival's stop), how Plan A follows the race, the margin and
anti-flip-flop rules for the Plan B call, and that the call reaches the radio (also in
quiet mode), the HUD summary and the AI context. No game needed: made-up fields only.
"""

import pathlib
import sys
import time
import unittest
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import race_sim as sim
import race_strategy as rs
import strategy_replan as rp
from analysis import AnalysisEngine, PROACTIVE_KEYS
from engineer_brain import format_context, rule_based_answer
from test_analysis import SilentVoice
from test_race_sim import RACE, SILVERSTONE, candidates, field, lap, status

INTER = 7


def cand(compounds, pit_laps):
    return sim.Candidate(sim.plan_label(len(pit_laps), compounds), len(pit_laps), list(compounds), list(pit_laps))


def outcome(c, ref=False, gain=0.0, places=0.0, p=0.0):
    return SimpleNamespace(label=c.label, stops=c.stops, compounds=list(c.compounds), pit_laps=list(c.pit_laps),
                           is_reference=ref,
                           vs_reference=None if ref else {'time_gain_s': gain, 'time_gain_range_s': [0, 0],
                                                          'position_gain': places, 'p_better': p})


PLAN_A = cand(['MEDIUM', 'HARD'], [30])
PLAN_B = cand(['MEDIUM', 'HARD', 'SOFT'], [24, 40])


class Rig:
    """A Replanner with Plan A adopted and a helper to feed it fake re-run results."""

    def __init__(self, **config):
        self.r = rp.Replanner(rp.ReplanConfig(**config))
        self.r.adopt(PLAN_A, 0)
        self.n = 0

    def run(self, lap_num, gain, places=0.0, p=0.8, reasons=('lap',), challenger=PLAN_B):
        self.n += 1
        self.r.pending = list(reasons)
        self.r.note_submitted(self.n)
        result = SimpleNamespace(lap=lap_num, runs=100, outcomes=[
            outcome(self.r.plan_a, ref=True), outcome(challenger, gain=gain, places=places, p=p)])
        return self.r.consider(result, self.n, lap_num)


class PairedComparisonTests(unittest.TestCase):
    def test_reference_plan_and_paired_gains(self):
        state = sim.RaceState(SILVERSTONE, 50, 21, field(6, player=2, compound='SOFT', age=15))
        stay = sim.Candidate('No-stop (S)', 0, ['SOFT'], [])
        stop = sim.Candidate('One-stop (S-H)', 1, ['SOFT', 'HARD'], [21])
        twin = sim.Candidate('Twin', 0, ['SOFT'], [])
        result = sim.simulate(state, [stay, stop, twin], reference=0, runs=150, seed=5)
        by = {o.label: o for o in result.outcomes}
        self.assertTrue(by['No-stop (S)'].is_reference)
        self.assertIsNone(by['No-stop (S)'].vs_reference)
        self.assertEqual(by['Twin'].vs_reference['time_gain_s'], 0.0)       # same draws, same plan
        self.assertEqual(by['Twin'].vs_reference['p_better'], 0.0)
        better = by['One-stop (S-H)'].vs_reference
        self.assertGreater(better['time_gain_s'], 5.0)
        self.assertGreater(better['p_better'], 0.85)
        self.assertGreaterEqual(better['position_gain'], 0.0)
        lo, hi = better['time_gain_range_s']
        self.assertLessEqual(lo, hi)
        self.assertIn('vs_reference', result.summary()['options'][0])

    def test_no_reference_keeps_the_old_shape(self):
        result = sim.simulate(sim.RaceState(SILVERSTONE, 52, 11, field()), candidates(42), runs=30, seed=1)
        self.assertTrue(all(o.vs_reference is None and not o.is_reference for o in result.outcomes))
        with self.assertRaises(ValueError):
            sim.simulate(sim.RaceState(SILVERSTONE, 52, 11, field()), candidates(42), reference=9, runs=5)

    def test_raining_now_slicks_lose_to_inters(self):
        state = sim.RaceState(SILVERSTONE, 30, 11, field(6, player=2), raining=True)
        stay = cand(['MEDIUM', 'HARD'], [20])
        inters = rp._candidate(['MEDIUM', 'INTERMEDIATE'], [11], ', box now')
        result = sim.simulate(state, [stay, inters], reference=0, runs=80, seed=2,
                              profile={'sc_probability': 0})
        self.assertEqual(result.p_rain, 1.0)
        vs = next(o for o in result.outcomes if not o.is_reference).vs_reference
        self.assertGreater(vs['time_gain_s'], 30)
        self.assertGreater(vs['p_better'], 0.9)


class TriggerTests(unittest.TestCase):
    def test_lap_safety_car_vsc_rain(self):
        r = rp.Replanner()
        self.assertEqual(r.observe(10, 0, 0), [])          # first look: baseline only
        self.assertEqual(r.observe(10, 0, 0), [])
        self.assertEqual(r.observe(11, 0, 0), ['lap'])
        self.assertEqual(r.events, 0)
        self.assertEqual(r.observe(11, 1, 0), ['safety_car'])
        self.assertEqual(r.observe(11, 2, 0), ['vsc'])
        self.assertEqual(r.observe(11, 0, 0), ['green_flag'])
        self.assertEqual(r.observe(11, 0, 3), ['rain'])
        self.assertEqual(r.observe(11, 0, 4), [])          # heavier rain: still wet, no new trigger
        self.assertEqual(r.observe(11, 0, 2), ['drying'])
        self.assertEqual(r.events, 3)                       # SC, VSC, rain
        self.assertTrue(r.has_event_pending)
        r.note_submitted('k')
        self.assertEqual(r.pending, [])

    def test_nearby_rival_stop_only(self):
        r = rp.Replanner()
        laps = [lap(i + 1, 20) for i in range(10)]
        r.observe(20, 0, 0, laps, 4, lambda i: f'D{i}')
        laps[5] = replace(laps[5], num_pit_stops=1)          # P6, one place behind P5: nearby
        self.assertEqual(r.observe(20, 0, 0, laps, 4, lambda i: f'D{i}'), ['rival_stop'])
        self.assertEqual(r.rival_name, 'D5')
        laps[9] = replace(laps[9], num_pit_stops=1)          # P10, five places back: not a trigger
        self.assertEqual(r.observe(20, 0, 0, laps, 4, lambda i: f'D{i}'), [])
        laps[4] = replace(laps[4], num_pit_stops=1)          # our own stop is not a rival's
        self.assertEqual(r.observe(20, 0, 0, laps, 4), [])
        self.assertEqual(r.events, 1)


class PlanATests(unittest.TestCase):
    def setUp(self):
        self.plan = rs.plan_race(32, 'MEDIUM', 10, 21, rs.pit_loss_s(SILVERSTONE))
        self.r = rp.Replanner()

    def test_adopts_the_first_plan_and_keeps_it(self):
        a = self.r.sync_plan_a(self.plan, 21, 0, 'Medium', 32)
        self.assertEqual(a.compounds, self.plan.best.compounds)
        other = rs.plan_race(31, 'MEDIUM', 11, 22, rs.pit_loss_s(SILVERSTONE), deg_scale=3.0)
        self.assertIs(self.r.sync_plan_a(other, 22, 0, 'Medium', 31), a)   # the plan we are on stays

    def test_stop_made_moves_plan_on(self):
        self.r.adopt(cand(['MEDIUM', 'HARD', 'SOFT'], [25, 40]), 0)
        a = self.r.sync_plan_a(self.plan, 26, 1, 'Hard', 27)
        self.assertEqual((a.compounds, a.pit_laps, a.stops), (['HARD', 'SOFT'], [40], 1))
        a = self.r.sync_plan_a(self.plan, 26, 1, 'Soft', 27)                # fitted softs instead
        self.assertEqual(a.compounds, ['SOFT', 'SOFT'])

    def test_unplanned_stop_and_flashback_start_again(self):
        self.r.adopt(cand(['MEDIUM'], []), 0)
        a = self.r.sync_plan_a(self.plan, 26, 1, 'Medium', 27)
        self.assertEqual(a.compounds, self.plan.best.compounds)
        self.r.sync_plan_a(self.plan, 26, 0, 'Medium', 27)
        self.assertEqual(self.r.stops_done, 0)

    def test_missed_stop_slides_to_this_lap(self):
        self.r.adopt(cand(['MEDIUM', 'HARD', 'SOFT'], [25, 26]), 0)
        a = self.r.sync_plan_a(self.plan, 27, 0, 'Medium', 26)
        self.assertEqual(a.pit_laps, [27, 28])
        a = self.r.sync_plan_a(self.plan, 51, 0, 'Medium', 2)                 # one lap left: no time to stop twice
        self.assertEqual(a.pit_laps, [51])
        self.assertEqual(a.compounds, ['MEDIUM', 'HARD'])

    def test_candidates_plan_a_first_and_event_options(self):
        self.r.adopt(cand(['MEDIUM', 'HARD'], [26]), 0)
        plain = self.r.candidates(self.plan, 21, False, False, 'Medium')
        self.assertIs(plain[0], self.r.plan_a)
        self.assertFalse(any(c.label.endswith('box now') for c in plain))
        keys = [(tuple(c.compounds), tuple(c.pit_laps)) for c in plain]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertLessEqual(len(plain), rp.MAX_CANDIDATES + 1)
        sc = self.r.candidates(self.plan, 21, True, False, 'Medium')
        now = [c for c in sc if c.label.endswith('box now')]
        self.assertIn([21], [c.pit_laps[:1] for c in now])
        self.assertEqual(now[0].compounds, ['MEDIUM', 'HARD'])
        wet = self.r.candidates(self.plan, 21, False, True, 'Medium')
        self.assertIn((['MEDIUM', 'INTERMEDIATE'], [21]), [(c.compounds, c.pit_laps) for c in wet])
        self.r.pending = ['rival_stop']
        self.assertTrue(any(c.label.endswith('box now') for c in self.r.candidates(self.plan, 21, False, False, 'Medium')))

    def test_pit_now_and_same_plan(self):
        self.assertEqual(rp.pit_now(cand(['M', 'H', 'S'], [25, 26]), 24).pit_laps, [24, 26])
        self.assertIsNone(rp.pit_now(cand(['M', 'H', 'S'], [25, 26]), 25))         # already this lap
        self.assertIsNone(rp.pit_now(cand(['M'], []), 20))
        self.assertTrue(rp.same_plan(cand(['M', 'H'], [24]), cand(['M', 'H'], [26]), 20))
        self.assertFalse(rp.same_plan(cand(['M', 'H'], [24]), cand(['M', 'H'], [27]), 20))
        self.assertFalse(rp.same_plan(cand(['M', 'H'], [24]), cand(['M', 'S'], [24]), 20))
        self.assertFalse(rp.same_plan(cand(['M', 'H'], [21]), cand(['M', 'H'], [20]), 20))   # box now is a call


class DecisionTests(unittest.TestCase):
    def test_below_margin_holds(self):
        rig = Rig()
        self.assertIsNone(rig.run(20, gain=2.0, places=0.2))
        self.assertEqual(rig.r.status, 'holding')
        self.assertFalse(rig.r.challenger['clears_margin'])
        self.assertIsNone(rig.run(21, gain=8.0, p=0.5))          # quicker on average, not often enough
        self.assertEqual(rig.r.plan_a.label, PLAN_A.label)

    def test_routine_rerun_needs_two_in_a_row(self):
        rig = Rig()
        self.assertIsNone(rig.run(20, gain=5.0))
        self.assertEqual(rig.r.status, 'watching')
        self.assertIsNone(rig.run(21, gain=1.0))                  # lost it: streak resets
        self.assertIsNone(rig.run(22, gain=5.0))
        call = rig.run(23, gain=5.0)
        self.assertIsNotNone(call)
        self.assertEqual(rig.r.plan_a.compounds, PLAN_B.compounds)   # Plan B is now the plan
        self.assertEqual(rig.r.status, 'switched')
        self.assertIn('Plan B, Plan B', call.message)
        self.assertIn('We go two stops. Box next lap for hards.', call.message)
        self.assertIn('Gains about 5 seconds.', call.message)

    def test_places_count_as_margin(self):
        rig = Rig()
        rig.run(20, gain=0.5, places=1.2)
        call = rig.run(21, gain=0.5, places=1.2)
        self.assertIn('Worth 1 place.', call.message)

    def test_event_calls_at_once_and_ignores_cooldown(self):
        rig = Rig()
        box_now = rp._candidate(['MEDIUM', 'HARD'], [20], ', box now')
        call = rig.run(20, gain=9.0, reasons=('safety_car',), challenger=box_now)
        self.assertIsNotNone(call)
        self.assertEqual(call.message, 'Safety car. Plan B, Plan B. Box this lap for hards. Box, box. '
                                       'Gains about 9 seconds.')
        self.assertEqual(call.reasons, ['safety_car'])
        again = rig.run(20, gain=6.0, reasons=('rain',), challenger=PLAN_B)
        self.assertTrue(again.message.startswith('Rain is here. Plan B'))

    def test_cooldown_after_a_call(self):
        rig = Rig()
        rig.run(20, gain=5.0)
        self.assertIsNotNone(rig.run(21, gain=5.0))
        for n in (22, 23):
            self.assertIsNone(rig.run(n, gain=5.0, challenger=PLAN_A))
        self.assertEqual(rig.r.status, 'cooldown')
        self.assertIsNotNone(rig.run(24, gain=5.0, challenger=PLAN_A))

    def test_same_plan_is_never_a_plan_b(self):
        rig = Rig()
        nudged = cand(['MEDIUM', 'HARD'], [31])
        for n in (20, 21, 22):
            self.assertIsNone(rig.run(n, gain=10.0, challenger=nudged))
        self.assertIsNone(rig.r.challenger)

    def test_margin_is_configurable(self):
        rig = Rig(margin_s=20.0, margin_positions=5.0)
        for n in (20, 21, 22):
            self.assertIsNone(rig.run(n, gain=10.0, places=1.0))
        loose = Rig(margin_s=1.0, confirm_runs=1)
        self.assertIsNotNone(loose.run(20, gain=1.5))

    def test_stale_or_unknown_results_are_ignored(self):
        rig = Rig()
        rig.r.pending = ['safety_car']
        rig.r.note_submitted('old')
        result = SimpleNamespace(lap=20, runs=100, outcomes=[outcome(PLAN_A, ref=True), outcome(PLAN_B, gain=9, p=0.9)])
        self.assertIsNone(rig.r.consider(result, 'old', 23))      # two laps late
        self.assertIsNone(rig.r.consider(result, 'never-sent', 20))
        rig.r.pending = ['safety_car']
        rig.r.note_submitted('moved')
        rig.r.adopt(PLAN_B, 0)                                     # Plan A changed while it ran
        result2 = SimpleNamespace(lap=20, runs=100, outcomes=result.outcomes)
        self.assertIsNone(rig.r.consider(result2, 'moved', 20))

    def test_summary(self):
        rig = Rig()
        self.assertEqual(rig.r.summary(20)['radio'], 'Plan A: one stop, box lap 30 for hards.')
        rig.run(20, gain=9.0, reasons=('vsc',))
        s = rig.r.summary(20)
        self.assertTrue(s['call_active'])
        self.assertEqual(s['last_call']['reasons'], ['vsc'])
        self.assertFalse(rig.r.summary(22)['call_active'])
        self.assertIsNone(rp.Replanner().summary(1))


class LiveWiringTests(unittest.TestCase):
    def setUp(self):
        self.voice = SilentVoice()
        self.engine = AnalysisEngine(self.voice, race_simulation='sync', sim_runs=40, on_demand=True)
        self.engine.update_session(RACE, 1, 0)
        self.laps = [lap(i + 1, 20) for i in range(6)]
        self.engine.update_car_status(status(12), [status(12 + i) for i in range(6)])
        self.engine.update_field(self.laps, 2, 1000.0)
        self.engine.latest_lap = self.laps[2]

    def test_plan_b_breaks_quiet_mode(self):
        self.assertIn('plan_b', PROACTIVE_KEYS)

    def test_reruns_on_safety_car_within_the_lap(self):
        e = self.engine
        e.update_race_simulation()
        first = e.sim_runner.latest
        self.assertIsNotNone(first)
        self.assertTrue(first.outcomes[0].is_reference or any(o.is_reference for o in first.outcomes))
        self.assertIsNotNone(e.replanner.plan_a)
        e.update_race_simulation()
        self.assertIs(e.sim_runner.latest, first)                # nothing changed: no re-run
        e.session = replace(RACE, safety_car_status=1)
        e.update_race_simulation()
        self.assertIsNot(e.sim_runner.latest, first)             # SC: re-run straight away
        self.assertTrue(any(o.pit_laps[:1] == [20] for o in e.sim_runner.latest.outcomes))   # a stop under it
        self.assertEqual(e.sim_runner.latest.lap, 20)

    def test_event_reruns_are_throttled(self):
        e = self.engine
        e.update_race_simulation()
        first = e.sim_runner.latest
        laps = list(self.laps)
        laps[3] = replace(laps[3], num_pit_stops=1)
        e.field_laps = laps
        e.update_race_simulation()
        second = e.sim_runner.latest
        self.assertIsNot(second, first)                          # rival stop: re-run
        laps[1] = replace(laps[1], num_pit_stops=1)
        e.update_race_simulation()
        self.assertIs(e.sim_runner.latest, second)               # a second one moments later waits
        e._last_event_submit -= rp.MIN_EVENT_INTERVAL_S + 0.1
        e.update_race_simulation()
        self.assertIsNot(e.sim_runner.latest, second)            # ... then runs

    def test_plan_b_call_reaches_radio_hud_and_ai(self):
        e = self.engine
        e.update_race_simulation()
        # A Plan A that is clearly worse than the alternatives (three stops in three laps),
        # and a VSC to trigger the re-run: Plan B must be called straight away.
        e.replanner.adopt(cand(['MEDIUM', 'HARD', 'SOFT', 'MEDIUM'], [21, 22, 23]), 0)
        e.replanner.config.confirm_runs = 1
        e.session = replace(RACE, safety_car_status=2)
        e.update_race_simulation()
        spoken = [m for m in self.voice.messages if 'Plan B' in m[0]]
        self.assertEqual(len(spoken), 1, self.voice.messages)
        message, priority, options = spoken[0]
        self.assertTrue(message.startswith('VSC. Plan B, Plan B.'))
        self.assertTrue(options['key'].endswith(':plan_b'))
        summary = e.strategy_summary()
        self.assertTrue(summary['replan']['call_active'])
        self.assertEqual(summary['last_call']['kind'], 'plan_b')
        text = format_context(e.build_context())
        self.assertIn('Current plan (Plan A, the plan we are on)', text)
        self.assertIn('Plan B called on lap 20', text)
        answer = rule_based_answer('strategy', {'session_mode': 'race', 'strategy': summary})
        self.assertTrue(answer.startswith('VSC. Plan B'))

    def test_cleared_on_new_session_and_off_without_sim(self):
        e = self.engine
        e.update_race_simulation()
        e.update_session(replace(RACE, track_id=11), 2, 0)
        self.assertIsNone(e.replanner.plan_a)
        self.assertEqual(e.replanner.events, 0)
        plain = AnalysisEngine(SilentVoice())
        plain.update_session(RACE, 1, 0)
        plain.update_car_status(status(12), [status(12 + i) for i in range(6)])
        plain.latest_lap = self.laps[2]
        plain.update_race_simulation()
        self.assertIsNone(plain.replanner.plan_a)
        self.assertIsNone(plain.strategy_summary()['replan'])

    def test_rerun_cost_stays_live(self):
        """A full field, Plan A plus alternatives and event options, 300 races: well under a second
        or two on a laptop core (live it runs off the telemetry thread once a lap)."""
        state = sim.RaceState(SILVERSTONE, 52, 11, field(20, player=8))
        plan = rs.plan_race(42, 'MEDIUM', 10, 11, rs.pit_loss_s(SILVERSTONE))
        r = rp.Replanner()
        r.sync_plan_a(plan, 11, 0, 'Medium', 42)
        cands = r.candidates(plan, 11, True, False, 'Medium')
        start = time.perf_counter()
        sim.simulate(state, cands, reference=0, runs=300, seed=1)
        self.assertLess(time.perf_counter() - start, 5.0)


if __name__ == '__main__':
    unittest.main()
