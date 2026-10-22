"""Tyre/fuel models fitted per compound and track, with ranges (ai_engineer/fitted_models.py).

The sandbox has no F1 2021 data, so everything here runs on stints with known true values:
small hand-made ones for the maths, and noisy simulator races (tests/golden/synth.py) for the
recording-to-fit pipeline.
"""

import json
import pathlib
import random
import statistics
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for path in (ROOT / 'ai_engineer', ROOT / 'tools', ROOT / 'tests' / 'golden'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import fitted_models as fm  # noqa: E402
import strategy_ml  # noqa: E402
from analysis import AnalysisEngine  # noqa: E402
from engineer_brain import format_context, rule_based_answer  # noqa: E402
from history import TelemetryHistory  # noqa: E402
from packets import SessionData, parse_player_car_damage, parse_player_car_status  # noqa: E402
from simulate_telemetry import build_car_damage_packet, build_car_status_packet  # noqa: E402
from test_analysis import SilentVoice  # noqa: E402

MONZA, MONZA_M = 11, 5793  # tyre severity Low (0.8)
SPAIN, SPAIN_M = 4, 4655  # High (1.15)
SILVERSTONE = 7  # Very High (1.3)


def stints(
    rng,
    track,
    compound,
    rate,
    n_stints=4,
    laps=12,
    noise=0.15,
    stint_sd=0.1,
    growth=0.0,
    session='s',
    fuel=1.8,
    fuel_sd=0.04,
    length=SPAIN_M,
):
    """Lap records for ``n_stints`` stints with known wear rate, growth and noise."""
    out = []
    for s in range(n_stints):
        factor = 1.0 + rng.gauss(0, stint_sd)
        wear = 0.0
        for age in range(laps):
            inc = max(0.0, rate * factor * (1 + growth * age) + rng.gauss(0, noise))
            out.append(
                fm.LapRecord(
                    source='test',
                    session=f'{session}{s // 2}',
                    stint=f'{session}{track}{compound}{s}',
                    track_id=track,
                    track_length=length,
                    session_type=10,
                    race_laps=30,
                    compound=compound,
                    lap=age + 2,
                    tyre_age=age,
                    wear_start=wear,
                    wear_end=wear + inc,
                    fuel_start=100 - 2 * age,
                    fuel_end=100 - 2 * age - (fuel + rng.gauss(0, fuel_sd)),
                    lap_time_ms=int(90000 + 60 * age + rng.gauss(0, 150)),
                    fuel_ok=True,
                    time_ok=True,
                )
            )
            wear += inc
    return out


class RangeAndPriorTests(unittest.TestCase):
    def test_empty_store_falls_back_to_wide_built_in_estimates(self):
        store = fm.ModelStore()
        soft, hard = store.wear_model(SPAIN, 'C4'), store.wear_model(SPAIN, 'C2')
        self.assertEqual(soft.basis, 'prior')
        self.assertGreater(soft.rate(0).mid, hard.rate(0).mid)
        r = soft.rate(0)
        self.assertGreater(r.width / r.mid, 0.8)  # +/- 50%: honest about knowing nothing
        severe = store.wear_model(SILVERSTONE, 'C3').rate(0).mid
        gentle = store.wear_model(MONZA, 'C3').rate(0).mid
        self.assertGreater(severe, gentle)  # track tyre severity scales the prior
        fuel = store.fuel_model(MONZA, MONZA_M).per_lap()
        self.assertTrue(fuel.contains(1.9))
        self.assertEqual(store.deg_model(MONZA, 'C3').basis, 'prior')

    def test_compound_keys(self):
        self.assertEqual(fm.compound_key(16), 'C5')
        self.assertEqual(fm.compound_key(19), 'C2')
        self.assertEqual(fm.compound_key(7), 'INTER')
        self.assertEqual(fm.compound_key('Intermediate'), 'INTER')
        self.assertEqual(fm.compound_key('wet'), 'WET')
        self.assertIsNone(fm.compound_key(99))
        self.assertIsNone(fm.compound_key('Medium'))  # visual names differ per track: not a key

    def test_range_helpers(self):
        r = fm.Range(1.0, 2.0, 3.0)
        self.assertTrue(r.contains(2.5))
        self.assertFalse(r.contains(3.5))
        self.assertEqual(r.width, 2.0)
        self.assertEqual(fm.Range(None, 1.0, None).as_list(), [None, 1.0, None])
        self.assertAlmostEqual(fm.z_score(0.8), 1.2816, places=3)


class WearFitTests(unittest.TestCase):
    def test_fit_recovers_rate_and_growth_per_track_and_compound(self):
        rng = random.Random(1)
        records = (
            stints(rng, SPAIN, 'C3', 3.0, growth=0.02)
            + stints(rng, MONZA, 'C3', 2.0, session='m', length=MONZA_M)
            + stints(rng, SPAIN, 'C4', 4.2, session='q')
        )
        store = fm.ModelStore.fit(records)
        spain = store.wear_model(SPAIN, 'C3')
        self.assertEqual(spain.basis, 'track')
        self.assertEqual((spain.n_laps, spain.n_stints), (48, 4))
        self.assertTrue(spain.rate(0).contains(3.0), spain.rate(0))
        self.assertTrue(spain.rate(10).contains(3.6), spain.rate(10))
        self.assertGreater(spain.mean[1], 0)  # wear accelerates with age
        self.assertTrue(store.wear_model(MONZA, 'C3').rate(0).contains(2.0))
        self.assertTrue(store.wear_model(SPAIN, 'C4').rate(0).contains(4.2))
        self.assertLess(spain.rate(0).width, fm.ModelStore().wear_model(SPAIN, 'C3').rate(0).width / 3)
        self.assertIn('*/C3', store.wear)

    def test_unseen_track_uses_pooled_fit_scaled_by_severity(self):
        rng = random.Random(2)
        records = stints(rng, SPAIN, 'C3', 3.0 * 1.15) + stints(rng, MONZA, 'C3', 3.0 * 0.8, session='m')
        store = fm.ModelStore.fit(records)
        silverstone = store.wear_model(SILVERSTONE, 'C3')
        self.assertEqual(silverstone.basis, 'pooled')
        self.assertTrue(silverstone.rate(0).contains(3.0 * 1.3), silverstone.rate(0))
        own = store.wear_model(SPAIN, 'C3').rate(0)
        self.assertGreater(silverstone.rate(0).width, own.width)  # less sure on a new track
        self.assertEqual(store.wear_model(SILVERSTONE, 'C5').basis, 'prior')  # no data anywhere

    def test_ranges_are_calibrated_on_held_out_stints(self):
        rng = random.Random(3)
        hits = total = 0
        for trial in range(6):
            train = stints(rng, SPAIN, 'C3', 2.8, n_stints=5, noise=0.2, stint_sd=0.12, session=f't{trial}')
            model = fm.ModelStore.fit(train).wear_model(SPAIN, 'C3')
            for test in range(6):
                laps = stints(
                    rng, SPAIN, 'C3', 2.8, n_stints=1, noise=0.2, stint_sd=0.12, session=f'x{trial}{test}'
                )
                ages = [r.tyre_age for r in laps]
                incs = [r.wear_increment for r in laps]
                for j in (0, 2, 5):
                    forecast = model.condition(ages[:j], incs[:j]).wear_after(ages[j], 0.0, 5)
                    hits += forecast.contains(sum(incs[j : j + 5]))
                    total += 1
        self.assertGreater(hits / total, 0.65)
        self.assertLess(hits / total, 0.95)

    def test_stint_laps_narrow_the_range_and_pull_towards_this_stint(self):
        model = fm.ModelStore().wear_model(SPAIN, 'C3')  # prior: about 2.9 %/lap
        fresh = model.condition([], [])
        worn = model.condition([0, 1, 2, 3, 4], [4.0, 4.1, 3.9, 4.0, 4.0])  # this set wears fast
        self.assertLess(worn.rate(5).width, fresh.rate(5).width / 2)
        self.assertAlmostEqual(worn.rate(5).mid, 4.0, delta=0.3)
        self.assertEqual(worn.stint_laps, 5)

    def test_laps_until_and_probability(self):
        stint = fm.ModelStore().wear_model(SPAIN, 'C3').condition(list(range(6)), [3.0] * 6)
        r = stint.laps_until(60.0, 6, 18.0)
        self.assertLessEqual(r.low, r.mid)
        self.assertLessEqual(r.mid, r.high)
        self.assertAlmostEqual(r.mid, 14.0, delta=0.6)  # 42 more points at ~3/lap
        self.assertEqual(stint.laps_until(60.0, 6, 61.0).as_list(), [0.0, 0.0, 0.0])
        self.assertGreater(stint.prob_below(75.0, 6, 18.0, 5), 0.99)
        self.assertLess(stint.prob_below(75.0, 6, 18.0, 40), 0.01)
        w = stint.wear_after(6, 18.0, 3)
        self.assertGreaterEqual(w.low, 18.0)
        self.assertAlmostEqual(w.mid, 27.0, delta=0.8)

    def test_garbage_increments_are_ignored(self):
        rec = fm.LapRecord('t', 's', 'st', SPAIN, SPAIN_M, 10, 0, 'C3', 2, 1, wear_start=40.0, wear_end=0.0)
        self.assertIsNone(rec.wear_increment)  # a tyre change, not wear
        rec.wear_end = 80.0
        self.assertIsNone(rec.wear_increment)  # puncture/damage, not wear

    def test_race_distance_the_fit_has_not_seen_widens_the_range(self):
        store = fm.ModelStore.fit(stints(random.Random(4), SPAIN, 'C3', 3.0))
        same = store.wear_model(SPAIN, 'C3', race_laps=30).rate(0)
        other = store.wear_model(SPAIN, 'C3', race_laps=50).rate(0)
        self.assertGreater(other.width, same.width * 1.5)
        self.assertAlmostEqual(other.mid, same.mid)


class DegAndFuelTests(unittest.TestCase):
    def test_degradation_slope_is_fitted_per_compound_with_a_range(self):
        store = fm.ModelStore.fit(stints(random.Random(5), SPAIN, 'C3', 3.0, n_stints=6))
        deg = store.deg_model(SPAIN, 'C3')
        self.assertEqual(deg.basis, 'track')
        # lap time rises 60 ms a lap; the fuel correction adds 2 kg/lap * 0.03 s/kg = 0.06 s/lap
        self.assertTrue(deg.slope().contains(0.12), deg.slope())
        this_stint = deg.condition(list(range(8)), [90.0 + 0.3 * a for a in range(8)])
        self.assertGreater(this_stint.mid, deg.slope().mid)  # this stint falls off faster

    def test_fuel_fit_margin_and_session_conditioning(self):
        store = fm.ModelStore.fit(stints(random.Random(6), SPAIN, 'C3', 3.0, fuel=1.8))
        model = store.fuel_model(SPAIN, SPAIN_M)
        self.assertEqual(model.basis, 'track')
        self.assertTrue(model.per_lap().contains(1.8))
        fuel = model.condition([1.95, 1.96, 1.94, 1.95, 1.96])  # this race burns more
        self.assertGreater(fuel.per_lap().mid, 1.85)
        kg, laps = fuel.margin(20.0, 10)
        self.assertLess(kg.low, kg.mid)
        self.assertLess(kg.mid, kg.high)
        self.assertAlmostEqual(kg.mid, 20.0 - 10 * fuel.mean, places=6)
        self.assertAlmostEqual(laps.mid, kg.mid / fuel.mean, places=6)
        self.assertLess(fuel.needed(10).width, fuel.needed(20).width)
        stale = model.condition([1.8, 1.8, 0.0, 3.6, 1.8, 1.8])  # a stale lap-end snapshot
        self.assertAlmostEqual(stale.per_lap().mid, 1.8, delta=0.05)
        unseen = store.fuel_model(MONZA, MONZA_M)
        self.assertEqual(unseen.basis, 'pooled')  # kg per km from Spain
        self.assertAlmostEqual(unseen.mean, 1.8 / SPAIN_M * MONZA_M, delta=0.15)


class PersistenceTests(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        rng = random.Random(7)
        store = fm.ModelStore.fit(
            stints(rng, SPAIN, 'C3', 3.0),
            sources=[{'path': 'x.f1rec.gz', 'laps': 48, 'simulated': True}],
            consumed_captures={'abc'},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'models' / 'fitted.json'
            store.save(path)
            data = json.loads(path.read_text())
            self.assertEqual(data['version'], fm.STORE_VERSION)
            self.assertTrue(data['simulated_only'])
            self.assertIn(f'{SPAIN}/C3', data['wear'])
            loaded = fm.ModelStore.load(path)
            self.assertEqual(loaded.consumed_captures, {'abc'})
            self.assertEqual(
                loaded.wear_model(SPAIN, 'C3').rate(4).as_list(4),
                store.wear_model(SPAIN, 'C3').rate(4).as_list(4),
            )
            self.assertEqual(
                loaded.fuel_model(SPAIN, SPAIN_M).per_lap().as_list(4),
                store.fuel_model(SPAIN, SPAIN_M).per_lap().as_list(4),
            )
            self.assertEqual(
                loaded.deg_model(SPAIN, 'C3').slope().as_list(4),
                store.deg_model(SPAIN, 'C3').slope().as_list(4),
            )

    def test_missing_corrupt_or_newer_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            self.assertEqual(fm.ModelStore.load_or_empty(tmp / 'none.json').wear, {})
            (tmp / 'bad.json').write_text('{not json')
            self.assertEqual(fm.ModelStore.load_or_empty(tmp / 'bad.json').wear, {})
            (tmp / 'new.json').write_text(json.dumps({'version': fm.STORE_VERSION + 1}))
            with self.assertRaises(ValueError):
                fm.ModelStore.load(tmp / 'new.json')


class LapRowTests(unittest.TestCase):
    def rows(self, spec):
        return [
            {
                'lap_num': lap,
                'compound': comp,
                'tyre_age_laps': age,
                'worst_wear': wear,
                'fuel_start': 50 - lap,
                'fuel_end': 49 - lap,
                'lap_time_ms': 90000,
                'eligible': True,
            }
            for lap, comp, age, wear in spec
        ]

    def test_stale_and_skipped_snapshots_still_give_per_lap_wear(self):
        rows = self.rows(
            [
                (1, 18, 1, 2.5),
                (2, 18, 1, 2.5),  # lap 2 row is a stale copy
                (3, 18, 3, 7.5),
                (4, 18, 4, 10.0),
                (5, 19, 0, 0.0),
                (6, 19, 2, 4.0),
                (7, 19, 3, 6.0),
            ]
        )  # stop, new set
        recs = fm.records_from_lap_rows(rows, source='t', session='s', track_id=SPAIN, track_length=SPAIN_M)
        wear = [
            (r.compound, r.tyre_age, round(r.wear_increment, 2)) for r in recs if r.wear_increment is not None
        ]
        self.assertEqual(
            wear,
            [
                ('C3', 0, 2.5),
                ('C3', 1, 2.5),
                ('C3', 2, 2.5),
                ('C3', 3, 2.5),
                ('C2', 0, 2.0),
                ('C2', 1, 2.0),
                ('C2', 2, 2.0),
            ],
        )
        self.assertEqual(len({r.stint for r in recs}), 2)
        current = fm.current_stint(recs, 19, 4)
        self.assertEqual(len(fm.stint_increments(current)[1]), 3)
        self.assertEqual(fm.current_stint(recs, 18, 4), [])  # not on that set any more

    def test_history_database_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = TelemetryHistory(pathlib.Path(tmp) / 'h.sqlite3')
            try:
                session = SessionData(0, 30, 20, 30, SPAIN_M, 10, SPAIN, 0)
                a, b = history.start_session(session, 1), history.start_session(session, 2)
                for capture in (a, b):
                    for lap in range(1, 7):
                        history.record_lap_stats(
                            capture,
                            {
                                'lap_num': lap,
                                'lap_time_ms': 90000,
                                'eligible': True,
                                'compound': 18,
                                'tyre_age_laps': lap,
                                'wear': (3.0 * lap,) * 4,
                                'fuel_start': 60 - 2 * lap,
                                'fuel_end': 58 - 2 * lap,
                            },
                        )
                recs, captures = fm.laps_from_history(history)
                self.assertEqual(captures, {a, b})
                self.assertEqual(sum(r.wear_increment is not None for r in recs), 12)
                recs, captures = fm.laps_from_history(history, exclude_captures={a})
                self.assertEqual(captures, {b})
                self.assertEqual(fm.laps_from_history(history, track_id=MONZA)[0], [])
                store = fm.ModelStore.fit(recs)
                self.assertAlmostEqual(store.wear_model(SPAIN, 'C3').rate(0).mid, 3.0, delta=0.15)
                self.assertAlmostEqual(store.fuel_model(SPAIN, SPAIN_M).per_lap().mid, 2.0, delta=0.05)
            finally:
                history.close()


class RecordingPipelineTests(unittest.TestCase):
    """Noisy simulator races -> .f1rec recordings -> lap records -> fitted models."""

    @classmethod
    def setUpClass(cls):
        import recording
        import synth

        cls.tmp = tempfile.TemporaryDirectory()
        cls.paths = []
        for seed, (compound, pits) in enumerate((('soft', {5: 'medium'}), ('medium', {6: 'hard'}))):
            scenario = synth.Scenario(
                name=f'fit_test_{seed}',
                track_id=SPAIN,
                track_length=SPAIN_M,
                session_type=10,
                total_laps=10,
                cars=[
                    synth.Car('Player', compound=compound, grid=1, pits=pits),
                    synth.Car('Rival', pace=0.5, grid=2),
                ],
                fuel_kg=10 * 1.8 + 3,
                wear_scale=1.1,
                wear_noise=0.05,
                stint_noise=0.05,
                fuel_per_lap_kg=1.8,
                fuel_noise=0.03,
                lap_noise_s=0.1,
                seed=seed,
                script=[(7.3, 'safety_car', 1), (8.6, 'safety_car', 0)] if seed else [],
            )
            path = pathlib.Path(cls.tmp.name) / f'{scenario.name}.f1rec.gz'
            with recording.ReproducibleWriter(path, {'source': 'synthetic'}) as writer:
                for t, data in synth.simulate(scenario):
                    writer.write(data, round(t * 1e9))
            cls.paths.append(path)
        cls.records = [r for p in cls.paths for r in fm.laps_from_recording(p)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_records_carry_wear_fuel_and_flags(self):
        recs = self.records
        self.assertTrue(all(r.simulated and r.track_id == SPAIN for r in recs))
        self.assertEqual({r.compound for r in recs}, {'C4', 'C3', 'C2'})
        # The damage packet's worst wheel wears 1.08 x the car's rate; medium is 2.4 %/lap.
        medium = [r.wear_increment for r in recs if r.compound == 'C3' and r.wear_increment is not None]
        self.assertAlmostEqual(statistics.fmean(medium), 2.4 * 1.1 * 1.08, delta=0.25)
        burns = [r.fuel_used for r in recs if r.fuel_used is not None]
        self.assertAlmostEqual(statistics.median(burns), 1.8, delta=0.1)
        first = [r for r in recs if r.source.endswith('fit_test_0.f1rec.gz')]
        self.assertFalse(next(r for r in first if r.lap == 5).fuel_ok)  # in-lap: not a racing lap
        self.assertTrue(next(r for r in first if r.lap == 3).fuel_ok)
        self.assertFalse(any(r.time_ok for r in recs if r.lap == 1))  # standing start
        sc_laps = [r for r in recs if r.source.endswith('fit_test_1.f1rec.gz') and r.lap == 8]
        self.assertTrue(sc_laps and not sc_laps[0].fuel_ok)  # Safety Car lap

    def test_fit_and_hold_out_check(self):
        import fit_models

        store = fm.ModelStore.fit(self.records, [{'path': str(p), 'simulated': True} for p in self.paths])
        self.assertTrue(store.simulated_only)
        medium = store.wear_model(SPAIN, 'C3')
        self.assertEqual(medium.basis, 'track')
        self.assertTrue(medium.simulated)
        self.assertTrue(medium.rate(0).contains(2.4 * 1.1 * 1.08), medium.rate(0))
        self.assertTrue(store.fuel_model(SPAIN, SPAIN_M).per_lap().contains(1.8))
        checks = fit_models.validate(self.records)
        self.assertTrue(any(c.kind == 'wear' for c in checks))
        self.assertTrue(any(c.kind == 'fuel' for c in checks))
        report = fit_models.render_report(store, checks, self.records)
        self.assertIn('Simulator data only', report)
        self.assertIn('| Catalunya | C3 |', report)
        self.assertIn('Hold-out check', report)

    def test_cli_writes_models_and_report(self):
        import fit_models

        with tempfile.TemporaryDirectory() as tmp:
            out, report = pathlib.Path(tmp) / 'm.json', pathlib.Path(tmp) / 'r.md'
            code = fit_models.main(
                [*map(str, self.paths), '--out', str(out), '--report', str(report), '--no-validate']
            )
            self.assertEqual(code, 0)
            self.assertIn(f'{SPAIN}/C3', fm.ModelStore.load(out).wear)
            self.assertIn('# Fitted tyre and fuel models', report.read_text())


class PitWindowTests(unittest.TestCase):
    def stint(self, rate=2.5, laps=8):
        return fm.ModelStore().wear_model(SPAIN, 'C3').condition(list(range(laps)), [rate] * laps)

    def test_window_is_a_range_of_laps(self):
        p = strategy_ml.predict_pit_window(self.stint(), 8, 20.0, current_lap=9, total_laps=53)
        soonest, expected, latest = p.ideal_lap_range
        self.assertLessEqual(soonest, expected)
        self.assertLessEqual(expected, latest)
        self.assertAlmostEqual(expected, 25, delta=1)  # 40 more points at 2.5/lap
        self.assertEqual(p.ideal_lap, expected)
        self.assertGreater(p.latest_lap, p.ideal_lap)
        self.assertFalse(p.finish_without_stop)
        self.assertLess(p.finish_without_stop_prob, 0.05)
        self.assertEqual((p.basis, p.stint_laps), ('prior', 8))
        self.assertEqual(len(p.wear_per_lap_range), 3)

    def test_gentle_wear_reaches_the_flag(self):
        p = strategy_ml.predict_pit_window(self.stint(rate=0.8), 8, 6.4, current_lap=9, total_laps=30)
        self.assertTrue(p.finish_without_stop)
        self.assertGreater(p.finish_without_stop_prob, 0.9)

    def test_window_after_the_flag_is_no_window(self):
        p = strategy_ml.predict_pit_window(self.stint(laps=2), 2, 5.0, current_lap=3, total_laps=7)
        self.assertIsNone(p.ideal_lap)
        self.assertEqual(p.ideal_lap_range, [None, None, None])
        self.assertTrue(p.finish_without_stop)
        last = strategy_ml.predict_pit_window(self.stint(laps=6), 6, 15.0, current_lap=7, total_laps=7)
        self.assertTrue(last.finish_without_stop)
        self.assertIsNone(strategy_ml.predict_pit_window(None, 3, 5.0, 4))


class EngineIntegrationTests(unittest.TestCase):
    """The engine uses a fitted store, history and this stint, and passes ranges downstream."""

    def setUp(self):
        rng = random.Random(8)
        self.store = fm.ModelStore.fit(
            stints(rng, MONZA, 'C4', 2.5, n_stints=6, length=MONZA_M, fuel=1.9)
            + stints(rng, MONZA, 'C3', 1.8, session='h', length=MONZA_M, fuel=1.9)
        )
        self.engine = AnalysisEngine(SilentVoice(), fitted=self.store)
        self.engine.update_session(SessionData(0, 30, 20, 53, MONZA_M, 10, MONZA, 0), 77, 0)

    def feed(self, laps, rate=2.5):
        for lap in range(1, laps + 1):
            self.engine.update_car_status(
                parse_player_car_status(build_car_status_packet(0, 1, 60.0 - 1.9 * lap, 17, lap), 0)
            )
            self.engine.update_car_damage(
                parse_player_car_damage(build_car_damage_packet(0, 1, (rate * lap,) * 4), 0)
            )
            self.engine.lap_start_fuel = 60.0 - 1.9 * (lap - 1)
            self.engine._record_lap_stats(lap, 82000 + 50 * lap, True, {1: 27000, 2: 28000, 3: 27000})
        self.engine.latest_lap = type(
            'L',
            (),
            {
                'current_lap_num': laps + 1,
                'car_position': 4,
                'last_lap_time_ms': 82000,
                'num_pit_stops': 0,
                'pit_status': 0,
            },
        )()

    def test_fitted_store_gives_a_window_from_the_first_lap(self):
        self.feed(1)
        p = self.engine.pit_prediction()
        self.assertIsNotNone(p)
        self.assertEqual(p.basis, 'track')
        self.assertGreater(p.samples, 70)
        fresh_width = p.ideal_lap_range[2] - p.ideal_lap_range[0]
        self.engine.update_session(SessionData(0, 30, 20, 53, MONZA_M, 10, MONZA, 0), 78, 0)
        self.feed(8)
        later = self.engine.pit_prediction()
        self.assertLessEqual(later.ideal_lap_range[2] - later.ideal_lap_range[0], fresh_width)
        self.assertEqual(later.stint_laps, 8)

    def test_summary_context_prompt_and_answers_carry_ranges(self):
        self.feed(6)
        summary = self.engine.model_summary()
        self.assertEqual(summary['compound'], 'C4')
        self.assertEqual(summary['basis'], 'track')
        for key in (
            'wear_per_lap_pct',
            'wear_in_5_laps_pct',
            'deg_s_per_lap',
            'fuel_per_lap_kg',
            'fuel_margin_laps',
            'pit_ideal_lap',
        ):
            self.assertEqual(len(summary[key]), 3, key)
        low, mid, high = summary['wear_per_lap_pct']
        self.assertLess(low, mid)
        self.assertLess(mid, high)
        fuel = self.engine.fuel_prediction()
        self.assertEqual(fuel['basis'], 'track')
        self.assertAlmostEqual(fuel['per_lap_kg'], 1.9, delta=0.1)
        ctx = self.engine.build_context()
        self.assertIsNotNone(ctx['tyre_fuel_model'])
        text = format_context(ctx)
        self.assertIn('fitted on this track', text)
        self.assertIn('Tyre model ranges [low, expected, high]', text)
        self.assertIn('Fuel: ', text)
        answer = rule_based_answer('pit_window', ctx)
        self.assertIsNotNone(ctx['pit_prediction']['ideal_lap'])
        self.assertIn('window lap', answer)

    def test_strategy_plan_reports_the_range_of_outcomes(self):
        self.engine.session_mode = 'race'
        self.feed(6)
        summary = self.engine.strategy_summary()
        self.assertIsNotNone(summary)
        spread = summary['uncertainty']
        self.assertTrue(set(spread) >= {'worse', 'expected', 'better', 'stops_agree'})
        self.assertEqual(spread['expected']['stops'], summary['best']['stops'])
        if spread['next_stop_lap_range']:
            lo, hi = spread['next_stop_lap_range']
            self.assertLessEqual(lo, hi)

    def test_history_laps_refine_an_unfitted_track(self):
        with tempfile.TemporaryDirectory() as tmp:
            history = TelemetryHistory(pathlib.Path(tmp) / 'h.sqlite3')
            try:
                engine = AnalysisEngine(SilentVoice(), history=history)
                session = SessionData(0, 30, 20, 30, SPAIN_M, 10, SPAIN, 0)
                engine.update_session(session, 1, 0)
                self.engine = engine
                self.feed(8, rate=3.3)
                engine.update_session(session, 2, 0)  # a new session at the same track
                self.feed(1, rate=3.3)
                p = engine.pit_prediction()
                self.assertEqual(p.basis, 'track')  # learned from the earlier session
                self.assertAlmostEqual(p.wear_per_lap_pct, 3.3, delta=0.4)
            finally:
                history.close()


if __name__ == '__main__':
    unittest.main()
