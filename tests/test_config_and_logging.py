"""config.toml loading, command-line overrides, and the structured logging setup."""

import io
import json
import logging
import pathlib
import tempfile
import unittest
from types import SimpleNamespace

import log_setup
import main as app
import race_sim
import strategy_replan
from analysis import PROACTIVE_KEYS, TYRE_TEMP_CRITICAL, AnalysisEngine
from packets import parse_player_telemetry
from pipeline import config as cfg
from simulate_telemetry import build_telemetry_packet
from test_analysis import SilentVoice  # noqa: F401  (puts ai_engineer/ on the import path)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def write_toml(text: str) -> pathlib.Path:
    handle = tempfile.NamedTemporaryFile('w', suffix='.toml', delete=False, encoding='utf-8')
    with handle:
        handle.write(text)
    return pathlib.Path(handle.name)


class ConfigFileTests(unittest.TestCase):
    def test_shipped_config_is_exactly_the_built_in_defaults(self):
        self.assertEqual(cfg.load_config(ROOT / 'config.toml', required=True), cfg.EngineConfig())

    def test_defaults_match_the_modules_they_replace(self):
        defaults = cfg.EngineConfig()
        self.assertEqual(defaults.simulation.runs, race_sim.DEFAULT_RUNS)
        self.assertEqual(defaults.simulation.plan_b_margin_s, strategy_replan.REPLAN_MARGIN_S)
        self.assertEqual(defaults.simulation.plan_b_places, strategy_replan.REPLAN_MARGIN_POSITIONS)
        self.assertEqual(frozenset(defaults.radio.proactive_keys), PROACTIVE_KEYS)
        self.assertEqual(defaults.tyres.critical_c, TYRE_TEMP_CRITICAL)

    def test_missing_default_file_gives_defaults_but_a_named_file_must_exist(self):
        missing = ROOT / 'no-such-config.toml'
        self.assertEqual(cfg.load_config(missing), cfg.EngineConfig())
        with self.assertRaises(cfg.ConfigError):
            cfg.load_config(missing, required=True)

    def test_partial_file_keeps_other_defaults_and_accepts_ints_for_floats(self):
        loaded = cfg.load_config(write_toml('[radio]\nbusy_lateral_g = 2\n[tyres]\ncritical_c = 115\n'))
        self.assertEqual(loaded.radio.busy_lateral_g, 2.0)
        self.assertIsInstance(loaded.radio.busy_lateral_g, float)
        self.assertEqual(loaded.tyres.critical_c, 115)
        self.assertEqual(loaded.tyres.high_c, cfg.TyreConfig().high_c)
        self.assertEqual(loaded.network, cfg.NetworkConfig())

    def test_typos_and_wrong_types_are_errors(self):
        for text in (
            '[radio]\nbusy_lateral_gee = 1.0\n',
            '[radios]\n',
            '[tyres]\ncritical_c = "hot"\n',
            '[radio]\non_demand = 1\n',
            '[radio]\nproactive_keys = "plan_b"\n',
            'not toml [',
        ):
            with self.subTest(text=text), self.assertRaises(cfg.ConfigError):
                cfg.load_config(write_toml(text))

    def test_overrides_beat_the_file_and_none_means_not_given(self):
        base = cfg.load_config(write_toml('[network]\nudp_port = 30000\n[simulation]\nruns = 50\n'))
        merged = cfg.apply_overrides(base, {'network.udp_port': 20888, 'simulation.runs': None})
        self.assertEqual((merged.network.udp_port, merged.simulation.runs), (20888, 50))
        self.assertEqual(base.network.udp_port, 30000)  # the original is untouched
        with self.assertRaises(cfg.ConfigError):
            cfg.apply_overrides(base, {'network.nope': 1})


class CommandLineTests(unittest.TestCase):
    def settings(self, *argv):
        return app.settings_from(app.build_parser().parse_args(list(argv)))

    def test_flags_override_the_config_file(self):
        path = write_toml(
            '[network]\nudp_port = 30000\n[radio]\non_demand = true\n[logging]\nlevel = "WARNING"\n'
        )
        settings = self.settings(
            '--config', str(path), '--chatty', '--log-level', 'debug', '--sim-runs', '40', '--no-race-sim'
        )
        self.assertEqual(settings.network.udp_port, 30000)  # from the file
        self.assertFalse(settings.radio.on_demand)  # --chatty
        self.assertEqual(settings.logging.level, 'DEBUG')  # --log-level, case-insensitive
        self.assertEqual(settings.simulation.runs, 40)
        self.assertFalse(settings.simulation.enabled)

    def test_no_flags_means_config_values(self):
        path = write_toml('[ai]\ntimeout_s = 12\n[voice]\ntts = "system"\n')
        settings = self.settings('--config', str(path))
        self.assertEqual((settings.ai.timeout_s, settings.voice.tts), (12.0, 'system'))
        self.assertTrue(settings.radio.on_demand)

    def test_debug_flag_is_a_log_level_shortcut(self):
        self.assertEqual(self.settings('--config', str(write_toml('')), '--debug').logging.level, 'DEBUG')


class EngineConfigTests(unittest.TestCase):
    def test_busy_filter_reads_the_lateral_g_threshold(self):
        motion = SimpleNamespace(g_force_lateral=1.5)
        telemetry = parse_player_telemetry(build_telemetry_packet(1, 1, 200, 1, 0, 6, 90), 0)
        for threshold, busy in ((1.2, True), (2.0, False)):
            config = cfg.apply_overrides(cfg.EngineConfig(), {'radio.busy_lateral_g': threshold})
            engine = AnalysisEngine(SilentVoice(), config=config)
            engine.latest_telemetry, engine.latest_motion = telemetry, motion
            with self.subTest(threshold=threshold):
                self.assertEqual(engine.is_driver_busy(), busy)

    def test_proactive_keys_come_from_config(self):
        config = cfg.apply_overrides(cfg.EngineConfig(), {'radio.proactive_keys': ['lap_time']})
        engine = AnalysisEngine(SilentVoice(), on_demand=True, config=config)
        engine.speak('Safety car deployed.', key='safety_car')
        engine.speak('One minute thirty.', key='lap_time')
        self.assertEqual([m for m, *_ in engine.voice.messages], ['One minute thirty.'])


class LoggingTests(unittest.TestCase):
    def tearDown(self):
        root = logging.getLogger(log_setup.ROOT)
        for handler in list(root.handlers):
            root.removeHandler(handler)
        root.setLevel(logging.NOTSET)
        root.propagate = True

    def test_json_lines_carry_level_logger_message_and_extra_fields(self):
        stream = io.StringIO()
        log_setup.configure('INFO', 'json', stream=stream)
        log = log_setup.get_logger('history')
        log.debug('hidden')
        log.warning('Could not save %s', 'lap', extra={'lap': 12})
        lines = stream.getvalue().splitlines()
        self.assertEqual(len(lines), 1)
        entry = json.loads(lines[0])
        self.assertEqual(
            (entry['level'], entry['logger'], entry['msg'], entry['lap']),
            ('WARNING', 'f1.history', 'Could not save lap', 12),
        )
        self.assertTrue(entry['ts'].endswith('Z'))

    def test_text_lines_are_short_and_readable(self):
        stream = io.StringIO()
        log_setup.configure('debug', 'text', stream=stream)
        log_setup.get_logger('voice').debug('Heard: "box"', extra={'confidence': 0.9})
        line = stream.getvalue().strip()
        self.assertRegex(line, r'^\d\d:\d\d:\d\d DEBUG \[voice\] Heard: "box"  confidence=0.9$')

    def test_bad_level_or_format_is_rejected(self):
        with self.assertRaises(ValueError):
            log_setup.configure('LOUD')
        with self.assertRaises(ValueError):
            log_setup.configure('INFO', 'xml')

    def test_engine_history_failures_are_logged_not_raised(self):
        class BrokenHistory:
            def record_lap_stats(self, capture, stats):
                raise OSError('disk full')

        engine = AnalysisEngine(SilentVoice(), history=BrokenHistory())
        engine.capture_id = 'cap'
        with self.assertLogs('f1.history', 'WARNING') as logs:
            engine._record_lap_stats(3, 90000, True, {1: 30000})
        self.assertIn('Could not save lap stats: disk full', logs.output[0])
        self.assertEqual(len(engine.session_lap_stats), 1)


if __name__ == '__main__':
    unittest.main()
