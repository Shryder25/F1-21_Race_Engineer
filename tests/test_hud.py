import pathlib
import shutil
import subprocess
import unittest


class HUDClientTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser script regression test')
    def test_distance_alignment_and_baseline_reset(self):
        result = subprocess.run(['node', str(pathlib.Path(__file__).with_name('hud_client_test.js'))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser script regression test')
    def test_tyre_and_fuel_model_card_shows_ranges(self):
        result = subprocess.run(['node', str(pathlib.Path(__file__).with_name('hud_model_card_test.js'))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser script regression test')
    def test_strategy_card_lists_rival_stop_predictions(self):
        result = subprocess.run(['node', str(pathlib.Path(__file__).with_name('hud_rival_card_test.js'))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser script regression test')
    def test_track_database_shows_qualifying_plan(self):
        result = subprocess.run(['node', str(pathlib.Path(__file__).with_name('hud_quali_setup_test.js'))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js required for browser script regression test')
    def test_track_map_draws_stored_circuit_pit_lane_and_flashback(self):
        result = subprocess.run(['node', str(pathlib.Path(__file__).with_name('hud_track_map_test.js'))],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
