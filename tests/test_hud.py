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
