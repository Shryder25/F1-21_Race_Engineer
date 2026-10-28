"""The HUD page layout: every element the browser script looks up is in the page, and the
live and pit wall tabs are built to fit one screen."""

import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'


class HudLayoutTests(unittest.TestCase):
    def setUp(self):
        self.html = (ROOT / 'templates' / 'index.html').read_text(encoding='utf-8')
        self.js = (ROOT / 'static' / 'app.js').read_text(encoding='utf-8')
        self.css = (ROOT / 'static' / 'style.css').read_text(encoding='utf-8')

    def test_every_id_the_script_uses_is_in_the_page(self):
        used = set(re.findall(r"getElementById\('([\w-]+)'\)", self.js))
        present = set(re.findall(r'id="([\w-]+)"', self.html))
        self.assertTrue(used)
        self.assertEqual(sorted(used - present), [])

    def test_live_and_pit_wall_tabs_fit_the_screen(self):
        for tab in ('tab-live', 'tab-pitwall'):
            tag = re.search(rf'<main[^>]*id="{tab}"', self.html)
            self.assertIsNotNone(tag, tab)
            self.assertIn('fit-screen', tag.group(0))
        # Four quadrants on the live tab: car & timing, track map, race, strategy
        for cls in ('quad-car', 'track-map-panel', 'quad-race', 'quad-strategy'):
            self.assertIn(cls, self.html)
        # The page itself never scrolls at desktop size; tabs that need it scroll inside.
        body = re.search(r'^body\.f1-theme \{[^}]*\}', self.css, re.M).group(0)
        self.assertIn('height: 100vh', body)
        self.assertIn('overflow: hidden', body)
        fit = re.search(r'\.tab-content\.fit-screen \{[^}]*\}', self.css).group(0)
        self.assertIn('overflow: hidden', fit)


if __name__ == '__main__':
    unittest.main()
