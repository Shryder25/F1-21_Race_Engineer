"""The fine-tune chats look exactly like a live engineer question, and don't teach made-up numbers."""

import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ai_engineer'))
sys.path.insert(0, str(ROOT / 'tools' / 'training'))
import make_sft

from engineer_brain import SYSTEM_PROMPT, build_prompt

SITUATION = (
    'Lap 20/57. P3. Medium, 18 laps old. 0 stops so far. Ahead VER +2.0s, closing 0.3s in the last lap. '
    'Behind NOR -1.4s, steady. Last lap 90.000s. HAM just pitted. Safety car deployed. Raining. Track 41C.'
)


class MakeSftTest(unittest.TestCase):
    def test_situation_becomes_the_live_context(self):
        ctx = make_sft.context_from_situation(SITUATION)
        self.assertEqual((ctx['lap'], ctx['total_laps'], ctx['position']), (20, 57, 3))
        self.assertEqual((ctx['compound'], ctx['tyre_age_laps']), ('Medium', 18))
        self.assertEqual((ctx['car_ahead_name'], ctx['gap_ahead_s']), ('VER', 2.0))
        self.assertEqual((ctx['car_behind_name'], ctx['gap_behind_s']), ('NOR', 1.4))
        self.assertEqual(ctx['last_lap_ms'], 90000)
        self.assertTrue(ctx['safety_car'])
        self.assertEqual(ctx['weather'], 'Light Rain')
        self.assertEqual(ctx['track_temp_c'], 41)
        self.assertEqual(ctx['recent_race_events'], ['HAM just pitted', 'Safety car deployed'])

    def test_chat_matches_the_prompt_the_engineer_sends(self):
        chat = make_sft.to_chat(
            {'situation': SITUATION, 'engineer': 'Box this lap, box.', 'tags': ['pit_call']}
        )
        system, user, answer = (m['content'] for m in chat['messages'])
        self.assertEqual(system, SYSTEM_PROMPT)
        self.assertEqual(user, build_prompt('pit_window', make_sft.context_from_situation(SITUATION))[1])
        self.assertEqual(json.loads(answer), {'voice': 'Box this lap, box.', 'advice': ''})

    def test_lines_with_made_up_numbers_are_left_out(self):
        self.assertIsNotNone(
            make_sft.to_chat({'situation': SITUATION, 'engineer': 'Gap is 2.0, push.', 'tags': ['gap']})
        )
        self.assertIsNone(
            make_sft.to_chat({'situation': SITUATION, 'engineer': 'Gap is 3.4, push.', 'tags': ['gap']})
        )

    def test_question_follows_the_line_topic(self):
        self.assertEqual(make_sft.question_for(['tyres', 'pit_call']), 'pit_window')
        self.assertEqual(make_sft.question_for([]), make_sft.DEFAULT_QUESTION)

    def test_cli_writes_train_and_test_chats(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = pathlib.Path(tmp)
            for name in ('train', 'test'):
                rows = [
                    {'situation': SITUATION, 'engineer': 'Stay out, tyres are fine.', 'tags': ['tyres']},
                    {'situation': SITUATION, 'engineer': 'Target 1:29.5 this lap.', 'tags': ['pace']},
                ]
                (data / f'examples_{name}.jsonl').write_text(
                    '\n'.join(json.dumps(r) for r in rows) + '\nbad\n'
                )
            self.assertEqual(make_sft.main(['--data', tmp]), 0)
            lines = (data / 'sft_train.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(len(json.loads(lines[0])['messages']), 3)
            self.assertTrue((data / 'sft_test.jsonl').exists())

    def test_missing_dataset_says_what_to_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(make_sft.main(['--data', tmp]), 1)

    def test_notebook_is_valid_json_and_uses_the_converter(self):
        nb = json.loads((ROOT / 'tools' / 'training' / 'finetune_engineer_colab.ipynb').read_text())
        source = ''.join(''.join(c['source']) for c in nb['cells'])
        self.assertIn('make_sft.py', source)
        self.assertIn('f1-engineer', source)


if __name__ == '__main__':
    unittest.main()
