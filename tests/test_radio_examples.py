"""Real F1 radio lines from similar moments are shown to the AI so it talks like an engineer."""

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from engineer_brain import EngineerBrain, build_prompt
from radio_examples import RadioExamples, moment_from_context, moment_from_situation, short_situation

ROWS = [
    {
        'situation': 'Lap 20/57. P3. Medium, 18 laps old. 0 stops so far. Ahead LEC +1.2s, closing 0.2s '
        'in the last lap. Last lap 95.100s. Track 41C.',
        'engineer': 'Box, box. Box this lap for hards.',
        'tags': ['pit_call', 'tyres'],
    },
    {
        'situation': 'Lap 21/57. P4. Medium, 19 laps old. 0 stops so far. Track 40C.',
        'engineer': 'Stay out, stay out. Tyres look fine, we go longer.',
        'tags': ['pit_call', 'tyres'],
    },
    {
        'situation': 'Lap 5/57. P10. Soft, 5 laps old. 0 stops so far. Track 38C.',
        'engineer': 'Gap to Norris is one point five, you have DRS next lap.',
        'tags': ['gap'],
    },
    {
        'situation': 'Lap 33/57. P2. Hard, 12 laps old. 1 stops so far. Safety car deployed. Track 39C.',
        'engineer': 'Safety car, safety car. Stay out, we keep track position.',
        'tags': ['pit_call', 'flags'],
    },
    {
        'situation': 'Lap 40/57. P2. Intermediate, 6 laps old. 1 stops so far. Raining. Track 25C.',
        'engineer': 'Rain is getting heavier at turn ten, stay on the inters and keep it tidy.',
        'tags': ['weather', 'tyres'],
    },
    {
        'situation': 'Lap 22/57. P3. Medium, 20 laps old. 0 stops so far.',
        'engineer': 'Box, box. Box this lap for hards.',  # duplicate wording: shown once
        'tags': ['pit_call', 'tyres'],
    },
    {
        'situation': 'Lap 10/57. P6. Medium, 10 laps old.',
        'engineer': ' '.join(['word'] * 40),  # too long to be a useful example
        'tags': ['pace'],
    },
    {'situation': '', 'engineer': 'No situation, skipped.', 'tags': []},
]

CTX = {
    'lap': 19,
    'total_laps': 50,
    'position': 3,
    'compound': 'Medium',
    'tyre_age_laps': 17,
    'weather': 'Clear',
    'safety_car': False,
    'gap_ahead_s': 1.4,
}


def write_rows(folder: str, rows=ROWS) -> pathlib.Path:
    path = pathlib.Path(folder) / 'examples_train.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
        f.write('not json\n')
    return path


class LoadingTest(unittest.TestCase):
    def test_missing_file_means_no_examples(self):
        self.assertIsNone(RadioExamples.load('/nonexistent/examples_train.jsonl'))

    def test_bad_rows_and_long_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            examples = RadioExamples.load(write_rows(tmp))
        self.assertIsNotNone(examples)
        self.assertEqual(len(examples), 6)

    def test_reads_the_situation_text_from_the_dataset_builder(self):
        moment = moment_from_situation(ROWS[3]['situation'])
        self.assertAlmostEqual(moment.phase, 33 / 57)
        self.assertEqual((moment.position, moment.compound, moment.tyre_age), (2, 'hard', 12))
        self.assertTrue(moment.safety_car)
        self.assertFalse(moment.rain)
        self.assertTrue(moment_from_situation(ROWS[4]['situation']).rain)

    def test_reads_the_live_game_context(self):
        moment = moment_from_context({**CTX, 'weather': 'Light Rain', 'safety_car': 'VSC'})
        self.assertEqual((moment.position, moment.compound, moment.tyre_age), (3, 'medium', 17))
        self.assertTrue(moment.rain)
        self.assertTrue(moment.safety_car)
        self.assertFalse(moment_from_context({'safety_car': 'NONE'}).safety_car)


class PickingTest(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.examples = RadioExamples.load(write_rows(tmp))

    def test_pit_question_gets_pit_calls_from_a_similar_moment(self):
        picked = self.examples.similar('should I box this lap?', CTX, count=2)
        self.assertEqual(picked[0].engineer, 'Box, box. Box this lap for hards.')
        self.assertTrue(all('pit_call' in e.tags for e in picked))
        # The safety-car call doesn't fit a green-flag moment.
        self.assertNotIn('Safety car', ' '.join(e.engineer for e in picked))

    def test_same_line_is_never_shown_twice(self):
        picked = self.examples.similar('pit_window', CTX, count=4)
        texts = [e.engineer.lower() for e in picked]
        self.assertEqual(len(texts), len(set(texts)))

    def test_safety_car_moment_prefers_safety_car_call(self):
        picked = self.examples.similar('do we pit?', {**CTX, 'safety_car': True}, count=1)
        self.assertIn('Safety car', picked[0].engineer)

    def test_lines_about_other_topics_are_left_out(self):
        picked = self.examples.similar('gap_ahead', CTX)
        self.assertEqual([e.engineer for e in picked], [ROWS[2]['engineer']])
        self.assertEqual(self.examples.prompt_block('how is my battery', CTX), '')

    def test_prompt_block_is_short_and_says_not_to_copy_numbers(self):
        block = self.examples.prompt_block('should I box?', CTX)
        self.assertIn('never copy their numbers or names', block)
        self.assertIn('"Box, box. Box this lap for hards."', block)
        self.assertLessEqual(len(block.splitlines()), 5)
        self.assertNotIn('Last lap 95.100s', block)  # detail the style doesn't need

    def test_short_situation_keeps_lap_tyre_and_flags(self):
        self.assertEqual(
            short_situation(ROWS[3]['situation']),
            'Lap 33/57. P2. Hard, 12 laps old. Safety car deployed',
        )


class PromptTest(unittest.TestCase):
    def setUp(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.examples = RadioExamples.load(write_rows(tmp))

    def test_examples_go_into_the_ai_prompt(self):
        _, user = build_prompt('should I box this lap?', CTX, examples=self.examples)
        self.assertIn('Real F1 engineers in similar moments', user)
        self.assertTrue(user.endswith('Answer as the race engineer.'))
        self.assertLess(user.index('Current data'), user.index('Real F1 engineers'))

    def test_prompt_unchanged_without_examples(self):
        _, without = build_prompt('should I box this lap?', CTX)
        self.assertNotIn('Real F1 engineers', without)
        _, with_none = build_prompt('should I box this lap?', CTX, examples=None)
        self.assertEqual(without, with_none)

    def test_brain_sends_examples_to_the_model(self):
        class Client:
            supports_fast = False
            model = 'fake'

            def __init__(self):
                self.prompts = []

            def chat_json(self, system, user, **kwargs):
                self.prompts.append(user)
                return {'voice': 'Box this lap.', 'advice': ''}

            def chat(self, system, user, **kwargs):
                self.prompts.append(user)
                return 'Box this lap.'

        client = Client()
        brain = EngineerBrain(lambda: dict(CTX), client, examples=self.examples)
        brain.answer_question('should I box this lap?')
        self.assertTrue(client.prompts)
        self.assertIn('Real F1 engineers in similar moments', client.prompts[0])


if __name__ == '__main__':
    unittest.main()
