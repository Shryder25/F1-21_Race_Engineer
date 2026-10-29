"""Turn the real-radio dataset into chat examples for fine-tuning the engineer model.

    python tools/training/make_sft.py --data data/openf1

Reads examples_train.jsonl and examples_test.jsonl (from tools/openf1_radio.py build) and
writes sft_train.jsonl and sft_test.jsonl next to them. Each example is a chat in exactly
the shape the engineer sends a live question: the same system prompt and the same
"Driver asks ... Current data ..." message (built by engineer_brain.build_prompt from the
real race situation), with the real engineer's line as the JSON answer.

Lines that quote a number the situation doesn't have are left out, so the model doesn't
learn to make up gaps or lap times.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'ai_engineer'))
from engineer_brain import build_prompt  # noqa: E402

Row = dict

_LAP = re.compile(r'^Lap (\d+)(?:/(\d+))?$')
_POS = re.compile(r'^P(\d+)$')
_TYRE = re.compile(r'^([A-Za-z]+), (\d+) laps? old$')
_AHEAD = re.compile(r'^Ahead (\S+) \+([\d.]+)s')
_BEHIND = re.compile(r'^Behind (\S+) -([\d.]+)s')
_LAST = re.compile(r'^Last lap ([\d.]+)s$')
_TRACK = re.compile(r'^Track (-?\d+)C$')
_DIGITS = re.compile(r'\d+(?:\.\d+)?')

# Which question the line answers, by its first matching topic.
TAG_QUESTIONS = [
    ('pit_call', 'pit_window'),
    ('flags', 'Anything happening on track?'),
    ('weather', 'weather'),
    ('threat', 'threats'),
    ('gap', 'gap_ahead'),
    ('tyres', 'tyre_temps'),
    ('pace', 'pace'),
    ('car', 'Anything I need to do with the car?'),
]
DEFAULT_QUESTION = 'Anything I need to know?'


def context_from_situation(text: str) -> dict:
    """The live-game context dict that matches a situation line from openf1_radio.situation."""
    ctx: dict = {'session_mode': 'race', 'session_type': 'Race', 'weather': 'Clear'}
    events = []
    for sentence in (s.strip().rstrip('.') for s in text.split('. ')):
        if not sentence:
            continue
        if m := _LAP.match(sentence):
            ctx['lap'] = int(m.group(1))
            if m.group(2):
                ctx['total_laps'] = int(m.group(2))
        elif m := _POS.match(sentence):
            ctx['position'] = int(m.group(1))
        elif m := _TYRE.match(sentence):
            ctx['compound'] = m.group(1).capitalize()
            ctx['tyre_age_laps'] = int(m.group(2))
        elif m := _AHEAD.match(sentence):
            ctx['car_ahead_name'], ctx['gap_ahead_s'] = m.group(1), float(m.group(2))
        elif m := _BEHIND.match(sentence):
            ctx['car_behind_name'], ctx['gap_behind_s'] = m.group(1), float(m.group(2))
        elif m := _LAST.match(sentence):
            ctx['last_lap_ms'] = round(float(m.group(1)) * 1000)
        elif m := _TRACK.match(sentence):
            ctx['track_temp_c'] = int(m.group(1))
        elif sentence == 'Raining':
            ctx['weather'] = 'Light Rain'
        elif 'stop' in sentence and 'so far' in sentence:
            continue
        else:
            if re.search(r'\b(safety car|vsc)\b', sentence, re.I) and 'ending' not in sentence.lower():
                ctx['safety_car'] = (
                    'VSC' if 'vsc' in sentence.lower() or 'virtual' in sentence.lower() else True
                )
            events.append(sentence)
    if events:
        ctx['recent_race_events'] = events
    return ctx


def question_for(tags: list[str]) -> str:
    for tag, question in TAG_QUESTIONS:
        if tag in tags:
            return question
    return DEFAULT_QUESTION


def grounded(line: str, prompt: str) -> bool:
    """True when every number in the line also appears in the prompt."""
    known = set(_DIGITS.findall(prompt))
    return all(number in known for number in _DIGITS.findall(line))


def to_chat(row: Row) -> Row | None:
    line = ' '.join(str(row.get('engineer') or '').split())
    situation = str(row.get('situation') or '')
    if not line or not situation:
        return None
    ctx = context_from_situation(situation)
    system, user = build_prompt(question_for(list(row.get('tags') or [])), ctx)
    if not grounded(line, user):
        return None
    answer = json.dumps({'voice': line, 'advice': ''}, ensure_ascii=False)
    return {
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
            {'role': 'assistant', 'content': answer},
        ],
        'session_key': row.get('session_key'),
    }


def convert(source: Path, target: Path) -> tuple[int, int]:
    kept = dropped = 0
    with source.open(encoding='utf-8') as f, target.open('w', encoding='utf-8') as out:
        for raw in f:
            try:
                row = json.loads(raw)
            except ValueError:
                continue
            chat = to_chat(row) if isinstance(row, dict) else None
            if chat is None:
                dropped += 1
                continue
            out.write(json.dumps(chat, ensure_ascii=False) + '\n')
            kept += 1
    return kept, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data', type=Path, default=Path('data/openf1'))
    args = parser.parse_args(argv)
    for name in ('train', 'test'):
        source = args.data / f'examples_{name}.jsonl'
        if not source.exists():
            print(f'missing {source}: run tools/openf1_radio.py build first')
            return 1
        kept, dropped = convert(source, args.data / f'sft_{name}.jsonl')
        print(f'{name}: {kept} chat examples, {dropped} left out (made-up numbers or empty)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
