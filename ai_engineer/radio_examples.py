"""Real F1 engineer radio lines from similar race moments, shown to the AI as examples.

``tools/openf1_radio.py build`` writes ``data/openf1/examples_train.jsonl``: real team-radio
lines with the race situation they were said in. When the driver asks something, the few lines
whose situation and topic look most like the current one go into the prompt, so a small local
model copies the tone and length of a real engineer. No training is needed.

The file is local only. If it is missing, nothing changes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from log_setup import get_logger

log = get_logger('engineer')

DEFAULT_PATH = Path(__file__).resolve().parent.parent / 'data' / 'openf1' / 'examples_train.jsonl'
DEFAULT_COUNT = 4
MAX_WORDS = 28  # longer lines teach the model to ramble

# Same topic words as tools/openf1_radio.py uses to tag the real lines.
TAGS: dict[str, re.Pattern[str]] = {
    'pit_call': re.compile(r'\b(box|pit|pits|stop|stops|stay out|plan [a-z]|in this lap|window)\b', re.I),
    'gap': re.compile(r'\b(gap|interval|ahead|behind|tenths?|drs|catching|far)\b', re.I),
    'tyres': re.compile(
        r'\b(tyres?|tires?|deg|degradation|wear|graining|softs?|mediums?|hards?|inters?|temps?)\b', re.I
    ),
    'pace': re.compile(
        r'\b(pace|push|lap ?times?|purple|fastest|target|delta|quicker|slower|faster)\b', re.I
    ),
    'threat': re.compile(r'\b(under ?cut|over ?cut|defend|attack|coming|closing|threat|battle|pass)\b', re.I),
    'flags': re.compile(r'\b(safety car|vsc|yellow|red flag|green flag|sc|flag)\b', re.I),
    'weather': re.compile(r'\b(rain|wet|drying|dry line|spits?|drizzle|weather)\b', re.I),
    'car': re.compile(
        r'\b(mode|strat|engine|brakes?|battery|ers|fuel|lift and coast|lico|diff|damage)\b', re.I
    ),
}

# The fixed radio commands and the topics their answers are about.
COMMAND_TAGS = {
    'weather': {'weather'},
    'tyre_temps': {'tyres'},
    'pit_window': {'pit_call', 'tyres'},
    'gap_ahead': {'gap'},
    'gap_behind': {'gap', 'threat'},
    'pace': {'pace'},
    'threats': {'threat', 'gap'},
    'strategy': {'pit_call', 'tyres', 'threat'},
    'rivals': {'pit_call', 'threat'},
}

_LAP = re.compile(r'\bLap (\d+)/(\d+)')
_POS = re.compile(r'\bP(\d+)\b')
_TYRE = re.compile(r'\b([A-Za-z]+), (\d+) laps? old')
_SC = re.compile(r'\b(safety car|vsc|virtual safety car)\b', re.I)
_RAIN = re.compile(r'\b(rain|raining|wet)\b', re.I)
_WET_TYRE = re.compile(r'\b(inter|intermediate|wet)\b', re.I)


@dataclass(frozen=True)
class Moment:
    """The parts of a race situation used to find similar moments (None = unknown)."""

    phase: float | None = None  # 0 at the start, 1 at the flag
    position: int | None = None
    compound: str | None = None
    tyre_age: int | None = None
    safety_car: bool = False
    rain: bool = False


@dataclass(frozen=True)
class Example:
    situation: str
    engineer: str
    tags: frozenset[str]
    moment: Moment


def question_tags(question: str) -> set[str]:
    if question in COMMAND_TAGS:
        return set(COMMAND_TAGS[question])
    return {name for name, pattern in TAGS.items() if pattern.search(question or '')}


def moment_from_situation(text: str) -> Moment:
    lap = _LAP.search(text)
    pos = _POS.search(text)
    tyre = _TYRE.search(text)
    return Moment(
        phase=int(lap.group(1)) / int(lap.group(2)) if lap and int(lap.group(2)) else None,
        position=int(pos.group(1)) if pos else None,
        compound=tyre.group(1).lower() if tyre else None,
        tyre_age=int(tyre.group(2)) if tyre else None,
        safety_car=bool(_SC.search(text)),
        rain=bool(_RAIN.search(text)) or bool(tyre and _WET_TYRE.search(tyre.group(1))),
    )


def moment_from_context(ctx: dict) -> Moment:
    lap, total = ctx.get('lap'), ctx.get('total_laps')
    compound = ctx.get('compound') or ctx.get('tyre_compound')
    weather = str(ctx.get('weather') or '')
    safety_car = ctx.get('safety_car')
    return Moment(
        phase=lap / total
        if isinstance(lap, int | float) and isinstance(total, int | float) and total
        else None,
        position=ctx.get('position') if isinstance(ctx.get('position'), int) else None,
        compound=str(compound).lower() if compound else None,
        tyre_age=ctx.get('tyre_age_laps') if isinstance(ctx.get('tyre_age_laps'), int) else None,
        safety_car=bool(safety_car) and str(safety_car).upper() != 'NONE',
        rain=bool(_RAIN.search(weather)) or bool(compound and _WET_TYRE.search(str(compound))),
    )


def similarity(now: Moment, then: Moment) -> float:
    """Bigger is more alike. Unknown parts neither help nor hurt."""
    score = 0.0
    if now.phase is not None and then.phase is not None:
        score += 1.5 * (1 - min(abs(now.phase - then.phase), 1.0))
    if now.position is not None and then.position is not None:
        score += 1 - min(abs(now.position - then.position), 10) / 10
    if now.tyre_age is not None and then.tyre_age is not None:
        score += 1 - min(abs(now.tyre_age - then.tyre_age), 20) / 20
    if now.compound and then.compound and now.compound[:3] == then.compound[:3]:
        score += 0.5
    score += 1.5 if now.safety_car == then.safety_car else -1.5
    score += 1.0 if now.rain == then.rain else -1.0
    return score


def short_situation(text: str) -> str:
    """The situation without the detail the model doesn't need to copy the style."""
    keep = []
    for sentence in (s.strip() for s in text.split('. ')):
        if not sentence:
            continue
        if _LAP.match(sentence) or _POS.fullmatch(sentence.rstrip('.')) or 'laps old' in sentence:
            keep.append(sentence.rstrip('.'))
        elif _SC.search(sentence) or _RAIN.search(sentence) or 'pitted' in sentence or 'Ahead' in sentence:
            keep.append(sentence.rstrip('.'))
    return '. '.join(keep[:5])


class RadioExamples:
    """The real radio lines, loaded once, and the lookup of the most similar ones."""

    def __init__(self, examples: list[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    @classmethod
    def load(cls, path: Path | str | None = None) -> RadioExamples | None:
        """The examples in ``path``, or None when the file is missing or has none usable."""
        path = Path(path) if path else DEFAULT_PATH
        if not path.is_file():
            return None
        examples = []
        try:
            with path.open(encoding='utf-8') as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    situation = str(row.get('situation') or '').strip()
                    engineer = ' '.join(str(row.get('engineer') or '').split())
                    if not situation or not engineer or len(engineer.split()) > MAX_WORDS:
                        continue
                    tags = row.get('tags') or [
                        name for name, pattern in TAGS.items() if pattern.search(engineer)
                    ]
                    examples.append(
                        Example(situation, engineer, frozenset(tags), moment_from_situation(situation))
                    )
        except OSError as error:
            log.warning(f'Could not read real radio examples {path}: {error}')
            return None
        if not examples:
            return None
        log.info(f'Real radio examples: {len(examples)} lines from {path}')
        return cls(examples)

    def similar(self, question: str, ctx: dict, count: int = DEFAULT_COUNT) -> list[Example]:
        wanted = question_tags(question)
        now = moment_from_context(ctx)
        scored = []
        for i, example in enumerate(self.examples):
            topic = len(wanted & example.tags)
            if wanted and not topic:
                continue  # a line about something else teaches the wrong content
            score = 3.0 * topic + similarity(now, example.moment)
            score += 0.3 if len(example.engineer.split()) <= 18 else 0.0
            scored.append((score, -i, example))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        picked: list[Example] = []
        seen: set[str] = set()
        for _, _, example in scored:
            key = example.engineer.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append(example)
            if len(picked) == count:
                break
        return picked

    def prompt_block(self, question: str, ctx: dict, count: int = DEFAULT_COUNT) -> str:
        """Lines to add to the AI prompt, or '' when nothing fits."""
        picked = self.similar(question, ctx, count)
        if not picked:
            return ''
        lines = [
            'Real F1 engineers in similar moments (copy their tone and length; never copy their '
            'numbers or names, use only the current data):'
        ]
        for example in picked:
            situation = short_situation(example.situation)
            lines.append(f'- {situation}: "{example.engineer}"' if situation else f'- "{example.engineer}"')
        return '\n'.join(lines)
