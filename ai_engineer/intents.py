"""Intent matching for radio requests: which fixed command a sentence is asking for.

Instead of one regex per command, every sentence is turned into *concepts* (TYRE, TEMP,
PIT, BEHIND...) from a word list, and each intent scores the concepts it cares about.
Two things make this hold up to natural speech:

  * Many ways to say the same thing map to one concept: "rears", "boots" and "rubber" are
    all TYRE; "holding up", "worn" and "grip" are CONDITION. So "how are my tyres holding
    up?" and "tyre temps" both land on ``tyre_temps``.
  * Words the speech recogniser misspells ("temprature", "strategie", "forcast") are
    snapped to the nearest known word when they are close enough.

Combinations decide close calls: PIT with a rival word ("when will they box") is a rivals
question, PIT on its own ("when do we box") is the pit window, and "will it stop raining"
is weather because rain beats the weak "stop".
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

# ── concepts ──
TYRE, TEMP, CONDITION = 'tyre', 'temp', 'condition'
PIT, PIT_WEAK, WINDOW = 'pit', 'pit_weak', 'window'
SELF, RIVAL, NAMED_STOP, DOING = 'self', 'rival', 'named_stop', 'doing'
AHEAD, BEHIND, GAP = 'ahead', 'behind', 'gap'
THREAT, STRATEGY, WEATHER, PACE = 'threat', 'strategy', 'weather', 'pace'

# Multi-word phrases first (matched on the whole sentence), then single words.
PHRASES = {
    'holding up': {CONDITION},
    'how many stops': {STRATEGY},
    'one stop': {STRATEGY},
    'two stop': {STRATEGY},
    'which tyre': {STRATEGY},
    'which tyres': {STRATEGY},
    'what tyre': {STRATEGY},
    'what tyres': {STRATEGY},
    'what compound': {STRATEGY},
    'win probability': {STRATEGY},
    'chances of': {STRATEGY},
    'our chances': {STRATEGY},
    'the chances': {STRATEGY},
    'my chances': {STRATEGY},
    'the odds': {STRATEGY},
    'in front': {AHEAD},
    'car ahead': {RIVAL, AHEAD},
    'car in front': {RIVAL, AHEAD},
    'car behind': {RIVAL, BEHIND},
    'other cars': {RIVAL},
    'the field': {RIVAL},
    'the others': {RIVAL},
    'coming in': {PIT},
    'lap time': {PACE},
    'lap times': {PACE},
    'how am i doing': {PACE},
    'how are we doing': {PACE},
    'how far': {GAP},
    'how close': {GAP},
    'under pressure': {THREAT},
    'closing in': {THREAT},
    'is it going to rain': {WEATHER},
    'what lap': {WINDOW},
    'which lap': {WINDOW},
    'stay out': {PIT_WEAK},
}
WORDS = {
    # tyres
    'tyre': {TYRE},
    'tyres': {TYRE},
    'rubber': {TYRE},
    'boots': {TYRE},
    'rears': {TYRE},
    'fronts': {TYRE},
    'tread': {TYRE},
    'temp': {TEMP},
    'temps': {TEMP},
    'temperature': {TEMP},
    'temperatures': {TEMP},
    'hot': {TEMP},
    'cold': {TEMP},
    'overheating': {TEMP},
    'warm': {TEMP},
    'cool': {TEMP},
    'wear': {CONDITION},
    'worn': {CONDITION},
    'deg': {CONDITION},
    'degradation': {CONDITION},
    'grip': {CONDITION},
    'condition': {CONDITION},
    'shape': {CONDITION},
    'lasting': {CONDITION},
    'holding': {CONDITION},
    'feel': {CONDITION},
    'feeling': {CONDITION},
    'life': {CONDITION},
    # pit stops
    'pit': {PIT},
    'pitting': {PIT},
    'box': {PIT},
    'boxing': {PIT},
    'window': {PIT, WINDOW},
    'stop': {PIT_WEAK},
    'stops': {PIT_WEAK},
    'stopping': {PIT_WEAK},
    # who
    'i': {SELF},
    'we': {SELF},
    'me': {SELF},
    'my': {SELF},
    'us': {SELF},
    'our': {SELF},
    'rival': {RIVAL},
    'rivals': {RIVAL},
    'others': {RIVAL},
    'everyone': {RIVAL},
    'opponent': {RIVAL},
    'opponents': {RIVAL},
    'they': {RIVAL},
    "they're": {RIVAL},
    'theyre': {RIVAL},
    'them': {RIVAL},
    'doing': {DOING},
    # gaps
    'ahead': {AHEAD},
    'front': {AHEAD},
    'leader': {AHEAD},
    'behind': {BEHIND},
    'chasing': {BEHIND},
    'following': {BEHIND},
    'gap': {GAP},
    'gaps': {GAP},
    'interval': {GAP},
    'distance': {GAP},
    # threats
    'threat': {THREAT},
    'threats': {THREAT},
    'attacking': {THREAT},
    'attack': {THREAT},
    'pressure': {THREAT},
    'catching': {THREAT},
    'closing': {THREAT},
    'defend': {THREAT},
    'danger': {THREAT},
    # strategy
    'strategy': {STRATEGY},
    'plan': {STRATEGY},
    'undercut': {STRATEGY},
    'overcut': {STRATEGY},
    'compound': {STRATEGY},
    'odds': {STRATEGY},
    'chances': {STRATEGY},
    'podium': {STRATEGY},
    'strategies': {STRATEGY},
    # weather
    'weather': {WEATHER},
    'rain': {WEATHER},
    'raining': {WEATHER},
    'forecast': {WEATHER},
    'wet': {WEATHER},
    'drizzle': {WEATHER},
    'storm': {WEATHER},
    'clouds': {WEATHER},
    'sky': {WEATHER},
    'shower': {WEATHER},
    'showers': {WEATHER},
    # pace
    'pace': {PACE},
    'laptime': {PACE},
    'quick': {PACE},
    'slow': {PACE},
    'delta': {PACE},
}

# Each intent's score is the sum of its weights for the concepts present.
WEIGHTS = {
    'tyre_temps': {TYRE: 2.0, TEMP: 3.0, CONDITION: 1.0},
    'pit_window': {PIT: 3.0, PIT_WEAK: 2.0, WINDOW: 1.0, SELF: 0.5},
    'gap_ahead': {GAP: 1.5, AHEAD: 2.0},
    'gap_behind': {GAP: 1.0, BEHIND: 3.0},
    'threats': {THREAT: 3.5},
    'strategy': {STRATEGY: 4.0},
    'weather': {WEATHER: 3.0},
    'pace': {PACE: 3.0},
}
# Ties go to the earlier intent (the old regex order).
ORDER = (
    'rivals',
    'threats',
    'strategy',
    'gap_behind',
    'pit_window',
    'gap_ahead',
    'tyre_temps',
    'weather',
    'pace',
)
MIN_SCORE = 2.0

_WAKE = re.compile(r'^(?:hey |ok |okay )?(?:engineer|race engineer|box box)\b[, ]*')
# "when will Hamilton box", "is Leclerc going to pit", "is Verstappen coming in"
_NAMED_STOP = re.compile(
    r'\b(?:when (?:is|will|does)|is|will) (?!(?:i|we|me|my|us|it|the|there|this|that|he|she)\b)'
    r"[a-z']+ (?:going to |gonna )?(?:pit|box|stop|coming in)"
)
_FUZZY_VOCAB = sorted(w for w in WORDS if len(w) >= 4)


def normalise(text: str) -> str:
    text = str(text or '').lower().replace('tire', 'tyre')
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def _fuzzy(word: str) -> str | None:
    """A misheard word ('temprature') -> the known word it's closest to, if it's close."""
    if len(word) < 5:
        return None
    match = difflib.get_close_matches(word, _FUZZY_VOCAB, n=1, cutoff=0.8)
    return match[0] if match else None


def concepts(text: str) -> set[str]:
    t = _WAKE.sub('', normalise(text))
    found: set[str] = set()
    padded = f' {t} '
    for phrase, tags in PHRASES.items():
        if f' {phrase} ' in padded:
            found |= tags
    for word in t.split():
        tags = WORDS.get(word)
        if tags is None:
            near = _fuzzy(word)
            tags = WORDS[near] if near else set()
        found |= tags
    if _NAMED_STOP.search(t):
        found.add(NAMED_STOP)
    return found


@dataclass
class IntentMatch:
    intent: str | None
    score: float
    scores: dict = field(default_factory=dict)
    concepts: set = field(default_factory=set)


def score_intents(found: set[str]) -> dict[str, float]:
    scores = {intent: sum(w for c, w in weights.items() if c in found) for intent, weights in WEIGHTS.items()}
    # Rivals: a stop/tyre/strategy question about someone else. With "we"/"I" in it
    # ("should we pit before the car ahead", "can we undercut them") it's our own call.
    rival = 0.0
    if NAMED_STOP in found:
        rival = 6.0
    elif RIVAL in found and SELF not in found and found & {PIT, PIT_WEAK, TYRE, STRATEGY, DOING}:
        rival = 5.0
    elif RIVAL in found and found & {PIT, PIT_WEAK, TYRE, DOING}:
        rival = 3.0
    scores['rivals'] = rival
    if WEATHER in found and PIT not in found:
        scores['pit_window'] = min(scores['pit_window'], 1.5)  # "will it stop raining"
    if STRATEGY in found:
        scores['tyre_temps'] = min(scores['tyre_temps'], 2.0)  # "which tyres next"
    return scores


def match_intent(text: str) -> IntentMatch:
    found = concepts(text) if text else set()
    if not found:
        return IntentMatch(None, 0.0, {}, found)
    scores = score_intents(found)
    best = max(ORDER, key=lambda name: (scores.get(name, 0.0), -ORDER.index(name)))
    score = scores.get(best, 0.0)
    return IntentMatch(best if score >= MIN_SCORE else None, score, scores, found)


def match(text: str) -> str | None:
    return match_intent(text).intent
