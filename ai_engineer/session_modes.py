"""Classify F1 2021 session types so the engineer behaves differently in each.

Game session ids: 1-4 practice, 5-9 qualifying, 10-12 race, 13 time trial.
"""

from __future__ import annotations

PRACTICE, QUALIFYING, RACE, TIME_TRIAL, UNKNOWN = 'practice', 'qualifying', 'race', 'time_trial', 'unknown'

_MODES = {1: PRACTICE, 2: PRACTICE, 3: PRACTICE, 4: PRACTICE,
          5: QUALIFYING, 6: QUALIFYING, 7: QUALIFYING, 8: QUALIFYING, 9: QUALIFYING,
          10: RACE, 11: RACE, 12: RACE, 13: TIME_TRIAL}

MODE_LABELS = {PRACTICE: 'Free Practice', QUALIFYING: 'Qualifying', RACE: 'Race',
               TIME_TRIAL: 'Time Trial', UNKNOWN: 'Session'}

# What each mode makes meaningful.
_FEATURES = {
    #              rival gaps, pit window, race pace
    RACE:        (True,  True,  True),
    PRACTICE:    (False, True,  False),   # long runs: tyre wear still matters
    QUALIFYING:  (False, False, False),
    TIME_TRIAL:  (False, False, False),   # alone on track, no stops
    UNKNOWN:     (False, False, False),
}


def session_mode(session_type: int | None) -> str:
    return _MODES.get(session_type, UNKNOWN)


def mode_label(session_type: int | None) -> str:
    return MODE_LABELS[session_mode(session_type)]


def reports_gaps(mode: str) -> bool:
    return _FEATURES.get(mode, _FEATURES[UNKNOWN])[0]


def has_pit_strategy(mode: str) -> bool:
    return _FEATURES.get(mode, _FEATURES[UNKNOWN])[1]


def has_race_pace(mode: str) -> bool:
    return _FEATURES.get(mode, _FEATURES[UNKNOWN])[2]


def welcome_line(mode: str, track: str) -> str:
    """Opening radio call tailored to what the driver is about to do."""
    return {
        RACE: f'Engineer online. {track}, race. Gaps, pit window and race pace are live. Let\'s go.',
        QUALIFYING: f'Engineer online. {track}, qualifying. Deltas and sector splits are live. Build the lap.',
        PRACTICE: f'Engineer online. {track}, free practice. Tyre wear and lap deltas are live. Let\'s learn the car.',
        TIME_TRIAL: f'Engineer online. {track}, time trial. Chase your personal best, delta is live.',
    }.get(mode, f'Engineer online. Welcome to {track}. Braking points and live delta tracker online. Let\'s go.')
