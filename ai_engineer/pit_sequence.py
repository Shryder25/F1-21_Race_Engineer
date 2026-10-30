"""The pit stop as a real engineer calls it: box, confirm, stay-out abort, out-lap brief.

A race engineer never makes a stop with one line. The sequence is:

  1. Box call     the decision, early in the lap ("Box this lap, box box"). The existing
                  strategy and damage calls make it; ``box_called()`` arms the sequence.
  2. Confirm      a few seconds before pit entry: "Box, box. Pit entry coming up."
  3. Stay out     until the car is almost at the entry line the call can be taken back
                  ("Stay out, stay out") when a rival's stop or a re-plan means the stop
                  no longer makes sense. Calls made for damage or worn-out tyres can't be.
  4. Out-lap brief once the car is back on track: places gained or lost in the stop, the cars
                  either side with their gap and tyre age, and what to do on the out lap.

Pit entry is a lap distance learned from every car that turns into the pit lane (median of
the field's entries), so it is right for any circuit after the first stop of the race, and
saved per track. Until then it falls back to a typical spot just before the line.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import radio_phrases as say
from voice_engine import Priority

RACE_SESSIONS = (10, 11, 12)
ACTIVE = 2

# Where pit entry is when it hasn't been seen yet: this far before the line.
DEFAULT_ENTRY_BEFORE_LINE_M = 400.0
# The confirm call goes this far before pit entry (about 8-10 s at racing speed).
CONFIRM_LEAD_M = 700.0
# A box call made this close to the entry already does the confirm's job.
CONFIRM_SKIP_M = CONFIRM_LEAD_M + 150.0
# Closer than this to the entry line a box call is for the next time round...
LATE_CALL_M = 150.0
# ...and a stay-out call is too late to take: the car is committed.
ABORT_MIN_M = 120.0
# Entries seen before the learned spot is trusted over the default, and the most kept.
MIN_ENTRY_SAMPLES = 1
MAX_ENTRY_SAMPLES = 25
SAVE_AFTER_SAMPLES = 3

# Out-lap brief: wait for the stop-time call and for gaps to settle, but never longer than this.
BRIEF_DELAY_S = 4.0
BRIEF_TIMEOUT_S = 12.0
GAP_FRESH_S = 2.0
FRESH_TYRE_LAPS = 2
TYRE_EDGE_LAPS = 8  # this many laps fresher than the car around = a tyre advantage worth saying
CLOSE_BEHIND_S = 2.0

# Why the box was called. Only these may be taken back with a stay-out call: a stop for
# damage, worn-out tyres or to cover a car that has already boxed has to happen.
ABORTABLE_REASONS = frozenset({'window', 'undercut', 'plan_b', 'plan'})

COMPOUND_WORDS = {
    'SOFT': 'softs',
    'MEDIUM': 'mediums',
    'HARD': 'hards',
    'INTERMEDIATE': 'inters',
    'WET': 'wets',
}
_GAME_NAMES = {
    'Soft': 'SOFT',
    'Medium': 'MEDIUM',
    'Hard': 'HARD',
    'Intermediate': 'INTERMEDIATE',
    'Wet': 'WET',
}

IDLE, BOXING, ABORTED, IN_PIT, OUT_LAP = 'idle', 'boxing', 'aborted', 'in_pit', 'out_lap'


def tyre_words(compound_name: str | None) -> str | None:
    if not compound_name:
        return None
    key = _GAME_NAMES.get(compound_name, compound_name.upper())
    return COMPOUND_WORDS.get(key)


@dataclass
class Neighbour:
    side: str  # 'ahead' or 'behind'
    car_index: int
    name: str | None
    gap_s: float | None
    compound: str | None  # spoken plural ("mediums"), None if unknown
    tyre_age: int | None
    stops: int


@dataclass
class PitCall:
    """The box call in progress, from the decision until the out-lap brief."""

    reason: str
    lap: int
    car_index: int | None = None  # the rival the call is about (undercut target, car covered)
    called_remaining_m: float = 0.0
    wraps_allowed: int = 0  # entry passes to skip (a call made too late for this lap)
    confirmed: bool = False
    prev_remaining_m: float | None = None


@dataclass
class StopRecord:
    """The player's own stop, for the out-lap brief."""

    entry_position: int
    entry_stops: int
    exit_time: float | None = None
    briefed: bool = False


@dataclass
class PitSequence:
    """Box -> confirm -> (stay out) -> in pit -> out-lap brief. One per session."""

    speak: Callable[..., Any]
    track_length: float = 0.0
    racing: bool = False
    state: str = IDLE
    call: PitCall | None = None
    stop: StopRecord | None = None
    entry_samples: list[float] = field(default_factory=list)
    stored_entry_m: float | None = None
    last_event: dict[str, Any] | None = None
    _prev_pit: dict[int, int] = field(default_factory=dict)
    _player_pit: int | None = None
    _last_lap_num: int | None = None
    _unsaved_samples: int = 0

    # ── session ──

    def reset(self, track_length: float, racing: bool, stored_entry_m: float | None = None) -> None:
        self.track_length = float(track_length or 0.0)
        self.racing = racing
        self.state, self.call, self.stop = IDLE, None, None
        self.entry_samples = []
        self.stored_entry_m = stored_entry_m
        self.last_event = None
        self._prev_pit = {}
        self._player_pit = None
        self._last_lap_num = None
        self._unsaved_samples = 0

    # ── pit entry ──

    @property
    def entry_m(self) -> float | None:
        """Lap distance of pit entry: the field's median entry, the stored one, or a default."""
        if self.track_length <= 0:
            return None
        if len(self.entry_samples) >= MIN_ENTRY_SAMPLES:
            return statistics.median(self.entry_samples)
        if self.stored_entry_m is not None:
            return self.stored_entry_m
        return max(0.0, self.track_length - DEFAULT_ENTRY_BEFORE_LINE_M)

    @property
    def entry_source(self) -> str:
        if self.entry_samples:
            return 'observed'
        return 'stored' if self.stored_entry_m is not None else 'default'

    def to_entry_m(self, lap_distance: float) -> float | None:
        """Metres from here to pit entry, going forwards round the lap."""
        entry = self.entry_m
        if entry is None or not math.isfinite(lap_distance):
            return None
        return (entry - lap_distance) % self.track_length

    def take_entry_to_save(self) -> float | None:
        """The learned entry once it is worth saving to the track store (then forgets it's new)."""
        if self._unsaved_samples < SAVE_AFTER_SAMPLES or not self.entry_samples:
            return None
        self._unsaved_samples = 0
        return round(statistics.median(self.entry_samples), 1)

    def _learn_entries(self, laps: list[Any]) -> None:
        for index, lap in enumerate(laps):
            if lap.result_status != ACTIVE:
                self._prev_pit.pop(index, None)
                continue
            previous = self._prev_pit.get(index)
            self._prev_pit[index] = lap.pit_status
            if previous != 0 or not lap.pit_status:
                continue  # only a car seen on track that now turns in (not one starting in the pits)
            distance = lap.lap_distance
            if not math.isfinite(distance) or self.track_length <= 0:
                continue
            distance %= self.track_length
            self.entry_samples.append(distance)
            if len(self.entry_samples) > MAX_ENTRY_SAMPLES:
                self.entry_samples.pop(0)
            self._unsaved_samples += 1

    # ── step 1: the box call ──

    @property
    def pending(self) -> bool:
        """A box call is live and the car hasn't turned in yet."""
        return self.state == BOXING and self.call is not None

    def box_called(self, reason: str, lap: Any, car_index: int | None = None) -> None:
        """A box-this-lap call has gone out: arm the confirm, the abort and the brief."""
        if not self.racing or lap is None or lap.pit_status:
            return
        remaining = self.to_entry_m(lap.lap_distance)
        if remaining is None:
            return
        if self.pending and self.call is not None and self.call.lap == lap.current_lap_num:
            # Already boxing this lap: keep the first reason unless this one can't be taken back.
            if reason not in ABORTABLE_REASONS:
                self.call.reason = reason
                self.call.car_index = car_index
            return
        self.state = BOXING
        self.call = PitCall(
            reason,
            lap.current_lap_num,
            car_index,
            called_remaining_m=remaining,
            wraps_allowed=1 if remaining < LATE_CALL_M else 0,
            prev_remaining_m=remaining,
        )
        self._event('box', reason=reason)

    # ── step 3: stay out ──

    def can_abort(self, lap: Any) -> bool:
        if not self.pending or self.call is None or self.call.reason not in ABORTABLE_REASONS:
            return False
        if lap is None or lap.pit_status:
            return False
        remaining = self.to_entry_m(lap.lap_distance)
        return remaining is not None and (remaining >= ABORT_MIN_M or self.call.wraps_allowed > 0)

    def cancel(self, why: str) -> None:
        """Drop a pending box call that another call has already taken back on air."""
        if self.pending:
            self.state = ABORTED
            self._event('stay_out', why=why)

    def abort(self, message: str, why: str) -> None:
        """Take the box call back: "Stay out, stay out." plus why."""
        self.speak(f'Stay out, stay out. {message}', Priority.CRITICAL, key='pit_abort', ttl=5, cooldown=0)
        self.state = ABORTED
        self._event('stay_out', why=why)

    # ── every frame ──

    def update(
        self,
        lap: Any,
        laps: list[Any] | None,
        player_index: int,
        session_time: float,
        neighbours: Callable[[], list[Neighbour]] | None = None,
    ) -> None:
        if laps is not None:
            self._learn_entries(laps)
        if not self.racing or lap is None:
            return
        if self._last_lap_num is not None and lap.current_lap_num < self._last_lap_num:
            # Flashback to an earlier lap: whatever was called no longer stands.
            self.state, self.call, self.stop = IDLE, None, None
        self._last_lap_num = lap.current_lap_num
        was_in_pit, in_pit = self._player_pit, lap.pit_status
        self._player_pit = in_pit
        if in_pit and not was_in_pit:
            self._entered_pits(lap, was_in_pit is None)
        elif was_in_pit and not in_pit:
            self._left_pits(lap, session_time)
        if self.pending:
            self._watch_entry(lap)
        if self.state == OUT_LAP and neighbours is not None:
            self._maybe_brief(lap, session_time, neighbours)

    def _entered_pits(self, lap: Any, first_frame: bool) -> None:
        if first_frame:
            return  # starting from the pit lane: not a stop we called or need to brief
        self.state, self.call = IN_PIT, None
        self.stop = StopRecord(lap.car_position, lap.num_pit_stops)
        self._event('pit_entry')

    def _left_pits(self, lap: Any, session_time: float) -> None:
        stop = self.stop
        if stop is None or lap.num_pit_stops <= stop.entry_stops:
            # A drive-through, or out of a pit-lane start: nothing changed on the car.
            self.state, self.stop = IDLE, None
            return
        stop.exit_time = session_time
        self.state = OUT_LAP
        self._event('pit_exit')

    # ── step 2: confirm before the entry ──

    def _watch_entry(self, lap: Any) -> None:
        call = self.call
        assert call is not None
        remaining = self.to_entry_m(lap.lap_distance)
        if remaining is None:
            return
        previous, call.prev_remaining_m = call.prev_remaining_m, remaining
        wrapped = (
            previous is not None and previous < self.track_length / 4 and remaining > self.track_length / 2
        )
        if wrapped:
            if call.wraps_allowed > 0:
                call.wraps_allowed -= 1
                call.called_remaining_m = self.track_length  # the next entry is a full lap away
                return
            # Past the entry line without turning in: the driver stayed out.
            self.state, self.call = IDLE, None
            self._event('missed')
            return
        if (
            not call.confirmed
            and call.wraps_allowed == 0
            and remaining <= CONFIRM_LEAD_M
            and call.called_remaining_m > CONFIRM_SKIP_M
        ):
            call.confirmed = True
            self.speak(
                'Box, box. Pit entry coming up, confirm box.',
                Priority.CRITICAL,
                key='pit_confirm',
                ttl=4,
                cooldown=0,
            )
            self._event('confirm')

    # ── step 4: the out-lap brief ──

    def _maybe_brief(self, lap: Any, session_time: float, neighbours: Callable[[], list[Neighbour]]) -> None:
        stop = self.stop
        if stop is None or stop.exit_time is None or stop.briefed:
            return
        waited = session_time - stop.exit_time
        if waited < BRIEF_DELAY_S:
            return
        around = neighbours()
        if waited < BRIEF_TIMEOUT_S and any(n.gap_s is None for n in around):
            return  # the gap either side settles once the cars around pass the pit exit
        stop.briefed = True
        self.state = IDLE
        message = out_lap_brief(stop.entry_position, lap.car_position, around, lap.num_pit_stops)
        self.speak(message, Priority.RACE, key='pit_brief', ttl=10, cooldown=0)
        self._event('brief', message=message)

    def _event(self, kind: str, **details: Any) -> None:
        self.last_event = {'kind': kind, **details}

    def summary(self) -> dict[str, Any]:
        """For the HUD and the AI prompt: where the stop is and where pit entry is."""
        call = self.call
        return {
            'state': self.state,
            'reason': call.reason if call else None,
            'lap': call.lap if call else None,
            'confirmed': bool(call and call.confirmed),
            'entry_m': None if self.entry_m is None else round(self.entry_m, 1),
            'entry_source': self.entry_source,
            'last_event': self.last_event,
        }


def _car_line(n: Neighbour, my_stops: int) -> str:
    who = n.name or 'Car'
    text = f'{who} {n.side}'
    if n.gap_s is not None:
        text += f', {say.gap_value(n.gap_s)}'
    if n.compound:
        if n.tyre_age is not None and n.tyre_age <= FRESH_TYRE_LAPS:
            text += f', on fresh {n.compound}'
        elif n.tyre_age is not None:
            text += f', on {say.number_words(n.tyre_age)}-lap-old {n.compound}'
        else:
            text += f', on {n.compound}'
    if n.stops == 0 and my_stops > 0:
        text += ', yet to stop'
    return text + '.'


def out_lap_brief(entry_position: int, position: int, around: list[Neighbour], my_stops: int) -> str:
    """ "Lost one in the stop. Hamilton ahead, two point one, on twenty-lap-old hards..." """
    parts = []
    if entry_position and position:
        change = entry_position - position
        if change:
            places = 'place' if abs(change) == 1 else 'places'
            verb = 'Gained' if change > 0 else 'Lost'
            parts.append(f'{verb} {say.number_words(abs(change))} {places} in the stop, P{position}.')
    for n in sorted(around, key=lambda n: n.side != 'ahead'):
        parts.append(_car_line(n, my_stops))
    behind = next((n for n in around if n.side == 'behind'), None)
    ahead = next((n for n in around if n.side == 'ahead'), None)
    if behind is not None and behind.gap_s is not None and abs(behind.gap_s) < CLOSE_BEHIND_S:
        parts.append(f'Get the tyres in, but keep {behind.name or "the car behind"} behind.')
    elif (
        ahead is not None
        and ahead.tyre_age is not None
        and ahead.tyre_age >= TYRE_EDGE_LAPS
        and ahead.stops == 0
    ):
        parts.append(f'Tyres in, then go and get {ahead.name or "the car ahead"}.')
    elif ahead is not None and ahead.tyre_age is not None and ahead.tyre_age >= TYRE_EDGE_LAPS:
        parts.append('Tyres in, then push. You have the tyre advantage.')
    else:
        parts.append('Bring the tyres in, then push.')
    return ' '.join(parts)
