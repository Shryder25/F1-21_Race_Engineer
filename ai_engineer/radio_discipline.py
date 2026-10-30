"""Radio discipline: how much the engineer talks, and when.

Real race engineers keep the radio short and never talk while the driver is through a
fast corner. Three pieces enforce that:

  * ``CallBudget``  at most N proactive callouts per lap (default 4 in practice and
                    qualifying, 2 in a race). Urgent calls (Safety Car, flags, penalties,
                    damage, box calls, pit window, strategy calls, a car attacking with DRS)
                    always get through and don't use the budget.
  * ``race_quiet_reason`` the moments in a race when only urgent calls go out: lap 1, a DRS
                    fight (a car within a second either side) and the last two laps.
  * ``DriverLoadGate`` says when the driver is in a high-load corner (fast and high lateral
                    G) or braking hard. The voice queue holds everything except critical
                    calls until the car is back on a straight, then delivers them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from voice_engine import Priority

DEFAULT_CALLS_PER_LAP = 4
DEFAULT_RACE_CALLS_PER_LAP = 2  # in a race the driver is fighting: half as many proactive calls

# Race quiet windows: only urgent calls on these laps / in a fight.
QUIET_FINAL_LAPS = 2
DRS_FIGHT_S = 1.0
# Calls that describe the fight itself, so the DRS-fight window must not swallow them.
FIGHT_TOPICS = frozenset({'drs_ahead', 'drs_behind'})

# Topics that are never budgeted, whatever their priority. CRITICAL and STRATEGY priority
# calls (box box, pit window, undercut, Plan B, a rival's stop) are always urgent too.
URGENT_TOPICS = frozenset(
    {
        'safety_car',
        'safety_car_in',
        'flag_red',
        'flag_yellow',
        'penalty',
        'critical_wear',
        'wing_damage',
        'weather_rain',
        'tyre_temperature',
        'pit_window',
        'undercut',
        'rival_pit',
        'rival_preempt',
        'plan_b',
        'drs_behind',
        'own_pit_stop',
        'pit_confirm',
        'pit_abort',
        'pit_brief',
        'lights_out',
        'chequered',
        'race_winner',
        'race_finish',
        'lap_deleted',
    }
)


def is_urgent(topic: str | None, priority) -> bool:
    return Priority(priority) <= Priority.STRATEGY or (topic or '') in URGENT_TOPICS


def race_quiet_reason(
    lap_num: int | None,
    total_laps: int | None,
    gap_ahead_s: float | None,
    gap_behind_s: float | None,
    topic: str | None = None,
) -> str | None:
    """Why only urgent calls should go out right now in a race, or None.

    Lap 1 is chaos, a car within a second either side is a fight the driver is concentrating
    on, and the last two laps are about bringing it home. Pace, gaps and briefings wait.
    """
    if lap_num is not None and 0 < lap_num <= 1:
        return 'lap_1'
    if lap_num and total_laps and lap_num > total_laps - QUIET_FINAL_LAPS:
        return 'final_laps'
    if (topic or '') not in FIGHT_TOPICS and any(
        gap is not None and gap < DRS_FIGHT_S for gap in (gap_ahead_s, gap_behind_s)
    ):
        return 'drs_fight'
    return None


class CallBudget:
    """Counts proactive callouts per lap. ``per_lap`` 0 or None means no limit.

    ``race_per_lap`` is the tighter limit used in a race (``set_race``); it never raises
    the base limit, and a base of 0 (no limit) stays unlimited in every session.
    """

    def __init__(
        self,
        per_lap: int | None = DEFAULT_CALLS_PER_LAP,
        race_per_lap: int | None = DEFAULT_RACE_CALLS_PER_LAP,
    ):
        self.base_per_lap = per_lap if per_lap and per_lap > 0 else None
        self.race_per_lap = race_per_lap if race_per_lap and race_per_lap > 0 else None
        self.per_lap = self.base_per_lap
        self.lap = None
        self.used = 0
        self.skipped = 0  # over-budget calls dropped this lap (HUD/debug)

    def set_race(self, race: bool) -> None:
        """Use the race limit in a race, the base limit in every other session."""
        if race and self.base_per_lap is not None and self.race_per_lap is not None:
            self.per_lap = min(self.base_per_lap, self.race_per_lap)
        else:
            self.per_lap = self.base_per_lap

    def reset(self) -> None:
        self.lap, self.used, self.skipped = None, 0, 0

    def new_lap(self, lap_num: int) -> None:
        if lap_num != self.lap:
            self.lap, self.used, self.skipped = lap_num, 0, 0

    def allows(self, topic: str | None, priority) -> bool:
        """True if this call may be queued now. Doesn't count it: call ``spend`` once queued."""
        if self.per_lap is None or is_urgent(topic, priority):
            return True
        if self.used >= self.per_lap:
            self.skipped += 1
            return False
        return True

    def spend(self, topic: str | None, priority):
        if self.per_lap is not None and not is_urgent(topic, priority):
            self.used += 1

    @property
    def remaining(self):
        return None if self.per_lap is None else max(0, self.per_lap - self.used)


@dataclass
class LoadThresholds:
    corner_speed_kph: float = 200.0  # a high-load corner is at least this fast...
    corner_g: float = 3.5  # ...with at least this much lateral G
    straight_g: float = 1.5  # back on a straight once lateral G stays below this...
    straight_s: float = 0.4  # ...for this long (so an S-bend counts as one corner)
    brake: float = 0.5  # brake pedal (0-1) counted as a heavy braking zone
    stale_s: float = 1.0  # no telemetry for this long: don't hold the radio


class DriverLoadGate:
    """Tracks whether the driver is busy: high-load corner or heavy braking.

    ``update`` runs on the telemetry thread with each sample; ``busy`` is read by the voice
    thread. Once a high-load corner starts, the driver stays busy until lateral G has been
    below ``straight_g`` for ``straight_s`` seconds, so queued calls land on the straight.
    """

    def __init__(self, thresholds: LoadThresholds | None = None, clock=time.monotonic):
        self.t = thresholds or LoadThresholds()
        self.clock = clock
        self.in_corner = False
        self.braking = False
        self._calm_since = None
        self._last_update = None

    def reset(self) -> None:
        self.in_corner = self.braking = False
        self._calm_since = self._last_update = None

    def update(
        self,
        speed_kph: float | None,
        lateral_g: float | None,
        brake: float | None = None,
        now: float | None = None,
    ):
        now = self.clock() if now is None else now
        self._last_update = now
        g = abs(lateral_g or 0.0)
        speed = speed_kph or 0.0
        self.braking = (brake or 0.0) >= self.t.brake
        if speed >= self.t.corner_speed_kph and g >= self.t.corner_g:
            self.in_corner = True
            self._calm_since = None
        elif self.in_corner:
            if g < self.t.straight_g:
                if self._calm_since is None:
                    self._calm_since = now
                elif now - self._calm_since >= self.t.straight_s:
                    self.in_corner = False
                    self._calm_since = None
            else:
                self._calm_since = None

    def busy(self, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        if self._last_update is None or now - self._last_update > self.t.stale_s:
            return False
        return self.in_corner or self.braking
