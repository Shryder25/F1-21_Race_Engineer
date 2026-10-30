import math
import time
from collections import deque
from collections.abc import Callable
from typing import Any

import radio_phrases as say
from physics_models import evaluate_opponent_threat
from voice_engine import Priority

# How many cars ahead/behind of the player get a threat score each frame (beyond immediate
# neighbours, Box-Box scores the wider nearby field rather than only P+1/P-1).
THREAT_WINDOW_POSITIONS = 3

# Gap calls: one update per lap at the line, with the trend per lap, plus an instant call
# when a car gets within DRS range (with hysteresis so a gap hovering at 1.0s doesn't spam).
DRS_RANGE_S = 1.0
DRS_REARM_S = 1.3
GAP_TREND_MIN_S = 0.2
RACE_PACE_EVERY_LAPS = 3


class RaceRadio:
    """Authentic F1 race radio protocol with silence enforcement, concise calls, and priority interrupts."""

    def __init__(self, speak: Callable[..., Any], on_box: Callable[..., Any] | None = None) -> None:
        self.speak = speak
        # Told when a box-this-lap call goes on air, so the pit sequence can confirm it
        # before pit entry (pit_sequence.py). Gets the reason: 'window', 'wear', 'damage', or
        # 'safety_car' after a Safety Car / VSC decision (box or stay out) went on air.
        self.on_box = on_box
        self.phrases = say.Phrasebook()
        # Car index -> short tyre note ("on eighteen-lap-old mediums"), set by the engine,
        # which knows every car's status. DRS calls add it as information for the driver.
        self.tyre_note: Callable[[int], str | None] | None = None
        # Car index -> radio name from the Participants packet. Kept across reset() because a
        # flashback rewinds the timeline but not who is in the field; cleared on session change.
        self.driver_names: dict[int, str] = {}
        self.reset(None)

    def set_driver_names(self, names: dict[int, str]):
        self.driver_names = dict(names)

    def driver_name(self, index) -> str | None:
        return None if index is None else self.driver_names.get(index)

    def reset(self, session: Any) -> None:
        self.session = session
        self.trajectories = {}
        self.last_session_time = None
        self.gap_ahead_s = self.gap_behind_s = None
        self.car_ahead_index = self.car_behind_index = None
        self.pit_notices = set()
        self.pit_window = None
        self.last_pit_stops = None
        self.served_window = None
        self.sectors_reported = set()
        self.race_laps = deque(maxlen=3)
        self.race_pace_ms = None
        self.last_safety_car = session.safety_car_status if session else 0
        self.last_weather = session.weather if session else 0
        self.last_critical_alert = 0.0
        self.last_lap_times_ms: dict[int, int] = {}  # car index -> most recent completed lap time
        self.opponent_threats: list = []
        self.reported_gaps = (None, None)  # (ahead, behind) at the last lap-end gap call
        self.reported_gap_cars = (None, None)  # (ahead, behind) car index at that call
        self.drs_armed = {'ahead': True, 'behind': True}
        self.clean_race_laps = 0
        self.reported_race_pace_ms = None
        self.rival_gaps: dict[int, tuple[float, float]] = {}  # car index -> (signed gap s, session time)
        self.rival_in_pits: dict[int, bool] = {}
        self.rival_pit_events: list[dict] = []
        # Car index -> lap numbers on which it was seen in the pit lane. A lap on or right
        # after one of those is an in-lap or out-lap and says nothing about the car's pace.
        self.pit_laps: dict[int, set[int]] = {}

    def update_strategy(self, lap):
        """Monitor pit stop windows with concise radio timing."""
        if not self.session or self.session.session_type not in (10, 11, 12):
            return
        ideal = self.session.pit_stop_window_ideal_lap
        latest = self.session.pit_stop_window_latest_lap
        window = (ideal, latest)
        if self.last_pit_stops is not None and lap.num_pit_stops > self.last_pit_stops:
            self.served_window = window
        self.last_pit_stops = lap.num_pit_stops
        self.pit_window = window if ideal > 0 and latest >= ideal else None
        if not self.pit_window or lap.pit_status or window == self.served_window:
            return
        if lap.current_lap_num == latest:
            phase, message = 'last', 'Box, box. Box this lap, last lap of the window.'
        elif ideal <= lap.current_lap_num < latest:
            phase, message = (
                'open',
                (
                    f'Pit window is open. Target lap {say.number_words(ideal)}, '
                    f'no later than {say.number_words(latest)}.'
                ),
            )
        elif lap.current_lap_num == ideal - 1:
            phase, message = (
                'next',
                (f'Pit window opens next lap. Plan is to box lap {say.number_words(ideal)}.'),
            )
        else:
            return
        notice = (window, lap.num_pit_stops, phase)
        if notice not in self.pit_notices:
            self.pit_notices.add(notice)
            spoken = self.speak(message, Priority.STRATEGY, key='pit_window', ttl=8, cooldown=0)
            if phase == 'last' and spoken:
                self._box('window')

    def _box(self, reason: str) -> None:
        if self.on_box is not None:
            self.on_box(reason)

    # ──────────────────────────────────────────
    # Priority Interrupts (Critical Events)
    # ──────────────────────────────────────────

    def update_safety_car(
        self, safety_car_status: int, decide: Callable[[int], str | None] | None = None
    ) -> None:
        """Immediate priority interrupt when Safety Car or VSC is deployed/withdrawn.

        ``decide(status)`` gives the box / stay-out decision for a deployment ("Box, box, cheap
        stop..." or "Stay out..."). A real engineer says it in the same breath as the Safety
        Car call: the pit entry may be seconds away, there is no time for a follow-up.
        """
        if safety_car_status == self.last_safety_car:
            return
        prev = self.last_safety_car
        self.last_safety_car = safety_car_status
        decision = decide(safety_car_status) if decide and safety_car_status in (1, 2) else None
        if safety_car_status == 1:
            spoken = self.speak(
                'Safety Car, Safety Car. ' + (decision or 'Keep the delta positive, no overtaking.'),
                Priority.CRITICAL,
                key='safety_car',
                ttl=10,
                cooldown=10,
            )
            if decision and spoken:
                self._box('safety_car')
        elif safety_car_status == 2:
            spoken = self.speak(
                'VSC, VSC. ' + (decision or 'Virtual Safety Car deployed, keep the delta positive.'),
                Priority.CRITICAL,
                key='safety_car',
                ttl=8,
                cooldown=10,
            )
            if decision and spoken:
                self._box('safety_car')
        elif prev == 2 and safety_car_status == 0:
            self.speak(
                "VSC ending, VSC ending. Green flag, push now.",
                Priority.CRITICAL,
                key='safety_car_in',
                ttl=8,
                cooldown=10,
            )
        elif prev == 1 and safety_car_status == 0:
            self.speak(
                "Safety Car in this lap. Get the tyres warm and be ready for the restart.",
                Priority.CRITICAL,
                key='safety_car_in',
                ttl=8,
                cooldown=10,
            )

    def update_weather_alert(self, weather_code: int):
        """Immediate priority interrupt on sudden rain arrival."""
        if self.last_weather is None:
            self.last_weather = weather_code
            return
        prev = self.last_weather
        self.last_weather = weather_code
        # 0=clear, 1=light cloud, 2=overcast, 3=light rain, 4=heavy rain, 5=storm
        if prev <= 2 and weather_code >= 3:
            self.speak(
                "Rain starting to fall. Grip will drop, stay off the kerbs and white lines.",
                Priority.CRITICAL,
                key='weather_rain',
                ttl=10,
                cooldown=30,
            )

    def update_damage_alert(self, worst_tyre_wear: float, front_wing_damage: int, now=None):
        """Priority interrupt on severe tyre wear or front wing damage."""
        now = time.monotonic() if now is None else now
        if now - self.last_critical_alert < 25.0:
            return
        if worst_tyre_wear >= 75.0:
            self.last_critical_alert = now
            if self.speak(
                f"Box, box. Box this lap, tyres are finished. "
                f"{say.number_words(int(worst_tyre_wear))} percent wear.",
                Priority.CRITICAL,
                key='critical_wear',
                ttl=8,
                cooldown=20,
            ):
                self._box('wear')
        elif front_wing_damage >= 50:
            self.last_critical_alert = now
            if self.speak(
                "We can see front wing damage. Box this lap for a new nose. Box, box.",
                Priority.CRITICAL,
                key='wing_damage',
                ttl=8,
                cooldown=25,
            ):
                self._box('damage')

    @staticmethod
    def _crossing_time(history, distance):
        if not history or not history[0][0] <= distance <= history[-1][0]:
            return None
        previous = None
        for point in reversed(history):
            if point[0] == distance:
                return point[1]
            if point[0] < distance and previous is not None:
                gap = previous[0] - point[0]
                if gap > 150 or previous[1] - point[1] > 2:
                    return None
                fraction = (distance - point[0]) / gap
                return point[1] + fraction * (previous[1] - point[1])
            previous = point
        return None

    def update_field(self, laps, player_index, session_time, now=None):
        now = time.monotonic() if now is None else now
        if not self.session or self.session.session_type not in (10, 11, 12):
            return
        if self.last_session_time is not None and session_time < self.last_session_time:
            self.trajectories.clear()
        self.last_session_time = session_time
        active = {}
        for index, lap in enumerate(laps):
            if (
                lap.result_status != 2
                or lap.car_position == 0
                or not math.isfinite(lap.total_distance)
                or lap.pit_status
            ):
                self.trajectories.pop(index, None)
                continue
            active[index] = lap
            history = self.trajectories.setdefault(index, deque(maxlen=6000))
            if history and (lap.total_distance < history[-1][0] or session_time - history[-1][1] > 2):
                history.clear()
            if not history or (lap.total_distance > history[-1][0] and session_time > history[-1][1]):
                history.append((lap.total_distance, session_time))
        for index, lap in enumerate(laps):
            if lap.pit_status:
                self.pit_laps.setdefault(index, set()).add(lap.current_lap_num)
        for index, lap in active.items():
            if lap.last_lap_time_ms and not self.near_pit_stop(index, lap.current_lap_num - 1):
                self.last_lap_times_ms[index] = lap.last_lap_time_ms
        self._track_rival_stops(laps, player_index, session_time)
        self.gap_ahead_s = self.gap_behind_s = None
        self.car_ahead_index = self.car_behind_index = None
        player = active.get(player_index)
        if player is None:
            self.opponent_threats = []
            return
        threats = []
        for index, lap in active.items():
            if index == player_index:
                continue
            offset = lap.car_position - player.car_position
            if abs(offset) > THREAT_WINDOW_POSITIONS:
                continue
            # Positive gap = ahead of the player (offset < 0, i.e. a better position number);
            # crossing time is read from whichever car's trajectory the other car's distance
            # falls within, matching the existing immediate-neighbour calculation below.
            if offset < 0:
                crossing = self._crossing_time(self.trajectories[index], player.total_distance)
                gap = None if crossing is None else max(0.0, session_time - crossing)
            else:
                crossing = self._crossing_time(self.trajectories[player_index], lap.total_distance)
                gap = None if crossing is None else -max(0.0, session_time - crossing)
            if gap is None:
                continue
            self.rival_gaps[index] = (gap, session_time)
            if offset == -1:
                self.gap_ahead_s, self.car_ahead_index = gap, index
            elif offset == 1:
                self.gap_behind_s, self.car_behind_index = -gap, index
            pace_delta = None
            player_best = self.last_lap_times_ms.get(player_index)
            opponent_best = self.last_lap_times_ms.get(index)
            if player_best and opponent_best:
                pace_delta = (opponent_best - player_best) / 1000.0
            threats.append(
                evaluate_opponent_threat(index, lap.car_position, gap, pace_delta, self.driver_name(index))
            )
        threats.sort(key=lambda t: t.threat_score, reverse=True)
        self.opponent_threats = threats[:5]
        self._call_drs_range(now)

    def near_pit_stop(self, index: int, lap_num: int) -> bool:
        """True if ``lap_num`` was an in-lap or out-lap for this car (within a lap of a stop)."""
        stops = self.pit_laps.get(index)
        return bool(stops) and (lap_num in stops or lap_num - 1 in stops)

    def _call_drs_range(self, now):
        """Instant call when a car comes within a second, either side.

        Information, not a driving instruction: the gap, who it is and what tyres they are on.
        The driver knows what to do with DRS.
        """
        for side, gap in (('ahead', self.gap_ahead_s), ('behind', self.gap_behind_s)):
            if gap is None:
                continue
            if gap >= DRS_REARM_S:
                self.drs_armed[side] = True
            elif gap < DRS_RANGE_S and self.drs_armed[side]:
                self.drs_armed[side] = False
                amount = say.gap_value(gap)
                index = self.car_ahead_index if side == 'ahead' else self.car_behind_index
                name = self.driver_name(index)
                tyres = self.tyre_note(index) if self.tyre_note and index is not None else None
                if side == 'ahead':
                    who = name or 'the car ahead'
                    message = self.phrases.pick(
                        'drs_ahead',
                        (
                            f'DRS available, {amount} to {who}.',
                            f'{amount.capitalize()} to {who}, you are in DRS range.',
                        ),
                    )
                else:
                    who = name or 'Car behind'
                    message = self.phrases.pick(
                        'drs_behind',
                        (
                            f'{who} is within DRS, {amount}.',
                            f'{who} has DRS, {amount} back.',
                        ),
                    )
                if tyres:
                    message += f' They are {tyres}.'
                self.speak(message, Priority.RACE, key=f'drs_{side}', ttl=4, cooldown=30)

    def call_lap_gaps(self) -> None:
        """Once a lap at the line: the gap either side and how it moved over the lap.

        "Gap to Hamilton ahead two point four, you're catching, three tenths that lap."
        The trend only counts when it is the same car as at the last call.
        """
        last_ahead, last_behind = self.reported_gaps
        car_ahead, car_behind = self.reported_gap_cars
        messages = []
        if self.gap_ahead_s is not None:
            name = self.driver_name(self.car_ahead_index)
            who = f'Gap to {name} ahead' if name else 'Gap ahead'
            text = f'{who} {say.gap_value(self.gap_ahead_s)}'
            if last_ahead is not None and car_ahead == self.car_ahead_index:
                change = last_ahead - self.gap_ahead_s
                if change >= GAP_TREND_MIN_S:
                    text += f", you're catching, {say.time_amount(change)} that lap"
                elif -change >= GAP_TREND_MIN_S:
                    text += f", they're pulling away, {say.time_amount(change)} that lap"
            messages.append(text + '.')
        if self.gap_behind_s is not None:
            name = self.driver_name(self.car_behind_index)
            who = f'{name} behind' if name else 'Car behind'
            text = f'{who} {say.gap_value(self.gap_behind_s)}'
            if last_behind is not None and car_behind == self.car_behind_index:
                change = last_behind - self.gap_behind_s
                if change >= GAP_TREND_MIN_S:
                    text += f", they're closing, {say.time_amount(change)} that lap"
                elif -change >= GAP_TREND_MIN_S:
                    text += f", you're pulling away, {say.time_amount(change)} that lap"
            messages.append(text + '.')
        self.reported_gaps = (self.gap_ahead_s, self.gap_behind_s)
        self.reported_gap_cars = (self.car_ahead_index, self.car_behind_index)
        if messages:
            self.speak(' '.join(messages), Priority.RACE, key='race_gaps', ttl=8, cooldown=0)

    def _track_rival_stops(self, laps, player_index, session_time):
        """Queue an event when a nearby rival turns into the pit lane.

        The gap is the last one measured before the car entered the pit lane (cars in the pit
        lane are dropped from the gap calculation), so it is only trusted if it is fresh.
        """
        for index, lap in enumerate(laps):
            if index == player_index or lap.result_status != 2 or lap.car_position == 0:
                self.rival_in_pits.pop(index, None)
                continue
            in_pits = lap.pit_status in (1, 2)
            was_in_pits = self.rival_in_pits.get(index)
            self.rival_in_pits[index] = in_pits
            if was_in_pits is None or was_in_pits or not in_pits:
                continue
            measured = self.rival_gaps.get(index)
            if measured is None or session_time - measured[1] > 2.0:
                continue
            self.rival_pit_events.append(
                {
                    'car_index': index,
                    'gap_s': measured[0],
                    'lap': lap.current_lap_num,
                    'position': lap.car_position,
                    'session_time': session_time,
                }
            )

    def pop_rival_pit_events(self) -> list[dict]:
        events, self.rival_pit_events = self.rival_pit_events, []
        return events

    def report_sector(self, lap_num: int, sector: int, milliseconds: int, personal_ms: int | None) -> None:
        token = (lap_num, sector)
        if token in self.sectors_reported or milliseconds <= 0:
            return
        self.sectors_reported.add(token)
        if self.session and self.session.session_type in (10, 11, 12):
            return  # sector deltas to your best are qualifying coaching, not race radio
        name = say.sector_name(sector)
        if not personal_ms:
            message = f'{name.capitalize()}, {say.sector_time(milliseconds)}.'
        else:
            difference = (milliseconds - personal_ms) / 1000
            amount = say.time_amount(difference)
            if abs(difference) < 0.05:
                message = f'{name.capitalize()}, matching your best.'
            elif difference < 0:
                message = self.phrases.pick(
                    'sector_up',
                    (
                        f'Green {name}, {amount} up.',
                        f'{name.capitalize()}, {amount} quicker. Good.',
                    ),
                )
            else:
                message = f'{name.capitalize()}, {amount} down.'
        self.speak(message, Priority.TIMING, key=f'sector_{sector}', ttl=8, cooldown=0)

    def completed_lap(self, milliseconds: int, lap_num: int, sectors: dict, personal_sectors: dict) -> None:
        if 3 in sectors:
            self.report_sector(lap_num, 3, sectors[3], personal_sectors.get(3))
        if not self.session or self.session.session_type not in (10, 11, 12):
            return
        if lap_num <= 1:
            return  # the standing start makes lap 1 seconds slow: not race pace
        self.race_laps.append(milliseconds)
        self.clean_race_laps += 1
        if len(self.race_laps) == 3:
            self.race_pace_ms = round(sum(self.race_laps) / 3)
            # Real engineers give a pace summary every few laps, not after every lap.
            if self.clean_race_laps % RACE_PACE_EVERY_LAPS:
                return
            previous_pace, self.reported_race_pace_ms = self.reported_race_pace_ms, self.race_pace_ms
            pace = say.lap_time(self.race_pace_ms)
            message = f'Last three laps averaging {pace}.'
            if previous_pace:
                change = (self.race_pace_ms - previous_pace) / 1000
                if change <= -0.1:
                    message += f' {say.time_amount(change).capitalize()} quicker than before, good pace.'
                elif change >= 0.3:
                    message += f' Dropping off, {say.time_amount(change)} slower. Look after the tyres.'
            comparison = self.pace_comparison(milliseconds)
            if comparison:
                message += ' ' + comparison
            self.speak(message, Priority.RACE, key='race_pace', ttl=12, cooldown=0)

    def pace_comparison(self, player_lap_ms) -> str:
        """Compare the player's last lap with the cars directly ahead and behind, by name."""
        parts = []
        for side, index in (('ahead', self.car_ahead_index), ('behind', self.car_behind_index)):
            other = self.last_lap_times_ms.get(index) if index is not None else None
            if not other or not player_lap_ms:
                continue
            who = self.driver_name(index) or 'Car'
            delta = (other - player_lap_ms) / 1000.0
            if abs(delta) < 0.05:
                parts.append(f'{who} {side} on the same pace.')
            else:
                parts.append(f"{who} {side} {say.time_amount(delta)} {'slower' if delta > 0 else 'quicker'}.")
        return ' '.join(parts)
