import math
import time
from collections import deque

from physics_models import evaluate_opponent_threat
import radio_phrases as say
from voice_engine import Priority

# How many cars ahead/behind of the player get a threat score each frame (beyond immediate
# neighbours, Box-Box scores the wider nearby field rather than only P+1/P-1).
THREAT_WINDOW_POSITIONS = 3

# Gap calls: a routine update roughly once a lap, plus an instant call when a car
# gets within DRS range (with hysteresis so a gap hovering at 1.0s doesn't spam).
GAP_REPORT_INTERVAL_S = 45
DRS_RANGE_S = 1.0
DRS_REARM_S = 1.3
GAP_TREND_MIN_S = 0.2
RACE_PACE_EVERY_LAPS = 3


class RaceRadio:
    """Authentic F1 race radio protocol with silence enforcement, concise calls, and priority interrupts."""

    def __init__(self, speak):
        self.speak = speak
        self.phrases = say.Phrasebook()
        self.reset(None)

    def set_driver_names(self, names: dict[int, str]):
        self.driver_names = dict(names)

    def driver_name(self, index) -> str | None:
        return None if index is None else self.driver_names.get(index)

    def _gap_call(self, side: str, gap: float, index) -> str:
        name = self.driver_name(index)
        return f'Gap to {name} {side} is {gap:.1f}.' if name else f'Gap {side} is {gap:.1f}.'

    def reset(self, session):
        self.session = session
        self.trajectories = {}
        self.last_session_time = None
        self.last_gap_report = -math.inf
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
        self.last_lap_times_ms: dict[int, int] = {}    # car index -> most recent completed lap time
        self.opponent_threats: list = []
        self.reported_gaps = (None, None)       # (ahead, behind) at the last routine gap call
        self.drs_armed = {'ahead': True, 'behind': True}
        self.clean_race_laps = 0
        self.reported_race_pace_ms = None

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
            phase, message = 'open', (f'Pit window is open. Target lap {say.number_words(ideal)}, '
                                      f'no later than {say.number_words(latest)}.')
        elif lap.current_lap_num == ideal - 1:
            phase, message = 'next', (f'Pit window opens next lap. Plan is to box '
                                      f'lap {say.number_words(ideal)}.')
        else:
            return
        notice = (window, lap.num_pit_stops, phase)
        if notice not in self.pit_notices:
            self.pit_notices.add(notice)
            self.speak(message, Priority.STRATEGY, key='pit_window', ttl=8, cooldown=0)

    # ──────────────────────────────────────────
    # Priority Interrupts (Critical Events)
    # ──────────────────────────────────────────

    def update_safety_car(self, safety_car_status: int):
        """Immediate priority interrupt when Safety Car or VSC is deployed/withdrawn."""
        if safety_car_status == self.last_safety_car:
            return
        prev = self.last_safety_car
        self.last_safety_car = safety_car_status
        if safety_car_status == 1:
            self.speak("Safety Car, Safety Car. Keep the delta positive, no overtaking.",
                       Priority.CRITICAL, key='safety_car', ttl=10, cooldown=10)
        elif safety_car_status == 2:
            self.speak("VSC, VSC. Virtual Safety Car deployed, keep the delta positive.",
                       Priority.CRITICAL, key='safety_car', ttl=8, cooldown=10)
        elif prev == 2 and safety_car_status == 0:
            self.speak("VSC ending, VSC ending. Green flag, push now.",
                       Priority.CRITICAL, key='safety_car_in', ttl=8, cooldown=10)
        elif prev == 1 and safety_car_status == 0:
            self.speak("Safety Car in this lap. Get the tyres warm and be ready for the restart.",
                       Priority.CRITICAL, key='safety_car_in', ttl=8, cooldown=10)

    def update_weather_alert(self, weather_code: int):
        """Immediate priority interrupt on sudden rain arrival."""
        if self.last_weather is None:
            self.last_weather = weather_code
            return
        prev = self.last_weather
        self.last_weather = weather_code
        # 0=clear, 1=light cloud, 2=overcast, 3=light rain, 4=heavy rain, 5=storm
        if prev <= 2 and weather_code >= 3:
            self.speak("Rain starting to fall. Grip will drop, stay off the kerbs and white lines.",
                       Priority.CRITICAL, key='weather_rain', ttl=10, cooldown=30)

    def update_damage_alert(self, worst_tyre_wear: float, front_wing_damage: int, now=None):
        """Priority interrupt on severe tyre wear or front wing damage."""
        now = time.monotonic() if now is None else now
        if now - self.last_critical_alert < 25.0:
            return
        if worst_tyre_wear >= 75.0:
            self.last_critical_alert = now
            self.speak(f"Box, box. Box this lap, tyres are finished. "
                       f"{say.number_words(int(worst_tyre_wear))} percent wear.",
                       Priority.CRITICAL, key='critical_wear', ttl=8, cooldown=20)
        elif front_wing_damage >= 50:
            self.last_critical_alert = now
            self.speak("We can see front wing damage. Box this lap for a new nose. Box, box.",
                       Priority.CRITICAL, key='wing_damage', ttl=8, cooldown=25)

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
            if (lap.result_status != 2 or lap.car_position == 0
                    or not math.isfinite(lap.total_distance) or lap.pit_status):
                self.trajectories.pop(index, None)
                continue
            active[index] = lap
            history = self.trajectories.setdefault(index, deque(maxlen=6000))
            if history and (lap.total_distance < history[-1][0] or session_time - history[-1][1] > 2):
                history.clear()
            if not history or (lap.total_distance > history[-1][0] and session_time > history[-1][1]):
                history.append((lap.total_distance, session_time))
        for index, lap in active.items():
            if lap.last_lap_time_ms:
                self.last_lap_times_ms[index] = lap.last_lap_time_ms
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
            if offset == -1:
                self.gap_ahead_s, self.car_ahead_index = gap, index
            elif offset == 1:
                self.gap_behind_s, self.car_behind_index = -gap, index
            pace_delta = None
            player_best = self.last_lap_times_ms.get(player_index)
            opponent_best = self.last_lap_times_ms.get(index)
            if player_best and opponent_best:
                pace_delta = (opponent_best - player_best) / 1000.0
            threats.append(evaluate_opponent_threat(index, lap.car_position, gap, pace_delta,
                                                   self.driver_name(index)))
        threats.sort(key=lambda t: t.threat_score, reverse=True)
        self.opponent_threats = threats[:5]
        self._call_drs_range(now)
        if now - self.last_gap_report >= GAP_REPORT_INTERVAL_S:
            self._call_gaps(now)

    def _call_drs_range(self, now):
        """Instant call when a car comes within a second, either side."""
        for side, gap in (('ahead', self.gap_ahead_s), ('behind', self.gap_behind_s)):
            if gap is None:
                continue
            if gap >= DRS_REARM_S:
                self.drs_armed[side] = True
            elif gap < DRS_RANGE_S and self.drs_armed[side]:
                self.drs_armed[side] = False
                amount = say.gap_value(gap)
                if side == 'ahead':
                    message = self.phrases.pick('drs_ahead', (
                        f'You are within a second, {amount} to the car ahead. DRS is yours.',
                        f'{amount.capitalize()} to the car ahead, you have DRS. Go get them.',
                    ))
                else:
                    message = self.phrases.pick('drs_behind', (
                        f'Car behind is within a second, {amount}. Defend the braking zones.',
                        f'Car behind has DRS, {amount}. Cover the inside.',
                    ))
                # Counts as a gap update so the routine call doesn't repeat it moments later.
                self.last_gap_report = now
                self.reported_gaps = (self.gap_ahead_s, self.gap_behind_s)
                self.speak(message, Priority.RACE, key=f'drs_{side}', ttl=4, cooldown=30)

    def _call_gaps(self, now):
        """Routine gap update with the trend since the last one, like 'you're catching'."""
        last_ahead, last_behind = self.reported_gaps
        messages = []
        if self.gap_ahead_s is not None:
            text = f'Gap ahead {say.gap_value(self.gap_ahead_s)}'
            if last_ahead is not None and last_ahead - self.gap_ahead_s >= GAP_TREND_MIN_S:
                text += ", you're catching"
            elif last_ahead is not None and self.gap_ahead_s - last_ahead >= GAP_TREND_MIN_S:
                text += ", they're pulling away"
            messages.append(text + '.')
        if self.gap_behind_s is not None:
            text = f'Car behind {say.gap_value(self.gap_behind_s)}'
            if last_behind is not None and last_behind - self.gap_behind_s >= GAP_TREND_MIN_S:
                text += ", they're closing"
            elif last_behind is not None and self.gap_behind_s - last_behind >= GAP_TREND_MIN_S:
                text += ", you're pulling away"
            messages.append(text + '.')
        if messages:
            self.last_gap_report = now
            self.reported_gaps = (self.gap_ahead_s, self.gap_behind_s)
            self.speak(' '.join(messages), Priority.RACE, key='traffic_gaps', ttl=5,
                       cooldown=GAP_REPORT_INTERVAL_S)

    def report_sector(self, lap_num, sector, milliseconds, personal_ms):
        token = (lap_num, sector)
        if token in self.sectors_reported or milliseconds <= 0:
            return
        self.sectors_reported.add(token)
        name = say.sector_name(sector)
        if not personal_ms:
            message = f'{name.capitalize()}, {say.sector_time(milliseconds)}.'
        else:
            difference = (milliseconds - personal_ms) / 1000
            amount = say.time_amount(difference)
            if abs(difference) < 0.05:
                message = f'{name.capitalize()}, matching your best.'
            elif difference < 0:
                message = self.phrases.pick('sector_up', (
                    f'Green {name}, {amount} up.',
                    f'{name.capitalize()}, {amount} quicker. Good.',
                ))
            else:
                message = f'{name.capitalize()}, {amount} down.'
        self.speak(message, Priority.TIMING, key=f'sector_{sector}', ttl=8, cooldown=0)

    def completed_lap(self, milliseconds, lap_num, sectors, personal_sectors):
        if 3 in sectors:
            self.report_sector(lap_num, 3, sectors[3], personal_sectors.get(3))
        if not self.session or self.session.session_type not in (10, 11, 12):
            return
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
            self.speak(message, Priority.RACE, key='race_pace', ttl=12, cooldown=0)
