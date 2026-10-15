import math
import time
from collections import deque

from voice_engine import Priority


class RaceRadio:
    def __init__(self, speak):
        self.speak = speak
        self.reset(None)

    def reset(self, session):
        self.session = session
        self.trajectories = {}
        self.last_session_time = None
        self.last_gap_report = -math.inf
        self.gap_ahead_s = self.gap_behind_s = None
        self.pit_notices = set()
        self.pit_window = None
        self.last_pit_stops = None
        self.served_window = None
        self.sectors_reported = set()
        self.race_laps = deque(maxlen=3)
        self.race_pace_ms = None

    def update_strategy(self, lap):
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
            phase, message = 'last', 'Last lap of your planned pit window.'
        elif ideal <= lap.current_lap_num < latest:
            phase, message = 'open', f'Pit window open. Planned stop lap {ideal}, latest lap {latest}.'
        elif lap.current_lap_num == ideal - 1:
            phase, message = 'next', f'Pit window opens next lap. Target lap {ideal}, latest lap {latest}.'
        else:
            return
        notice = (window, lap.num_pit_stops, phase)
        if notice not in self.pit_notices:
            self.pit_notices.add(notice)
            self.speak(message, Priority.STRATEGY, key='pit_window', ttl=8, cooldown=0)

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
        self.gap_ahead_s = self.gap_behind_s = None
        player = active.get(player_index)
        if player is None:
            return
        for index, lap in active.items():
            crossing = None
            if lap.car_position == player.car_position - 1:
                crossing = self._crossing_time(self.trajectories[index], player.total_distance)
                if crossing is not None:
                    self.gap_ahead_s = max(0, session_time - crossing)
            elif lap.car_position == player.car_position + 1:
                crossing = self._crossing_time(self.trajectories[player_index], lap.total_distance)
                if crossing is not None:
                    self.gap_behind_s = max(0, session_time - crossing)
        if now - self.last_gap_report >= 20:
            messages = []
            if self.gap_ahead_s is not None:
                messages.append(f'Gap ahead, {self.gap_ahead_s:.1f} seconds.')
            if self.gap_behind_s is not None:
                messages.append(f'Gap behind, {self.gap_behind_s:.1f} seconds.')
            if messages:
                self.last_gap_report = now
                self.speak(' '.join(messages), Priority.RACE, key='traffic_gaps', ttl=5, cooldown=15)

    def report_sector(self, lap_num, sector, milliseconds, personal_ms):
        token = (lap_num, sector)
        if token in self.sectors_reported or milliseconds <= 0:
            return
        self.sectors_reported.add(token)
        message = f'Sector {sector}, {milliseconds / 1000:.3f} seconds.'
        if personal_ms:
            difference = milliseconds - personal_ms
            if abs(difference) < 50:
                message += ' Matching your personal best sector.'
            else:
                direction = 'faster' if difference < 0 else 'slower'
                message += f' {abs(difference) / 1000:.2f} seconds {direction} than your personal best sector.'
        self.speak(message, Priority.TIMING, key=f'sector_{sector}', ttl=8, cooldown=0)

    def completed_lap(self, milliseconds, lap_num, sectors, personal_sectors):
        if 3 in sectors:
            self.report_sector(lap_num, 3, sectors[3], personal_sectors.get(3))
        if not self.session or self.session.session_type not in (10, 11, 12):
            return
        self.race_laps.append(milliseconds)
        if len(self.race_laps) == 3:
            self.race_pace_ms = round(sum(self.race_laps) / 3)
            seconds = self.race_pace_ms / 1000
            self.speak(f'Race pace over the last three clean laps, {int(seconds // 60)} minute '
                       f'{seconds % 60:.3f} average.', Priority.RACE,
                       key='race_pace', ttl=12, cooldown=0)
