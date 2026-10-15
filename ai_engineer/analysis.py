"""
Analysis Engine — The Race Engineer's Brain
=============================================
Real-time telemetry analysis producing:
  1. Verbal callouts (Braking points, tyre temps, lap/sector deltas, setups)
  2. Live Web HUD telemetry streaming
  3. Corner-by-corner analysis of "Why You Are Slower"
  4. Real-time speed trace comparison vs session best
"""

import time
import copy
import sqlite3
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Optional

from packets import CarTelemetryData, LapData, SessionData, TRACK_NAMES, SESSION_TYPES
from track_data import get_braking_points, get_setup_advice
from voice_engine import VoiceEngine, Priority
from history import TelemetryHistory
from race_radio import RaceRadio
from web_hud import update_web_state, add_voice_log


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TYRE_TEMP_HIGH = 104     # °C – surface temp above this = overheating
TYRE_TEMP_LOW = 80       # °C – surface temp below this = cold
TYRE_TEMP_CRITICAL = 112 # °C – critical degradation
TYRE_TEMP_SUSTAIN_SECS = 2.5

BRAKING_WARN_DISTANCE = 220  # Metres before braking zone to announce

DELTA_SAMPLE_INTERVAL = 10.0  # Metres between samples for trace
DELTA_REPORT_INTERVAL = 14.0  # Seconds between voice delta callouts

WHEEL_NAMES = ("Rear Left", "Rear Right", "Front Left", "Front Right")

# Corner definitions for Silverstone (Track ID 7)
SILVERSTONE_CORNERS = [
    {"name": "Turn 1 Abbey", "start": 180, "end": 450, "ref_apex": 275, "ref_brake": 290},
    {"name": "Turn 3 Village", "start": 700, "end": 1000, "ref_apex": 110, "ref_brake": 850},
    {"name": "Turn 4 The Loop", "start": 1000, "end": 1200, "ref_apex": 88, "ref_brake": 1050},
    {"name": "Turn 6 Brooklands", "start": 1600, "end": 1950, "ref_apex": 140, "ref_brake": 1750},
    {"name": "Turn 7 Luffield", "start": 1950, "end": 2250, "ref_apex": 115, "ref_brake": 2000},
    {"name": "Turn 9 Copse", "start": 2700, "end": 3050, "ref_apex": 255, "ref_brake": 2820},
    {"name": "Maggots-Becketts", "start": 3250, "end": 3850, "ref_apex": 220, "ref_brake": 3350},
    {"name": "Turn 15 Stowe", "start": 4800, "end": 5200, "ref_apex": 185, "ref_brake": 4950},
    {"name": "Turn 16 Vale Chicane", "start": 5300, "end": 5600, "ref_apex": 95, "ref_brake": 5420},
    {"name": "Turn 17-18 Club", "start": 5600, "end": 5850, "ref_apex": 150, "ref_brake": 5680},
]


@dataclass
class TyreState:
    hot_since: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    cold_since: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    last_alert_time: float = 0.0


@dataclass
class DeltaTracker:
    # Best lap: distance -> elapsed_time_ms
    best_lap_times: dict[float, int] = field(default_factory=dict)
    best_lap_speeds: list[list[float]] = field(default_factory=list)  # [[dist, speed], ...]
    best_lap_time_ms: int = 0
    best_lap_num: int = 0

    # Current lap
    current_lap_times: dict[float, int] = field(default_factory=dict)
    current_lap_speeds: list[list[float]] = field(default_factory=list)
    current_lap_samples: list[dict] = field(default_factory=list)
    current_lap_num: int = 0
    current_lap_invalid: bool = False
    recorded_from_start: bool = False
    last_distance: float | None = None
    last_elapsed_ms: int = 0
    last_sample_distance: float = -999.0

    # Timing
    last_delta_report_time: float = 0.0
    last_reported_delta_ms: int | None = None


class AnalysisEngine:
    """The brain of the AI Race Engineer."""

    def __init__(self, voice: VoiceEngine, history: TelemetryHistory | None = None):
        self.voice = voice
        self.radio = RaceRadio(self.speak)
        self.history = history
        self.capture_id: str | None = None
        self.session_uid: int | None = None
        self.session_context = None
        self.best_sector_times: dict[int, int] = {}
        self.current_sector_times: dict[int, int] = {}

        self.tyre_state = TyreState()
        self.delta = DeltaTracker()
        self.session: Optional[SessionData] = None
        self.track_id: int = -1
        self.track_name: str = "Waiting for session"
        self.braking_points: list[tuple[float, str, str]] = []
        self.announced_braking: set[str] = set()
        self.last_lap_num: int = 0
        self.last_position: int = 0
        self.last_sector: int = 0
        self.session_started = False
        self.setup_advice_given = False

        # Corner tracking
        self.corners = []
        self.current_corner_data: dict[str, dict] = {}
        self.best_corner_data: dict[str, dict] = {}
        self.corner_diagnostics: list[dict] = []
        self.previous_corner_sample = None
        self.brake_start_distance = None

    # ──────────────────────────────────────────
    # Session Handling
    # ──────────────────────────────────────────

    def update_session(self, session: SessionData, session_uid: int | None = None, player_index=None):
        context = (session_uid, session.track_id, session.track_length, session.formula,
                   session.session_type, TelemetryHistory.weather_group(session.weather), player_index)
        if context != self.session_context:
            self.session_context = context
            self.session_uid = session_uid
            self.delta = DeltaTracker()
            self.tyre_state = TyreState()
            self.current_corner_data.clear()
            self.best_corner_data.clear()
            self.corner_diagnostics.clear()
            self.previous_corner_sample = None
            self.brake_start_distance = None
            self.current_sector_times.clear()
            self.best_sector_times.clear()
            self.announced_braking.clear()
            self.last_sector = 0
            self.session_started = False
            self.capture_id = None
            if hasattr(self.voice, 'clear'):
                self.voice.clear()
            if self.history and session.track_id >= 0:
                try:
                    self.capture_id = self.history.start_session(session, session_uid)
                    reference = self.history.best_reference(session)
                    if reference:
                        self.delta.best_lap_time_ms = reference['lap_time_ms']
                        self.delta.best_lap_num = reference['lap_num']
                        self.delta.best_lap_times = {float(k): v for k, v in reference['times'].items()}
                        self.delta.best_lap_speeds = reference['speeds']
                        self.best_corner_data = reference['corners']
                        self.best_sector_times = self.history.personal_sectors(session)
                except (OSError, ValueError, KeyError, sqlite3.Error) as error:
                    print(f'[History] Could not load track history: {error}')
            update_web_state({
                'connected': False, 'corner_analysis': [], 'voice_log': [],
                'speed_trace_current': [], 'speed_trace_best': self.delta.best_lap_speeds,
                'best_lap_time_ms': self.delta.best_lap_time_ms, 'delta_ms': None,
                'gap_ahead_s': None, 'gap_behind_s': None, 'race_pace_ms': None,
                'pit_window': None, 'last_update_monotonic': None,
                'speed': 0, 'gear': 0, 'rpm': 0, 'drs': False, 'throttle': 0, 'brake': 0,
                'lap_num': 0, 'lap_time_ms': 0, 'last_lap_time_ms': 0, 'position': 0,
                'tyre_surface_temps': [0] * 4, 'tyre_inner_temps': [0] * 4,
                'tyre_pressures': [0] * 4, 'engine_temp': 0,
                'personal_sectors_ms': self.best_sector_times,
            })
            self.radio.reset(session)
        self.session = session
        self.radio.session = session
        self.track_id = session.track_id
        self.track_name = TRACK_NAMES.get(session.track_id, f"Track {session.track_id}")
        session_type = SESSION_TYPES.get(session.session_type, "Session")

        # Load braking points
        self.braking_points = get_braking_points(session.track_id)
        update_web_state({'track_name': self.track_name, 'track_id': self.track_id,
                          'track_length': session.track_length, 'session_uid': session_uid,
                          'capture_id': self.capture_id})
        if session.track_id == 7:
            self.corners = SILVERSTONE_CORNERS
        else:
            self.corners = []

        if not self.session_started:
            self.session_started = True
            msg = (
                f"Engineer online. Welcome to {self.track_name}. {session_type}. "
                f"Track temperature {session.track_temperature} degrees. "
                f"Braking points and live delta tracker online. Let's go."
            )
            self.speak(msg, Priority.LOW, key='session', ttl=10, cooldown=0)

    def speak(self, message: str, priority: Priority = Priority.NORMAL, **options):
        """Queue speech; the voice worker logs utterances when they actually start."""
        topic = options.get('key', message)
        options['key'] = f'{self.capture_id or self.session_context}:{topic}'
        queued = self.voice.say(message, priority, **options)
        if not queued and not getattr(self.voice, 'is_running', False):
            add_voice_log(message, status='muted')

    # ──────────────────────────────────────────
    # Tyre Temperature Analysis
    # ──────────────────────────────────────────

    def analyze_tyres(self, telemetry: CarTelemetryData, lap: LapData):
        now = time.time()
        temps = telemetry.tyres_surface_temperature

        if telemetry.speed < 40:
            return

        hot_wheels = []
        cold_wheels = []
        critical_wheels = []

        for i in range(4):
            temp = temps[i]
            if temp >= TYRE_TEMP_CRITICAL:
                critical_wheels.append((WHEEL_NAMES[i], temp))
            elif temp > TYRE_TEMP_HIGH:
                if self.tyre_state.hot_since[i] == 0.0:
                    self.tyre_state.hot_since[i] = now
                elif now - self.tyre_state.hot_since[i] >= TYRE_TEMP_SUSTAIN_SECS:
                    hot_wheels.append((WHEEL_NAMES[i], temp))
            else:
                self.tyre_state.hot_since[i] = 0.0

            if temp < TYRE_TEMP_LOW:
                if self.tyre_state.cold_since[i] == 0.0:
                    self.tyre_state.cold_since[i] = now
                elif now - self.tyre_state.cold_since[i] >= TYRE_TEMP_SUSTAIN_SECS:
                    cold_wheels.append((WHEEL_NAMES[i], temp))
            else:
                self.tyre_state.cold_since[i] = 0.0

        if now - self.tyre_state.last_alert_time < 25.0:
            return

        if critical_wheels:
            names = " and ".join(w[0] for w in critical_wheels)
            self.speak(
                f"Critical! {names} overheating at {critical_wheels[0][1]} degrees! Manage your tyres!",
                Priority.CRITICAL, key='tyre_temperature', ttl=5, cooldown=25
            )
            self.tyre_state.last_alert_time = now
        elif hot_wheels:
            fronts = [w for w in hot_wheels if "Front" in w[0]]
            rears = [w for w in hot_wheels if "Rear" in w[0]]
            if fronts and rears:
                self.speak(f"All tyres running hot above {TYRE_TEMP_HIGH} degrees. Manage entry speeds.", Priority.NORMAL)
            elif fronts:
                avg = sum(w[1] for w in fronts) // len(fronts)
                self.speak(f"Front tyres overheating at {avg} degrees. Ease off entry steering angle.", Priority.NORMAL)
            elif rears:
                avg = sum(w[1] for w in rears) // len(rears)
                self.speak(f"Rear tyres overheating at {avg} degrees. Smooth throttle on corner exit.", Priority.NORMAL)
            self.tyre_state.last_alert_time = now

    # ──────────────────────────────────────────
    # Braking Point Callouts
    # ──────────────────────────────────────────

    def check_braking_points(self, lap: LapData, telemetry: CarTelemetryData):
        if not self.braking_points or telemetry.speed < 60:
            return

        dist = lap.lap_distance
        if dist < 50:
            # Clear announcements near start/finish
            self.announced_braking.clear()
            return

        for bp_dist, corner_name, message in self.braking_points:
            approach_dist = bp_dist - dist
            if 0 < approach_dist <= BRAKING_WARN_DISTANCE:
                if corner_name not in self.announced_braking:
                    self.announced_braking.add(corner_name)
                    dist_rnd = int(round(approach_dist / 10.0) * 10)
                    self.speak(f"{message}, {dist_rnd} metres.", Priority.CRITICAL,
                               key='braking', ttl=1.0, cooldown=0)

    # ──────────────────────────────────────────
    # Corner Diagnostic Engine ("Why You Are Slower")
    # ──────────────────────────────────────────

    def update_corners(self, telemetry: CarTelemetryData, lap: LapData):
        """Measure matched corner boundaries; never time a partial corner."""
        sample = (lap.lap_distance, lap.current_lap_time_ms, telemetry.speed, telemetry.brake)
        previous = self.previous_corner_sample
        self.previous_corner_sample = sample
        if previous is None or sample[0] <= previous[0]:
            return
        dist, elapsed, speed, brake = sample
        prev_dist, prev_time, _, prev_brake = previous
        if brake > 0.3 and prev_brake <= 0.3:
            fraction = (0.3 - prev_brake) / (brake - prev_brake)
            self.brake_start_distance = round(prev_dist + (dist - prev_dist) * fraction)
        elif brake <= 0.3:
            self.brake_start_distance = None

        def crossing_time(boundary):
            return prev_time + (elapsed - prev_time) * (boundary - prev_dist) / (dist - prev_dist)

        for c in self.corners:
            c_name = c["name"]
            c_start = c["start"]
            c_end = c["end"]
            if prev_dist <= c_start < dist and c_name not in self.current_corner_data:
                self.current_corner_data[c_name] = {
                    'min_speed': speed, 'brake_point': self.brake_start_distance,
                    'start_time_ms': crossing_time(c_start), 'end_time_ms': None,
                }
            data = self.current_corner_data.get(c_name)
            if data is None or data['end_time_ms'] is not None:
                continue
            if c_start <= dist <= c_end:
                data['min_speed'] = min(data['min_speed'], speed)
                if data['brake_point'] is None and brake > 0.3:
                    data['brake_point'] = self.brake_start_distance
            if prev_dist <= c_end < dist:
                data['end_time_ms'] = crossing_time(c_end)
                self._diagnose_corner(c_name, data, c)

    def _diagnose_corner(self, c_name: str, current_data: dict, ref_def: dict):
        """Diagnose time loss in corner and update HUD."""
        start_t = current_data.get("start_time_ms", 0)
        end_t = current_data.get("end_time_ms", 0)
        curr_dur = (end_t - start_t) / 1000.0

        ref_apex = ref_def["ref_apex"]
        ref_brake = ref_def["ref_brake"]

        curr_apex = current_data.get("min_speed", 0)
        curr_brake = current_data.get("brake_point", ref_brake)

        # Baseline comparison
        best_d = self.best_corner_data.get(c_name)
        if best_d and best_d.get('end_time_ms') is not None:
            best_apex = best_d["min_speed"]
            ref_brake = best_d.get('brake_point')
            best_dur = (best_d["end_time_ms"] - best_d["start_time_ms"]) / 1000.0
            delta_s = curr_dur - best_dur
            reference_source = 'recorded lap'
        else:
            best_apex = ref_apex
            delta_s = None
            reference_source = 'track guide'

        # Diagnosis string
        advice = []
        if curr_brake is not None and ref_brake is not None and curr_brake < (ref_brake - 20):
            diff = ref_brake - curr_brake
            advice.append(f"Braked {diff}m before the {reference_source} braking point.")
        if curr_apex < (best_apex - 8):
            diff = best_apex - curr_apex
            advice.append(f"Apex speed {diff} km/h below the {reference_source}.")
        elif curr_apex >= best_apex:
            advice.append("Good apex speed maintained!")

        if not advice:
            advice_text = "Good execution through corner."
        else:
            advice_text = " ".join(advice)

        # Update HUD corner list
        diag_entry = {
            "name": c_name,
            "delta": delta_s,
            "reference_source": reference_source,
            "current_apex": curr_apex,
            "best_apex": best_apex,
            "brake_point": curr_brake,
            "advice": advice_text,
        }

        # Keep latest corner diagnostics
        existing_idx = next((i for i, item in enumerate(self.corner_diagnostics) if item["name"] == c_name), None)
        if existing_idx is not None:
            self.corner_diagnostics[existing_idx] = diag_entry
        else:
            self.corner_diagnostics.append(diag_entry)

    # ──────────────────────────────────────────
    # Lap Delta Tracker & Sector Reporting
    # ──────────────────────────────────────────

    def update_delta(self, lap: LapData, telemetry: CarTelemetryData):
        now = time.time()
        dist = lap.lap_distance
        elapsed_ms = lap.current_lap_time_ms

        if dist < 0 or elapsed_ms <= 0:
            return

        # New lap check
        if lap.current_lap_num != self.delta.current_lap_num:
            self._on_new_lap(lap)

        self.delta.current_lap_invalid |= bool(
            lap.current_lap_invalid or lap.pit_status or lap.driver_status in (0, 2, 3)
            or (self.session and self.session.safety_car_status))
        if self.delta.last_distance is None:
            self.delta.recorded_from_start = dist <= 100 and elapsed_ms <= 5000
        else:
            max_gap = max(100, (self.session.track_length if self.session else 0) * 0.025)
            if (dist < self.delta.last_distance - 1 or
                    dist - self.delta.last_distance > max_gap or
                    elapsed_ms < self.delta.last_elapsed_ms):
                self.delta.current_lap_invalid = True
        self.delta.last_distance = dist
        self.delta.last_elapsed_ms = elapsed_ms
        if lap.sector >= 1 and lap.sector1_time_ms > 0:
            self.current_sector_times[1] = lap.sector1_time_ms
        if lap.sector >= 2 and lap.sector2_time_ms > 0:
            self.current_sector_times[2] = lap.sector2_time_ms

        # Sector reporting (Sector 0 -> 1 -> 2)
        if lap.sector != self.last_sector:
            self._on_sector_change(lap)
            self.last_sector = lap.sector

        # Sample at regular distance intervals
        if dist - self.delta.last_sample_distance >= DELTA_SAMPLE_INTERVAL:
            self.delta.last_sample_distance = dist
            key = dist
            self.delta.current_lap_times[key] = elapsed_ms
            self.delta.current_lap_speeds.append([round(dist, 1), telemetry.speed])
            self.delta.current_lap_samples.append({
                'distance': dist, 'elapsed_ms': elapsed_ms, 'speed': telemetry.speed,
                'throttle': telemetry.throttle, 'brake': telemetry.brake, 'gear': telemetry.gear,
                'rpm': telemetry.engine_rpm, 'tyre_surface_temps': list(telemetry.tyres_surface_temperature),
                'tyre_inner_temps': list(telemetry.tyres_inner_temperature),
                'tyre_pressures': list(telemetry.tyres_pressure),
            })

            # Calculate live delta if baseline exists
            reference = self.delta.best_lap_times
            keys = sorted(reference)
            index = bisect_left(keys, dist)
            reference_ms = reference.get(dist)
            if reference_ms is None and 0 < index < len(keys):
                left, right = keys[index - 1], keys[index]
                if right - left <= max(100, (self.session.track_length if self.session else 0) * 0.025):
                    fraction = (dist - left) / (right - left)
                    reference_ms = reference[left] + fraction * (reference[right] - reference[left])
            self.delta.last_reported_delta_ms = None
            if reference_ms is not None:
                delta_ms = round(elapsed_ms - reference_ms)
                self.delta.last_reported_delta_ms = delta_ms

                # Voice delta callout periodically
                if now - self.delta.last_delta_report_time >= DELTA_REPORT_INTERVAL:
                    self.delta.last_delta_report_time = now
                    delta_s = abs(delta_ms) / 1000.0
                    if delta_ms < -100:
                        self.speak(f"Minus {delta_s:.1f} seconds to your best. Strong pace.", Priority.LOW,
                                   key='live_delta', ttl=6, cooldown=14)
                    elif delta_ms > 150:
                        self.speak(f"Plus {delta_s:.1f} seconds to your best.", Priority.LOW,
                                   key='live_delta', ttl=6, cooldown=14)

    def _on_sector_change(self, lap: LapData):
        """Announce sector timing and delta."""
        if not self.delta.recorded_from_start or self.delta.current_lap_invalid:
            return
        if lap.sector in (1, 2):
            milliseconds = lap.sector1_time_ms if lap.sector == 1 else lap.sector2_time_ms
            self.radio.report_sector(lap.current_lap_num, lap.sector, milliseconds,
                                     self.best_sector_times.get(lap.sector))

    def _on_new_lap(self, lap: LapData):
        """Handle lap completion and save best baseline."""
        prev_lap_num = self.delta.current_lap_num
        self.delta.current_lap_num = lap.current_lap_num
        self.announced_braking.clear()

        # If previous lap completed with valid lap time
        track_length = self.session.track_length if self.session else 0
        complete = (
            prev_lap_num > 0 and lap.current_lap_num == prev_lap_num + 1
            and self.delta.recorded_from_start and not self.delta.current_lap_invalid
            and track_length > 0 and self.delta.last_distance is not None
            and self.delta.last_distance >= track_length - 100
            and lap.last_lap_time_ms >= self.delta.last_elapsed_ms
        )
        sectors = dict(self.current_sector_times)
        if 1 in sectors and 2 in sectors and lap.last_lap_time_ms > sectors[1] + sectors[2]:
            sectors[3] = lap.last_lap_time_ms - sectors[1] - sectors[2]
        if prev_lap_num > 0 and self.delta.current_lap_samples:
            recording = {
                'lap_time_ms': lap.last_lap_time_ms, 'lap_num': prev_lap_num,
                'times': self.delta.current_lap_times, 'speeds': self.delta.current_lap_speeds,
                'samples': self.delta.current_lap_samples,
                'corners': self.current_corner_data, 'sectors': sectors,
            }
            if self.history and self.capture_id:
                try:
                    self.history.record_lap(self.capture_id, prev_lap_num,
                                            lap.last_lap_time_ms, complete, recording)
                except (OSError, ValueError, sqlite3.Error) as error:
                    print(f'[History] Could not save lap: {error}')
        if complete and lap.last_lap_time_ms > 0 and self.delta.current_lap_times:
            last_ms = lap.last_lap_time_ms
            self.radio.completed_lap(last_ms, prev_lap_num, sectors, self.best_sector_times)
            for sector, milliseconds in sectors.items():
                self.best_sector_times[sector] = min(self.best_sector_times.get(sector, milliseconds), milliseconds)
            secs = last_ms / 1000.0
            mins = int(secs // 60)
            secs_rem = secs % 60

            # Store as best if first completed lap or faster
            is_best = (self.delta.best_lap_time_ms == 0 or last_ms < self.delta.best_lap_time_ms)

            if is_best:
                self.delta.best_lap_times = dict(self.delta.current_lap_times)
                self.delta.best_lap_speeds = list(self.delta.current_lap_speeds)
                self.delta.best_lap_time_ms = last_ms
                self.delta.best_lap_num = prev_lap_num
                self.best_corner_data = copy.deepcopy(self.current_corner_data)

                self.speak(f"Personal best! {mins} minute {secs_rem:.3f}. New reference lap set.", Priority.TIMING,
                           key='lap_time', ttl=10, cooldown=0)
            else:
                diff_s = (last_ms - self.delta.best_lap_time_ms) / 1000.0
                self.speak(f"Lap complete: {mins} {secs_rem:.3f}. Plus {diff_s:.2f} seconds to your best.", Priority.TIMING,
                           key='lap_time', ttl=10, cooldown=0)

        # Reset current recording lap
        self.delta.current_lap_times.clear()
        self.delta.current_lap_speeds.clear()
        self.delta.current_lap_samples.clear()
        self.current_sector_times.clear()
        self.delta.last_sample_distance = -999.0
        self.delta.current_lap_invalid = False
        self.delta.recorded_from_start = False
        self.delta.last_distance = None
        self.delta.last_elapsed_ms = 0
        self.delta.last_reported_delta_ms = None
        self.current_corner_data.clear()
        self.previous_corner_sample = None
        self.brake_start_distance = None

    # ──────────────────────────────────────────
    # Main Update
    # ──────────────────────────────────────────

    def update_telemetry(self, telemetry: CarTelemetryData, lap: LapData):
        """Main update called on each telemetry packet."""
        self.update_delta(lap, telemetry)
        self.analyze_tyres(telemetry, lap)
        self.check_braking_points(lap, telemetry)
        self.update_corners(telemetry, lap)
        self.radio.update_strategy(lap)

        # Stream snapshot to Web HUD
        update_web_state({
            "connected": True,
            "last_update_monotonic": time.monotonic(),
            "track_name": self.track_name,
            "track_id": self.track_id,
            "speed": telemetry.speed,
            "gear": telemetry.gear,
            "rpm": telemetry.engine_rpm,
            "drs": bool(telemetry.drs),
            "throttle": round(telemetry.throttle, 2),
            "brake": round(telemetry.brake, 2),
            "tyre_surface_temps": list(telemetry.tyres_surface_temperature),
            "tyre_inner_temps": list(telemetry.tyres_inner_temperature),
            "tyre_pressures": list(telemetry.tyres_pressure),
            "engine_temp": telemetry.engine_temperature,
            "lap_num": lap.current_lap_num,
            "position": lap.car_position,
            "lap_time_ms": lap.current_lap_time_ms,
            "last_lap_time_ms": lap.last_lap_time_ms,
            "best_lap_time_ms": self.delta.best_lap_time_ms,
            "delta_ms": self.delta.last_reported_delta_ms,
            "lap_distance": round(lap.lap_distance, 1),
            "sector": lap.sector,
            'gap_ahead_s': self.radio.gap_ahead_s,
            'gap_behind_s': self.radio.gap_behind_s,
            'race_pace_ms': self.radio.race_pace_ms,
            'pit_window': self.radio.pit_window,
            'personal_sectors_ms': self.best_sector_times,
            "corner_analysis": self.corner_diagnostics,
            "speed_trace_current": self.delta.current_lap_speeds[-120:],
            "speed_trace_best": self.delta.best_lap_speeds
        })
