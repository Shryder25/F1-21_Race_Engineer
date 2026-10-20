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

from packets import (CarTelemetryData, LapData, SessionData, CarStatusData, CarDamageData,
                     CarMotionData, TRACK_NAMES, SESSION_TYPES, ACTUAL_COMPOUNDS)
from track_data import get_braking_points, get_setup_advice, corner_zones, resolve_track_id
import physics_models
import session_modes
from voice_engine import VoiceEngine, Priority
from history import TelemetryHistory
from race_radio import RaceRadio
from race_events import RaceControl
from web_hud import update_web_state, add_voice_log
import strategy_ml


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TYRE_TEMP_HIGH = 104     # °C – surface temp above this = overheating
TYRE_TEMP_LOW = 80       # °C – surface temp below this = cold
TYRE_TEMP_CRITICAL = 112 # °C – critical degradation
TYRE_TEMP_SUSTAIN_SECS = 2.5

# Critical-overheat alert: require the temperature to actually be climbing, not just
# momentarily above the threshold, and respect a cooldown so one real overheat event
# doesn't repeat every single frame while the tyre stays hot.
TYRE_TEMP_RISE_RATE_C_PER_S = 1.5   # minimum ΔT/Δt to treat a spike as genuine overheating
TYRE_TEMP_ALERT_COOLDOWN_SECS = 40.0  # between 30-45s, per the requested fix

BRAKING_WARN_DISTANCE = 220  # Metres before braking zone to announce

DELTA_SAMPLE_INTERVAL = 10.0  # Metres between samples for trace
DELTA_REPORT_INTERVAL = 14.0  # Seconds between voice delta callouts

WHEEL_NAMES = ("Rear Left", "Rear Right", "Front Left", "Front Right")
WHEEL_CODES = ("RL", "RR", "FL", "FR")   # same order as the telemetry tuples

WEATHER_NAMES = {0: "clear", 1: "light cloud", 2: "overcast", 3: "light rain",
                 4: "heavy rain", 5: "storm"}

# In on-demand mode the engineer stays silent unless the driver asks, except for critical events.
PROACTIVE_KEYS = frozenset({'tyre_temperature', 'safety_car', 'safety_car_in', 'weather_rain', 'critical_wear', 'wing_damage',
                            'penalty', 'flag_yellow', 'flag_red'})

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
    # Rate-of-change tracking for the critical-overheat alert: a momentary noisy spike
    # (common in the game's telemetry, e.g. a brief lockup reading) should not trigger a
    # radio call on its own — only a temperature that is genuinely climbing fast should.
    last_temp_sample: list[float] | None = None
    last_temp_sample_time: float = 0.0
    last_critical_alert_time: float = 0.0


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

    def __init__(self, voice: VoiceEngine, history: TelemetryHistory | None = None,
                 on_demand: bool = False):
        self.voice = voice
        self.on_demand = on_demand
        self.latest_telemetry: CarTelemetryData | None = None
        self.latest_lap: LapData | None = None
        self.latest_status: CarStatusData | None = None
        self.latest_damage: CarDamageData | None = None
        self.lap_start_fuel: float | None = None
        self.current_lap_speed_trap_kph: float | None = None
        self.last_speed_trap_kph: float | None = None
        self.session_lap_stats: list[dict] = []
        self._ml_cache: tuple | None = None
        self.session_mode = session_modes.UNKNOWN
        self._corner_key: tuple | None = None
        self.radio = RaceRadio(self.speak)
        self.race_control = RaceControl(self.speak)
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

        # Live Motion & Track Mapping
        self.latest_motion: CarMotionData | None = None
        self.current_lap_map_points: list[list[float]] = []
        self.track_map_points: list[list[float]] = []
        self.current_lap_braking_zones: list[list[float]] = []
        self.current_lap_ers_zones: list[list[float]] = []
        self.braking_zones_best: list[list[float]] = []
        self.ers_zones_best: list[list[float]] = []
        self.last_map_dist: float = -999.0

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
            self.session_lap_stats.clear()
            self._ml_cache = None
            self.latest_status = self.latest_damage = None
            self.lap_start_fuel = None
            self.current_lap_speed_trap_kph = None
            self.last_speed_trap_kph = None
            self.announced_braking.clear()
            self.last_sector = 0
            self.session_started = False
            self.capture_id = None
            self.latest_motion = None
            self.current_lap_map_points.clear()
            self.track_map_points.clear()
            self.current_lap_braking_zones.clear()
            self.current_lap_ers_zones.clear()
            self.braking_zones_best.clear()
            self.ers_zones_best.clear()
            self.last_map_dist = -999.0
            self.radio.set_driver_names({})
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
                'car_ahead_name': None, 'car_behind_name': None,
                'pit_window': None, 'last_update_monotonic': None,
                'speed': 0, 'gear': 0, 'rpm': 0, 'drs': False, 'throttle': 0, 'brake': 0,
                'lap_num': 0, 'lap_time_ms': 0, 'last_lap_time_ms': 0, 'position': 0,
                'tyre_surface_temps': [0] * 4, 'tyre_inner_temps': [0] * 4,
                'tyre_pressures': [0] * 4, 'engine_temp': 0,
                'personal_sectors_ms': self.best_sector_times,
            })
            self.radio.reset(session)
            self.race_control.reset(session)
            update_web_state({'race_control': []})
        self.session = session
        self.radio.session = session
        self.race_control.session = session
        self.track_id = session.track_id
        self.track_name = TRACK_NAMES.get(session.track_id, f"Track {session.track_id}")
        self.session_mode = session_modes.session_mode(session.session_type)

        # Load braking points and corner zones for whichever circuit the game reports
        self.braking_points = get_braking_points(session.track_id)
        update_web_state({'track_name': self.track_name, 'track_id': self.track_id,
                          'track_length': session.track_length, 'session_uid': session_uid,
                          'capture_id': self.capture_id,
                          'session_type': SESSION_TYPES.get(session.session_type, 'Session'),
                          'session_mode': self.session_mode,
                          'session_label': session_modes.MODE_LABELS[self.session_mode],
                          'setup_track_id': resolve_track_id(session.track_id)})
        corner_key = (session.track_id, session.track_length)
        if corner_key != self._corner_key:
            self._corner_key = corner_key
            if session.track_id == 7:
                self.corners = SILVERSTONE_CORNERS      # hand-measured apex speeds
            else:
                self.corners = corner_zones(session.track_id, session.track_length)

        if not self.session_started:
            self.session_started = True
            self.speak(session_modes.welcome_line(self.session_mode, self.track_name),
                       Priority.LOW, key='session', ttl=10, cooldown=0)

    def speak(self, message: str, priority: Priority = Priority.NORMAL, **options):
        """Queue speech; priority interrupts in PROACTIVE_KEYS break on-demand silence."""
        topic = options.get('key', message)
        if self.on_demand and topic not in PROACTIVE_KEYS:
            return  # radio silence until the driver asks or critical priority event
        options['key'] = f'{self.capture_id or self.session_context}:{topic}'
        queued = self.voice.say(message, priority, **options)
        if not queued and not getattr(self.voice, 'is_running', False):
            add_voice_log(message, status='muted')

    def update_motion(self, motion: CarMotionData):
        """Update live world position coordinates and lateral/longitudinal G-forces."""
        self.latest_motion = motion
        x = round(motion.world_position_x, 2)
        z = round(motion.world_position_z, 2)
        if self.latest_lap and self.latest_telemetry:
            dist = self.latest_lap.lap_distance
            if abs(dist - self.last_map_dist) >= 6.0:
                self.last_map_dist = dist
                self.current_lap_map_points.append([x, z])
                if self.latest_telemetry.brake > 0.08:
                    self.current_lap_braking_zones.append([x, z, round(self.latest_telemetry.brake, 2)])
                if self.latest_status and (self.latest_status.ers_deploy_mode > 0 or self.latest_status.ers_deployed_this_lap > 0):
                    self.current_lap_ers_zones.append([x, z])

    def is_driver_busy(self) -> bool:
        """Check if driver is under heavy braking or in a high-G corner (silence filter)."""
        if self.latest_telemetry is None:
            return False
        if self.latest_telemetry.brake > 0.15:
            return True
        if abs(self.latest_telemetry.steer) > 0.20:
            return True
        if self.latest_motion and abs(self.latest_motion.g_force_lateral) > 1.2:
            return True
        return False

    # ──────────────────────────────────────────
    # Tyre Temperature Analysis
    # ──────────────────────────────────────────

    def analyze_tyres(self, telemetry: CarTelemetryData, lap: LapData):
        now = time.time()
        temps = telemetry.tyres_surface_temperature

        if telemetry.speed < 40:
            return

        # Rate-of-change (ΔT/Δt) per wheel since the last sample, used to tell a genuine
        # overheat climb apart from a single noisy spike that happens to be above the
        # critical threshold for one frame (a false positive the static-threshold check
        # had no way to rule out).
        rise_rates = [0.0, 0.0, 0.0, 0.0]
        prev_sample, prev_time = self.tyre_state.last_temp_sample, self.tyre_state.last_temp_sample_time
        if prev_sample is not None and now > prev_time:
            dt = now - prev_time
            rise_rates = [(temps[i] - prev_sample[i]) / dt for i in range(4)]
        self.tyre_state.last_temp_sample = list(temps)
        self.tyre_state.last_temp_sample_time = now

        hot_wheels = []
        cold_wheels = []
        critical_wheels = []

        for i in range(4):
            temp = temps[i]
            if temp >= TYRE_TEMP_CRITICAL and rise_rates[i] >= TYRE_TEMP_RISE_RATE_C_PER_S:
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

        if critical_wheels and now - self.tyre_state.last_critical_alert_time >= TYRE_TEMP_ALERT_COOLDOWN_SECS:
            names = " and ".join(w[0] for w in critical_wheels)
            self.speak(
                f"Critical! {names} overheating at {critical_wheels[0][1]} degrees! Manage your tyres!",
                Priority.CRITICAL, key='tyre_temperature', ttl=5, cooldown=25
            )
            self.tyre_state.last_critical_alert_time = now
            self.tyre_state.last_alert_time = now
            return

        if now - self.tyre_state.last_alert_time < 25.0:
            return

        if hot_wheels:
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

        ref_apex = ref_def.get("ref_apex")      # None for corners without a measured apex speed
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
        if best_apex is None:
            pass        # no reference apex yet: this lap becomes the reference
        elif curr_apex < (best_apex - 8):
            diff = best_apex - curr_apex
            advice.append(f"Apex speed {diff} km/h below the {reference_source}.")
        elif curr_apex >= best_apex:
            advice.append("Good apex speed maintained!")

        if not advice:
            advice_text = ("Reference lap not set yet; this lap will be your benchmark."
                           if best_apex is None else "Good execution through corner.")
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
        if prev_lap_num > 0 and lap.last_lap_time_ms > 0:
            self._record_lap_stats(prev_lap_num, lap.last_lap_time_ms, complete, sectors,
                                   self.current_lap_speed_trap_kph)
        self.last_speed_trap_kph = self.current_lap_speed_trap_kph
        self.current_lap_speed_trap_kph = None
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
                if self.current_lap_map_points:
                    self.track_map_points = list(self.current_lap_map_points)
                    self.braking_zones_best = list(self.current_lap_braking_zones)
                    self.ers_zones_best = list(self.current_lap_ers_zones)

                self.speak(f"Personal best! {mins} minute {secs_rem:.3f}. New reference lap set.", Priority.TIMING,
                           key='lap_time', ttl=10, cooldown=0)
            else:
                diff_s = (last_ms - self.delta.best_lap_time_ms) / 1000.0
                self.speak(f"Lap complete: {mins} {secs_rem:.3f}. Plus {diff_s:.2f} seconds to your best.", Priority.TIMING,
                           key='lap_time', ttl=10, cooldown=0)

        self.lap_start_fuel = self.latest_status.fuel_in_tank if self.latest_status else None

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
        self.current_lap_map_points.clear()
        self.current_lap_braking_zones.clear()
        self.current_lap_ers_zones.clear()
        self.last_map_dist = -999.0

    # ──────────────────────────────────────────
    # Learning Engine: tyre wear, fuel and ML predictions
    # ──────────────────────────────────────────

    def reset_map(self):
        """Manually wipe the recorded track map/braking/ERS trail (driver pressed 'Reset Map').

        Unlike a new-lap or new-session reset, this does not touch lap timing, tyre or
        AI state — it only clears the coordinate trail so a dirty out-lap or a spin that
        left stray points doesn't junk up the map for the rest of the session.
        """
        self.current_lap_map_points.clear()
        self.track_map_points.clear()
        self.current_lap_braking_zones.clear()
        self.current_lap_ers_zones.clear()
        self.braking_zones_best.clear()
        self.ers_zones_best.clear()
        self.last_map_dist = -999.0
        update_web_state({'track_map': [], 'braking_zones': [], 'ers_zones': []})

    def update_car_status(self, status: CarStatusData):
        self.latest_status = status
        if self.lap_start_fuel is None:
            self.lap_start_fuel = status.fuel_in_tank

    def update_car_damage(self, damage: CarDamageData):
        self.latest_damage = damage

    def update_participants(self, participants):
        """Driver names from the Participants packet, so radio calls can say who is where."""
        self.radio.set_driver_names({index: p.radio_name for index, p in enumerate(participants)
                                     if p.radio_name})
        self.race_control.update_participants(participants)

    def update_speed_trap(self, speed_kph: float):
        """Record the fastest Speed Trap event ('SPTP') seen so far on the current lap."""
        if self.current_lap_speed_trap_kph is None or speed_kph > self.current_lap_speed_trap_kph:
            self.current_lap_speed_trap_kph = speed_kph

    def _record_lap_stats(self, lap_num: int, lap_time_ms: int, eligible: bool, sectors: dict,
                          speed_trap_kph: float | None = None):
        """Log tyre wear, fuel burn and sector deltas for the lap that just finished."""
        status, damage = self.latest_status, self.latest_damage
        deltas = {sector: ms - self.best_sector_times[sector]
                  for sector, ms in sectors.items() if sector in self.best_sector_times}
        stats = {
            'lap_num': lap_num, 'lap_time_ms': lap_time_ms, 'eligible': eligible,
            'compound': status.actual_tyre_compound if status else None,
            'tyre_age_laps': status.tyres_age_laps if status else None,
            'wear': tuple(damage.tyres_wear) if damage else None,
            'fuel_start': self.lap_start_fuel,
            'fuel_end': status.fuel_in_tank if status else None,
            'sectors': dict(sectors), 'sector_deltas': deltas,
            'speed_trap_kph': speed_trap_kph,
        }
        self.session_lap_stats.append(stats)
        self._ml_cache = None
        if self.history and self.capture_id:
            try:
                self.history.record_lap_stats(self.capture_id, stats)
            except (OSError, ValueError, sqlite3.Error) as error:
                print(f'[History] Could not save lap stats: {error}')

    def _history_args(self):
        return (self.session, self.capture_id) if self.history and self.session else None

    def tyre_model(self):
        """Wear curve for the current compound from this session plus earlier sessions."""
        status = self.latest_status
        if status is None:
            return None
        compound = status.actual_tyre_compound
        samples = [(st['tyre_age_laps'], max(st['wear']))
                   for st in self.session_lap_stats
                   if st['compound'] == compound and st['wear'] is not None
                   and st['tyre_age_laps'] is not None]
        args = self._history_args()
        if args:
            try:
                samples += self.history.tyre_wear_samples(args[0], compound, exclude_capture=args[1])
            except sqlite3.Error as error:
                print(f'[History] Could not load wear history: {error}')
        return strategy_ml.fit_wear_curve(samples)

    def pit_prediction(self):
        """ML pit window (lap numbers) from tyre age, or None until there is enough data."""
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None:
            return None
        if not session_modes.has_pit_strategy(self.session_mode):
            return None     # no stops in qualifying or time trial
        key = (lap.current_lap_num, status.actual_tyre_compound, len(self.session_lap_stats))
        if self._ml_cache and self._ml_cache[0] == key:
            return self._ml_cache[1]
        prediction = strategy_ml.predict_pit_window(
            self.tyre_model(), status.tyres_age_laps, lap.current_lap_num, session.total_laps)
        self._ml_cache = (key, prediction)
        return prediction

    def fuel_prediction(self) -> dict | None:
        """Learned fuel burn per lap and how many laps the current load lasts."""
        status = self.latest_status
        if status is None:
            return None
        samples = [st['fuel_start'] - st['fuel_end'] for st in self.session_lap_stats
                   if st['eligible'] and st['fuel_start'] is not None and st['fuel_end'] is not None]
        args = self._history_args()
        if args:
            try:
                samples += self.history.fuel_use_samples(args[0], exclude_capture=args[1])
            except sqlite3.Error as error:
                print(f'[History] Could not load fuel history: {error}')
        per_lap = strategy_ml.fuel_per_lap(samples)
        result = {'fuel_kg': round(status.fuel_in_tank, 1), 'per_lap_kg': per_lap,
                  'game_laps_remaining': round(status.fuel_remaining_laps, 1), 'laps_of_fuel': None,
                  'margin_laps': None}
        if per_lap:
            result['laps_of_fuel'] = round(status.fuel_in_tank / per_lap, 1)
            lap, session = self.latest_lap, self.session
            if lap and session and session.total_laps > 0:
                result['margin_laps'] = round(
                    result['laps_of_fuel'] - (session.total_laps - lap.current_lap_num + 1), 1)
        return result

    def pace_trend(self) -> float | None:
        """Fuel-corrected tyre fall-off in s/lap for the current stint."""
        status = self.latest_status
        if status is None:
            return None
        stint = [(st['tyre_age_laps'], st['lap_time_ms'], st['fuel_start'])
                 for st in self.session_lap_stats
                 if st['eligible'] and st['compound'] == status.actual_tyre_compound
                 and st['tyre_age_laps'] is not None]
        return strategy_ml.pace_degradation(stint[-8:])

    def tire_health(self):
        """Instant physics-model tyre read: grip, health and overheat risk, no lap history needed.

        Complements pit_prediction() (which learns a wear curve from observed laps and needs
        several of them): this scores the current instant from a model of the compound, so it
        is available from the first telemetry frame of a fresh set.
        """
        if self.latest_status is None:
            return None
        return physics_models.evaluate_tire(
            self.latest_status.compound_name,
            self.latest_status.tyres_age_laps,
            self.latest_damage.tyres_wear if self.latest_damage else None,
            self.latest_telemetry.tyres_surface_temperature if self.latest_telemetry else None)

    def opponent_threats(self) -> list[dict]:
        """Nearby cars ranked by how much pressure they put on the player right now."""
        return [t.__dict__ for t in self.radio.opponent_threats]

    def race_report(self) -> list[dict]:
        """This session's completed laps for the HUD race-report table: lap number, lap
        time, each sector, speed trap and tyre compound. Reads from the in-memory
        session_lap_stats (so it works even with no history DB attached) rather than
        querying SQLite on every telemetry frame.
        """
        report = []
        for st in self.session_lap_stats:
            sectors = st.get('sectors') or {}
            report.append({
                'lap_num': st['lap_num'], 'lap_time_ms': st['lap_time_ms'],
                'eligible': bool(st.get('eligible')),
                'compound': ACTUAL_COMPOUNDS.get(st.get('compound'), '--'),
                's1_ms': sectors.get(1), 's2_ms': sectors.get(2), 's3_ms': sectors.get(3),
                'speed_trap_kph': st.get('speed_trap_kph'),
            })
        return report

    def build_context(self) -> dict:
        """Snapshot of live telemetry plus ML predictions, for the AI prompt and fallbacks."""
        t, lap, s, st, dmg = (self.latest_telemetry, self.latest_lap, self.session,
                              self.latest_status, self.latest_damage)
        ctx: dict = {'track': self.track_name,
                     'session_type': SESSION_TYPES.get(s.session_type, 'Session') if s else None,
                     'session_mode': self.session_mode,
                     'weather': WEATHER_NAMES.get(s.weather) if s else None,
                     'track_temp_c': s.track_temperature if s else None,
                     'air_temp_c': s.air_temperature if s else None,
                     'safety_car': bool(s and s.safety_car_status)}
        if lap:
            ctx.update(lap=lap.current_lap_num, total_laps=s.total_laps if s else None,
                       position=lap.car_position, last_lap_ms=lap.last_lap_time_ms or None,
                       best_lap_ms=self.delta.best_lap_time_ms or None,
                       delta_to_best_ms=self.delta.last_reported_delta_ms,
                       race_pace_ms=self.radio.race_pace_ms,
                       gap_ahead_s=self.radio.gap_ahead_s, gap_behind_s=self.radio.gap_behind_s,
                       car_ahead_name=self.radio.driver_name(self.radio.car_ahead_index),
                       car_behind_name=self.radio.driver_name(self.radio.car_behind_index),
                       game_pit_window=self.radio.pit_window)
        if t:
            ctx['tyre_surface_c'] = dict(zip(WHEEL_CODES, t.tyres_surface_temperature))
            ctx['tyre_inner_c'] = dict(zip(WHEEL_CODES, t.tyres_inner_temperature))
        if dmg:
            ctx['tyre_wear_pct'] = {c: round(w, 1) for c, w in zip(WHEEL_CODES, dmg.tyres_wear)}
        if st:
            ctx.update(compound=st.compound_name, tyre_age_laps=st.tyres_age_laps)
        prediction = self.pit_prediction()
        ctx['pit_prediction'] = prediction.as_dict() if prediction else None
        ctx['fuel'] = self.fuel_prediction()
        ctx['pace_trend_s_per_lap'] = self.pace_trend()
        tire = self.tire_health()
        ctx['tire_health'] = None if tire is None else {
            'grip_factor': tire.grip_factor, 'health_score': tire.health_score,
            'deg_rate_s_per_lap': tire.deg_rate_s_per_lap, 'remaining_laps': tire.remaining_laps,
            'performance_loss_s': tire.performance_loss_s, 'overheat_risk': tire.overheat_risk.value,
            'limiting_factor': tire.limiting_factor}
        threats = self.opponent_threats()
        ctx['opponent_threats'] = [
            {'threat_level': t['threat_level'], 'threat_score': t['threat_score'],
             'gap_s': t['gap_s'], 'pace_delta_s': t['pace_delta_s'],
             'laps_to_catch': t['laps_to_catch'], 'factors': t['factors'],
             'driver_name': t['driver_name']}
            for t in threats] or None
        ctx['recent_race_events'] = [event['text'] for event in self.race_control.recent(5)] or None
        if self.session_lap_stats:
            last = self.session_lap_stats[-1]
            ctx['last_lap_sectors_ms'] = last['sectors'] or None
            ctx['last_lap_sector_deltas_ms'] = last['sector_deltas'] or None
        return ctx

    # ──────────────────────────────────────────
    # Main Update
    # ──────────────────────────────────────────

    def update_telemetry(self, telemetry: CarTelemetryData, lap: LapData):
        """Main update called on each telemetry packet."""
        self.latest_telemetry, self.latest_lap = telemetry, lap
        self.update_delta(lap, telemetry)
        self.analyze_tyres(telemetry, lap)
        self.check_braking_points(lap, telemetry)
        self.update_corners(telemetry, lap)
        self.radio.update_strategy(lap)

        # Priority Interrupts
        if self.session:
            self.radio.update_safety_car(self.session.safety_car_status)
            self.radio.update_weather_alert(self.session.weather)
        if self.latest_damage:
            self.radio.update_damage_alert(
                self.latest_damage.worst_tyre_wear,
                self.latest_damage.front_left_wing + self.latest_damage.front_right_wing
            )

        # Live Motion & Track Map coordinates
        car_x = self.latest_motion.world_position_x if self.latest_motion else 0.0
        car_z = self.latest_motion.world_position_z if self.latest_motion else 0.0
        car_yaw = self.latest_motion.yaw if self.latest_motion else 0.0
        g_lat = round(self.latest_motion.g_force_lateral, 2) if self.latest_motion else 0.0
        g_lon = round(self.latest_motion.g_force_longitudinal, 2) if self.latest_motion else 0.0

        active_map = self.track_map_points if self.track_map_points else self.current_lap_map_points
        active_braking = self.braking_zones_best if self.braking_zones_best else self.current_lap_braking_zones
        active_ers = self.ers_zones_best if self.ers_zones_best else self.current_lap_ers_zones

        # Stream snapshot to Web HUD
        update_web_state({
            "connected": True,
            "last_update_monotonic": time.monotonic(),
            "track_name": self.track_name,
            "track_id": self.track_id,
            "track_length": self.session.track_length if self.session else 0,
            "speed": telemetry.speed,
            "gear": telemetry.gear,
            "rpm": telemetry.engine_rpm,
            "drs": bool(telemetry.drs),
            "throttle": round(telemetry.throttle, 2),
            "brake": round(telemetry.brake, 2),
            "steer": round(telemetry.steer, 2),
            "tyre_surface_temps": list(telemetry.tyres_surface_temperature),
            "tyre_inner_temps": list(telemetry.tyres_inner_temperature),
            "tyre_pressures": list(telemetry.tyres_pressure),
            "tyre_wear": list(self.latest_damage.tyres_wear) if self.latest_damage else [0.0, 0.0, 0.0, 0.0],
            "tyre_compound": self.latest_status.compound_name if self.latest_status else "Medium",
            "ers_store_energy": self.latest_status.ers_store_energy if self.latest_status else 2000000.0,
            "ers_deploy_mode": self.latest_status.ers_deploy_mode if self.latest_status else 1,
            "ers_percent": int(round((self.latest_status.ers_store_energy / 4000000.0) * 100)) if self.latest_status else 50,
            "engine_temp": telemetry.engine_temperature,
            "lap_num": lap.current_lap_num,
            "total_laps": self.session.total_laps if self.session else 0,
            "position": lap.car_position,
            "lap_time_ms": lap.current_lap_time_ms,
            "last_lap_time_ms": lap.last_lap_time_ms,
            "best_lap_time_ms": self.delta.best_lap_time_ms,
            "delta_ms": self.delta.last_reported_delta_ms,
            "lap_distance": round(lap.lap_distance, 1),
            "sector": lap.sector,
            "current_sectors_ms": dict(self.current_sector_times),
            "personal_sectors_ms": dict(self.best_sector_times),
            "gap_ahead_s": self.radio.gap_ahead_s,
            "gap_behind_s": self.radio.gap_behind_s,
            "car_ahead_name": self.radio.driver_name(self.radio.car_ahead_index),
            "car_behind_name": self.radio.driver_name(self.radio.car_behind_index),
            "race_pace_ms": self.radio.race_pace_ms,
            "pit_window": self.radio.pit_window,
            "opponent_threats": self.opponent_threats(),
            "race_control": self.race_control.recent(),
            "race_report": self.race_report(),
            "tire_health": (lambda t: None if t is None else {
                'grip_factor': t.grip_factor, 'health_score': t.health_score,
                'remaining_laps': t.remaining_laps, 'overheat_risk': t.overheat_risk.value,
                'limiting_factor': t.limiting_factor})(self.tire_health()),
            "safety_car": ("VSC" if self.session and self.session.safety_car_status == 2
                           else "SAFETY CAR" if self.session and self.session.safety_car_status == 1
                           else "NONE"),
            "weather": WEATHER_NAMES.get(self.session.weather, "Clear") if self.session else "Clear",
            "corner_analysis": self.corner_diagnostics,
            "speed_trace_current": self.delta.current_lap_speeds[-120:],
            "speed_trace_best": self.delta.best_lap_speeds,
            "car_x": car_x,
            "car_z": car_z,
            "car_yaw": car_yaw,
            "g_force_lat": g_lat,
            "g_force_lon": g_lon,
            "track_map": active_map[-400:],
            "braking_zones": active_braking[-100:],
            "ers_zones": active_ers[-100:]
        })
