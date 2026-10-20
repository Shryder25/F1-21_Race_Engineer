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
import radio_phrases as say
from web_hud import update_web_state, add_voice_log
import strategy_ml
import race_strategy
import rival_strategy
import fitted_models
import race_sim
import track_maps
from session_history import SessionLog, lap_rows, stint_rows


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
# Strategy calls that cannot wait for the driver to ask: covering an undercut or taking one is
# a this-lap decision, so they break on-demand silence like a real pit wall would.
PROACTIVE_KEYS = frozenset({'tyre_temperature', 'safety_car', 'safety_car_in', 'weather_rain', 'critical_wear',
                            'wing_damage', 'penalty', 'flag_yellow', 'flag_red', 'rival_pit', 'undercut',
                            'rival_preempt'})
# Only react to a rival's stop when they were this close; further away it changes nothing.
RIVAL_REACT_GAP_S = 6.0
# Rival stop forecasts: say who is about to box when their predicted stop is this close
# (laps), and pre-empt an undercut from the car behind when it is due this lap or next.
RIVAL_FORECAST_LAPS = 2
RIVAL_PREEMPT_LAPS = 1
# A pre-emptive stop needs the rival's own tyre wear behind the prediction and at least this
# chance that they box by the end of next lap. No rival calls on lap one or behind the Safety Car.
RIVAL_PREEMPT_PROB = 0.5
# Cars this many places either side of the player go into the strategy summary.
RIVAL_SUMMARY_POSITIONS = 3

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
                 on_demand: bool = False, fitted: fitted_models.ModelStore | None = None,
                 race_simulation: str = 'off', sim_runs: int = race_sim.DEFAULT_RUNS,
                 track_store: track_maps.TrackMapStore | None = None):
        self.voice = voice
        # Saved outline + pit lane per circuit (memory-only unless main.py gives it a file).
        self.track_store = track_store if track_store is not None else track_maps.TrackMapStore()
        # Monte Carlo race simulator (race_sim.py): 'async' live, 'sync' in tests, 'off'.
        self.sim_runner = race_sim.SimRunner(race_simulation, sim_runs)
        self.field_tracker = race_sim.FieldTracker()
        self.field_laps = None
        self.player_index = 0
        # Tyre/fuel models fitted to the game (tools/fit_models.py); empty = built-in priors.
        self.fitted = fitted if fitted is not None else fitted_models.ModelStore()
        self._history_records: list | None = None
        self._session_captures: set[str] = set()
        self._forecast_cache: tuple | None = None
        self.on_demand = on_demand
        self.latest_telemetry: CarTelemetryData | None = None
        self.latest_lap: LapData | None = None
        self.latest_status: CarStatusData | None = None
        self.latest_damage: CarDamageData | None = None
        self.all_status: list[CarStatusData] | None = None    # every car, for rival tyre age/compound
        self.compounds_used: set[str] = set()
        self._strategy_cache: tuple | None = None
        self.last_strategy_call: dict | None = None
        self.undercut_calls: set[tuple] = set()
        # Every car's stints, wear and stops, and the next-stop/compound model built on them.
        self.rivals = rival_strategy.RivalTracker()
        self.rival_calls: set[tuple] = set()
        self.player_start_compound: str | None = None
        self.lap_start_fuel: float | None = None
        self.current_lap_speed_trap_kph: float | None = None
        self.last_speed_trap_kph: float | None = None
        self.session_lap_stats: list[dict] = []
        self._ml_cache: tuple | None = None
        self.session_mode = session_modes.UNKNOWN
        self._corner_key: tuple | None = None
        self.radio = RaceRadio(self.speak)
        self.phrases = self.radio.phrases
        self.race_control = RaceControl(self.speak)
        self.history = history
        self.capture_id: str | None = None
        # Every car's laps, tyre stints, classification and setups (packets 11, 8 and 5).
        self.session_log = SessionLog()
        self._session_log_capture: str | None = None
        self.session_uid: int | None = None
        self.session_context = None
        self.weather_group: str | None = None
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
        # Lap distance of each trail point, so a flashback can cut the trail back.
        self.current_lap_map_dists: list[float] = []
        self.current_lap_braking_dists: list[float] = []
        self.current_lap_ers_dists: list[float] = []
        self.track_outline: list[list[float]] = []      # whole circuit, stored or first clean lap
        self.pit_lane_points: list[list[float]] = []
        self.current_pit_points: list[list[float]] = []
        self.map_lap_has_pit = False
        self.map_rewind_pending = False
        self.flashback_count = 0
        self._map_revision = 0
        self._sent_map_revision = -1

    # ──────────────────────────────────────────
    # Session Handling
    # ──────────────────────────────────────────

    def update_session(self, session: SessionData, session_uid: int | None = None, player_index=None):
        # Weather is deliberately not part of the session identity: rain arriving mid-race is
        # the same session (a reset here would wipe the race state and swallow the rain call).
        context = (session_uid, session.track_id, session.track_length, session.formula,
                   session.session_type, player_index)
        weather_group = TelemetryHistory.weather_group(session.weather)
        if context != self.session_context:
            self.session_context = context
            self.weather_group = weather_group
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
            self.session_log = SessionLog(session_uid, player_index)
            self._session_log_capture = None
            self._ml_cache = None
            self._forecast_cache = None
            self._history_records = None
            self._session_captures = set()
            self.latest_status = self.latest_damage = None
            self.all_status = None
            self.compounds_used.clear()
            self._strategy_cache = None
            self.last_strategy_call = None
            self.field_tracker.reset()
            self.field_laps = None
            self.sim_runner.reset()
            self.undercut_calls.clear()
            self.rivals.reset()
            self.rival_calls.clear()
            self.player_start_compound = None
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
            self._clear_trail_distances()
            self.current_pit_points.clear()
            self.map_lap_has_pit = False
            self.map_rewind_pending = False
            self._load_track_map(session)
            self.radio.set_driver_names({})
            if hasattr(self.voice, 'clear'):
                self.voice.clear()
            self._load_history_reference(session, session_uid)
            update_web_state({
                'connected': False, 'corner_analysis': [], 'voice_log': [],
                'speed_trace_current': [], 'speed_trace_best': self.delta.best_lap_speeds,
                'best_lap_time_ms': self.delta.best_lap_time_ms, 'delta_ms': None,
                'gap_ahead_s': None, 'gap_behind_s': None, 'race_pace_ms': None,
                'car_ahead_name': None, 'car_behind_name': None,
                'pit_window': None, 'race_strategy': None, 'last_update_monotonic': None,
                'speed': 0, 'gear': 0, 'rpm': 0, 'drs': False, 'throttle': 0, 'brake': 0,
                'lap_num': 0, 'lap_time_ms': 0, 'last_lap_time_ms': 0, 'position': 0,
                'tyre_surface_temps': [0] * 4, 'tyre_inner_temps': [0] * 4,
                'tyre_pressures': [0] * 4, 'engine_temp': 0,
                'personal_sectors_ms': self.best_sector_times,
            })
            self.radio.reset(session)
            self.race_control.reset(session)
            update_web_state({'race_control': []})
        elif weather_group != self.weather_group:
            self._change_conditions(session)
            self.weather_group = weather_group
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

    def _load_history_reference(self, session: SessionData, session_uid):
        """Open a history capture for these conditions and load the best reference lap for them."""
        if not (self.history and session.track_id >= 0):
            return
        try:
            self.capture_id = self.history.start_session(session, session_uid,
                                                         self.session_log.player_index)
            self._session_captures.add(self.capture_id)
            self._history_records = None
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

    def _change_conditions(self, session: SessionData):
        """Dry -> intermediate -> wet within one session: keep the race, swap the references.

        A dry best lap, sector or corner is no yardstick on a wet track, and laps recorded
        from here on belong with wet history, so only the timing references and the history
        capture change. Race state (gaps, positions, strategy, radio) carries on.
        """
        best = DeltaTracker()
        self.delta.best_lap_times, self.delta.best_lap_speeds = best.best_lap_times, best.best_lap_speeds
        self.delta.best_lap_time_ms, self.delta.best_lap_num = 0, 0
        self.delta.current_lap_invalid = True          # this lap was run in mixed conditions
        self.best_corner_data = {}
        self.best_sector_times = {}
        self._load_history_reference(session, self.session_uid)
        update_web_state({'speed_trace_best': self.delta.best_lap_speeds,
                          'best_lap_time_ms': self.delta.best_lap_time_ms,
                          'personal_sectors_ms': self.best_sector_times, 'capture_id': self.capture_id})

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
        if self.latest_lap and self.latest_telemetry and not self.map_rewind_pending:
            lap = self.latest_lap
            if lap.pit_status:
                # Pit lane is its own path, never part of the circuit outline.
                self.map_lap_has_pit = True
                track_maps.append_pit_point(self.current_pit_points, x, z)
                return
            if self.current_pit_points:
                self._finish_pit_lane()
            if self.delta.current_lap_num and lap.current_lap_num != self.delta.current_lap_num:
                return                        # lap change not processed yet: keep last lap intact
            dist = lap.lap_distance
            if dist < self.last_map_dist - 30.0:
                self._trim_trail(dist)        # car went backwards along the lap: a rewind
            if abs(dist - self.last_map_dist) >= 6.0:
                self.last_map_dist = dist
                self.current_lap_map_points.append([x, z])
                self.current_lap_map_dists.append(dist)
                self._map_revision += 1
                if self.latest_telemetry.brake > 0.08:
                    self.current_lap_braking_zones.append([x, z, round(self.latest_telemetry.brake, 2)])
                    self.current_lap_braking_dists.append(dist)
                if self.latest_status and (self.latest_status.ers_deploy_mode > 0 or self.latest_status.ers_deployed_this_lap > 0):
                    self.current_lap_ers_zones.append([x, z])
                    self.current_lap_ers_dists.append(dist)

    # ──────────────────────────────────────────
    # Track map: stored outline, pit lane, flashbacks
    # ──────────────────────────────────────────

    def _load_track_map(self, session: SessionData):
        stored = self.track_store.get(session.track_id, session.track_length)
        self.track_outline = stored['outline']
        self.pit_lane_points = stored['pit_lane']
        self._map_revision += 1
        self._push_track_map()

    def _push_track_map(self):
        update_web_state({'track_outline': self.track_outline,
                          'pit_lane': self.pit_lane_points,
                          'track_map_source': 'stored' if self.track_outline else 'recording'})

    def _clear_trail_distances(self):
        self.current_lap_map_dists.clear()
        self.current_lap_braking_dists.clear()
        self.current_lap_ers_dists.clear()

    def _trim_trail(self, dist: float):
        """Drop every trail point recorded beyond `dist` on this lap (the rewound future)."""
        for points, dists in ((self.current_lap_map_points, self.current_lap_map_dists),
                              (self.current_lap_braking_zones, self.current_lap_braking_dists),
                              (self.current_lap_ers_zones, self.current_lap_ers_dists)):
            keep = next((i for i, d in enumerate(dists) if d > dist), len(dists))
            del points[keep:], dists[keep:]
        self.last_map_dist = self.current_lap_map_dists[-1] if self.current_lap_map_dists else -999.0
        self._map_revision += 1

    def _finish_pit_lane(self):
        """Car left the pit lane: keep the path as the pit lane if this track has none yet."""
        points, self.current_pit_points = self.current_pit_points, []
        if self.pit_lane_points or len(points) < track_maps.MIN_PIT_POINTS or not self.session:
            return
        self.pit_lane_points = points
        self.track_store.save(self.track_id, self.session.track_length, pit_lane=points)
        self._push_track_map()

    def _capture_outline(self, lap_num: int, new_lap_num: int):
        """After a full lap, store it as the circuit outline if this track has none yet."""
        if (self.track_outline or self.map_lap_has_pit or not self.session
                or new_lap_num != lap_num + 1):
            return
        outline = track_maps.build_outline(self.current_lap_map_dists, self.current_lap_map_points,
                                           self.session.track_length)
        if outline:
            self.track_outline = outline
            self.track_store.save(self.track_id, self.session.track_length, outline=outline)
            self._push_track_map()

    def on_flashback(self):
        """The game rewound (F1 2021 flashback). Stop recording until a lap packet from the
        restored timeline says where the car is, then cut the trail back to there."""
        self.map_rewind_pending = True
        self.current_pit_points.clear()
        self.flashback_count += 1
        update_web_state({'flashback_count': self.flashback_count})

    def _resolve_map_rewind(self, lap: LapData):
        self.map_rewind_pending = False
        if lap.current_lap_num == self.delta.current_lap_num:
            self._trim_trail(lap.lap_distance)
        # A rewind into an earlier lap is handled by the lap change in update_delta.

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
            names = " and ".join(w[0].lower() for w in critical_wheels)
            verb = "is" if len(critical_wheels) == 1 else "are"
            self.speak(
                f"Careful, {names} {verb} overheating, {say.number_words(int(critical_wheels[0][1]))} degrees. "
                f"Cool it down, lift and coast into the next braking zone.",
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
                self.speak("Tyres are running hot all round. Back off for a couple of corners "
                           "and let them recover.", Priority.NORMAL)
            elif fronts:
                avg = sum(w[1] for w in fronts) // len(fronts)
                self.speak(f"Fronts are getting hot, {say.number_words(int(avg))}. Be gentle on turn-in.",
                           Priority.NORMAL)
            elif rears:
                avg = sum(w[1] for w in rears) // len(rears)
                self.speak(f"Rears are getting hot, {say.number_words(int(avg))}. Smooth on the throttle "
                           f"out of the slow corners.", Priority.NORMAL)
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
                    amount = say.time_amount(delta_ms / 1000.0)
                    if delta_ms < -100:
                        self.speak(self.phrases.pick('delta_up', (
                            f"{amount.capitalize()} up on your best. Keep it going.",
                            f"You're {amount} up, good lap so far.",
                        )), Priority.LOW, key='live_delta', ttl=6, cooldown=14)
                    elif delta_ms > 150:
                        self.speak(f"{amount.capitalize()} down on your best.", Priority.LOW,
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
        if prev_lap_num > 0:
            self._capture_outline(prev_lap_num, lap.current_lap_num)
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
            spoken = say.lap_time(last_ms)

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

                self.speak(self.phrases.pick('personal_best', (
                    f"Good lap! {spoken.capitalize()}, that's a personal best.",
                    f"{spoken.capitalize()}. Personal best, nice job.",
                )), Priority.TIMING,
                           key='lap_time', ttl=10, cooldown=0)
            else:
                diff_s = (last_ms - self.delta.best_lap_time_ms) / 1000.0
                if diff_s < 0.15:
                    message = f"{spoken.capitalize()}. Just {say.time_amount(diff_s)} off your best."
                else:
                    message = f"{spoken.capitalize()}. {say.time_amount(diff_s).capitalize()} off your best."
                self.speak(message, Priority.TIMING,
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
        self._clear_trail_distances()
        self.map_lap_has_pit = bool(lap.pit_status)
        self.last_map_dist = -999.0
        self._map_revision += 1

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
        self._clear_trail_distances()
        self.current_pit_points.clear()
        self.last_map_dist = -999.0
        # Also forget the stored outline/pit lane, so a bad one can be mapped again.
        self.track_outline, self.pit_lane_points = [], []
        if self.session:
            self.track_store.forget(self.track_id)
        self._map_revision += 1
        self._sent_map_revision = self._map_revision
        update_web_state({'track_map': [], 'braking_zones': [], 'ers_zones': []})
        self._push_track_map()

    def update_car_status(self, status: CarStatusData, all_status: list[CarStatusData] | None = None):
        self.latest_status = status
        if all_status is not None:
            self.all_status = all_status
            self.rivals.update_status(all_status)
        compound = race_strategy.normalise_compound(status.compound_name)
        if compound:
            self.compounds_used.add(compound)
            if self.player_start_compound is None:
                self.player_start_compound = compound
        if self.lap_start_fuel is None:
            self.lap_start_fuel = status.fuel_in_tank

    def update_field(self, laps, player_index: int, session_time: float):
        """Every car's Lap Data: rival stints for the stop model, and line-crossing times and
        clean-lap pace for the race simulator."""
        self.rivals.update_laps(laps, player_index)
        self.field_laps, self.player_index = laps, player_index
        if self.session is None or self.session_mode != session_modes.RACE:
            return
        self.field_tracker.update(laps, session_time, self.all_status,
                                  neutralised=self.session.safety_car_status in (1, 2))

    def update_car_damage(self, damage: CarDamageData, all_damage: list[CarDamageData] | None = None):
        self.latest_damage = damage
        if all_damage is not None:
            self.rivals.update_damage(all_damage)

    def update_participants(self, participants):
        """Driver names from the Participants packet, so radio calls can say who is where."""
        self.radio.set_driver_names({index: p.radio_name for index, p in enumerate(participants)
                                     if p.radio_name})
        self.race_control.update_participants(participants)
        for index, p in enumerate(participants):
            if p.name:
                self.session_log.names[index] = p.radio_name or p.name

    # ── Every car's laps and stints (Session History), result and setups ──

    def update_session_history(self, history):
        """One car's Session History packet: log its laps and tyre stints."""
        self._sync_session_log_capture()
        changed = self.session_log.update_history(history)
        if changed is None or not (self.history and self.capture_id):
            return
        laps, stints, last_lap = changed
        try:
            self.history.record_car_history(self.capture_id, history.car_index, laps, stints, last_lap,
                                            driver=self.session_log.names.get(history.car_index))
        except (OSError, ValueError, sqlite3.Error) as error:
            print(f'[History] Could not save car history: {error}')

    def update_final_classification(self, classification):
        self._sync_session_log_capture()
        if not self.session_log.update_classification(classification):
            return
        if self.history and self.capture_id:
            try:
                self.history.record_final_classification(self.capture_id,
                                                         self.session_log.classification_rows())
            except (OSError, ValueError, sqlite3.Error) as error:
                print(f'[History] Could not save final classification: {error}')

    def update_car_setups(self, setups):
        self._sync_session_log_capture()
        changed = self.session_log.update_setups(setups)
        if not (changed and self.history and self.capture_id):
            return
        try:
            for index, setup in changed.items():
                self.history.record_car_setup(self.capture_id, index, setup.as_dict())
        except (OSError, ValueError, sqlite3.Error) as error:
            print(f'[History] Could not save car setup: {error}')

    @property
    def player_setup(self):
        """The player's latest setup from the Car Setups packet, or None."""
        return self.session_log.setups.get(self.session_log.player_index)

    def _sync_session_log_capture(self):
        """A new history capture mid-session (rain arrived) gets every car's rows so far."""
        if self._session_log_capture == self.capture_id:
            return
        self._session_log_capture = self.capture_id
        if not (self.history and self.capture_id):
            return
        log = self.session_log
        try:
            for car, history in log.histories.items():
                self.history.record_car_history(self.capture_id, car, lap_rows(history),
                                                stint_rows(history), driver=log.names.get(car))
            if log.classification:
                self.history.record_final_classification(self.capture_id, log.classification_rows())
            for index, setup in log.setups.items():
                self.history.record_car_setup(self.capture_id, index, setup.as_dict())
        except (OSError, ValueError, sqlite3.Error) as error:
            print(f'[History] Could not save car history: {error}')

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
        self._forecast_cache = None
        if self.history and self.capture_id:
            try:
                self.history.record_lap_stats(self.capture_id, stats)
            except (OSError, ValueError, sqlite3.Error) as error:
                print(f'[History] Could not save lap stats: {error}')

    # ── Fitted tyre / fuel models (fitted_models.py): ranges, not single numbers ──

    def _past_records(self) -> list:
        """Laps from earlier sessions on this track that the fitted store has not seen yet."""
        if self._history_records is None:
            records = []
            if self.history and self.session:
                try:
                    records, _ = fitted_models.laps_from_history(
                        self.history, self.fitted.consumed_captures | self._session_captures,
                        track_id=self.session.track_id)
                except sqlite3.Error as error:
                    print(f'[History] Could not load lap history: {error}')
                records = [r for r in records if r.track_length == self.session.track_length]
            self._history_records = records
        return self._history_records

    def _session_records(self) -> list:
        s = self.session
        if s is None:
            return []
        return fitted_models.records_from_lap_rows(
            self.session_lap_stats, source='live', session='live', track_id=s.track_id,
            track_length=s.track_length, session_type=s.session_type,
            race_laps=self._race_laps(), wet=s.weather >= 3)

    def _race_laps(self) -> int:
        s = self.session
        return s.total_laps if s and session_modes.has_pit_strategy(self.session_mode) else 0

    def model_forecast(self) -> dict | None:
        """Wear, degradation and fuel models for the car's current tyres and track.

        Each model is the stored fit (or its fallback), refined with earlier sessions in the
        history database and then with this stint's / session's own laps. Cached per lap.
        """
        status, session = self.latest_status, self.session
        if status is None or session is None:
            return None
        key = (status.actual_tyre_compound, status.tyres_age_laps, len(self.session_lap_stats),
               session.track_id, session.weather >= 3)
        if self._forecast_cache and self._forecast_cache[0] == key:
            return self._forecast_cache[1]
        compound = status.actual_tyre_compound
        past = [r for r in self._past_records()]
        session_records = self._session_records()
        stint = fitted_models.current_stint(session_records, compound, status.tyres_age_laps)
        ckey = fitted_models.compound_key(compound)
        same_compound_past = [r for r in past + session_records
                              if r.compound == ckey and r.stint not in {x.stint for x in stint}]

        wear_model = fitted_models.refine(
            self.fitted.wear_model(session.track_id, compound, self._race_laps()), same_compound_past)
        stint_wear = wear_model.condition(*fitted_models.stint_increments(stint))
        deg_model = fitted_models.refine(self.fitted.deg_model(session.track_id, compound),
                                         same_compound_past)
        deg_ages, deg_times = fitted_models.stint_times(stint)
        deg = deg_model.condition(deg_ages, deg_times)
        fuel_model = fitted_models.refine(
            self.fitted.fuel_model(session.track_id, session.track_length, self._race_laps()),
            [r for r in past if r.session != 'live'])
        session_fuel = fuel_model.condition([u for r in session_records
                                             if (u := r.fuel_used) is not None])
        forecast = {'stint_wear': stint_wear, 'deg': deg, 'deg_laps': len(deg_ages),
                    'deg_basis': deg_model.basis, 'deg_fit_laps': deg_model.n_laps,
                    'fuel': session_fuel, 'compound': ckey}
        self._forecast_cache = (key, forecast)
        return forecast

    def wear_forecast(self) -> fitted_models.StintWear | None:
        """The wear model conditioned on this stint, or None until there is any data at all."""
        forecast = self.model_forecast()
        if forecast is None:
            return None
        stint = forecast['stint_wear']
        if stint.model.n_laps == 0 and stint.stint_laps < 2:
            return None             # built-in estimate only: not enough to quote a window from
        return stint

    def tyre_model(self):
        """Backwards-compatible name for the current stint's wear forecast."""
        return self.wear_forecast()

    def _worst_wear(self) -> float:
        return max(self.latest_damage.tyres_wear) if self.latest_damage else 0.0

    def pit_prediction(self):
        """Pit window (lap ranges) from the fitted wear model, or None until there is data."""
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None:
            return None
        if not session_modes.has_pit_strategy(self.session_mode):
            return None     # no stops in qualifying or time trial
        key = (lap.current_lap_num, status.actual_tyre_compound, status.tyres_age_laps,
               len(self.session_lap_stats))
        if self._ml_cache and self._ml_cache[0] == key:
            return self._ml_cache[1]
        prediction = strategy_ml.predict_pit_window(
            self.wear_forecast(), status.tyres_age_laps, self._worst_wear(), lap.current_lap_num,
            session.total_laps)
        self._ml_cache = (key, prediction)
        return prediction

    def fuel_prediction(self) -> dict | None:
        """Fuel burn per lap and the margin at the flag, as ranges (low end = the worry)."""
        status = self.latest_status
        if status is None:
            return None
        result = {'fuel_kg': round(status.fuel_in_tank, 1), 'per_lap_kg': None,
                  'game_laps_remaining': round(status.fuel_remaining_laps, 1), 'laps_of_fuel': None,
                  'margin_laps': None, 'per_lap_kg_range': None, 'laps_of_fuel_range': None,
                  'margin_laps_range': None, 'margin_kg_range': None, 'basis': None,
                  'level': fitted_models.DEFAULT_LEVEL}
        forecast = self.model_forecast()
        if forecast is None:
            return result
        fuel = forecast['fuel']
        result['basis'] = fuel.model.basis
        result['fit_laps'] = fuel.model.n_laps
        result['session_laps'] = fuel.session_laps
        per = fuel.per_lap()
        laps_of_fuel = fuel.laps_of_fuel(status.fuel_in_tank)
        result['per_lap_kg_range'] = per.as_list(3)
        result['laps_of_fuel_range'] = laps_of_fuel.as_list(1)
        if fuel.model.n_laps or fuel.session_laps:     # a fit or this session's laps: quote it
            result['per_lap_kg'] = round(per.mid, 3)
            result['laps_of_fuel'] = round(laps_of_fuel.mid, 1)
        lap, session = self.latest_lap, self.session
        if lap and session and session.total_laps > 0:
            to_go = session.total_laps - lap.current_lap_num + 1
            kg, laps = fuel.margin(status.fuel_in_tank, to_go)
            result['margin_kg_range'] = kg.as_list(2)
            result['margin_laps_range'] = laps.as_list(1)
            if result['per_lap_kg'] is not None:
                result['margin_laps'] = round(laps.mid, 1)
        return result

    def deg_forecast(self) -> fitted_models.Range | None:
        forecast = self.model_forecast()
        return forecast['deg'] if forecast else None

    def pace_trend(self) -> float | None:
        """Fuel-corrected tyre fall-off in s/lap for the current stint (fitted model + stint laps).

        None until the stint has three clean laps and there is no fitted model to lean on.
        """
        forecast = self.model_forecast()
        if forecast is None:
            return None
        if forecast['deg_laps'] < 3 and forecast['deg_fit_laps'] == 0:
            return None
        return round(forecast['deg'].mid, 3)

    def model_summary(self) -> dict | None:
        """Ranges for the HUD and the AI prompt: wear, degradation, pit window, fuel."""
        forecast = self.model_forecast()
        status = self.latest_status
        if forecast is None or status is None:
            return None
        stint = forecast['stint_wear']
        model = stint.model
        age, wear = status.tyres_age_laps, self._worst_wear()
        prediction = self.pit_prediction()
        fuel = self.fuel_prediction() or {}
        return {
            'level': fitted_models.DEFAULT_LEVEL,
            'compound': forecast['compound'], 'compound_name': status.compound_name,
            'basis': model.basis, 'fit_laps': model.n_laps, 'fit_stints': model.n_stints,
            'stint_laps': stint.stint_laps, 'simulated_fit': bool(model.simulated and model.n_laps),
            'wear_now_pct': round(wear, 1),
            'wear_per_lap_pct': stint.rate(age).as_list(2),
            'wear_in_5_laps_pct': stint.wear_after(age, wear, 5).as_list(1),
            'deg_s_per_lap': forecast['deg'].as_list(3), 'deg_basis': forecast['deg_basis'],
            'pit_ideal_lap': prediction.ideal_lap_range if prediction else None,
            'pit_latest_lap': prediction.latest_lap_range if prediction else None,
            'finish_without_stop_prob': prediction.finish_without_stop_prob if prediction else None,
            'fuel_per_lap_kg': fuel.get('per_lap_kg_range'),
            'fuel_margin_laps': fuel.get('margin_laps_range'),
            'fuel_basis': fuel.get('basis'),
        }

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

    # ──────────────────────────────────────────
    # Race strategy (stops, compounds, undercut / overcut)
    # ──────────────────────────────────────────

    def _strategy_calibration(self, case: str = 'expected') -> tuple[float, float]:
        """(deg_scale, life_scale) for the expected case, or the ends of the model ranges:
        'worse' = tyres fall off faster and wear out sooner, 'better' = the opposite."""
        status = self.latest_status
        prediction = self.pit_prediction()
        deg = self.pace_trend()
        to_limit = prediction.laps_to_latest if prediction else None
        if case != 'expected':
            spread = self.deg_forecast()
            if deg is not None and spread is not None:
                deg = spread.high if case == 'worse' else spread.low
            if prediction is not None and prediction.latest_lap_range:
                lap = self.latest_lap.current_lap_num
                soonest, _mid, latest = prediction.latest_lap_range
                bound = soonest if case == 'worse' else latest
                if bound is not None:
                    to_limit = bound - lap
                elif case == 'better':
                    to_limit = None if to_limit is None else to_limit * 1.5
        return race_strategy.calibration(status.compound_name, status.tyres_age_laps, deg, to_limit)

    def strategy_plan(self):
        """Best stop count and compound sequence for the rest of the race, or None outside a race."""
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None or self.session_mode != session_modes.RACE:
            return None
        laps_remaining = session.total_laps - lap.current_lap_num + 1
        key = (lap.current_lap_num, status.compound_name, status.tyres_age_laps,
               frozenset(self.compounds_used), session.track_id, laps_remaining, len(self.session_lap_stats))
        if self._strategy_cache and self._strategy_cache[0] == key:
            return self._strategy_cache[1]
        args = (laps_remaining, status.compound_name, status.tyres_age_laps, lap.current_lap_num,
                race_strategy.pit_loss_s(session.track_id), self.compounds_used)
        scales = self._strategy_calibration()
        plan = race_strategy.plan_race(*args, *scales)
        # The same plan at both ends of the tyre-model ranges: does the call survive?
        cases = {}
        for case in ('worse', 'better'):
            case_scales = self._strategy_calibration(case)
            cases[case] = plan if case_scales == scales else race_strategy.plan_race(*args, *case_scales)
        self._strategy_cache = (key, plan, scales, cases)
        return plan

    def strategy_uncertainty(self) -> dict | None:
        """How the plan changes if the tyres turn out at either end of the model's range."""
        plan = self.strategy_plan()
        if plan is None or not self._strategy_cache or len(self._strategy_cache) < 4:
            return None
        cases = self._strategy_cache[3]
        out = {'level': fitted_models.DEFAULT_LEVEL}
        stops, next_stop = [], []
        for case in ('worse', 'expected', 'better'):
            option = (cases.get(case) if case != 'expected' else plan)
            option = option.best if option is not None else None
            out[case] = None if option is None else {
                'stops': option.stops, 'compounds': option.compounds, 'pit_laps': option.pit_laps}
            if option is not None:
                stops.append(option.stops)
                if option.pit_laps:
                    next_stop.append(option.pit_laps[0])
        out['stops_agree'] = len(set(stops)) <= 1
        out['next_stop_lap_range'] = [min(next_stop), max(next_stop)] if next_stop else None
        return out

    def strategy_scales(self) -> tuple[float, float]:
        """(deg_scale, life_scale) the current plan was built with."""
        return self._strategy_cache[2] if self._strategy_cache else (1.0, 1.0)

    def _rival_status(self, index):
        if self.all_status is None or index is None or not 0 <= index < len(self.all_status):
            return None
        return self.all_status[index]

    # ── Rival stop / compound predictions ──

    def _race_context(self) -> rival_strategy.RaceContext | None:
        session, lap = self.session, self.latest_lap
        if session is None or self.session_mode != session_modes.RACE or session.total_laps <= 0:
            return None
        window = self.radio.pit_window
        return rival_strategy.RaceContext(
            track_id=session.track_id, total_laps=session.total_laps, track_length=session.track_length,
            pit_loss_s=race_strategy.pit_loss_s(session.track_id),
            game_window=tuple(window) if window else None,
            player_start_compound=self.player_start_compound,
            player_stops=getattr(lap, 'num_pit_stops', 0) if lap else 0, weather=session.weather)

    def rival_prediction(self, index) -> rival_strategy.RivalPrediction | None:
        """Next stop lap and compound for one rival (None outside a race or without data)."""
        ctx = self._race_context()
        if ctx is None or index is None or index == self.rivals.player_index:
            return None
        return self.rivals.predict(index, ctx, self.fitted, self.radio.driver_name(index))

    def rival_predictions(self) -> list[rival_strategy.RivalPrediction]:
        """Every rival's predicted next stop and compound, in race order."""
        ctx = self._race_context()
        if ctx is None:
            return []
        return self.rivals.predict_all(ctx, self.fitted, self.radio.driver_names,
                                       exclude=self.rivals.player_index)

    def _rival_stop_in_laps(self, prediction, lap_num: int) -> int | None:
        """Laps a rival is expected to stay out after we box this lap (soonest end of the range)."""
        if prediction is None:
            return None
        if not prediction.will_stop:
            return race_strategy.MAX_COVER_LAPS
        soonest = prediction.stop_lap_range[0]
        return None if soonest is None else soonest - lap_num

    def update_rival_forecasts(self, lap: LapData, plan, deg_scale: float, life_scale: float):
        """Proactive calls from the rival stop model, for the cars directly ahead and behind.

        * Car behind due to box this lap or next with an undercut that would work: pre-empt it
          (box now) when our own stop is close, otherwise warn to be ready to cover.
        * Otherwise, when the car ahead or behind is due within a couple of laps, a heads-up:
          "Hamilton likely to box lap 22, onto hards."
        Each car gets at most one call per stint.
        """
        status = self.latest_status
        if lap.current_lap_num < 2 or (self.session and self.session.safety_car_status):
            return
        wet = race_strategy.normalise_compound(status.compound_name) in race_strategy.WET_COMPOUNDS
        preempt_token = ('pre_empt', lap.num_pit_stops)
        for side, index, gap in (('behind', self.radio.car_behind_index, self.radio.gap_behind_s),
                                 ('ahead', self.radio.car_ahead_index, self.radio.gap_ahead_s)):
            prediction = self.rival_prediction(index)
            if prediction is None or not prediction.will_stop or gap is None or gap > RIVAL_REACT_GAP_S:
                continue
            due_in = prediction.laps_to_stop(lap.current_lap_num)
            token = (index, prediction.stops_done)
            if token in self.rival_calls or due_in is None:
                continue
            who = f'{prediction.name} {side}' if prediction.name else f'Car {side}'
            onto = f' for {prediction.next_compound.lower()}s' if prediction.next_compound else ''
            if (side == 'behind' and not wet and prediction.basis == 'wear'
                    and prediction.prob_stop_by(lap.current_lap_num + RIVAL_PREEMPT_LAPS) >= RIVAL_PREEMPT_PROB
                    and preempt_token not in self.rival_calls
                    and plan.best.stops > 0 and plan.laps_remaining > 3):
                threat = race_strategy.undercut_gain(status.compound_name, status.tyres_age_laps,
                                                     prediction.next_compound, 1, deg_scale, life_scale)
                if threat >= gap - race_strategy.CALL_MARGIN_S:
                    self.rival_calls.update((token, preempt_token))
                    my_stop_in = plan.best.pit_laps[0] - lap.current_lap_num
                    if my_stop_in <= race_strategy.UNDERCUT_WINDOW_LAPS:
                        call = race_strategy.StrategyCall(
                            'pre_empt', f'Expect {who} to box this lap or next{onto}. Undercut threat, '
                                        f'box this lap to stay ahead, box box.', threat, gap, index)
                        self._make_strategy_call(call, 'rival_preempt')
                    else:
                        call = race_strategy.StrategyCall(
                            'cover_ready', f'{who} likely to box lap {prediction.stop_lap}{onto}. '
                                           'Undercut would work, be ready to cover.', threat, gap, index)
                        self._make_strategy_call(call, 'rival_forecast')
                    continue
            if due_in <= RIVAL_FORECAST_LAPS:
                self.rival_calls.add(token)
                call = race_strategy.StrategyCall(
                    'rival_forecast', f'{who} likely to box lap {prediction.stop_lap}{onto}.', None, gap, index)
                self.last_strategy_call = {**call.as_dict(), 'lap': lap.current_lap_num}
                self.speak(call.message, Priority.RACE, key='rival_forecast', ttl=8, cooldown=20)

    def rival_strategy_summary(self) -> dict | None:
        """Nearby rivals' predicted stops for the HUD and the AI prompt."""
        lap = self.latest_lap
        predictions = self.rival_predictions()
        if lap is None or not predictions:
            return None
        nearby = [p for p in predictions if p.position and abs(p.position - lap.car_position) <= RIVAL_SUMMARY_POSITIONS]
        out = {'field': self.rivals.field.as_dict(), 'nearby': [], 'ahead': None, 'behind': None}
        for p in nearby:
            row = {k: getattr(p, k) for k in (
                'car_index', 'name', 'position', 'compound', 'tyre_age', 'wear_pct', 'will_stop',
                'stop_lap', 'stop_lap_range', 'next_compound', 'next_compound_probs', 'basis', 'summary')}
            out['nearby'].append(row)
            if p.car_index == self.radio.car_ahead_index:
                out['ahead'] = row
            elif p.car_index == self.radio.car_behind_index:
                out['behind'] = row
        return out

    def update_strategy_calls(self, lap: LapData):
        """React to nearby rivals pitting and call an undercut when one is on."""
        events = self.radio.pop_rival_pit_events()
        status = self.latest_status
        if self.session_mode != session_modes.RACE or status is None or lap.pit_status:
            return
        plan = self.strategy_plan()
        if plan is None:
            return
        deg_scale, life_scale = self.strategy_scales()
        for event in events:
            if abs(event['gap_s']) > RIVAL_REACT_GAP_S:
                continue
            rival = self._rival_status(event['car_index'])
            # The stop model's last prediction for this car (made before it turned in).
            last = self.rivals.last_prediction(event['car_index'])
            expected = last.next_compound if last is not None else None
            call = race_strategy.react_to_rival_stop(
                'ahead' if event['gap_s'] > 0 else 'behind', event['gap_s'], status.compound_name,
                status.tyres_age_laps, rival.compound_name if rival else None, None,
                plan.laps_remaining, plan.best.stops, deg_scale, life_scale, event['car_index'],
                rival_name=self.radio.driver_name(event['car_index']), assumed_new_compound=expected)
            self._make_strategy_call(call, 'rival_pit')

        self.update_rival_forecasts(lap, plan, deg_scale, life_scale)
        if plan.best.stops == 0 or race_strategy.normalise_compound(status.compound_name) in race_strategy.WET_COMPOUNDS:
            return          # no undercut maths on a wet track: the crossover decides stops there
        token = (lap.num_pit_stops, lap.current_lap_num)
        if token in self.undercut_calls:
            return
        rival = self._rival_status(self.radio.car_ahead_index)
        if rival is None:
            return
        call = race_strategy.undercut_opportunity(
            self.radio.gap_ahead_s, status.compound_name, status.tyres_age_laps, rival.compound_name,
            rival.tyres_age_laps, plan.best.pit_laps[0] - lap.current_lap_num, plan.laps_remaining,
            deg_scale, life_scale, self.radio.car_ahead_index,
            rival_stop_in_laps=self._rival_stop_in_laps(
                self.rival_prediction(self.radio.car_ahead_index), lap.current_lap_num))
        if call is not None:
            self.undercut_calls.add(token)
            self._make_strategy_call(call, 'undercut')

    def _make_strategy_call(self, call, key: str):
        self.last_strategy_call = {**call.as_dict(), 'lap': self.latest_lap.current_lap_num if self.latest_lap else None}
        self.speak(call.message, Priority.STRATEGY, key=key, ttl=10, cooldown=0)

    # ── Monte Carlo race simulation (race_sim.py) ──

    def race_sim_state(self) -> race_sim.RaceState | None:
        """The race as it stands, for the simulator: every running car's gap, tyres and pace."""
        if self.session_mode != session_modes.RACE or self.latest_lap is None:
            return None
        laps = self.field_laps
        if not laps:                          # player only (no field data yet)
            laps, index = [self.latest_lap], 0
        else:
            index = self.player_index
        statuses = self.all_status if self.all_status else None
        if statuses is None and self.latest_status is not None:
            statuses = [None] * len(laps)
            statuses[index] = self.latest_status
        predictions = None
        rival_predictions = getattr(self, 'rival_predictions', None)
        if callable(rival_predictions):       # rival stop model, when there is one
            try:
                predictions = {p.car_index: p for p in (rival_predictions() or [])}
            except Exception:                 # a broken rival model must not stop the sim
                predictions = None
        names = {i: self.radio.driver_name(i) for i in range(len(laps))}
        return race_sim.build_state(laps, statuses, index, self.session, self.field_tracker, names,
                                    predictions, self.fitted)

    def update_race_simulation(self):
        """Re-run the simulation once a lap (and when the race changes: SC, a stop, new tyres)."""
        if not self.sim_runner.enabled or self.session_mode != session_modes.RACE:
            return
        status, lap, session = self.latest_status, self.latest_lap, self.session
        if status is None or lap is None or session is None or lap.pit_status:
            return
        key = (lap.current_lap_num, session.safety_car_status, lap.num_pit_stops, status.compound_name)
        if key == self.sim_runner.latest_key:
            return
        plan = self.strategy_plan()
        candidates = race_sim.candidates_from_plan(plan)
        if not candidates:
            return
        state = self.race_sim_state()
        if state is None:
            return
        deg_scale, life_scale = self.strategy_scales()
        seed = (lap.current_lap_num * 1009 + session.safety_car_status * 101 + lap.num_pit_stops) & 0xFFFFFFFF
        self.sim_runner.submit(key, state, candidates, models=self.fitted, seed=seed,
                               pit_loss_mean=race_strategy.pit_loss_s(session.track_id),
                               player_deg_scale=deg_scale, player_life_scale=life_scale,
                               player_deg=self.deg_forecast())

    def race_simulation(self) -> dict | None:
        """Latest simulation summary, or None when it is off, not run yet or out of date."""
        result = self.sim_runner.latest
        lap = self.latest_lap
        if result is None or lap is None or self.session_mode != session_modes.RACE:
            return None
        if lap.current_lap_num - result.lap > 3:
            return None
        return result.summary()

    def strategy_summary(self) -> dict | None:
        """Plan plus the latest undercut/overcut call, for the HUD and the AI prompt."""
        plan = self.strategy_plan()
        if plan is None:
            return None
        summary = plan.as_dict()
        summary['radio'] = race_strategy.describe_plan(plan)
        uncertainty = self.strategy_uncertainty()
        summary['uncertainty'] = uncertainty
        if uncertainty and not uncertainty['stops_agree'] and uncertainty['worse']:
            worse = uncertainty['worse']['stops']
            if worse != plan.best.stops:
                words = {0: 'no stop', 1: 'one stop', 2: 'two stops', 3: 'three stops'}
                summary['radio'] += f" If the tyres go off faster, it's {words.get(worse, worse)}."
        simulation = self.race_simulation()
        summary['simulation'] = simulation
        if simulation and simulation.get('radio'):
            summary['radio'] += ' ' + simulation['radio']
        summary['last_call'] = self.last_strategy_call
        status, rival = self.latest_status, self._rival_status(self.radio.car_ahead_index)
        wet = race_strategy.normalise_compound(status.compound_name if status else None) in race_strategy.WET_COMPOUNDS
        if self.radio.gap_ahead_s is not None and rival is not None and status is not None and not wet:
            deg_scale, life_scale = self.strategy_scales()
            lap = self.latest_lap
            cover = race_strategy.cover_laps(self._rival_stop_in_laps(
                self.rival_prediction(self.radio.car_ahead_index), lap.current_lap_num), plan.laps_remaining)
            gain = race_strategy.undercut_gain(rival.compound_name, rival.tyres_age_laps,
                                               race_strategy.next_compound(status.compound_name),
                                               cover, deg_scale, life_scale)
            summary['undercut_on_car_ahead'] = {
                'gap_s': round(self.radio.gap_ahead_s, 2), 'gain_s': gain,
                'works': gain - self.radio.gap_ahead_s >= race_strategy.CALL_MARGIN_S,
                'rival_compound': rival.compound_name, 'rival_tyre_age': rival.tyres_age_laps,
                'rival_stays_out_laps': cover}
        else:
            summary['undercut_on_car_ahead'] = None
        rivals = self.rival_strategy_summary()
        summary['rivals'] = rivals
        lines = [f"{side.capitalize()}: {rivals[side]['summary']}"
                 for side in ('ahead', 'behind') if rivals and rivals.get(side)]
        summary['rival_radio'] = ' '.join(lines) or None
        if self.session and self.session.pit_stop_rejoin_position:
            summary['game_rejoin_position'] = self.session.pit_stop_rejoin_position
        return summary

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
        ctx['tyre_fuel_model'] = self.model_summary()
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
        ctx['strategy'] = self.strategy_summary()
        rivals = (ctx['strategy'] or {}).get('rivals')
        ctx['rival_predictions'] = [r['summary'] for r in rivals['nearby']] if rivals and rivals['nearby'] else None
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
        if self.map_rewind_pending:
            self._resolve_map_rewind(lap)
        self.update_delta(lap, telemetry)
        self.analyze_tyres(telemetry, lap)
        self.check_braking_points(lap, telemetry)
        self.update_corners(telemetry, lap)
        self.radio.update_strategy(lap)
        self.update_strategy_calls(lap)
        self.update_race_simulation()

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

        # The live trail is only needed until the circuit outline is known, and only sent
        # when it changed: the whole lap (not a 400-point tail) so nothing gets erased.
        map_update = {}
        if self._map_revision != self._sent_map_revision:
            self._sent_map_revision = self._map_revision
            trail = [] if self.track_outline else (self.track_map_points or self.current_lap_map_points)
            map_update = {
                'track_map': trail,
                'current_lap_trail': self.current_lap_map_points,
                'braking_zones': self.braking_zones_best or self.current_lap_braking_zones,
                'ers_zones': self.ers_zones_best or self.current_lap_ers_zones,
            }

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
            "race_strategy": self.strategy_summary(),
            "model_ranges": self.model_summary(),
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
            "pit_status": lap.pit_status,
            **map_update,
        })
