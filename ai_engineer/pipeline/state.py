"""Engine state: every per-session and per-lap value the pipeline stages read and write.

The stages (``pipeline/*.py``) are layers of one engine object, so they share this state.
Everything is declared and typed here, in one place, so a new per-session field has an
obvious home: add it to ``EngineState.__init__`` with its default, and to the reset block
in ``IngestStage.update_session`` if it must clear when the session changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import fitted_models
import pit_wall
import race_briefing
import race_sim
import radio_discipline
import rival_strategy
import session_modes
import strategy_replan
import track_maps
import track_wetness
from history import TelemetryHistory
from packets import CarDamageData, CarMotionData, CarStatusData, CarTelemetryData, LapData, SessionData
from pipeline.config import EngineConfig
from race_events import RaceControl
from race_radio import RaceRadio
from radio_phrases import Phrasebook
from session_history import SessionLog
from voice_engine import Priority

JSON = dict[str, Any]

WHEEL_NAMES = ('Rear Left', 'Rear Right', 'Front Left', 'Front Right')
WHEEL_CODES = ('RL', 'RR', 'FL', 'FR')  # same order as the telemetry tuples
ERS_STORE_MAX_J = 4_000_000.0  # a full ERS battery (4 MJ)
ERS_MODE_NAMES = {0: 'none', 1: 'medium', 2: 'hotlap', 3: 'overtake'}

WEATHER_NAMES = {0: 'clear', 1: 'light cloud', 2: 'overcast', 3: 'light rain', 4: 'heavy rain', 5: 'storm'}

# Corner definitions for Silverstone (Track ID 7), with hand-measured apex speeds.
SILVERSTONE_CORNERS: list[JSON] = [
    {'name': 'Turn 1 Abbey', 'start': 180, 'end': 450, 'ref_apex': 275, 'ref_brake': 290},
    {'name': 'Turn 3 Village', 'start': 700, 'end': 1000, 'ref_apex': 110, 'ref_brake': 850},
    {'name': 'Turn 4 The Loop', 'start': 1000, 'end': 1200, 'ref_apex': 88, 'ref_brake': 1050},
    {'name': 'Turn 6 Brooklands', 'start': 1600, 'end': 1950, 'ref_apex': 140, 'ref_brake': 1750},
    {'name': 'Turn 7 Luffield', 'start': 1950, 'end': 2250, 'ref_apex': 115, 'ref_brake': 2000},
    {'name': 'Turn 9 Copse', 'start': 2700, 'end': 3050, 'ref_apex': 255, 'ref_brake': 2820},
    {'name': 'Maggots-Becketts', 'start': 3250, 'end': 3850, 'ref_apex': 220, 'ref_brake': 3350},
    {'name': 'Turn 15 Stowe', 'start': 4800, 'end': 5200, 'ref_apex': 185, 'ref_brake': 4950},
    {'name': 'Turn 16 Vale Chicane', 'start': 5300, 'end': 5600, 'ref_apex': 95, 'ref_brake': 5420},
    {'name': 'Turn 17-18 Club', 'start': 5600, 'end': 5850, 'ref_apex': 150, 'ref_brake': 5680},
]


class VoiceOutput(Protocol):
    """What the engine needs from a voice: VoiceEngine live, a recorder in tests."""

    def say(
        self,
        message: str,
        priority: Priority = ...,
        *,
        key: str | None = ...,
        ttl: float = ...,
        cooldown: float | None = ...,
    ) -> bool: ...


def wheel_group(wheels: list[int]) -> str:
    """Wheel indices (RL, RR, FL, FR order) as a driver would hear them: 'Fronts', 'Front left'."""
    ordered = sorted(set(wheels))
    if ordered == [0, 1, 2, 3]:
        return 'All four tyres'
    if ordered == [2, 3]:
        return 'Fronts'
    if ordered == [0, 1]:
        return 'Rears'
    names = [WHEEL_NAMES[i].lower() for i in ordered]
    text = names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]
    return text[0].upper() + text[1:]


def _four() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0]


@dataclass
class TyreState:
    hot_since: list[float] = field(default_factory=_four)
    cold_since: list[float] = field(default_factory=_four)
    last_alert_time: float = 0.0
    # Rate-of-change tracking for the critical-overheat alert: a momentary noisy spike
    # (common in the game's telemetry, e.g. a brief lockup reading) should not trigger a
    # radio call on its own — only a temperature that is genuinely climbing fast should.
    last_temp_sample: list[float] | None = None
    last_temp_sample_time: float = 0.0
    last_critical_alert_time: float = 0.0
    # Per-lap record for the end-of-lap summary (the driver can see live temps on screen,
    # so mid-lap the engineer only speaks for a critical overheat, once per lap at most).
    lap_peak: list[float] = field(default_factory=_four)
    lap_hot_secs: list[float] = field(default_factory=_four)
    lap_cold_secs: list[float] = field(default_factory=_four)
    critical_called_this_lap: bool = False
    last_summary: str | None = None
    laps_since_summary: int = 0


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
    current_lap_samples: list[JSON] = field(default_factory=list)
    current_lap_num: int = 0
    current_lap_invalid: bool = False
    recorded_from_start: bool = False
    last_distance: float | None = None
    last_elapsed_ms: int = 0
    last_sample_distance: float = -999.0

    # Timing
    last_delta_report_time: float = 0.0
    last_reported_delta_ms: int | None = None


class EngineState:
    """Every value the engine keeps, typed, with its starting default."""

    def __init__(
        self,
        voice: VoiceOutput,
        history: TelemetryHistory | None = None,
        on_demand: bool = False,
        fitted: fitted_models.ModelStore | None = None,
        race_simulation: str = 'off',
        sim_runs: int = race_sim.DEFAULT_RUNS,
        track_store: track_maps.TrackMapStore | None = None,
        replan: strategy_replan.ReplanConfig | None = None,
        config: EngineConfig | None = None,
        calls_per_lap: int | None = radio_discipline.DEFAULT_CALLS_PER_LAP,
        load_thresholds: radio_discipline.LoadThresholds | None = None,
    ) -> None:
        self.config = config if config is not None else EngineConfig()
        self.voice = voice
        # Radio discipline: a per-lap budget for proactive calls, and no talking through
        # high-load corners or heavy braking (the voice queue holds calls for the straight).
        # main.py passes config.toml's values; a bare engine uses radio_discipline's defaults.
        self.call_budget = radio_discipline.CallBudget(calls_per_lap)
        self.load_gate = radio_discipline.DriverLoadGate(load_thresholds)
        # Saved outline + pit lane per circuit (memory-only unless main.py gives it a file).
        self.track_store = track_store if track_store is not None else track_maps.TrackMapStore()
        # Monte Carlo race simulator (race_sim.py): 'async' live, 'sync' in tests, 'off'.
        self.sim_runner = race_sim.SimRunner(race_simulation, sim_runs)
        self.field_tracker = race_sim.FieldTracker()
        # Live re-planning on top of the simulator: Plan A, triggers and the Plan B call.
        self.replanner = strategy_replan.Replanner(replan)
        # How wet the track is and where it is heading: weather physics + the field's lap times.
        self.wetness = track_wetness.TrackWetness()
        self._last_sim_key: tuple[Any, ...] | None = None  # key of the last re-run sent
        self._last_event_submit = -1e9  # monotonic time of the last event-only re-run
        self.field_laps: list[LapData] | None = None
        self.player_index = 0
        # Tyre/fuel models fitted to the game (tools/fit_models.py); empty = built-in priors.
        self.fitted = fitted if fitted is not None else fitted_models.ModelStore()
        self._history_records: list[fitted_models.LapRecord] | None = None
        self._session_captures: set[str] = set()
        self._forecast_cache: tuple[tuple[Any, ...], JSON] | None = None
        # Speak only when asked (plus the proactive keys). main.py passes the config / --chatty
        # choice; a bare engine (tests, tools) talks freely.
        self.on_demand = on_demand
        self.latest_telemetry: CarTelemetryData | None = None
        self.latest_lap: LapData | None = None
        self.latest_status: CarStatusData | None = None
        self.latest_damage: CarDamageData | None = None
        self.all_status: list[CarStatusData] | None = None  # every car, for rival tyre age/compound
        self.compounds_used: set[str] = set()
        # (key, plan, (deg_scale, life_scale), {'worse': plan, 'better': plan})
        self._strategy_cache: tuple[Any, ...] | None = None
        self.last_strategy_call: JSON | None = None
        self.undercut_calls: set[tuple[int, int]] = set()
        # Every car's stints, wear and stops, and the next-stop/compound model built on them.
        self.rivals = rival_strategy.RivalTracker()
        self.briefing = race_briefing.RaceBriefing()  # chatty mode's per-lap race-info call
        self.rival_calls: set[tuple[Any, ...]] = set()
        self.player_start_compound: str | None = None
        self.lap_start_fuel: float | None = None
        self.current_lap_speed_trap_kph: float | None = None
        self.last_speed_trap_kph: float | None = None
        self.session_lap_stats: list[JSON] = []
        self._ml_cache: tuple[tuple[Any, ...], Any] | None = None
        self.session_mode = session_modes.UNKNOWN
        self._corner_key: tuple[int, int] | None = None
        self.radio = RaceRadio(self.speak)
        self.phrases: Phrasebook = self.radio.phrases
        self.race_control = RaceControl(self.speak)
        self.history = history
        self.capture_id: str | None = None
        # Every car's laps, tyre stints, classification and setups (packets 11, 8 and 5).
        self.session_log = SessionLog()
        self._session_log_capture: str | None = None
        self.session_uid: int | None = None
        self.session_context: tuple[Any, ...] | None = None
        self.weather_group: str | None = None
        self.best_sector_times: dict[int, int] = {}
        self.current_sector_times: dict[int, int] = {}
        # The last finished lap's sectors (S3 = lap time - S1 - S2, since the game never
        # sends S3 on its own) and the personal bests they were up against.
        self.last_lap_sectors: dict[int, int] = {}
        self.last_lap_sector_refs: dict[int, int] = {}
        # Lap distance (m) where sectors 2 and 3 start, learned while driving.
        self.sector_starts: dict[int, float] = {}

        self.tyre_state = TyreState()
        self.driver_profile: JSON | None = None
        self.delta = DeltaTracker()
        self.session: SessionData | None = None
        self.track_id: int = -1
        self.track_name: str = 'Waiting for session'
        self.braking_points: list[tuple[float, str, str]] = []
        self.announced_braking: set[str] = set()
        self.last_lap_num: int = 0
        self.last_position: int = 0
        self.last_sector: int = 0
        self.session_started = False
        self.setup_advice_given = False

        # Corner tracking
        self.corners: list[JSON] = []
        self.current_corner_data: dict[str, JSON] = {}
        self.best_corner_data: dict[str, JSON] = {}
        self.best_corner_runs: dict[str, JSON] = {}  # quickest run through each corner this session
        self.corner_diagnostics: list[JSON] = []
        # (lap distance, lap time ms, speed, brake) of the previous frame
        self.previous_corner_sample: tuple[float, int, int, float] | None = None
        self.brake_start_distance: int | None = None

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
        self.track_outline: list[list[float]] = []  # whole circuit, stored or first clean lap
        self.pit_lane_points: list[list[float]] = []
        self.current_pit_points: list[list[float]] = []
        self.field_mapper: track_maps.FieldMapper | None = None
        self.field_pit_points: dict[int, list[list[float]]] = {}  # other cars' pit runs
        self.car_teams: dict[int, int] = {}
        self.map_lap_has_pit = False
        self.map_rewind_pending = False
        self.flashback_count = 0
        self._map_revision = 0
        self.field_cars: list[list[float]] = []
        self._sent_map_revision = -1

        # Pit wall: full-field timing tower, gap-to-leader history, tyre stints, sim odds
        self.pit_wall = pit_wall.PitWall()
        self._sent_pit_wall = (-1, -1)
        self._odds_result: race_sim.SimResult | None = None
        self._odds: JSON | None = None
        self.odds_history: list[JSON] = []
        self._sent_best_trace: list[list[float]] | None = None
        self._sent_best_len = -1

    def speak(self, message: str, priority: Priority = Priority.NORMAL, **options: Any) -> None:
        """Queue a radio call. Implemented by the radio stage (pipeline/radio.py)."""
        raise NotImplementedError
