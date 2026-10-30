"""
Analysis Engine — The Race Engineer's Brain
=============================================
Real-time telemetry analysis producing:
  1. Verbal callouts (Braking points, tyre temps, lap/sector deltas, setups)
  2. Live Web HUD telemetry streaming
  3. Corner-by-corner analysis of "Why You Are Slower"
  4. Real-time speed trace comparison vs session best

The work is split into pipeline stages (``pipeline/``, see its ``__init__`` for the map).
This module wires them together: ``AnalysisEngine`` is the full stack of stages, and
``update_telemetry`` runs them in order for every telemetry frame.
"""

from __future__ import annotations

from packets import CarTelemetryData, LapData
from pipeline import ingest, laps, outputs, radio, strategy
from pipeline.config import DEFAULT_PROACTIVE_KEYS, EngineConfig
from pipeline.ingest import IngestStage
from pipeline.state import (
    ERS_MODE_NAMES,
    ERS_STORE_MAX_J,
    SILVERSTONE_CORNERS,
    WEATHER_NAMES,
    WHEEL_CODES,
    WHEEL_NAMES,
    DeltaTracker,
    TyreState,
)
from pipeline.state import wheel_group as _wheel_group

# Defaults, kept importable under their old names (tests and tools use some of them).
# The live values come from config.toml through ``engine.config``.
_DEFAULTS = EngineConfig()
TYRE_TEMP_HIGH = _DEFAULTS.tyres.high_c
TYRE_TEMP_LOW = _DEFAULTS.tyres.low_c
TYRE_TEMP_CRITICAL = _DEFAULTS.tyres.critical_c
TYRE_TEMP_RISE_RATE_C_PER_S = _DEFAULTS.tyres.rise_rate_c_per_s
TYRE_TEMP_ALERT_COOLDOWN_SECS = _DEFAULTS.tyres.alert_cooldown_s
TYRE_CRITICAL_SUSTAIN_SECS = _DEFAULTS.tyres.critical_sustain_s
TYRE_SUMMARY_MIN_SECS = _DEFAULTS.tyres.summary_min_s
TYRE_SUMMARY_REPEAT_LAPS = _DEFAULTS.tyres.summary_repeat_laps
PROFILE_REFRESH_LAPS = _DEFAULTS.laps.profile_refresh_laps
BRAKING_WARN_DISTANCE = _DEFAULTS.radio.braking_warn_distance_m
DELTA_SAMPLE_INTERVAL = _DEFAULTS.laps.delta_sample_interval_m
DELTA_REPORT_INTERVAL = _DEFAULTS.radio.delta_report_interval_s
PROACTIVE_KEYS = frozenset(DEFAULT_PROACTIVE_KEYS)
RIVAL_REACT_GAP_S = _DEFAULTS.strategy.rival_react_gap_s
RIVAL_FORECAST_LAPS = _DEFAULTS.strategy.rival_forecast_laps
RIVAL_PREEMPT_LAPS = _DEFAULTS.strategy.rival_preempt_laps
RIVAL_PREEMPT_PROB = _DEFAULTS.strategy.rival_preempt_prob
RIVAL_SUMMARY_POSITIONS = _DEFAULTS.strategy.rival_summary_positions

# Stage modules that read the clock (``time``): the golden-session harness swaps it for a
# simulated clock in each of them.
TIMED_MODULES = (radio, strategy, outputs, laps, ingest)

__all__ = [
    'AnalysisEngine',
    'DeltaTracker',
    'TyreState',
    'EngineConfig',
    'PROACTIVE_KEYS',
    'SILVERSTONE_CORNERS',
    'ERS_MODE_NAMES',
    'ERS_STORE_MAX_J',
    'WEATHER_NAMES',
    'WHEEL_CODES',
    'WHEEL_NAMES',
    '_wheel_group',
    'TYRE_TEMP_HIGH',
    'TYRE_TEMP_LOW',
    'TYRE_TEMP_CRITICAL',
    'TYRE_TEMP_RISE_RATE_C_PER_S',
    'TYRE_TEMP_ALERT_COOLDOWN_SECS',
    'TYRE_CRITICAL_SUSTAIN_SECS',
    'TYRE_SUMMARY_MIN_SECS',
    'TYRE_SUMMARY_REPEAT_LAPS',
    'PROFILE_REFRESH_LAPS',
    'BRAKING_WARN_DISTANCE',
    'DELTA_SAMPLE_INTERVAL',
    'DELTA_REPORT_INTERVAL',
    'RIVAL_REACT_GAP_S',
    'RIVAL_FORECAST_LAPS',
    'RIVAL_PREEMPT_LAPS',
    'RIVAL_PREEMPT_PROB',
    'RIVAL_SUMMARY_POSITIONS',
]


class AnalysisEngine(IngestStage):
    """The brain of the AI Race Engineer: every pipeline stage, run frame by frame.

    Constructor arguments are those of ``pipeline.state.EngineState``; ``config`` carries the
    tuneable thresholds from config.toml (defaults when omitted).
    """

    def update_telemetry(self, telemetry: CarTelemetryData, lap: LapData) -> None:
        """Main update called on each telemetry packet: run every stage for this frame."""
        self.latest_telemetry, self.latest_lap = telemetry, lap
        if self.map_rewind_pending:
            self._resolve_map_rewind(lap)
        # Laps: delta, sectors, lap completion
        self.update_delta(lap, telemetry)
        # Radio: callouts straight off the frame
        self.analyze_tyres(telemetry, lap)
        self.check_braking_points(lap, telemetry)
        self.update_car_management(telemetry, lap)
        # Laps: corner diagnosis
        self.update_corners(telemetry, lap)
        # Strategy: game pit window, rival stops, undercuts, race simulation
        self.radio.update_strategy(lap)
        self.update_strategy_calls(lap)
        self.update_race_simulation()

        # Radio: priority interrupts
        if self.session:
            self.radio.update_safety_car(self.session.safety_car_status, self.safety_car_call)
            self.radio.update_weather_alert(self.session.weather)
        if self.latest_damage:
            self.radio.update_damage_alert(
                self.latest_damage.worst_tyre_wear,
                self.latest_damage.front_left_wing + self.latest_damage.front_right_wing,
            )

        # Outputs: the web HUD snapshot
        self.publish_frame(telemetry, lap)
