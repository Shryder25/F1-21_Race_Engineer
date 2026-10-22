"""Ingest stage: one handler per decoded game packet, feeding the engine's state.

telemetry_stream.py validates raw UDP datagrams, pairs them per game frame and calls these
handlers. Each handler stores what arrived and hands it to the stage that uses it:
session changes reset per-session state, Motion feeds the track map, Car Status / Damage /
Lap Data feed the models and rival tracker, Session History / Final Classification / Car
Setups are logged and written to the history database.

The per-frame telemetry handler, ``update_telemetry``, is the orchestrator in analysis.py.
"""

from __future__ import annotations

import time

import race_strategy
import session_modes
import track_maps
import web_hud
from history import TelemetryHistory
from packets import (
    SESSION_TYPES,
    TRACK_NAMES,
    CarDamageData,
    CarMotionData,
    CarSetupData,
    CarStatusData,
    FinalClassificationData,
    LapData,
    ParticipantData,
    SessionData,
    SessionHistoryData,
)
from pipeline.laps import LapsStage
from pipeline.state import SILVERSTONE_CORNERS, DeltaTracker, TyreState
from session_history import SessionLog
from track_data import corner_zones, get_braking_points, resolve_track_id
from voice_engine import Priority


class IngestStage(LapsStage):
    # ── Session (packet 1) ──

    def update_session(
        self, session: SessionData, session_uid: int | None = None, player_index: int | None = None
    ) -> None:
        # Weather is deliberately not part of the session identity: rain arriving mid-race is
        # the same session (a reset here would wipe the race state and swallow the rain call).
        context = (
            session_uid,
            session.track_id,
            session.track_length,
            session.formula,
            session.session_type,
            player_index,
        )
        weather_group = TelemetryHistory.weather_group(session.weather)
        if context != self.session_context:
            self.session_context = context
            self.weather_group = weather_group
            self.session_uid = session_uid
            self._reset_session(session, session_uid, player_index)
        elif weather_group != self.weather_group:
            self._change_conditions(session)
            self.weather_group = weather_group
        self.session = session
        self.wetness.update_weather(
            session.weather,
            session.track_temperature,
            time.monotonic(),
            getattr(session, 'weather_forecast', ()),
        )
        self.radio.session = session
        self.race_control.session = session
        self.track_id = session.track_id
        self.track_name = TRACK_NAMES.get(session.track_id, f'Track {session.track_id}')
        self.session_mode = session_modes.session_mode(session.session_type)

        # Load braking points and corner zones for whichever circuit the game reports
        self.braking_points = get_braking_points(session.track_id)
        web_hud.update_web_state(
            {
                'track_name': self.track_name,
                'track_id': self.track_id,
                'track_length': session.track_length,
                'session_uid': session_uid,
                'capture_id': self.capture_id,
                'session_type': SESSION_TYPES.get(session.session_type, 'Session'),
                'session_mode': self.session_mode,
                'session_label': session_modes.MODE_LABELS[self.session_mode],
                'setup_track_id': resolve_track_id(session.track_id),
            }
        )
        corner_key = (session.track_id, session.track_length)
        if corner_key != self._corner_key:
            self._corner_key = corner_key
            if session.track_id == 7:
                self.corners = SILVERSTONE_CORNERS  # hand-measured apex speeds
            else:
                self.corners = corner_zones(session.track_id, session.track_length)

        if not self.session_started:
            self.session_started = True
            self.speak(
                session_modes.welcome_line(self.session_mode, self.track_name),
                Priority.LOW,
                key='session',
                ttl=10,
                cooldown=0,
            )

    def _reset_session(self, session: SessionData, session_uid: int | None, player_index: int | None) -> None:
        """A new session: every per-session value back to its default."""
        self.delta = DeltaTracker()
        self.tyre_state = TyreState()
        self.driver_profile = None
        self.current_corner_data.clear()
        self.best_corner_data.clear()
        self.corner_diagnostics.clear()
        self.previous_corner_sample = None
        self.brake_start_distance = None
        self.current_sector_times.clear()
        self.best_sector_times.clear()
        self.last_lap_sectors = {}
        self.last_lap_sector_refs = {}
        self.sector_starts = {}
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
        self.replanner.reset()
        self.wetness.reset()
        self._last_sim_key = None
        self._last_event_submit = -1e9
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
        self._clear_lap_trail()
        self.track_map_points.clear()
        self.braking_zones_best.clear()
        self.ers_zones_best.clear()
        self.last_map_dist = -999.0
        self.current_pit_points.clear()
        self.field_pit_points.clear()
        self.field_mapper = track_maps.FieldMapper(session.track_length)
        self.car_teams = {}
        self.field_cars = []
        self.map_lap_has_pit = False
        self.map_rewind_pending = False
        self.pit_wall.reset(
            session.track_length, race=session_modes.session_mode(session.session_type) == session_modes.RACE
        )
        self._sent_pit_wall = (self.pit_wall.history_revision, self.pit_wall.stint_revision)
        self._odds_result = self._odds = None
        self.odds_history = []
        self._load_track_map(session)
        self.radio.set_driver_names({})
        clear = getattr(self.voice, 'clear', None)
        if clear is not None:
            clear()
        self._load_history_reference(session, session_uid)
        self._reset_hud()
        self.radio.reset(session)
        self.race_control.reset(session)
        web_hud.update_web_state({'race_control': []})

    def _change_conditions(self, session: SessionData) -> None:
        """Dry -> intermediate -> wet within one session: keep the race, swap the references.

        A dry best lap, sector or corner is no yardstick on a wet track, and laps recorded
        from here on belong with wet history, so only the timing references and the history
        capture change. Race state (gaps, positions, strategy, radio) carries on.
        """
        best = DeltaTracker()
        self.delta.best_lap_times, self.delta.best_lap_speeds = best.best_lap_times, best.best_lap_speeds
        self.delta.best_lap_time_ms, self.delta.best_lap_num = 0, 0
        self.delta.current_lap_invalid = True  # this lap was run in mixed conditions
        self.best_corner_data = {}
        self.best_sector_times = {}
        self._load_history_reference(session, self.session_uid)
        web_hud.update_web_state(
            {
                'speed_trace_best': self.delta.best_lap_speeds,
                'best_lap_time_ms': self.delta.best_lap_time_ms,
                'personal_sectors_ms': self.best_sector_times,
                'capture_id': self.capture_id,
            }
        )

    # ── Motion (packet 0) ──

    def update_motion(
        self, motion: CarMotionData, all_positions: list[tuple[float, float]] | None = None
    ) -> None:
        """Update live world position coordinates and lateral/longitudinal G-forces.
        `all_positions` is every car's (x, z) from the same Motion packet: it maps the
        circuit from the whole field and places every car on the HUD map."""
        self.latest_motion = motion
        if all_positions is not None:
            self._update_field_map(all_positions)
        self._record_trail_point(round(motion.world_position_x, 2), round(motion.world_position_z, 2))

    # ── Car status, damage, every car's lap data (packets 7, 10, 2) ──

    def update_car_status(self, status: CarStatusData, all_status: list[CarStatusData] | None = None) -> None:
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

    def update_field(self, laps: list[LapData], player_index: int, session_time: float) -> None:
        """Every car's Lap Data: rival stints for the stop model, and line-crossing times and
        clean-lap pace for the race simulator."""
        self.rivals.update_laps(laps, player_index)
        self.field_laps, self.player_index = laps, player_index
        self.pit_wall.update(
            laps,
            session_time,
            player_index,
            self.all_status,
            names=self.radio.driver_names,
            teams=self.car_teams,
            histories=self.session_log.histories,
        )
        if self.session is None:
            return
        self.wetness.observe_field(
            laps, self.all_status, time.monotonic(), neutralised=self.session.safety_car_status in (1, 2)
        )
        if self.session_mode != session_modes.RACE:
            return
        self.field_tracker.update(
            laps, session_time, self.all_status, neutralised=self.session.safety_car_status in (1, 2)
        )

    def update_car_damage(self, damage: CarDamageData, all_damage: list[CarDamageData] | None = None) -> None:
        self.latest_damage = damage
        if all_damage is not None:
            self.rivals.update_damage(all_damage)

    # ── Participants (packet 4) ──

    def update_participants(self, participants: list[ParticipantData]) -> None:
        """Driver names from the Participants packet, so radio calls can say who is where."""
        self.radio.set_driver_names(
            {index: p.radio_name for index, p in enumerate(participants) if p.radio_name}
        )
        self.race_control.update_participants(participants)
        self.car_teams = {index: p.team_id for index, p in enumerate(participants)}
        for index, p in enumerate(participants):
            if p.name:
                self.session_log.names[index] = p.radio_name or p.name

    # ── Every car's laps and stints, result and setups (packets 11, 8, 5) ──

    def update_session_history(self, history: SessionHistoryData) -> None:
        """One car's Session History packet: log its laps and tyre stints."""
        self._sync_session_log_capture()
        if history.car_index == self.player_index:
            # Every set we have run this race, even before the engineer started: the planner's
            # two-compound rule must not demand a stop we already made.
            for stint in history.stints:
                compound = race_strategy.normalise_compound(stint.compound_name)
                if compound:
                    self.compounds_used.add(compound)
        changed = self.session_log.update_history(history)
        store, capture = self.history, self.capture_id
        if changed is None or not (store and capture):
            return
        laps, stints, last_lap = changed
        driver = self.session_log.names.get(history.car_index)
        self._write_history(
            'car history',
            lambda: store.record_car_history(
                capture, history.car_index, laps, stints, last_lap, driver=driver
            ),
        )

    def update_final_classification(self, classification: list[FinalClassificationData]) -> None:
        self._sync_session_log_capture()
        if not self.session_log.update_classification(classification):
            return
        store, capture = self.history, self.capture_id
        if store and capture:
            session_log = self.session_log
            self._write_history(
                'final classification',
                lambda: store.record_final_classification(capture, session_log.classification_rows()),
            )

    def update_car_setups(self, setups: list[CarSetupData]) -> None:
        self._sync_session_log_capture()
        changed = self.session_log.update_setups(setups)
        store, capture = self.history, self.capture_id
        if not (changed and store and capture):
            return

        def write() -> None:
            for index, setup in changed.items():
                store.record_car_setup(capture, index, setup.as_dict())

        self._write_history('car setup', write)

    @property
    def player_setup(self) -> CarSetupData | None:
        """The player's latest setup from the Car Setups packet, or None."""
        setup: CarSetupData | None = self.session_log.setups.get(self.session_log.player_index)
        return setup
