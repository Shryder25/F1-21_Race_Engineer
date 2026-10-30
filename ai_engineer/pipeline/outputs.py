"""Outputs stage: everything that leaves the engine except speech.

* Web HUD state: the per-frame snapshot (``publish_frame``), the reset on a new session,
  and the pit-wall pieces that are only sent when they change.
* History database writes (history.py): laps, lap stats, every car's stints, the final
  classification and setups. A failed write is logged and never stops the engine.
* The AI prompt context (``build_context``) for engineer_brain.py.

MoTeC export is not triggered from here: the channel store (channel_store.py) logs raw
packets beside the engine and ``main.py --export-motec`` converts the file afterwards.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from typing import Any

import driver_profile
import track_status
import web_hud
from log_setup import get_logger
from packets import ACTUAL_COMPOUNDS, SESSION_TYPES, CarTelemetryData, LapData, SessionData
from pipeline.state import ERS_MODE_NAMES, ERS_STORE_MAX_J, JSON, WEATHER_NAMES, WHEEL_CODES
from pipeline.track_map import TrackMapStage
from session_history import lap_rows, stint_rows

log = get_logger('history')

HISTORY_ERRORS = (OSError, ValueError, sqlite3.Error)


class OutputsStage(TrackMapStage):
    # ── History database ──

    def _write_history(self, what: str, write: Callable[[], object]) -> None:
        """Run one history write; a disk or database error is logged, never raised."""
        try:
            write()
        except HISTORY_ERRORS as error:
            log.warning('Could not save %s: %s', what, error, extra={'capture_id': self.capture_id})

    def _load_history_reference(self, session: SessionData, session_uid: int | None) -> None:
        """Open a history capture for these conditions and load the best reference lap for them."""
        history = self.history
        if not (history and session.track_id >= 0):
            return
        try:
            self.capture_id = history.start_session(session, session_uid, self.session_log.player_index)
            self._session_captures.add(self.capture_id)
            self._history_records = None
            reference = history.best_reference(session)
            if reference:
                self.delta.best_lap_time_ms = reference['lap_time_ms']
                self.delta.best_lap_num = reference['lap_num']
                self.delta.best_lap_times = {float(k): v for k, v in reference['times'].items()}
                self.delta.best_lap_speeds = reference['speeds']
                self.best_corner_data = reference['corners']
                self.best_sector_times = history.personal_sectors(session)
            self._refresh_driver_profile()
        except (*HISTORY_ERRORS, KeyError) as error:
            log.warning('Could not load track history: %s', error)

    def _refresh_driver_profile(self) -> None:
        """How this driver usually drives this track, from every saved clean lap (for the AI)."""
        try:
            self.driver_profile = driver_profile.build_profile(self.history, self.session)
        except (OSError, ValueError, sqlite3.Error) as error:
            log.warning('Could not build driver profile: %s', error)

    def _record_lap_stats(
        self,
        lap_num: int,
        lap_time_ms: int,
        eligible: bool,
        sectors: dict[int, int],
        speed_trap_kph: float | None = None,
    ) -> None:
        """Log tyre wear, fuel burn and sector deltas for the lap that just finished."""
        status, damage = self.latest_status, self.latest_damage
        deltas = {
            sector: ms - self.best_sector_times[sector]
            for sector, ms in sectors.items()
            if sector in self.best_sector_times
        }
        stats: JSON = {
            'lap_num': lap_num,
            'lap_time_ms': lap_time_ms,
            'eligible': eligible,
            'compound': status.actual_tyre_compound if status else None,
            'tyre_age_laps': status.tyres_age_laps if status else None,
            'wear': tuple(damage.tyres_wear) if damage else None,
            'fuel_start': self.lap_start_fuel,
            'fuel_end': status.fuel_in_tank if status else None,
            'sectors': dict(sectors),
            'sector_deltas': deltas,
            'speed_trap_kph': speed_trap_kph,
        }
        self.session_lap_stats.append(stats)
        self._ml_cache = None
        self._forecast_cache = None
        history, capture = self.history, self.capture_id
        if history and capture:
            self._write_history('lap stats', lambda: history.record_lap_stats(capture, stats))

    def _sync_session_log_capture(self) -> None:
        """A new history capture mid-session (rain arrived) gets every car's rows so far."""
        if self._session_log_capture == self.capture_id:
            return
        self._session_log_capture = self.capture_id
        history, capture = self.history, self.capture_id
        if not (history and capture):
            return
        session_log = self.session_log

        def write_all() -> None:
            for car, car_history in session_log.histories.items():
                history.record_car_history(
                    capture,
                    car,
                    lap_rows(car_history),
                    stint_rows(car_history),
                    driver=session_log.names.get(car),
                )
            if session_log.classification:
                history.record_final_classification(capture, session_log.classification_rows())
            for index, setup in session_log.setups.items():
                history.record_car_setup(capture, index, setup.as_dict())

        self._write_history('car history', write_all)

    # ── Web HUD ──

    def _reset_hud(self) -> None:
        """A new session: blank every live panel so nothing from the last one lingers."""
        web_hud.update_web_state(
            {
                'connected': False,
                'corner_analysis': [],
                'voice_log': [],
                'speed_trace_current': [],
                'speed_trace_best': self.delta.best_lap_speeds,
                'best_lap_time_ms': self.delta.best_lap_time_ms,
                'delta_ms': None,
                'gap_ahead_s': None,
                'gap_behind_s': None,
                'race_pace_ms': None,
                'car_ahead_name': None,
                'car_behind_name': None,
                'pit_window': None,
                'race_strategy': None,
                'last_update_monotonic': None,
                'speed': 0,
                'gear': 0,
                'rpm': 0,
                'drs': False,
                'throttle': 0,
                'brake': 0,
                'lap_num': 0,
                'lap_time_ms': 0,
                'last_lap_time_ms': 0,
                'position': 0,
                'tyre_surface_temps': [0] * 4,
                'tyre_inner_temps': [0] * 4,
                'tyre_pressures': [0] * 4,
                'engine_temp': 0,
                'personal_sectors_ms': self.best_sector_times,
                'current_sectors_ms': {},
                'last_lap_sectors_ms': {},
                'last_lap_sector_refs_ms': {},
                'track_status': dict(track_status.CLEAR),
                'safety_car': 'NONE',
                'timing_tower': [],
                'gap_history': [],
                'tyre_stints': [],
                'strategy_odds': None,
            }
        )

    def _best_trace_update(self) -> JSON:
        """The reference lap's speed trace, only when it was replaced (it is a new list each
        time; holding the sent one keeps its identity from being reused)."""
        trace = self.delta.best_lap_speeds
        if trace is self._sent_best_trace and len(trace) == self._sent_best_len:
            return {}
        self._sent_best_trace, self._sent_best_len = trace, len(trace)
        return {'speed_trace_best': trace}

    def pit_wall_update(self) -> JSON:
        """Timing tower every frame; gap chart and stints only when they changed."""
        names, wall = self.radio.driver_names, self.pit_wall
        update: JSON = {'timing_tower': wall.rows, 'strategy_odds': self.strategy_odds()}
        revisions = (wall.history_revision, wall.stint_revision)
        if revisions != self._sent_pit_wall:
            if revisions[0] != self._sent_pit_wall[0]:
                update['gap_history'] = wall.gap_chart(names, self.car_teams, self.player_index)
            if revisions[1] != self._sent_pit_wall[1]:
                update['tyre_stints'] = wall.stints()
            self._sent_pit_wall = revisions
        return update

    def opponent_threats(self) -> list[JSON]:
        """Nearby cars ranked by how much pressure they put on the player right now."""
        return [t.__dict__ for t in self.radio.opponent_threats]

    def race_report(self) -> list[JSON]:
        """This session's completed laps for the HUD race-report table: lap number, lap
        time, each sector, speed trap and tyre compound. Reads from the in-memory
        session_lap_stats (so it works even with no history DB attached) rather than
        querying SQLite on every telemetry frame.
        """
        report = []
        for st in self.session_lap_stats:
            sectors = st.get('sectors') or {}
            compound: Any = st.get('compound')
            report.append(
                {
                    'lap_num': st['lap_num'],
                    'lap_time_ms': st['lap_time_ms'],
                    'eligible': bool(st.get('eligible')),
                    'compound': ACTUAL_COMPOUNDS.get(compound, '--'),
                    's1_ms': sectors.get(1),
                    's2_ms': sectors.get(2),
                    's3_ms': sectors.get(3),
                    'speed_trap_kph': st.get('speed_trap_kph'),
                }
            )
        return report

    def _tire_health_hud(self) -> JSON | None:
        tire = self.tire_health()
        if tire is None:
            return None
        return {
            'grip_factor': tire.grip_factor,
            'health_score': tire.health_score,
            'remaining_laps': tire.remaining_laps,
            'overheat_risk': tire.overheat_risk.value,
            'limiting_factor': tire.limiting_factor,
        }

    def publish_frame(self, telemetry: CarTelemetryData, lap: LapData) -> None:
        """Stream this frame's snapshot to the web HUD."""
        session, status, damage, motion = (
            self.session,
            self.latest_status,
            self.latest_damage,
            self.latest_motion,
        )
        safety_car = session.safety_car_status if session else 0
        web_hud.update_web_state(
            {
                'connected': True,
                'last_update_monotonic': time.monotonic(),
                'track_name': self.track_name,
                'track_id': self.track_id,
                'track_length': session.track_length if session else 0,
                'speed': telemetry.speed,
                'gear': telemetry.gear,
                'rpm': telemetry.engine_rpm,
                'drs': bool(telemetry.drs),
                'throttle': round(telemetry.throttle, 2),
                'brake': round(telemetry.brake, 2),
                'steer': round(telemetry.steer, 2),
                'tyre_surface_temps': list(telemetry.tyres_surface_temperature),
                'tyre_inner_temps': list(telemetry.tyres_inner_temperature),
                'tyre_pressures': list(telemetry.tyres_pressure),
                'tyre_wear': list(damage.tyres_wear) if damage else [0.0, 0.0, 0.0, 0.0],
                'tyre_compound': status.compound_name if status else 'Medium',
                'ers_store_energy': status.ers_store_energy if status else 2000000.0,
                'ers_deploy_mode': status.ers_deploy_mode if status else 1,
                'ers_percent': int(round((status.ers_store_energy / 4000000.0) * 100)) if status else 50,
                'engine_temp': telemetry.engine_temperature,
                'car_management': self.car_manager.snapshot(status, damage),
                'lap_num': lap.current_lap_num,
                'total_laps': session.total_laps if session else 0,
                'position': lap.car_position,
                'lap_time_ms': lap.current_lap_time_ms,
                'last_lap_time_ms': lap.last_lap_time_ms,
                'best_lap_time_ms': self.delta.best_lap_time_ms,
                'delta_ms': self.delta.last_reported_delta_ms,
                'lap_distance': round(lap.lap_distance, 1),
                'sector': lap.sector,
                'current_sectors_ms': dict(self.current_sector_times),
                'personal_sectors_ms': dict(self.best_sector_times),
                'last_lap_sectors_ms': dict(self.last_lap_sectors),
                'last_lap_sector_refs_ms': dict(self.last_lap_sector_refs),
                'track_status': track_status.describe(
                    session, status.fia_flags if status else 0, self.sector_starts
                ),
                'gap_ahead_s': self.radio.gap_ahead_s,
                'gap_behind_s': self.radio.gap_behind_s,
                'car_ahead_name': self.radio.driver_name(self.radio.car_ahead_index),
                'car_behind_name': self.radio.driver_name(self.radio.car_behind_index),
                'race_pace_ms': self.radio.race_pace_ms,
                'pit_window': self.radio.pit_window,
                'opponent_threats': self.opponent_threats(),
                'race_strategy': self.strategy_summary(),
                'model_ranges': self.model_summary(),
                'race_control': self.race_control.recent(),
                'race_report': self.race_report(),
                'tire_health': self._tire_health_hud(),
                'safety_car': 'VSC' if safety_car == 2 else 'SAFETY CAR' if safety_car == 1 else 'NONE',
                'weather': WEATHER_NAMES.get(session.weather, 'Clear') if session else 'Clear',
                'corner_analysis': self.corner_diagnostics,
                'speed_trace_current': self.delta.current_lap_speeds[-120:],
                'car_x': motion.world_position_x if motion else 0.0,
                'car_z': motion.world_position_z if motion else 0.0,
                'car_yaw': motion.yaw if motion else 0.0,
                'g_force_lat': round(motion.g_force_lateral, 2) if motion else 0.0,
                'g_force_lon': round(motion.g_force_longitudinal, 2) if motion else 0.0,
                'pit_status': lap.pit_status,
                'cars': self.field_cars,
                **self.map_update(),
                **self._best_trace_update(),
                **self.pit_wall_update(),
            }
        )

    # ── AI prompt ──

    def standings_summary(self, limit: int = 22) -> list[JSON] | None:
        """The running order from the timing tower, compact enough for the AI prompt."""
        rows: list[JSON] | None = getattr(self.pit_wall, 'rows', None)
        if not rows:
            return None
        keys = (
            'position',
            'name',
            'gap_s',
            'interval_s',
            'laps_down',
            'last_lap_ms',
            'best_lap_ms',
            'compound',
            'tyre_age',
            'stops',
            'pit',
            'penalties_s',
            'status',
            'is_player',
            'fastest_lap',
        )
        return [{k: row.get(k) for k in keys} for row in rows[:limit]]

    @staticmethod
    def _car_context(lap: Any, t: Any, dmg: Any, st: Any) -> JSON:
        """Car health, current lap and penalties for the AI (fields missing on a packet are skipped)."""

        def get(obj: Any, name: str) -> Any:
            return getattr(obj, name, None) if obj is not None else None

        out: JSON = {}
        if lap is not None and get(lap, 'current_lap_time_ms') is not None:
            out['current_lap'] = {
                'time_ms': lap.current_lap_time_ms or None,
                'sector': (get(lap, 'sector') or 0) + 1,
                'invalid': bool(get(lap, 'current_lap_invalid')),
            }
        if lap is not None and get(lap, 'penalties') is not None:
            out['penalties'] = {
                'time_penalty_s': lap.penalties,
                'warnings': get(lap, 'warnings') or 0,
                'unserved_drive_through': get(lap, 'num_unserved_drive_through') or 0,
                'unserved_stop_go': get(lap, 'num_unserved_stop_go') or 0,
                'pit_stops_done': get(lap, 'num_pit_stops') or 0,
                'grid_position': get(lap, 'grid_position'),
            }
        if get(t, 'brakes_temperature') is not None:
            out['brake_temps_c'] = dict(zip(WHEEL_CODES, t.brakes_temperature, strict=False))
        if get(t, 'tyres_pressure') is not None:
            out['tyre_pressures_psi'] = {
                c: round(p, 1) for c, p in zip(WHEEL_CODES, t.tyres_pressure, strict=False)
            }
        if get(t, 'engine_temperature') is not None:
            out['engine_temp_c'] = t.engine_temperature
        if get(t, 'drs') is not None:
            out['drs_open'] = bool(t.drs)
        if get(dmg, 'front_left_wing') is not None:
            tyre_damage = get(dmg, 'tyres_damage') or ()
            out['damage_pct'] = {
                'front left wing': dmg.front_left_wing,
                'front right wing': get(dmg, 'front_right_wing'),
                'rear wing': get(dmg, 'rear_wing'),
                'floor': get(dmg, 'floor'),
                'gearbox wear': get(dmg, 'gearbox'),
                'engine wear': get(dmg, 'engine'),
                'tyre damage': max(tyre_damage) if tyre_damage else 0,
            }
        if get(st, 'ers_store_energy') is not None:
            mode = get(st, 'ers_deploy_mode')
            out['ers'] = {
                'battery_pct': round(st.ers_store_energy / ERS_STORE_MAX_J * 100),
                'deploy_mode': ERS_MODE_NAMES.get(mode, str(mode)),
                'used_this_lap_pct': round((get(st, 'ers_deployed_this_lap') or 0) / ERS_STORE_MAX_J * 100),
            }
        return out

    def build_context(self) -> JSON:
        """Snapshot of live telemetry plus ML predictions, for the AI prompt and fallbacks."""
        t, lap, s, st, dmg = (
            self.latest_telemetry,
            self.latest_lap,
            self.session,
            self.latest_status,
            self.latest_damage,
        )
        ctx: JSON = {
            'track': self.track_name,
            'session_type': SESSION_TYPES.get(s.session_type, 'Session') if s else None,
            'session_mode': self.session_mode,
            'weather': WEATHER_NAMES.get(s.weather) if s else None,
            'track_temp_c': s.track_temperature if s else None,
            'air_temp_c': s.air_temperature if s else None,
            'safety_car': bool(s and s.safety_car_status),
        }
        if lap:
            ctx.update(
                lap=lap.current_lap_num,
                total_laps=s.total_laps if s else None,
                position=lap.car_position,
                last_lap_ms=lap.last_lap_time_ms or None,
                best_lap_ms=self.delta.best_lap_time_ms or None,
                delta_to_best_ms=self.delta.last_reported_delta_ms,
                race_pace_ms=self.radio.race_pace_ms,
                gap_ahead_s=self.radio.gap_ahead_s,
                gap_behind_s=self.radio.gap_behind_s,
                car_ahead_name=self.radio.driver_name(self.radio.car_ahead_index),
                car_behind_name=self.radio.driver_name(self.radio.car_behind_index),
                game_pit_window=self.radio.pit_window,
            )
        ctx.update(self._car_context(lap, t, dmg, st))
        ctx['car_management'] = self.car_manager.snapshot(st, dmg)
        if t:
            ctx['tyre_surface_c'] = dict(zip(WHEEL_CODES, t.tyres_surface_temperature, strict=False))
            ctx['tyre_inner_c'] = dict(zip(WHEEL_CODES, t.tyres_inner_temperature, strict=False))
        if dmg:
            ctx['tyre_wear_pct'] = {c: round(w, 1) for c, w in zip(WHEEL_CODES, dmg.tyres_wear, strict=False)}
        if st:
            ctx.update(compound=st.compound_name, tyre_age_laps=st.tyres_age_laps)
        ctx['standings'] = self.standings_summary()
        prediction = self.pit_prediction()
        ctx['pit_prediction'] = prediction.as_dict() if prediction else None
        ctx['fuel'] = self.fuel_prediction()
        ctx['pace_trend_s_per_lap'] = self.pace_trend()
        ctx['tyre_fuel_model'] = self.model_summary()
        tire = self.tire_health()
        ctx['tire_health'] = (
            None
            if tire is None
            else {
                'grip_factor': tire.grip_factor,
                'health_score': tire.health_score,
                'deg_rate_s_per_lap': tire.deg_rate_s_per_lap,
                'remaining_laps': tire.remaining_laps,
                'performance_loss_s': tire.performance_loss_s,
                'overheat_risk': tire.overheat_risk.value,
                'limiting_factor': tire.limiting_factor,
            }
        )
        threats = self.opponent_threats()
        ctx['opponent_threats'] = [
            {
                'threat_level': th['threat_level'],
                'threat_score': th['threat_score'],
                'gap_s': th['gap_s'],
                'pace_delta_s': th['pace_delta_s'],
                'laps_to_catch': th['laps_to_catch'],
                'factors': th['factors'],
                'driver_name': th['driver_name'],
            }
            for th in threats
        ] or None
        ctx['strategy'] = self.strategy_summary()
        rivals: Any = (ctx['strategy'] or {}).get('rivals')
        ctx['rival_predictions'] = (
            [r['summary'] for r in rivals['nearby']] if rivals and rivals['nearby'] else None
        )
        ctx['track_conditions'] = self.conditions_summary()
        ctx['driver_profile'] = driver_profile.describe(self.driver_profile)
        ctx['recent_race_events'] = [event['text'] for event in self.race_control.recent(5)] or None
        if self.session_lap_stats:
            last = self.session_lap_stats[-1]
            ctx['last_lap_sectors_ms'] = last['sectors'] or None
            ctx['last_lap_sector_deltas_ms'] = last['sector_deltas'] or None
        return ctx
