"""Radio stage: what the engineer says, and when it is allowed to say it.

* ``speak()`` is the one door to the voice. On-demand mode keeps the radio silent except for
  the proactive keys in config (critical events and this-lap strategy calls).
* A per-lap budget for proactive calls (radio_discipline.CallBudget): urgent ones always pass.
  A race gets a tighter budget than practice or qualifying.
* Race quiet windows (radio_discipline.race_quiet_reason): lap 1, a DRS fight and the last
  two laps carry urgent calls only.
* ``is_driver_busy()`` is the silence filter: non-urgent calls wait out heavy braking
  and high-load corners, and land on the next straight.
* Tyre-temperature and braking-point callouts, which come straight off the telemetry frame.

Gaps, pace, sector times and race-control calls live in race_radio.py / race_events.py;
those objects call back into ``speak()`` so they share the same gating.
"""

from __future__ import annotations

import time
from typing import Any

import radio_discipline
import radio_phrases as say
import session_modes
import web_hud
from packets import CarTelemetryData, LapData
from pipeline.state import EngineState, wheel_group
from voice_engine import Priority


class RadioStage(EngineState):
    def speak(self, message: str, priority: Priority = Priority.NORMAL, **options: Any) -> bool:
        """Queue speech; priority interrupts in the proactive keys break on-demand silence.

        Returns False when the radio rules keep the call off the air (on-demand silence or
        the lap's budget), True when it went to the voice.
        """
        topic = options.get('key', message)
        if self.on_demand and topic not in self.config.radio.proactive_keys:
            return False  # radio silence until the driver asks or critical priority event
        if self.race_quiet_reason(topic, priority):
            return False  # lap 1, a DRS fight or the last two laps: urgent calls only
        if not self.call_budget.allows(topic, priority):
            return False  # this lap's proactive calls are used up; urgent calls still get through
        options['key'] = f'{self.capture_id or self.session_context}:{topic}'
        queued = self.voice.say(message, priority, **options)
        if queued:
            self.call_budget.spend(topic, priority)
        if not queued and not getattr(self.voice, 'is_running', False):
            web_hud.add_voice_log(message, status='muted')
        return True

    def race_quiet_reason(self, topic: str | None, priority: Priority) -> str | None:
        """Why a non-urgent race call is held back right now ('lap_1', 'drs_fight', 'final_laps')."""
        if (
            self.session_mode != session_modes.RACE
            or topic in ('session', 'lap_one')  # the radio check, and where you are after lap 1
            or radio_discipline.is_urgent(topic, priority)
        ):
            return None
        lap = self.latest_lap
        return radio_discipline.race_quiet_reason(
            lap.current_lap_num if lap else None,
            self.session.total_laps if self.session else None,
            self.radio.gap_ahead_s,
            self.radio.gap_behind_s,
            topic,
        )

    def is_driver_busy(self) -> bool:
        """True in a high-load corner or under heavy braking (the voice queue's silence filter).

        Read by the voice thread while a call is waiting. A high-load corner is fast and high
        lateral G (radio_discipline.LoadThresholds); once in one, the driver counts as busy
        until the car has been on a straight for a moment, so the call lands there.
        """
        telemetry = self.latest_telemetry
        if telemetry is None:
            return False
        motion = getattr(self, 'latest_motion', None)
        self.load_gate.update(telemetry.speed, motion.g_force_lateral if motion else 0.0, telemetry.brake)
        return bool(self.load_gate.busy())

    # ── Tyre temperatures ──

    def analyze_tyres(self, telemetry: CarTelemetryData, lap: LapData) -> None:
        cfg = self.config.tyres
        now = time.time()
        # The overheating call reads the carcass (inner) temperature: the surface flickers
        # with every slide and kerb, the carcass is what actually overheats the tyre.
        # The hot/cold lap summary stays on the surface temps the driver sees on the dash.
        temps = telemetry.tyres_inner_temperature
        surface = telemetry.tyres_surface_temperature

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

        dt = min(now - prev_time, 0.5) if prev_sample is not None and now > prev_time else 0.0
        critical_wheels = []
        for i in range(4):
            temp = temps[i]
            state = self.tyre_state
            state.lap_peak[i] = max(state.lap_peak[i], surface[i])
            if surface[i] > cfg.high_c:
                state.lap_hot_secs[i] += dt
            elif surface[i] < cfg.low_c:
                state.lap_cold_secs[i] += dt
            if temp >= cfg.critical_c and rise_rates[i] >= cfg.rise_rate_c_per_s:
                if state.hot_since[i] == 0.0:
                    state.hot_since[i] = now
                if now - state.hot_since[i] >= cfg.critical_sustain_s:
                    critical_wheels.append(i)
            elif temp < cfg.critical_c:
                state.hot_since[i] = 0.0

        # Mid-lap: one short call per lap, and only for a genuine, still-climbing overheat.
        if (
            critical_wheels
            and not self.tyre_state.critical_called_this_lap
            and now - self.tyre_state.last_critical_alert_time >= cfg.alert_cooldown_s
        ):
            pronoun = 'it' if len(critical_wheels) == 1 else 'them'
            self.speak(
                f'{wheel_group(critical_wheels)} overheating, cool {pronoun}.',
                Priority.CRITICAL,
                key='tyre_temperature',
                ttl=4,
                cooldown=25,
            )
            self.tyre_state.critical_called_this_lap = True
            self.tyre_state.last_critical_alert_time = now
            self.tyre_state.last_alert_time = now

    def tyre_lap_summary(self) -> str | None:
        """One short line on how the tyres ran that lap, or None if nothing worth saying."""
        state, min_secs = self.tyre_state, self.config.tyres.summary_min_s
        hot = [i for i in range(4) if state.lap_hot_secs[i] >= min_secs]
        cold = [i for i in range(4) if state.lap_cold_secs[i] >= min_secs and i not in hot]
        if hot:
            peak = int(max(state.lap_peak[i] for i in hot))
            return f'{wheel_group(hot)} ran hot that lap, peak {say.number_words(peak)}.'
        if cold:
            return f'{wheel_group(cold)} ran cold that lap.'
        return None

    def _report_tyre_lap(self) -> None:
        """End-of-lap tyre summary: said when the picture changes, else every few laps."""
        state = self.tyre_state
        summary = self.tyre_lap_summary()
        state.laps_since_summary += 1
        if summary and (
            summary.split(',')[0] != (state.last_summary or '').split(',')[0]
            or state.laps_since_summary >= self.config.tyres.summary_repeat_laps
        ):
            self.speak(summary, Priority.LOW, key='tyre_summary', ttl=10, cooldown=0)
            state.laps_since_summary = 0
        state.last_summary = summary
        state.lap_peak = [0.0, 0.0, 0.0, 0.0]
        state.lap_hot_secs = [0.0, 0.0, 0.0, 0.0]
        state.lap_cold_secs = [0.0, 0.0, 0.0, 0.0]
        state.critical_called_this_lap = False

    # ── Car management: fuel, ERS, brakes, engine, damage ──

    def update_car_management(self, telemetry: CarTelemetryData, lap: LapData) -> None:
        """Fuel, ERS, brake, engine-temperature and damage calls (car_management.py)."""
        session = self.session
        self.car_manager.update(
            time.monotonic(),
            lap,
            telemetry,
            self.latest_status,
            self.latest_damage,
            race=self.session_mode == session_modes.RACE,
            total_laps=session.total_laps if session else 0,
            corners=self.corners,
            gap_ahead_s=self.radio.gap_ahead_s,
        )

    # ── Braking points ──

    def check_braking_points(self, lap: LapData, telemetry: CarTelemetryData) -> None:
        if not self.config.radio.braking_calls or not self.braking_points or telemetry.speed < 60:
            return

        dist = lap.lap_distance
        if dist < 50:
            # Clear announcements near start/finish
            self.announced_braking.clear()
            return

        warn = self.config.radio.braking_warn_distance_m
        for bp_dist, corner_name, message in self.braking_points:
            approach_dist = bp_dist - dist
            if 0 < approach_dist <= warn and corner_name not in self.announced_braking:
                self.announced_braking.add(corner_name)
                dist_rnd = int(round(approach_dist / 10.0) * 10)
                self.speak(
                    f'{message}, {dist_rnd} metres.', Priority.CRITICAL, key='braking', ttl=1.0, cooldown=0
                )
