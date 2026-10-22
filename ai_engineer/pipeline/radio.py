"""Radio stage: what the engineer says, and when it is allowed to say it.

* ``speak()`` is the one door to the voice. On-demand mode keeps the radio silent except for
  the proactive keys in config (critical events and this-lap strategy calls).
* ``is_driver_busy()`` is the silence filter: non-urgent calls wait out heavy braking,
  big steering inputs and high lateral G.
* Tyre-temperature and braking-point callouts, which come straight off the telemetry frame.

Gaps, pace, sector times and race-control calls live in race_radio.py / race_events.py;
those objects call back into ``speak()`` so they share the same gating.
"""

from __future__ import annotations

import time
from typing import Any

import radio_phrases as say
import web_hud
from packets import CarTelemetryData, LapData
from pipeline.state import EngineState, wheel_group
from voice_engine import Priority


class RadioStage(EngineState):

    def speak(self, message: str, priority: Priority = Priority.NORMAL, **options: Any) -> None:
        """Queue speech; priority interrupts in the proactive keys break on-demand silence."""
        topic = options.get('key', message)
        if self.on_demand and topic not in self.config.radio.proactive_keys:
            return  # radio silence until the driver asks or critical priority event
        options['key'] = f'{self.capture_id or self.session_context}:{topic}'
        queued = self.voice.say(message, priority, **options)
        if not queued and not getattr(self.voice, 'is_running', False):
            web_hud.add_voice_log(message, status='muted')

    def is_driver_busy(self) -> bool:
        """Check if driver is under heavy braking or in a high-G corner (silence filter)."""
        busy = self.config.radio
        if self.latest_telemetry is None:
            return False
        if self.latest_telemetry.brake > busy.busy_brake:
            return True
        if abs(self.latest_telemetry.steer) > busy.busy_steer:
            return True
        return bool(self.latest_motion and abs(self.latest_motion.g_force_lateral) > busy.busy_lateral_g)

    # ── Tyre temperatures ──

    def analyze_tyres(self, telemetry: CarTelemetryData, lap: LapData) -> None:
        cfg = self.config.tyres
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

        dt = min(now - prev_time, 0.5) if prev_sample is not None and now > prev_time else 0.0
        critical_wheels = []
        for i in range(4):
            temp = temps[i]
            state = self.tyre_state
            state.lap_peak[i] = max(state.lap_peak[i], temp)
            if temp > cfg.high_c:
                state.lap_hot_secs[i] += dt
            elif temp < cfg.low_c:
                state.lap_cold_secs[i] += dt
            if temp >= cfg.critical_c and rise_rates[i] >= cfg.rise_rate_c_per_s:
                if state.hot_since[i] == 0.0:
                    state.hot_since[i] = now
                if now - state.hot_since[i] >= cfg.critical_sustain_s:
                    critical_wheels.append(i)
            elif temp < cfg.critical_c:
                state.hot_since[i] = 0.0

        # Mid-lap: one short call per lap, and only for a genuine, still-climbing overheat.
        if (critical_wheels and not self.tyre_state.critical_called_this_lap
                and now - self.tyre_state.last_critical_alert_time >= cfg.alert_cooldown_s):
            pronoun = 'it' if len(critical_wheels) == 1 else 'them'
            self.speak(f'{wheel_group(critical_wheels)} overheating, cool {pronoun}.',
                       Priority.CRITICAL, key='tyre_temperature', ttl=4, cooldown=25)
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
        if summary and (summary.split(',')[0] != (state.last_summary or '').split(',')[0]
                        or state.laps_since_summary >= self.config.tyres.summary_repeat_laps):
            self.speak(summary, Priority.LOW, key='tyre_summary', ttl=10, cooldown=0)
            state.laps_since_summary = 0
        state.last_summary = summary
        state.lap_peak = [0.0, 0.0, 0.0, 0.0]
        state.lap_hot_secs = [0.0, 0.0, 0.0, 0.0]
        state.lap_cold_secs = [0.0, 0.0, 0.0, 0.0]
        state.critical_called_this_lap = False

    # ── Braking points ──

    def check_braking_points(self, lap: LapData, telemetry: CarTelemetryData) -> None:
        if not self.braking_points or telemetry.speed < 60:
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
                self.speak(f'{message}, {dist_rnd} metres.', Priority.CRITICAL,
                           key='braking', ttl=1.0, cooldown=0)
