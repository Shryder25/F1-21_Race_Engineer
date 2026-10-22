"""Laps stage: live delta, sector and lap times, personal bests, corner-by-corner diagnosis.

Runs on every telemetry frame. A lap only becomes a reference (best lap, best sectors,
best corners) when it was recorded from the line, stayed valid, and ran the full distance.
"""

from __future__ import annotations

import copy
import time
from bisect import bisect_left

import radio_phrases as say
from packets import CarTelemetryData, LapData
from pipeline.outputs import OutputsStage
from pipeline.state import JSON
from voice_engine import Priority


class LapsStage(OutputsStage):
    # ── Lap delta and sectors ──

    def update_delta(self, lap: LapData, telemetry: CarTelemetryData) -> None:
        now = time.time()
        dist = lap.lap_distance
        elapsed_ms = lap.current_lap_time_ms

        if dist < 0 or elapsed_ms <= 0:
            return

        # New lap check
        if lap.current_lap_num != self.delta.current_lap_num:
            self._on_new_lap(lap)

        self.delta.current_lap_invalid |= bool(
            lap.current_lap_invalid
            or lap.pit_status
            or lap.driver_status in (0, 2, 3)
            or (self.session and self.session.safety_car_status)
        )
        max_gap = max(100, (self.session.track_length if self.session else 0) * 0.025)
        if self.delta.last_distance is None:
            self.delta.recorded_from_start = dist <= 100 and elapsed_ms <= 5000
        elif (
            dist < self.delta.last_distance - 1
            or dist - self.delta.last_distance > max_gap
            or elapsed_ms < self.delta.last_elapsed_ms
        ):
            self.delta.current_lap_invalid = True
        self.delta.last_distance = dist
        self.delta.last_elapsed_ms = elapsed_ms
        if lap.sector >= 1 and lap.sector1_time_ms > 0:
            self.current_sector_times[1] = lap.sector1_time_ms
        if lap.sector >= 2 and lap.sector2_time_ms > 0:
            self.current_sector_times[2] = lap.sector2_time_ms

        # Sector reporting (Sector 0 -> 1 -> 2)
        if lap.sector != self.last_sector:
            if lap.sector == self.last_sector + 1 and lap.lap_distance > 0:
                self.sector_starts[lap.sector + 1] = lap.lap_distance
            self._on_sector_change(lap)
            self.last_sector = lap.sector

        # Sample at regular distance intervals
        if dist - self.delta.last_sample_distance >= self.config.laps.delta_sample_interval_m:
            self._sample(lap, telemetry, now, max_gap)

    def _sample(self, lap: LapData, telemetry: CarTelemetryData, now: float, max_gap: float) -> None:
        """Record one speed-trace sample and compare it with the reference lap."""
        dist, elapsed_ms = lap.lap_distance, lap.current_lap_time_ms
        self.delta.last_sample_distance = dist
        self.delta.current_lap_times[dist] = elapsed_ms
        self.delta.current_lap_speeds.append([round(dist, 1), telemetry.speed])
        self.delta.current_lap_samples.append(
            {
                'distance': dist,
                'elapsed_ms': elapsed_ms,
                'speed': telemetry.speed,
                'throttle': telemetry.throttle,
                'brake': telemetry.brake,
                'gear': telemetry.gear,
                'rpm': telemetry.engine_rpm,
                'tyre_surface_temps': list(telemetry.tyres_surface_temperature),
                'tyre_inner_temps': list(telemetry.tyres_inner_temperature),
                'tyre_pressures': list(telemetry.tyres_pressure),
            }
        )

        # Calculate live delta if baseline exists
        reference = self.delta.best_lap_times
        keys = sorted(reference)
        index = bisect_left(keys, dist)
        reference_ms: float | None = reference.get(dist)
        if reference_ms is None and 0 < index < len(keys):
            left, right = keys[index - 1], keys[index]
            if right - left <= max_gap:
                fraction = (dist - left) / (right - left)
                reference_ms = reference[left] + fraction * (reference[right] - reference[left])
        self.delta.last_reported_delta_ms = None
        if reference_ms is None:
            return
        delta_ms = round(elapsed_ms - reference_ms)
        self.delta.last_reported_delta_ms = delta_ms

        # Voice delta callout periodically
        if now - self.delta.last_delta_report_time < self.config.radio.delta_report_interval_s:
            return
        self.delta.last_delta_report_time = now
        amount = say.time_amount(delta_ms / 1000.0)
        if delta_ms < -100:
            self.speak(
                self.phrases.pick(
                    'delta_up',
                    (
                        f'{amount.capitalize()} up on your best. Keep it going.',
                        f"You're {amount} up, good lap so far.",
                    ),
                ),
                Priority.LOW,
                key='live_delta',
                ttl=6,
                cooldown=14,
            )
        elif delta_ms > 150:
            self.speak(
                f'{amount.capitalize()} down on your best.',
                Priority.LOW,
                key='live_delta',
                ttl=6,
                cooldown=14,
            )

    def _on_sector_change(self, lap: LapData) -> None:
        """Announce sector timing and delta."""
        if not self.delta.recorded_from_start or self.delta.current_lap_invalid:
            return
        if lap.sector in (1, 2):
            milliseconds = lap.sector1_time_ms if lap.sector == 1 else lap.sector2_time_ms
            self.radio.report_sector(
                lap.current_lap_num, lap.sector, milliseconds, self.best_sector_times.get(lap.sector)
            )

    # ── Lap completion ──

    def _on_new_lap(self, lap: LapData) -> None:
        """Handle lap completion and save best baseline."""
        prev_lap_num = self.delta.current_lap_num
        if prev_lap_num > 0:
            self._capture_outline(prev_lap_num, lap.current_lap_num)
        self.delta.current_lap_num = lap.current_lap_num
        self.announced_braking.clear()
        if prev_lap_num > 0:
            self._report_tyre_lap()

        # If previous lap completed with valid lap time
        track_length = self.session.track_length if self.session else 0
        complete = (
            prev_lap_num > 0
            and lap.current_lap_num == prev_lap_num + 1
            and self.delta.recorded_from_start
            and not self.delta.current_lap_invalid
            and track_length > 0
            and self.delta.last_distance is not None
            and self.delta.last_distance >= track_length - 100
            and lap.last_lap_time_ms >= self.delta.last_elapsed_ms
        )
        sectors = dict(self.current_sector_times)
        if 1 in sectors and 2 in sectors and lap.last_lap_time_ms > sectors[1] + sectors[2]:
            sectors[3] = lap.last_lap_time_ms - sectors[1] - sectors[2]
        if prev_lap_num > 0 and lap.last_lap_time_ms > 0 and sectors:
            self.last_lap_sectors = dict(sectors)
            self.last_lap_sector_refs = dict(self.best_sector_times)
        if prev_lap_num > 0 and lap.last_lap_time_ms > 0:
            self._record_lap_stats(
                prev_lap_num, lap.last_lap_time_ms, complete, sectors, self.current_lap_speed_trap_kph
            )
        self.last_speed_trap_kph = self.current_lap_speed_trap_kph
        self.current_lap_speed_trap_kph = None
        if prev_lap_num > 0 and self.delta.current_lap_samples:
            self._save_lap_recording(lap, prev_lap_num, complete, sectors)
        if complete and lap.last_lap_time_ms > 0 and self.delta.current_lap_times:
            self._report_completed_lap(lap.last_lap_time_ms, prev_lap_num, sectors)

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
        self._clear_lap_trail()
        self.map_lap_has_pit = bool(lap.pit_status)
        self.last_map_dist = -999.0
        self._map_revision += 1

    def _save_lap_recording(
        self, lap: LapData, lap_num: int, complete: bool, sectors: dict[int, int]
    ) -> None:
        """The finished lap's trace and corners go to the history database."""
        history, capture = self.history, self.capture_id
        if not (history and capture):
            return
        recording: JSON = {
            'lap_time_ms': lap.last_lap_time_ms,
            'lap_num': lap_num,
            'times': self.delta.current_lap_times,
            'speeds': self.delta.current_lap_speeds,
            'samples': self.delta.current_lap_samples,
            'corners': self.current_corner_data,
            'sectors': sectors,
        }
        self._write_history(
            'lap', lambda: history.record_lap(capture, lap_num, lap.last_lap_time_ms, complete, recording)
        )
        if complete and lap_num % self.config.laps.profile_refresh_laps == 0:
            self._refresh_driver_profile()  # learns from the laps just driven

    def _report_completed_lap(self, last_ms: int, lap_num: int, sectors: dict[int, int]) -> None:
        """A clean, complete lap: radio the time, and keep it as the reference if it's a best."""
        self.radio.completed_lap(last_ms, lap_num, sectors, self.best_sector_times)
        for sector, milliseconds in sectors.items():
            self.best_sector_times[sector] = min(
                self.best_sector_times.get(sector, milliseconds), milliseconds
            )
        spoken = say.lap_time(last_ms)

        # Store as best if first completed lap or faster
        if self.delta.best_lap_time_ms == 0 or last_ms < self.delta.best_lap_time_ms:
            self.delta.best_lap_times = dict(self.delta.current_lap_times)
            self.delta.best_lap_speeds = list(self.delta.current_lap_speeds)
            self.delta.best_lap_time_ms = last_ms
            self.delta.best_lap_num = lap_num
            self.best_corner_data = copy.deepcopy(self.current_corner_data)
            if self.current_lap_map_points:
                self.track_map_points = list(self.current_lap_map_points)
                self.braking_zones_best = list(self.current_lap_braking_zones)
                self.ers_zones_best = list(self.current_lap_ers_zones)

            self.speak(
                self.phrases.pick(
                    'personal_best',
                    (
                        f"Good lap! {spoken.capitalize()}, that's a personal best.",
                        f'{spoken.capitalize()}. Personal best, nice job.',
                    ),
                ),
                Priority.TIMING,
                key='lap_time',
                ttl=10,
                cooldown=0,
            )
        else:
            diff_s = (last_ms - self.delta.best_lap_time_ms) / 1000.0
            if diff_s < 0.15:
                message = f'{spoken.capitalize()}. Just {say.time_amount(diff_s)} off your best.'
            else:
                message = f'{spoken.capitalize()}. {say.time_amount(diff_s).capitalize()} off your best.'
            self.speak(message, Priority.TIMING, key='lap_time', ttl=10, cooldown=0)

    def update_speed_trap(self, speed_kph: float) -> None:
        """Record the fastest Speed Trap event ('SPTP') seen so far on the current lap."""
        if self.current_lap_speed_trap_kph is None or speed_kph > self.current_lap_speed_trap_kph:
            self.current_lap_speed_trap_kph = speed_kph

    # ── Corner diagnostics ("Why You Are Slower") ──

    def update_corners(self, telemetry: CarTelemetryData, lap: LapData) -> None:
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

        def crossing_time(boundary: float) -> float:
            return prev_time + (elapsed - prev_time) * (boundary - prev_dist) / (dist - prev_dist)

        for c in self.corners:
            c_name = c['name']
            c_start = c['start']
            c_end = c['end']
            if prev_dist <= c_start < dist and c_name not in self.current_corner_data:
                self.current_corner_data[c_name] = {
                    'min_speed': speed,
                    'brake_point': self.brake_start_distance,
                    'start_time_ms': crossing_time(c_start),
                    'end_time_ms': None,
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

    def _diagnose_corner(self, c_name: str, current_data: JSON, ref_def: JSON) -> None:
        """Diagnose time loss in corner and update HUD."""
        start_t = current_data.get('start_time_ms', 0)
        end_t = current_data.get('end_time_ms', 0)
        curr_dur = (end_t - start_t) / 1000.0

        ref_apex = ref_def.get('ref_apex')  # None for corners without a measured apex speed
        ref_brake = ref_def['ref_brake']

        curr_apex = current_data.get('min_speed', 0)
        curr_brake = current_data.get('brake_point', ref_brake)

        # Baseline comparison
        best_d = self.best_corner_data.get(c_name)
        if best_d and best_d.get('end_time_ms') is not None:
            best_apex = best_d['min_speed']
            ref_brake = best_d.get('brake_point')
            best_dur = (best_d['end_time_ms'] - best_d['start_time_ms']) / 1000.0
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
            advice.append(f'Braked {diff}m before the {reference_source} braking point.')
        if best_apex is None:
            pass  # no reference apex yet: this lap becomes the reference
        elif curr_apex < (best_apex - 8):
            diff = best_apex - curr_apex
            advice.append(f'Apex speed {diff} km/h below the {reference_source}.')
        elif curr_apex >= best_apex:
            advice.append('Good apex speed maintained!')

        if not advice:
            advice_text = (
                'Reference lap not set yet; this lap will be your benchmark.'
                if best_apex is None
                else 'Good execution through corner.'
            )
        else:
            advice_text = ' '.join(advice)

        # Update HUD corner list
        diag_entry = {
            'name': c_name,
            'delta': delta_s,
            'reference_source': reference_source,
            'current_apex': curr_apex,
            'best_apex': best_apex,
            'brake_point': curr_brake,
            'advice': advice_text,
        }

        # Keep latest corner diagnostics
        existing_idx = next(
            (i for i, item in enumerate(self.corner_diagnostics) if item['name'] == c_name), None
        )
        if existing_idx is not None:
            self.corner_diagnostics[existing_idx] = diag_entry
        else:
            self.corner_diagnostics.append(diag_entry)
