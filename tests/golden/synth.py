"""Deterministic full-field F1 2021 session simulator for golden-session recordings.

The repo's simulate_telemetry.py streams a few seconds of one car to a socket. Golden
sessions need whole sessions instead: all 22 cars lapping with their own pace, tyre wear,
pit stops, fuel burn and positions, plus scripted moments (rain, Safety Car, flags,
retirements, penalties, damage) so every engineer feature has something to react to.

Everything here is pure arithmetic on a fixed time step, so the same scenario always
produces the same bytes. Packets are built with the struct formats from packets.py, so
they pass the same validation as real game datagrams.

This is NOT a substitute for real game captures: the car physics are a simple speed
profile, not the game's. It exists to pin down the engineer's behaviour on known inputs.
"""

from __future__ import annotations

import copy
import math
import random
import struct
from dataclasses import dataclass, field

import sys
from pathlib import Path

_AI = Path(__file__).resolve().parents[2] / 'ai_engineer'
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from packets import (HEADER_FORMAT, CAR_TELEMETRY_FORMAT, LAP_DATA_FORMAT, CAR_STATUS_FORMAT,
                     CAR_DAMAGE_FORMAT, CAR_MOTION_FORMAT, PARTICIPANT_FORMAT, SPEED_TRAP_FORMAT,
                     CAR_SETUP_FORMAT, FINAL_CLASSIFICATION_FORMAT, LAP_HISTORY_FORMAT,
                     SESSION_HISTORY_MAX_LAPS, SESSION_HISTORY_MAX_STINTS,
                     TYRE_STINT_HISTORY_FORMAT)
from track_data import get_braking_points

# ── Compounds: visual id, actual id, pace offset (s), deg (s per lap of age), wear (% per lap)
COMPOUNDS = {
    'soft': (16, 17, -0.6, 0.12, 3.4),
    'medium': (17, 18, 0.0, 0.07, 2.4),
    'hard': (18, 19, 0.4, 0.05, 1.7),
    'inter': (7, 7, 0.0, 0.08, 2.8),
    'wet': (8, 8, 0.0, 0.06, 2.2),
}
SLICKS = ('soft', 'medium', 'hard')

DT = 0.05                 # integration step (s)
FRAME_EVERY = 4           # lap + telemetry packets every 4 steps -> 5 Hz (keeps files small)
STATUS_EVERY = 10         # car status / damage / motion -> 2 Hz
SESSION_EVERY = 20        # session packet -> 1 Hz
PARTICIPANTS_EVERY = 100  # participants -> every 5 s
PIT_SPEED = 80 / 3.6      # pit lane limiter (m/s)
PIT_ENTRY_M = 250         # pit lane starts this far before the line and ends this far after it
PIT_STATIONARY_S = 2.4
LAUNCH_ACCEL = 11.0       # m/s^2
FUEL_PER_LAP_KG = 1.7
FUEL_TIME_S_PER_KG = 0.03
SC_PACE_FACTOR = 1.45     # Safety Car laps are this much slower than a racing lap
SC_FOLLOW_M = 18.0        # minimum spacing behind the car ahead under the Safety Car
FOLLOW_M = 14.0           # closest a car gets to the one ahead without passing (~0.2 s)
OVERTAKE_DELTA_S = 0.4    # a car this much quicker per lap gets past
SESSION_UID = 0x60_1D_E0_00


# ──────────────────────────────────────────────
# Scenario description
# ──────────────────────────────────────────────

@dataclass
class Car:
    name: str
    pace: float = 0.0                 # seconds off the track's reference lap (fresh mediums, no fuel)
    compound: str = 'medium'
    grid: int = 0                     # starting position (1 = pole)
    pits: dict = field(default_factory=dict)   # lap number -> compound fitted at the end of it
    race_number: int = 0


@dataclass
class Scenario:
    name: str
    track_id: int
    track_length: int
    session_type: int                 # 10 race, 8 short qualifying
    total_laps: int
    cars: list                        # index 0 is the player
    weather: int = 0
    track_temp: int = 35
    air_temp: int = 24
    pit_window: tuple = (0, 0)
    fuel_kg: float = 20.0
    # Scripted moments, each (player progress in laps from the start, kind, value...).
    # Progress 2.5 means half way round lap 3.
    script: list = field(default_factory=list)
    end_after_laps: float | None = None   # stop early (qualifying), in player laps
    description: str = ''
    # Tyre/fuel behaviour (defaults reproduce the golden sessions byte for byte). Noise is
    # drawn from a generator seeded with `seed`, so a scenario still always gives the same bytes.
    wear_scale: float = 1.0           # multiplies every compound's wear rate (track severity)
    wear_growth: float = 0.0          # wear rate grows by this fraction per lap of tyre age
    wear_noise: float = 0.0           # lap-to-lap wear scatter, relative sd
    stint_noise: float = 0.0          # each set wears differently, relative sd of its rate
    fuel_per_lap_kg: float = FUEL_PER_LAP_KG
    fuel_noise: float = 0.0           # lap-to-lap fuel burn scatter, kg sd
    lap_noise_s: float = 0.0          # lap-to-lap pace scatter, s sd
    seed: int = 0
    # Also send Session History (one car per frame), Car Setups and, at the end of a race,
    # Final Classification. Off by default so the checked-in golden recordings stay as they are.
    history_packets: bool = False


def _q(metres: float) -> float:
    """Distances to the nearest 1/8 m: exact in binary, so the recordings compress well."""
    return round(metres * 8) / 8


# ──────────────────────────────────────────────
# Packet builders
# ──────────────────────────────────────────────

def header(packet_id, session_time, frame, player_index=0):
    return struct.pack(HEADER_FORMAT, 2021, 1, 18, 1, packet_id, SESSION_UID,
                       float(session_time), frame, player_index, 255)


def session_packet(sc: Scenario, t, frame, weather, safety_car):
    body = bytearray(625 - 24)
    struct.pack_into('<BbbBHBbB', body, 0, weather, sc.track_temp, sc.air_temp, sc.total_laps,
                     sc.track_length, sc.session_type, sc.track_id, 0)
    body[124] = safety_car
    struct.pack_into('<BBB', body, 589, sc.pit_window[0], sc.pit_window[1], 0)
    return header(1, t, frame) + bytes(body)


def event_packet(t, frame, code: bytes, fmt='', *values, size=36):
    body = code + (struct.pack(fmt, *values) if fmt else b'')
    data = header(3, t, frame) + body
    return data + bytes(size - len(data))


def participants_packet(t, frame, cars):
    body = bytearray(struct.pack('<B', len(cars)))
    for index in range(22):
        car = cars[index] if index < len(cars) else None
        name = (car.name if car else '').encode('utf-8')[:47]
        body += struct.pack(PARTICIPANT_FORMAT, int(index != 0), index, 0, index // 2, 0,
                            car.race_number if car else 0, 0, name, 1)
    return header(4, t, frame) + bytes(body)


# ──────────────────────────────────────────────
# Simulator
# ──────────────────────────────────────────────

class _CarState:
    def __init__(self, index, car: Car, start_distance, fuel):
        self.index = index
        self.car = car
        self.total = start_distance       # metres from the start line (negative on the grid)
        self.speed = 0.0
        self.lap_num = 1
        self.lap_start_t = None           # set at lights out
        self.last_lap_ms = 0
        self.s1_ms = self.s2_ms = 0
        self.sector = 0
        self.compound = car.compound
        self.age = 0
        self.wear = 0.0
        self.fuel = fuel
        self.pit_status = 0
        self.pit_stops = 0
        self.pit_timer_ms = 0
        self.stationary_left = 0.0
        self.pit_done_this_entry = False
        self.result = 2                   # active
        self.position = car.grid
        self.lap_invalid = 0
        self.best_lap_ms = 0
        self.finish_t = None
        self.stint_factor = 1.0           # this set's wear relative to the compound's rate
        self.lap_jitter = 0.0
        self.history_laps = []            # (lap ms, s1, s2, s3, valid flags) per completed lap
        visual, actual, *_ = COMPOUNDS[car.compound]
        self.history_stints = [[255, actual, visual]]   # [end lap (255 = on the car), actual, visual]


class SessionSimulator:
    """Run a Scenario and yield (session_time, datagram) in arrival order."""

    def __init__(self, scenario: Scenario):
        scenario = copy.deepcopy(scenario)        # script actions change car pace
        self.sc = scenario
        self.length = scenario.track_length
        self.profile = self._speed_profile()
        self.base_lap_s = sum(10.0 / v for v in self.profile)
        self.is_race = scenario.session_type in (10, 11, 12)
        self.weather = scenario.weather
        self.safety_car = 0
        self.player_flag = 1 if self.is_race else 0
        self.wing_damage = 0
        self.hot = None                  # (front_c, rear_c) while a hot-tyre window is open
        self.script = sorted(scenario.script, key=lambda item: item[0])
        self.frame = 0
        self.t = 0.0
        self.out = []
        self.chequered = False
        self.finish_order = []
        self.fastest_ms = None
        grid = sorted(range(len(scenario.cars)), key=lambda i: scenario.cars[i].grid)
        self.cars = []
        for index, car in enumerate(scenario.cars):
            slot = grid.index(index)
            start = -8.0 * slot - 5.0 if self.is_race else -40.0 * slot - 300.0
            self.cars.append(_CarState(index, car, start, scenario.fuel_kg))
        self.lights_out_t = 5.0 if self.is_race else 0.0
        noisy = scenario.wear_noise or scenario.stint_noise or scenario.fuel_noise or scenario.lap_noise_s
        self.rng = random.Random(scenario.seed) if noisy else None
        for car in self.cars:
            self._new_set(car)
        self.history_cursor = 0
        self.trap_m = 10.0 * max(range(len(self.profile)), key=lambda i: (self.profile[i], -i))

    def _new_set(self, car):
        if self.rng is not None:
            car.stint_factor = max(0.2, 1.0 + self.rng.gauss(0.0, self.sc.stint_noise))

    # ── track model ──
    def _speed_profile(self):
        """Speed (m/s) every 10 m: flat out on straights, braking at 4 g into each corner."""
        points = sorted(get_braking_points(self.sc.track_id), key=lambda p: p[0])
        corners = []
        for k, (distance, _name, _msg) in enumerate(points):
            apex_kph = 75 + (k * 37) % 140            # varied corner speeds, 75-215 km/h
            corners.append((distance + 60.0, apex_kph / 3.6))
        v_max = 315 / 3.6
        profile = []
        for step in range(int(math.ceil(self.length / 10.0))):
            d = step * 10.0
            v = v_max
            for apex, v_c in corners:
                for shift in (-self.length, 0.0, self.length):
                    x = d - (apex + shift)
                    if x < 0:
                        v = min(v, math.sqrt(v_c ** 2 + 2 * 38.0 * -x))
                    else:
                        v = min(v, math.sqrt(v_c ** 2 + 2 * 9.0 * x))
            profile.append(v)
        return profile

    def v_at(self, lap_distance):
        d = lap_distance % self.length
        return self.profile[min(int(d / 10.0), len(self.profile) - 1)]

    def lap_distance(self, car):
        if car.total < 0:
            return car.total
        return car.total - (car.lap_num - 1) * self.length

    def target_lap_s(self, car):
        _vis, _act, offset, deg, _wear = COMPOUNDS[car.compound]
        lap = (self.base_lap_s + car.car.pace + offset + deg * car.age + FUEL_TIME_S_PER_KG * car.fuel
               + car.lap_jitter)
        wet = self.weather >= 3
        if wet and car.compound in SLICKS:
            lap += 4.5 if self.weather == 3 else 10.0
        elif not wet and car.compound in ('inter', 'wet'):
            lap += 3.5
        elif self.weather >= 4 and car.compound == 'inter':
            lap += 2.0
        if self.safety_car:
            lap = max(lap, self.base_lap_s * SC_PACE_FACTOR)
        return lap

    # ── emit ──
    def emit(self, data):
        self.out.append((self.t, data))

    def player_progress(self):
        return self.cars[0].total / self.length

    # ── main loop ──
    def run(self):
        sc = self.sc
        step = 0
        self.frame = 1
        self.emit(session_packet(sc, 0.0, self.frame, self.weather, self.safety_car))
        self.emit(participants_packet(0.0, self.frame, sc.cars))
        if self.is_race:
            self.emit(event_packet(0.0, self.frame, b'SSTA'))
        limit_s = 60 * 60
        while self.t < limit_s:
            self.t = round(step * DT, 3)
            if self.is_race and abs(self.t - self.lights_out_t) < DT / 2:
                self.emit(event_packet(self.t, self.frame, b'LGOT'))
            self._run_script()
            self._advance()
            if step % FRAME_EVERY == 0:
                self.frame += 1
                self._emit_frame(step)
            if self._finished():
                break
            step += 1
        if self.is_race:
            self.emit(event_packet(self.t, self.frame, b'SEND'))
            if sc.history_packets:
                for car in self.cars:
                    self.emit(self.session_history_packet(car))
                self.emit(self.final_classification_packet())
        return self.out

    def _finished(self):
        player = self.cars[0]
        if self.sc.end_after_laps is not None:
            return self.player_progress() >= self.sc.end_after_laps
        return player.result == 3 and self.t - player.finish_t >= 6.0

    def _run_script(self):
        progress = self.player_progress()
        while self.script and self.script[0][0] <= progress:
            _at, kind, *args = self.script.pop(0)
            getattr(self, f'_do_{kind}')(*args)

    # Script actions --------------------------------------------------
    def _do_weather(self, value):
        self.weather = value

    def _do_safety_car(self, status):
        self.safety_car = status
        if self.is_race:
            self.emit(event_packet(self.t, self.frame, b'DRSD' if status else b'DRSE'))

    def _do_flag(self, flag):
        self.player_flag = flag

    def _do_retire(self, index):
        car = self.cars[index]
        car.result = 7
        car.speed = 0.0
        self.emit(event_packet(self.t, self.frame, b'RTMT', '<B', index))

    def _do_penalty(self, penalty_type, infringement, index, other=255, seconds=0):
        lap = self.cars[index].lap_num
        self.emit(event_packet(self.t, self.frame, b'PENA', '<7B', penalty_type, infringement,
                               index, other, seconds, lap, 0))

    def _do_invalidate(self):
        self.cars[0].lap_invalid = 1
        self._do_penalty(10, 7, 0)          # lap invalidated for corner cutting

    def _do_wing_damage(self, percent):
        self.wing_damage = percent

    def _do_hot_tyres(self, front_c, rear_c):
        self.hot = (front_c, rear_c)

    def _do_cool_tyres(self):
        self.hot = None

    def _do_pace(self, index, pace):
        self.cars[index].car.pace = pace

    def _do_drs(self, enabled):
        self.emit(event_packet(self.t, self.frame, b'DRSE' if enabled else b'DRSD'))

    # Physics ---------------------------------------------------------
    def _advance(self):
        if self.t < self.lights_out_t:
            return
        for car in self.cars:
            if car.lap_start_t is None:
                car.lap_start_t = self.lights_out_t
            if car.result != 2:
                continue
            self._advance_car(car)
        # Following: a car can't drive through the one ahead. Under the Safety Car nobody
        # passes and the pack bunches up; otherwise a car only gets by when it is clearly
        # quicker (an overtake), else it sits in the dirty air just behind.
        running = sorted((c for c in self.cars if c.result == 2 and not c.pit_status and c.total > 0),
                         key=lambda c: -c.total)
        for ahead, car in zip(running, running[1:]):
            if self.safety_car:
                limit = SC_FOLLOW_M
            elif self.target_lap_s(ahead) - self.target_lap_s(car) >= OVERTAKE_DELTA_S:
                continue
            else:
                limit = FOLLOW_M
            if car.total > ahead.total - limit:
                car.total = ahead.total - limit
                car.speed = min(car.speed, ahead.speed)
        self._update_positions()

    def _advance_car(self, car):
        lap_d = self.lap_distance(car)
        pit_planned = car.lap_num in car.car.pits
        in_pit_zone = (pit_planned and lap_d >= self.length - PIT_ENTRY_M) or (
            car.pit_status and lap_d < PIT_ENTRY_M)
        if car.stationary_left > 0:
            car.stationary_left -= DT
            car.pit_status = 2
            car.pit_timer_ms = int(round((PIT_STATIONARY_S - max(0.0, car.stationary_left)) * 1000))
            car.speed = 0.0
            if car.stationary_left <= 0:
                car.pit_status = 1
                car.stationary_left = 0.0
            return
        if in_pit_zone:
            car.pit_status = car.pit_status or 1
            target = PIT_SPEED
        else:
            if car.pit_status:
                car.pit_status = 0
                car.pit_timer_ms = 0
                car.pit_done_this_entry = False
            scale = self.base_lap_s / self.target_lap_s(car)
            target = self.v_at(max(lap_d, 0.0)) * scale
        car.speed = min(target, car.speed + LAUNCH_ACCEL * DT)
        before = car.total
        car.total += car.speed * DT
        self._cross_marks(car, before)

    def _cross_marks(self, car, before):
        lap_start = (car.lap_num - 1) * self.length
        s1, s2 = lap_start + self.length / 3, lap_start + 2 * self.length / 3
        elapsed_ms = int(round((self.t - car.lap_start_t) * 1000))
        if before < s1 <= car.total:
            car.s1_ms, car.sector = elapsed_ms, 1
        if before < s2 <= car.total:
            car.s2_ms, car.sector = elapsed_ms - car.s1_ms, 2
        if car.index == 0 and not car.pit_status and before < lap_start + self.trap_m <= car.total:
            self.emit(self.speed_trap())
        line = car.lap_num * self.length
        if car.pit_status and car.lap_num in car.car.pits and not car.pit_done_this_entry \
                and before < line - 5 <= car.total:
            # Pit box sits just before the line: stop, change tyres, carry on.
            car.total = line - 5
            car.stationary_left = PIT_STATIONARY_S
            car.pit_done_this_entry = True
            car.history_stints[-1][0] = car.lap_num          # this is the in-lap
            car.compound = car.car.pits[car.lap_num]
            visual, actual, *_ = COMPOUNDS[car.compound]
            car.history_stints.append([255, actual, visual])
            car.age = 0
            car.wear = 0.0
            self._new_set(car)
            car.pit_stops += 1
            if car.index == 0:
                self.wing_damage = 0                # new nose
            return
        if before < line <= car.total:
            self._complete_lap(car, elapsed_ms)

    def _complete_lap(self, car, elapsed_ms):
        s1, s2 = car.s1_ms, car.s2_ms
        lap_valid = not car.lap_invalid and not car.pit_status
        car.history_laps.append((elapsed_ms, s1, s2, max(0, elapsed_ms - s1 - s2), 0x0F if lap_valid else 0))
        car.last_lap_ms = elapsed_ms
        car.lap_start_t = self.t
        car.lap_num += 1
        car.sector = 0
        car.s1_ms = car.s2_ms = 0
        sc = self.sc
        wear = COMPOUNDS[car.compound][4] * sc.wear_scale * (1.0 + sc.wear_growth * car.age)
        burn = sc.fuel_per_lap_kg
        if self.rng is not None:
            wear *= car.stint_factor * max(0.0, 1.0 + self.rng.gauss(0.0, sc.wear_noise))
            burn = max(0.0, burn + self.rng.gauss(0.0, sc.fuel_noise))
            car.lap_jitter = self.rng.gauss(0.0, sc.lap_noise_s)
        car.age += 1
        car.wear += wear
        car.fuel = max(0.0, car.fuel - burn)
        valid = not car.lap_invalid and not car.pit_status
        car.lap_invalid = 0
        if self.is_race and valid and car.lap_num > 2:      # lap one is a standing start
            if self.fastest_ms is None or elapsed_ms < self.fastest_ms:
                self.fastest_ms = elapsed_ms
                self.emit(event_packet(self.t, self.frame, b'FTLP', '<Bf', car.index, elapsed_ms / 1000))
        if self.is_race:
            completed = car.lap_num - 1
            if not self.chequered and completed >= self.sc.total_laps:
                self.chequered = True
                self.emit(event_packet(self.t, self.frame, b'CHQF'))
                self.emit(event_packet(self.t, self.frame, b'RCWN', '<B', car.index))
            if self.chequered:
                car.result = 3
                car.finish_t = self.t
                car.speed = 0.0
                self.finish_order.append(car.index)

    def _update_positions(self):
        finished = list(self.finish_order)
        running = sorted((c for c in self.cars if c.result == 2), key=lambda c: -c.total)
        retired = [c for c in self.cars if c.result not in (2, 3)]
        order = [self.cars[i] for i in finished] + running + retired
        if self.t < self.lights_out_t:
            order = sorted(self.cars, key=lambda c: c.car.grid)
        for position, car in enumerate(order, start=1):
            car.position = position

    # Packets ---------------------------------------------------------
    def _emit_frame(self, step):
        sc, t, frame = self.sc, self.t, self.frame
        if step % SESSION_EVERY == 0:
            self.emit(session_packet(sc, t, frame, self.weather, self.safety_car))
        if step % PARTICIPANTS_EVERY == 0:
            self.emit(participants_packet(t, frame, sc.cars))
            if sc.history_packets:
                self.emit(self.car_setups_packet())
        if sc.history_packets and self.t >= self.lights_out_t:
            self.emit(self.session_history_packet(self.cars[self.history_cursor % len(self.cars)]))
            self.history_cursor += 1
        if step % STATUS_EVERY == 0:
            self.emit(self._motion_packet())
            self.emit(self._status_packet())
            self.emit(self._damage_packet())
        self.emit(self._lap_packet())
        self.emit(self._telemetry_packet())

    def _lap_packet(self):
        body = bytearray()
        for index in range(22):
            if index >= len(self.cars):
                body += bytes(struct.calcsize(LAP_DATA_FORMAT))
                continue
            car = self.cars[index]
            lap_d = self.lap_distance(car)
            started = self.t >= self.lights_out_t
            current_ms = int(round((self.t - car.lap_start_t) * 1000)) if started and car.lap_start_t is not None else 0
            if car.result == 3:
                current_ms = 0
            driver_status = 4 if self.is_race else (3 if car.lap_num == 1 else 1)
            if car.pit_status:
                driver_status = 2 if car.stationary_left > 0 or self.lap_distance(car) > 0 else 4
            body += struct.pack(
                LAP_DATA_FORMAT, car.last_lap_ms, current_ms, car.s1_ms if car.sector >= 1 else 0,
                car.s2_ms if car.sector >= 2 else 0, _q(lap_d), _q(car.total),
                0.0, car.position, car.lap_num, car.pit_status, car.pit_stops, car.sector,
                car.lap_invalid, 0, 0, 0, 0, car.car.grid, driver_status, car.result,
                int(bool(car.pit_status)), 0, car.pit_timer_ms if car.pit_status == 2 else 0, 0)
        return header(2, self.t, self.frame) + bytes(body)

    def _player_inputs(self, car):
        lap_d = max(self.lap_distance(car), 0.0)
        now_v, next_v = self.v_at(lap_d), self.v_at(lap_d + 30.0)
        if car.speed < 1.0:
            return 0.0, 0.0, 0.0, 0 if car.total < 0 else 1
        if next_v < now_v - 2.0 and not car.pit_status:
            throttle, brake, steer = 0.0, 0.85, 0.05
        elif now_v < 200 / 3.6:
            throttle, brake, steer = 0.6, 0.0, 0.35
        else:
            throttle, brake, steer = 1.0, 0.0, 0.0
        kph = car.speed * 3.6
        gear = max(1, min(8, int(kph / 40) + 1))
        return throttle, brake, steer, gear

    def _telemetry_packet(self):
        body = bytearray()
        for index in range(22):
            if index >= len(self.cars) or index != 0:
                # Only the player's telemetry is read; rivals' blocks stay zero to keep the
                # checked-in recordings small (the game fills them, the engineer ignores them).
                body += bytes(struct.calcsize(CAR_TELEMETRY_FORMAT))
                continue
            car = self.cars[index]
            throttle, brake, steer, gear = self._player_inputs(car)
            kph = int(round(car.speed * 3.6))
            rpm = 4000 + int(min(1.0, (kph % 40) / 40 + 0.4) * 7500) if kph else 4000
            phase = (car.total % 1000) / 1000 * 2 * math.pi
            base = 92 + int(round(3 * math.sin(phase)))
            wet = self.weather >= 3 and car.compound in SLICKS
            front = rear = base - (12 if wet else 0)
            if index == 0 and self.hot:
                front, rear = self.hot
            inner = (rear - 4, rear - 4, front - 4, front - 4)
            body += struct.pack(
                CAR_TELEMETRY_FORMAT, kph, throttle, steer, brake, 0, gear, rpm, 0,
                min(100, kph // 3), 0, 450, 450, 520, 520,
                rear, rear, front, front, *inner, 105, 22.0, 22.0, 23.5, 23.5, 0, 0, 0, 0)
        return header(6, self.t, self.frame) + bytes(body) + b'\x00\x00\x00'

    def _status_packet(self):
        body = bytearray()
        for index in range(22):
            if index >= len(self.cars):
                body += bytes(struct.calcsize(CAR_STATUS_FORMAT))
                continue
            car = self.cars[index]
            visual, actual, *_ = COMPOUNDS[car.compound]
            flag = self.player_flag if index == 0 else (1 if self.is_race else 0)
            body += struct.pack(
                CAR_STATUS_FORMAT, 0, 0, 1, 56, int(bool(car.pit_status)), car.fuel, 110.0,
                car.fuel / FUEL_PER_LAP_KG, 13000, 4000, 8, 0, 0, actual, visual, car.age, flag,
                2_000_000.0, 1, 0.0, 0.0, 0.0, 0)
        return header(7, self.t, self.frame) + bytes(body)

    def _damage_packet(self):
        body = bytearray()
        for index in range(22):
            if index >= len(self.cars):
                body += bytes(struct.calcsize(CAR_DAMAGE_FORMAT))
                continue
            car = self.cars[index]
            w = round(car.wear, 2)
            wear = (w * 1.05, w * 1.08, w * 0.92, w)       # RL, RR, FL, FR
            wing = self.wing_damage if index == 0 else 0
            body += struct.pack(CAR_DAMAGE_FORMAT, *wear, 0, 0, 0, 0, 0, 0, 0, 0,
                                wing, wing // 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        return header(10, self.t, self.frame) + bytes(body)

    def _motion_packet(self):
        body = bytearray()
        radius = self.length / (2 * math.pi)
        for index in range(22):
            if index >= len(self.cars) or index != 0:      # only the player's motion is read
                body += bytes(struct.calcsize(CAR_MOTION_FORMAT))
                continue
            car = self.cars[index]
            angle = (max(self.lap_distance(car), 0.0) / self.length) * 2 * math.pi
            x, z = radius * math.cos(angle), radius * 0.6 * math.sin(angle)
            body += struct.pack(CAR_MOTION_FORMAT, x, 0.0, z, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0,
                                0.8 if car.speed < 50 else 0.3, 0.0, 1.0, angle, 0.0, 0.0)
        player_extra = bytes(1464 - 24 - len(body))
        return header(0, self.t, self.frame) + bytes(body) + player_extra

    def session_history_packet(self, car):
        """Session History (id 11) for one car: completed laps, the lap in progress, stints."""
        laps = list(car.history_laps)
        if car.result == 2:
            laps.append((0, car.s1_ms if car.sector >= 1 else 0, car.s2_ms if car.sector >= 2 else 0, 0, 0))
        laps = laps[:SESSION_HISTORY_MAX_LAPS]
        timed = [(t, n) for n, (t, *_rest) in enumerate(laps, start=1) if t and _rest[-1]]
        best = min(timed)[1] if timed else 0
        stints = car.history_stints[:SESSION_HISTORY_MAX_STINTS]
        body = struct.pack('<7B', car.index, len(laps), len(stints), best, best, best, best)
        for n in range(SESSION_HISTORY_MAX_LAPS):
            body += struct.pack(LAP_HISTORY_FORMAT, *(laps[n] if n < len(laps) else (0, 0, 0, 0, 0)))
        for n in range(SESSION_HISTORY_MAX_STINTS):
            body += struct.pack(TYRE_STINT_HISTORY_FORMAT, *(stints[n] if n < len(stints) else (0, 0, 0)))
        return header(11, self.t, self.frame) + body

    def car_setups_packet(self):
        body = bytearray()
        for index in range(22):
            if index >= len(self.cars):
                body += bytes(struct.calcsize(CAR_SETUP_FORMAT))
                continue
            wing = 5 + index % 3
            body += struct.pack(CAR_SETUP_FORMAT, wing, wing + 1, 60, 55, -3.0, -1.5, 0.05, 0.2,
                                5, 4, 6, 5, 3, 6, 100, 56, 22.0, 22.0, 23.5, 23.5, 0, self.sc.fuel_kg)
        return header(5, self.t, self.frame) + bytes(body)

    def final_classification_packet(self):
        body = bytearray(struct.pack('<B', len(self.cars)))
        for index in range(22):
            if index >= len(self.cars):
                body += bytes(struct.calcsize(FINAL_CLASSIFICATION_FORMAT))
                continue
            car = self.cars[index]
            timed = [t for t, *_rest in car.history_laps if t and _rest[-1]]
            stints = car.history_stints[:8]
            actual = [s[1] for s in stints] + [0] * (8 - len(stints))
            visual = [s[2] for s in stints] + [0] * (8 - len(stints))
            total = (car.finish_t or self.t) - self.lights_out_t
            body += struct.pack(FINAL_CLASSIFICATION_FORMAT, car.position, len(car.history_laps),
                                car.car.grid, 0, car.pit_stops, 3 if car.result == 3 else car.result,
                                min(timed) if timed else 0, total, 0, 0, len(stints), *actual, *visual)
        return header(8, self.t, self.frame) + bytes(body)

    def speed_trap(self):
        """Speed trap ('SPTP') event as the player passes the fastest point of the lap."""
        car = self.cars[0]
        return event_packet(self.t, self.frame, b'SPTP', SPEED_TRAP_FORMAT, 0,
                            float(round(car.speed * 3.6, 1)), 1, 1, 0, 0.0, size=40)


def simulate(scenario: Scenario):
    """All datagrams of a scenario as a list of (arrival_time_s, bytes)."""
    return SessionSimulator(scenario).run()
