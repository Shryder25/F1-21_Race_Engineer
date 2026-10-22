"""Validate datagrams and pair telemetry/lap readings from the same game frame."""

import math
import struct
import time

from packets import (
    PacketHeader,
    parse_all_car_damage,
    parse_all_car_status,
    parse_all_lap_data,
    parse_all_positions,
    parse_car_setups,
    parse_final_classification,
    parse_participants,
    parse_player_motion,
    parse_player_telemetry,
    parse_race_event,
    parse_session_data,
    parse_session_history,
    parse_speed_trap_event,
    race_event_body_size,
)
from session_history import valid_history
from web_hud import TELEMETRY_TIMEOUT, update_web_state

PACKET_SIZES = {0: 1464, 1: 625, 2: 970, 4: 1257, 5: 1102, 6: 1347, 7: 1058, 8: 839, 10: 882, 11: 1155}
# Packet ID 3 (Event) is a fixed header plus a variable-size, event-code-specific body, so
# unlike every other packet type it does not have one single valid length.
EVENT_PACKET_SIZES = {b'FLBK': 36, b'SPTP': 40}
# Race events (penalties, retirements, fastest lap...) arrive padded to the size of the event
# union, which is 36 or 40 bytes depending on the game patch; both carry every body we read.
RACE_EVENT_PACKET_SIZES = (36, 40)


def newer_frame(frame: int, previous: int) -> bool:
    return 0 < (frame - previous) % (2**32) < 2**31


class TelemetryStream:
    def __init__(self, analysis):
        self.analysis = analysis
        self.session_uid = None
        self.retired_sessions = set()
        self.latest_session = None
        self.latest_lap = None
        self.latest_telemetry = None
        self.last_frames = {}
        self.pending = {}
        self.last_processed_frame = None
        self.last_pair_time = None
        self.context_start_time = 0.0
        self.last_error = None
        self.player_index = None
        self.last_flashback = None
        self.rewind_anchor = None
        self.last_session_time = None

    def check_timeout(self, now=None):
        now = time.monotonic() if now is None else now
        if self.last_pair_time is not None and now - self.last_pair_time > TELEMETRY_TIMEOUT:
            self.last_pair_time = None
            update_web_state({'connected': False})
            self.analysis.delta.current_lap_invalid = True
            self.analysis.previous_corner_sample = None
            self.pending.clear()
            self.analysis.radio.trajectories.clear()
            self.analysis.radio.gap_ahead_s = self.analysis.radio.gap_behind_s = None
            if hasattr(self.analysis.voice, 'clear'):
                self.analysis.voice.clear()

    def handle(self, data: bytes, now=None) -> bool:
        now = time.monotonic() if now is None else now
        self.check_timeout(now)
        self.last_error = None
        try:
            header = PacketHeader.from_bytes(data)
            packet_id = header.packet_id
            if packet_id != 3 and packet_id not in PACKET_SIZES:
                return False
            race_event = None
            if packet_id == 3:
                code = bytes(data[24:28])
                expected_size = EVENT_PACKET_SIZES.get(code)
                if expected_size is None:
                    if race_event_body_size(code) is None:
                        return False  # an event sub-type we don't need (and don't know the size of)
                    expected_size = len(data) if len(data) in RACE_EVENT_PACKET_SIZES else -1
            else:
                expected_size = PACKET_SIZES[packet_id]
            if (
                header.packet_format != 2021
                or header.packet_version != 1
                or len(data) != expected_size
                or not 0 <= header.player_car_index < 22
                or not math.isfinite(header.session_time)
            ):
                raise ValueError('Wrong packet format, version, length, timestamp, or player index')
            if packet_id == 0:
                payload = parse_player_motion(data, header.player_car_index)
                all_positions = parse_all_positions(data)
                if not all(
                    math.isfinite(x)
                    for x in (payload.world_position_x, payload.world_position_z, payload.g_force_lateral)
                ):
                    raise ValueError('Invalid car motion')
            elif packet_id == 1:
                payload = parse_session_data(data)
                if not (payload.track_id >= 0 and payload.track_length > 0):
                    return False
            elif packet_id == 2:
                all_laps = parse_all_lap_data(data)
                payload = all_laps[header.player_car_index]
                if not all(
                    math.isfinite(value)
                    for value in (payload.lap_distance, payload.total_distance, payload.safety_car_delta)
                ):
                    raise ValueError('Non-finite lap data')
                if payload.sector not in (0, 1, 2) or payload.pit_status not in (0, 1, 2):
                    raise ValueError('Invalid lap state')
                payload = all_laps
            elif packet_id == 3:
                if data[24:28] == b'SPTP':
                    speed_trap = parse_speed_trap_event(data)
                    if (
                        speed_trap is None
                        or not math.isfinite(speed_trap.speed_kph)
                        or speed_trap.vehicle_index != header.player_car_index
                    ):
                        return False
                    self.analysis.update_speed_trap(speed_trap.speed_kph)
                    return False
                if data[24:28] != b'FLBK':
                    race_event = payload = parse_race_event(data)
                    if race_event is None:
                        return False
                else:
                    payload = struct.unpack_from('<If', data, 28)
                if race_event is None and not math.isfinite(payload[1]):
                    return False
            elif packet_id == 4:
                if data[24] > 22:
                    raise ValueError('Invalid participant count')
                payload = parse_participants(data)
            elif packet_id == 5:
                payload = parse_car_setups(data)
                if not all(
                    math.isfinite(x)
                    for s in payload
                    for x in (
                        s.front_camber,
                        s.rear_camber,
                        s.front_toe,
                        s.rear_toe,
                        s.fuel_load,
                        *s.tyre_pressures,
                    )
                ):
                    raise ValueError('Invalid car setups')
            elif packet_id == 8:
                if data[24] > 22:
                    raise ValueError('Invalid classified car count')
                payload = parse_final_classification(data)
                if not all(math.isfinite(c.total_race_time_s) for c in payload):
                    raise ValueError('Invalid final classification')
            elif packet_id == 11:
                payload = parse_session_history(data)
                if not valid_history(payload):
                    raise ValueError('Invalid session history')
            elif packet_id == 7:
                all_status = parse_all_car_status(data)
                payload = all_status[header.player_car_index]
                if not all(
                    math.isfinite(x) and x >= 0
                    for x in (payload.fuel_in_tank, payload.fuel_capacity, payload.fuel_remaining_laps)
                ):
                    raise ValueError('Invalid car status')
            elif packet_id == 10:
                all_damage = parse_all_car_damage(data)
                payload = all_damage[header.player_car_index]
                if not all(math.isfinite(w) and 0 <= w <= 100 for w in payload.tyres_wear):
                    raise ValueError('Invalid car damage')
            else:
                payload = parse_player_telemetry(data, header.player_car_index)
                controls = (payload.throttle, payload.brake)
                if (
                    not all(math.isfinite(x) and 0 <= x <= 1 for x in controls)
                    or not math.isfinite(payload.steer)
                    or not -1 <= payload.steer <= 1
                    or not all(math.isfinite(x) and x >= 0 for x in payload.tyres_pressure)
                    or not -1 <= payload.gear <= 8
                    or payload.speed > 600
                ):
                    raise ValueError('Invalid car telemetry')
        except (ValueError, struct.error) as error:
            self.last_error = str(error)
            return False

        if header.session_uid != self.session_uid:
            # Only a session packet can establish which track incoming readings belong to.
            if packet_id != 1 or header.session_uid in self.retired_sessions:
                return False
            if self.session_uid is not None:
                self.retired_sessions.add(self.session_uid)
            self.session_uid = header.session_uid
            self.last_frames.clear()
            self.last_processed_frame = None
            self.rewind_anchor = None
            self.last_session_time = None

        if packet_id == 3 and race_event is None:
            signature = (header.session_uid, header.frame_identifier, payload)
            if signature != self.last_flashback:
                self.last_flashback = signature
                self.rewind_anchor = (
                    payload[0],
                    payload[1],
                    now,
                    max(header.session_time, self.last_session_time or header.session_time),
                )
                self.last_frames.clear()
                self.pending.clear()
                self.last_processed_frame = self.last_pair_time = None
                self.context_start_time = payload[1]
                self.analysis.delta.current_lap_invalid = True
                self.analysis.delta.recorded_from_start = False
                self.analysis.previous_corner_sample = None
                self.analysis.radio.reset(self.latest_session)
                self.analysis.race_control.reset_tracking()
                if hasattr(self.analysis, 'on_flashback'):
                    self.analysis.on_flashback()
                if hasattr(self.analysis.voice, 'clear'):
                    self.analysis.voice.clear()
                update_web_state({'connected': False, 'gap_ahead_s': None, 'gap_behind_s': None})
            return False

        if self.rewind_anchor is not None:
            restored_frame, restored_time, received_at, _ = self.rewind_anchor
            # Old-timeline datagrams may arrive after FLBK. They must not advance
            # frame filters before the restored simulation can reach their time.
            if (
                (
                    header.frame_identifier != restored_frame
                    and not newer_frame(header.frame_identifier, restored_frame)
                )
                or header.session_time < restored_time
                or header.session_time > restored_time + max(0, now - received_at) + 0.25
            ):
                return False

        if packet_id != 1 and (
            header.player_car_index != self.player_index
            or self.latest_session.game_paused
            or self.latest_session.is_spectating
        ):
            return False
        if race_event is not None:
            # Several events can share one frame (two penalties at once), so no frame filter.
            if header.session_time >= self.context_start_time:
                self.analysis.race_control.handle_event(race_event, self.player_index, header.session_time)
            return False
        # Session History comes one car per packet, so its frame filter is per car.
        frame_key = (11, payload.car_index) if packet_id == 11 else packet_id
        previous = self.last_frames.get(frame_key)
        if previous is not None and not newer_frame(header.frame_identifier, previous):
            return False
        self.last_frames[frame_key] = header.frame_identifier

        if packet_id == 1:
            previous_context = self.analysis.session_context
            self.analysis.update_session(payload, header.session_uid, header.player_car_index)
            self.player_index = header.player_car_index
            self.latest_session = payload
            if previous_context != self.analysis.session_context:
                self.pending.clear()
                self.latest_lap = self.latest_telemetry = None
                self.last_pair_time = None
                self.last_processed_frame = None
                self.context_start_time = header.session_time
            update_web_state({'paused': bool(payload.game_paused), 'spectating': bool(payload.is_spectating)})
            if payload.game_paused or payload.is_spectating:
                self.pending.clear()
                if hasattr(self.analysis.voice, 'clear'):
                    self.analysis.voice.clear()
                update_web_state({'connected': False})
            return False

        if packet_id in (0, 4, 5, 7, 8, 10, 11):
            # Motion, participants, strategy inputs and every car's history arrive on their own
            # cadence; they need no frame pairing.
            if header.session_time >= self.context_start_time:
                if packet_id == 0:
                    self.analysis.update_motion(payload, all_positions)
                elif packet_id == 4:
                    self.analysis.update_participants(payload)
                elif packet_id == 5:
                    self.analysis.update_car_setups(payload)
                elif packet_id == 8:
                    self.analysis.update_final_classification(payload)
                elif packet_id == 11:
                    self.analysis.update_session_history(payload)
                elif packet_id == 7:
                    self.analysis.update_car_status(payload, all_status)
                    self.analysis.race_control.update_status(all_status, self.player_index)
                else:
                    self.analysis.update_car_damage(payload, all_damage)
            return False

        if header.session_time < self.context_start_time:
            return False
        frame = header.frame_identifier
        if self.last_processed_frame is not None and not newer_frame(frame, self.last_processed_frame):
            return False
        pair = self.pending.setdefault(frame, {})
        pair[packet_id] = (payload, now, header.session_time)
        while len(self.pending) > 8:
            self.pending.pop(next(iter(self.pending)))
        if 2 not in pair or 6 not in pair:
            return False
        self.pending.pop(frame, None)
        laps, lap_arrival, lap_time = pair[2]
        lap = laps[self.player_index]
        telemetry, telemetry_arrival, telemetry_time = pair[6]
        if abs(lap_arrival - telemetry_arrival) > 0.5 or abs(lap_time - telemetry_time) > 0.05:
            return False
        self.latest_lap, self.latest_telemetry = lap, telemetry
        self.last_processed_frame = frame
        self.last_pair_time = now
        self.last_session_time = lap_time
        if self.rewind_anchor is not None and lap_time >= self.rewind_anchor[3]:
            self.rewind_anchor = None
        self.analysis.radio.update_field(laps, self.player_index, lap_time, now=now)
        self.analysis.race_control.update_field(laps, self.player_index, lap_time)
        self.analysis.update_field(laps, self.player_index, lap_time)
        self.analysis.update_telemetry(telemetry, lap)
        update_web_state({'last_update_monotonic': now})
        return True
