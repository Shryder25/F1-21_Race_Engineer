"""
F1 2021 UDP Telemetry Packet Definitions
=========================================
Pure-Python struct-based parsing of the F1 2021 UDP telemetry protocol.
No external telemetry library needed — we parse the raw binary directly.

Reference: Codemasters F1 2021 UDP Specification
https://forums.codemasters.com/topic/80231-f1-2021-udp-specification/

Packet IDs:
  0 = Motion
  1 = Session
  2 = Lap Data
  3 = Event
  4 = Participants
  5 = Car Setups
  6 = Car Telemetry
  7 = Car Status
  8 = Final Classification
  9 = Lobby Info
 10 = Car Damage
 11 = Session History
"""

import math
import struct
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Packet Header (24 bytes)
# ──────────────────────────────────────────────
HEADER_FORMAT = "<HBBBBQfIBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes

# Packet ID 3 (Event) sub-type payloads start right after the header with a 4-byte
# ASCII event code, then a sub-type-specific body. Speed Trap ('SPTP') —
# vehicleIdx(B), speed_kph(f), isOverallFastestInSession(B), isDriverFastestInSession(B),
# fastestVehicleIdx(B), fastestSpeedInSession_kph(f) — 12 bytes, at offset HEADER_SIZE+4.
SPEED_TRAP_FORMAT = "<BfBBBf"
SPEED_TRAP_SIZE = struct.calcsize(SPEED_TRAP_FORMAT)


@dataclass
class SpeedTrapEvent:
    vehicle_index: int
    speed_kph: float
    is_overall_fastest: bool
    is_driver_fastest: bool
    fastest_vehicle_index: int
    fastest_speed_kph: float


def parse_speed_trap_event(data: bytes) -> SpeedTrapEvent | None:
    """Parse a Speed Trap ('SPTP') event packet's body, or None if it isn't one."""
    if data[HEADER_SIZE:HEADER_SIZE + 4] != b'SPTP':
        return None
    values = struct.unpack_from(SPEED_TRAP_FORMAT, data, HEADER_SIZE + 4)
    return SpeedTrapEvent(values[0], values[1], bool(values[2]), bool(values[3]), values[4], values[5])


# Buttons ('BUTN'): the controller buttons held right now, as a uint32 bit mask
# (game_buttons.BUTTON_FLAGS). Sent whenever the button state changes.
BUTTONS_FORMAT = "<I"


def parse_buttons_event(data: bytes) -> int | None:
    """The button mask of a Buttons ('BUTN') event, or None if it isn't one."""
    if data[HEADER_SIZE:HEADER_SIZE + 4] != b'BUTN' or len(data) < HEADER_SIZE + 8:
        return None
    return struct.unpack_from(BUTTONS_FORMAT, data, HEADER_SIZE + 4)[0]


# Every other Event ('EVNT') sub-type the race engineer reacts to. Body layouts follow the
# F1 2021 spec's EventDataDetails union; all fields are uint8 except FTLP's float lap time.
RACE_EVENT_BODY_FORMATS = {
    b'SSTA': '',          # session started
    b'SEND': '',          # session ended
    b'FTLP': '<Bf',       # fastest lap: vehicleIdx, lapTime (s)
    b'RTMT': '<B',        # retirement: vehicleIdx
    b'DRSE': '',          # DRS enabled
    b'DRSD': '',          # DRS disabled
    b'TMPT': '<B',        # team mate in pits: vehicleIdx
    b'CHQF': '',          # chequered flag
    b'RCWN': '<B',        # race winner: vehicleIdx
    b'PENA': '<7B',       # penalty: type, infringement, vehicleIdx, otherVehicleIdx, time, lapNum, placesGained
    b'STLG': '<B',        # start lights: numLights
    b'LGOT': '',          # lights out
    b'DTSV': '<B',        # drive through served: vehicleIdx
    b'SGSV': '<B',        # stop go served: vehicleIdx
}

PENALTY_TYPES = {
    0: 'Drive through', 1: 'Stop Go', 2: 'Grid penalty', 3: 'Penalty reminder',
    4: 'Time penalty', 5: 'Warning', 6: 'Disqualified', 7: 'Removed from formation lap',
    8: 'Parked too long timer', 9: 'Tyre regulations', 10: 'This lap invalidated',
    11: 'This and next lap invalidated', 12: 'This lap invalidated without reason',
    13: 'This and next lap invalidated without reason', 14: 'This and previous lap invalidated',
    15: 'This and previous lap invalidated without reason', 16: 'Retired', 17: 'Black flag timer',
}

INFRINGEMENT_TYPES = {
    0: 'blocking by slow driving', 1: 'blocking by wrong way driving',
    2: 'reversing off the start line', 3: 'causing a big collision', 4: 'causing a small collision',
    5: 'failing to hand back a position', 6: 'failing to hand back multiple positions',
    7: 'corner cutting', 8: 'gaining a place by corner cutting',
    9: 'gaining multiple places by corner cutting', 10: 'crossing the pit exit line',
    11: 'ignoring blue flags', 12: 'ignoring yellow flags', 13: 'ignoring a drive through',
    14: 'too many drive throughs', 15: 'drive through reminder', 16: 'drive through reminder',
    17: 'pit lane speeding', 18: 'parking for too long', 19: 'ignoring tyre regulations',
    20: 'too many penalties', 21: 'multiple warnings', 22: 'approaching disqualification',
    23: 'tyre regulations', 24: 'tyre regulations', 25: 'corner cutting', 26: 'running wide',
    27: 'gaining time by running wide', 28: 'gaining time by running wide',
    29: 'gaining time by running wide', 30: 'wall riding', 31: 'using a flashback',
    32: 'being reset to track', 33: 'blocking the pit lane', 34: 'a jump start',
    35: 'a collision with the safety car', 36: 'an illegal overtake under the safety car',
    37: 'exceeding the safety car pace', 38: 'exceeding the VSC pace',
    39: 'formation lap below allowed speed', 40: 'a mechanical failure', 41: 'terminal damage',
    42: 'falling too far behind the safety car', 43: 'black flag timer',
    44: 'an unserved stop go', 45: 'an unserved drive through', 46: 'an engine component change',
    47: 'a gearbox change', 48: 'a league grid penalty', 49: 'a retry penalty',
    50: 'an illegal time gain', 51: 'a mandatory pit stop',
}


@dataclass
class RaceEvent:
    code: str                          # four-letter event code, e.g. 'PENA'
    vehicle_index: int | None = None
    other_vehicle_index: int | None = None
    lap_time_s: float | None = None    # FTLP
    penalty_type: int | None = None    # PENA
    infringement_type: int | None = None
    time_s: int | None = None          # PENA: time gained or penalty seconds
    lap_num: int | None = None
    places_gained: int | None = None
    num_lights: int | None = None      # STLG

    @property
    def penalty_name(self) -> str:
        return PENALTY_TYPES.get(self.penalty_type, 'Penalty')

    @property
    def infringement_name(self) -> str:
        return INFRINGEMENT_TYPES.get(self.infringement_type, 'an infringement')


def race_event_body_size(code: bytes) -> int | None:
    fmt = RACE_EVENT_BODY_FORMATS.get(code)
    return None if fmt is None else struct.calcsize(fmt) if fmt else 0


def parse_race_event(data: bytes) -> RaceEvent | None:
    """Parse any race event sub-type in RACE_EVENT_BODY_FORMATS, or None for other codes."""
    code = bytes(data[HEADER_SIZE:HEADER_SIZE + 4])
    fmt = RACE_EVENT_BODY_FORMATS.get(code)
    if fmt is None:
        return None
    event = RaceEvent(code.decode('ascii'))
    if not fmt:
        return event
    v = struct.unpack_from(fmt, data, HEADER_SIZE + 4)
    if code == b'FTLP':
        event.vehicle_index, event.lap_time_s = v
    elif code == b'PENA':
        (event.penalty_type, event.infringement_type, event.vehicle_index, event.other_vehicle_index,
         event.time_s, event.lap_num, event.places_gained) = v
    elif code == b'STLG':
        event.num_lights = v[0]
    else:
        event.vehicle_index = v[0]
    return event


@dataclass
class PacketHeader:
    packet_format: int            # uint16 – 2021
    game_major_version: int       # uint8
    game_minor_version: int       # uint8
    packet_version: int           # uint8
    packet_id: int                # uint8 – Identifier for the packet type
    session_uid: int              # uint64 – Unique identifier for the session
    session_time: float           # float – Session timestamp
    frame_identifier: int         # uint32 – Identifier for the frame
    player_car_index: int         # uint8 – Index of player's car (0-21)
    secondary_player_car_index: int  # uint8

    @classmethod
    def from_bytes(cls, data: bytes) -> "PacketHeader":
        values = struct.unpack_from(HEADER_FORMAT, data, 0)
        return cls(*values)


# ──────────────────────────────────────────────
# Packet ID 0: Motion Data
# ──────────────────────────────────────────────
# Per-car motion data block: 60 bytes each
# 22 cars × 60 bytes + header(24) + extra player car data(120) = 1,464 bytes
CAR_MOTION_FORMAT = (
    "<"
    "f"       # worldPositionX
    "f"       # worldPositionY
    "f"       # worldPositionZ
    "f"       # worldVelocityX
    "f"       # worldVelocityY
    "f"       # worldVelocityZ
    "h"       # worldForwardDirX (int16 normalised)
    "h"       # worldForwardDirY
    "h"       # worldForwardDirZ
    "h"       # worldRightDirX
    "h"       # worldRightDirY
    "h"       # worldRightDirZ
    "f"       # gForceLateral
    "f"       # gForceLongitudinal
    "f"       # gForceVertical
    "f"       # yaw
    "f"       # pitch
    "f"       # roll
)
CAR_MOTION_SIZE = struct.calcsize(CAR_MOTION_FORMAT)  # 60 bytes


@dataclass
class CarMotionData:
    world_position_x: float
    world_position_y: float
    world_position_z: float
    world_velocity_x: float
    world_velocity_y: float
    world_velocity_z: float
    world_forward_dir_x: int
    world_forward_dir_y: int
    world_forward_dir_z: int
    world_right_dir_x: int
    world_right_dir_y: int
    world_right_dir_z: int
    g_force_lateral: float
    g_force_longitudinal: float
    g_force_vertical: float
    yaw: float
    pitch: float
    roll: float


def parse_car_motion(data: bytes, offset: int) -> CarMotionData:
    """Parse a single car's motion data block starting at offset."""
    values = struct.unpack_from(CAR_MOTION_FORMAT, data, offset)
    return CarMotionData(*values)


def parse_player_motion(data: bytes, player_index: int) -> CarMotionData:
    """Parse the player car motion from PacketMotionData."""
    offset = HEADER_SIZE + (player_index * CAR_MOTION_SIZE)
    return parse_car_motion(data, offset)


def parse_all_positions(data: bytes) -> list[tuple[float, float]]:
    """Every car's world (x, z) from PacketMotionData, for the HUD map."""
    return [struct.unpack_from('<fxxxxf', data, HEADER_SIZE + i * CAR_MOTION_SIZE)
            for i in range(22)]


# Extra data for the player's car only, after the 22 per-car blocks (120 bytes).
# Wheel arrays are in the game's order: RL, RR, FL, FR.
MOTION_EX_FORMAT = (
    "<"
    "4f"      # suspensionPosition (mm)
    "4f"      # suspensionVelocity (mm/s)
    "4f"      # suspensionAcceleration (mm/s^2)
    "4f"      # wheelSpeed (m/s)
    "4f"      # wheelSlip (ratio)
    "3f"      # localVelocity x, y, z (m/s)
    "3f"      # angularVelocity x, y, z (rad/s)
    "3f"      # angularAcceleration x, y, z (rad/s^2)
    "f"       # frontWheelsAngle (rad)
)
MOTION_EX_OFFSET = HEADER_SIZE + 22 * CAR_MOTION_SIZE  # 1,344


@dataclass
class PlayerMotionExtra:
    suspension_position: tuple
    suspension_velocity: tuple
    suspension_acceleration: tuple
    wheel_speed: tuple
    wheel_slip: tuple
    local_velocity: tuple
    angular_velocity: tuple
    angular_acceleration: tuple
    front_wheels_angle: float


def parse_player_motion_extra(data: bytes) -> PlayerMotionExtra:
    """The player-only block at the end of PacketMotionData (suspension, wheel speed/slip...)."""
    v = struct.unpack_from(MOTION_EX_FORMAT, data, MOTION_EX_OFFSET)
    return PlayerMotionExtra(v[0:4], v[4:8], v[8:12], v[12:16], v[16:20],
                             v[20:23], v[23:26], v[26:29], v[29])


# ──────────────────────────────────────────────
# Packet ID 6: Car Telemetry Data
# ──────────────────────────────────────────────
# Per-car telemetry data block: 60 bytes each
# 22 cars × 60 bytes + header(24) + footer(3) = 1,347 bytes
CAR_TELEMETRY_FORMAT = (
    "<"
    "H"       # speed (km/h)
    "f"       # throttle (0.0–1.0)
    "f"       # steer (-1.0 left to 1.0 right)
    "f"       # brake (0.0–1.0)
    "B"       # clutch (0–100)
    "b"       # gear (-1, 0, 1–8)
    "H"       # engineRPM
    "B"       # drs (0 or 1)
    "B"       # revLightsPercent (0–100)
    "H"       # revLightsBitValue
    "4H"      # brakesTemperature[4] – RL, RR, FL, FR
    "4B"      # tyresSurfaceTemperature[4] – RL, RR, FL, FR (uint8, 0–255°C)
    "4B"      # tyresInnerTemperature[4] – RL, RR, FL, FR (uint8, 0–255°C)
    "H"       # engineTemperature
    "4f"      # tyresPressure[4] – RL, RR, FL, FR
    "4B"      # surfaceType[4]
)
CAR_TELEMETRY_SIZE = struct.calcsize(CAR_TELEMETRY_FORMAT)  # 60 bytes


@dataclass
class CarTelemetryData:
    speed: int                         # km/h
    throttle: float
    steer: float
    brake: float
    clutch: int
    gear: int
    engine_rpm: int
    drs: int
    rev_lights_percent: int
    rev_lights_bit_value: int
    brakes_temperature: tuple          # (RL, RR, FL, FR)
    tyres_surface_temperature: tuple   # (RL, RR, FL, FR)
    tyres_inner_temperature: tuple     # (RL, RR, FL, FR)
    engine_temperature: int
    tyres_pressure: tuple              # (RL, RR, FL, FR)
    surface_type: tuple                # (RL, RR, FL, FR)


def parse_car_telemetry(data: bytes, offset: int) -> CarTelemetryData:
    """Parse a single car's telemetry data block starting at offset."""
    values = struct.unpack_from(CAR_TELEMETRY_FORMAT, data, offset)
    idx = 0
    speed = values[idx]; idx += 1
    throttle = values[idx]; idx += 1
    steer = values[idx]; idx += 1
    brake = values[idx]; idx += 1
    clutch = values[idx]; idx += 1
    gear = values[idx]; idx += 1
    engine_rpm = values[idx]; idx += 1
    drs = values[idx]; idx += 1
    rev_lights_percent = values[idx]; idx += 1
    rev_lights_bit_value = values[idx]; idx += 1
    brakes_temp = values[idx:idx+4]; idx += 4
    tyres_surface_temp = values[idx:idx+4]; idx += 4
    tyres_inner_temp = values[idx:idx+4]; idx += 4
    engine_temp = values[idx]; idx += 1
    tyres_pressure = values[idx:idx+4]; idx += 4
    surface_type = values[idx:idx+4]; idx += 4

    return CarTelemetryData(
        speed=speed,
        throttle=throttle,
        steer=steer,
        brake=brake,
        clutch=clutch,
        gear=gear,
        engine_rpm=engine_rpm,
        drs=drs,
        rev_lights_percent=rev_lights_percent,
        rev_lights_bit_value=rev_lights_bit_value,
        brakes_temperature=brakes_temp,
        tyres_surface_temperature=tyres_surface_temp,
        tyres_inner_temperature=tyres_inner_temp,
        engine_temperature=engine_temp,
        tyres_pressure=tyres_pressure,
        surface_type=surface_type,
    )


def parse_player_telemetry(data: bytes, player_index: int) -> CarTelemetryData:
    """Parse the player's car telemetry from a full PacketCarTelemetryData."""
    offset = HEADER_SIZE + (player_index * CAR_TELEMETRY_SIZE)
    return parse_car_telemetry(data, offset)


# ──────────────────────────────────────────────
# Packet ID 2: Lap Data
# ──────────────────────────────────────────────
# Per-car lap data block: 43 bytes each
# 22 cars × 43 bytes + header(24) = 970 bytes
LAP_DATA_FORMAT = (
    "<"
    "I"       # lastLapTimeInMS
    "I"       # currentLapTimeInMS
    "H"       # sector1TimeInMS
    "H"       # sector2TimeInMS
    "f"       # lapDistance (metres – can be negative before start/finish line)
    "f"       # totalDistance
    "f"       # safetyCarDelta
    "B"       # carPosition (1 = first)
    "B"       # currentLapNum
    "B"       # pitStatus (0=none, 1=pitting, 2=in pit area)
    "B"       # numPitStops
    "B"       # sector (0=sector1, 1=sector2, 2=sector3)
    "B"       # currentLapInvalid (0=valid, 1=invalid)
    "B"       # penalties (seconds accumulated)
    "B"       # warnings
    "B"       # numUnservedDriveThroughPens
    "B"       # numUnservedStopGoPens
    "B"       # gridPosition
    "B"       # driverStatus (0=garage, 1=flying lap, 2=in lap, 3=out lap, 4=on track)
    "B"       # resultStatus
    "B"       # pitLaneTimerActive
    "H"       # pitLaneTimeInLaneInMS
    "H"       # pitStopTimerInMS
    "B"       # pitStopShouldServePen
)
LAP_DATA_SIZE = struct.calcsize(LAP_DATA_FORMAT)  # 43 bytes


@dataclass
class LapData:
    last_lap_time_ms: int
    current_lap_time_ms: int
    sector1_time_ms: int
    sector2_time_ms: int
    lap_distance: float              # metres into the lap
    total_distance: float
    safety_car_delta: float
    car_position: int
    current_lap_num: int
    pit_status: int
    num_pit_stops: int
    sector: int
    current_lap_invalid: int
    penalties: int
    warnings: int
    num_unserved_drive_through: int
    num_unserved_stop_go: int
    grid_position: int
    driver_status: int
    result_status: int
    pit_lane_timer_active: int
    pit_lane_time_ms: int
    pit_stop_timer_ms: int
    pit_stop_should_serve_pen: int


def parse_lap_data(data: bytes, offset: int) -> LapData:
    """Parse a single car's lap data block starting at offset."""
    values = struct.unpack_from(LAP_DATA_FORMAT, data, offset)
    return LapData(*values)


def parse_player_lap_data(data: bytes, player_index: int) -> LapData:
    """Parse the player's lap data from a full PacketLapData."""
    offset = HEADER_SIZE + (player_index * LAP_DATA_SIZE)
    return parse_lap_data(data, offset)


def parse_all_lap_data(data: bytes) -> list[LapData]:
    return [parse_lap_data(data, HEADER_SIZE + index * LAP_DATA_SIZE) for index in range(22)]


# ──────────────────────────────────────────────
# Packet ID 1: Session Data (partial parse)
# ──────────────────────────────────────────────
# We only need the first few fields for track identification
SESSION_PARTIAL_FORMAT = (
    "<"
    "B"       # weather (0=clear..5=storm)
    "b"       # trackTemperature (°C)
    "b"       # airTemperature (°C)
    "B"       # totalLaps
    "H"       # trackLength (metres)
    "B"       # sessionType
    "b"       # trackId (-1 for unknown)
    "B"       # formula (0=F1, 1=F1Classic, 2=F2, 3=F1Generic)
)
SESSION_PARTIAL_SIZE = struct.calcsize(SESSION_PARTIAL_FORMAT)


@dataclass
class SessionData:
    weather: int
    track_temperature: int
    air_temperature: int
    total_laps: int
    track_length: int
    session_type: int
    track_id: int
    formula: int
    game_paused: int = 0
    is_spectating: int = 0
    safety_car_status: int = 0
    pit_stop_window_ideal_lap: int = 0
    pit_stop_window_latest_lap: int = 0
    pit_stop_rejoin_position: int = 0
    # The game's weather forecast for this session: WeatherSample tuples, soonest first.
    weather_forecast: tuple = ()
    # Marshal zones: (zone_start as a fraction of the lap 0-1, flag) per zone.
    # Flag: -1 invalid, 0 none, 1 green, 2 blue, 3 yellow, 4 red.
    marshal_zones: tuple = ()


@dataclass(frozen=True)
class WeatherSample:
    """One forecast sample: weather and rain chance ``offset_min`` minutes from now."""
    session_type: int
    offset_min: int
    weather: int              # 0 clear, 1 light cloud, 2 overcast, 3 light rain, 4 heavy rain, 5 storm
    track_temperature: int
    track_temperature_change: int   # 0 up, 1 down, 2 no change
    air_temperature: int
    air_temperature_change: int
    rain_percentage: int      # chance of rain, 0-100


# Session packet offsets (after the header) of the forecast block, F1 2021 spec:
# networkGame at 125, numWeatherForecastSamples at 126, then 56 samples of 8 bytes.
WEATHER_FORECAST_COUNT_OFFSET = 126
WEATHER_FORECAST_OFFSET = 127
WEATHER_SAMPLE_FORMAT = '<BBBbbbbB'
WEATHER_SAMPLE_SIZE = struct.calcsize(WEATHER_SAMPLE_FORMAT)
MAX_WEATHER_SAMPLES = 56


def parse_weather_forecast(data: bytes, session_type: int) -> tuple:
    """This session's forecast samples (the packet also carries the next sessions')."""
    if len(data) < HEADER_SIZE + WEATHER_FORECAST_OFFSET + WEATHER_SAMPLE_SIZE * MAX_WEATHER_SAMPLES:
        return ()
    count = min(data[HEADER_SIZE + WEATHER_FORECAST_COUNT_OFFSET], MAX_WEATHER_SAMPLES)
    samples = []
    for i in range(count):
        sample = WeatherSample(*struct.unpack_from(
            WEATHER_SAMPLE_FORMAT, data, HEADER_SIZE + WEATHER_FORECAST_OFFSET + i * WEATHER_SAMPLE_SIZE))
        if sample.session_type == session_type and sample.weather <= 5 and sample.rain_percentage <= 100:
            samples.append(sample)
    return tuple(sorted(samples, key=lambda w: w.offset_min))


# Session packet offsets (after the header) of the marshal zones, F1 2021 spec:
# numMarshalZones at 18, then 21 zones of (float zoneStart, int8 zoneFlag).
MARSHAL_ZONE_COUNT_OFFSET = 18
MARSHAL_ZONE_OFFSET = 19
MARSHAL_ZONE_FORMAT = '<fb'
MARSHAL_ZONE_SIZE = struct.calcsize(MARSHAL_ZONE_FORMAT)
MAX_MARSHAL_ZONES = 21


def parse_marshal_zones(data: bytes) -> tuple:
    """(zone_start, flag) for each marshal zone the game says is in use."""
    if len(data) < HEADER_SIZE + MARSHAL_ZONE_OFFSET + MARSHAL_ZONE_SIZE * MAX_MARSHAL_ZONES:
        return ()
    count = min(data[HEADER_SIZE + MARSHAL_ZONE_COUNT_OFFSET], MAX_MARSHAL_ZONES)
    zones = []
    for i in range(count):
        start, flag = struct.unpack_from(MARSHAL_ZONE_FORMAT, data,
                                         HEADER_SIZE + MARSHAL_ZONE_OFFSET + i * MARSHAL_ZONE_SIZE)
        if math.isfinite(start):
            zones.append((start, flag))
    return tuple(zones)


def parse_session_data(data: bytes) -> SessionData:
    """Parse session data (partial — just the fields we need)."""
    values = struct.unpack_from(SESSION_PARTIAL_FORMAT, data, HEADER_SIZE)
    session = SessionData(*values)
    # F1 2021 v1 has fixed arrays: 21 marshal zones and 56 weather samples.
    if len(data) >= 625:
        session.game_paused = data[HEADER_SIZE + 14]
        session.is_spectating = data[HEADER_SIZE + 15]
        session.safety_car_status = data[HEADER_SIZE + 124]
        session.marshal_zones = parse_marshal_zones(data)
        (session.pit_stop_window_ideal_lap, session.pit_stop_window_latest_lap,
         session.pit_stop_rejoin_position) = struct.unpack_from('<BBB', data, HEADER_SIZE + 589)
        session.weather_forecast = parse_weather_forecast(data, session.session_type)
    return session


# ──────────────────────────────────────────────
# Packet ID 7: Car Status (fuel, tyre compound, ERS)
# ──────────────────────────────────────────────
# Per-car block: 47 bytes; 22 cars x 47 + header(24) = 1058 bytes
CAR_STATUS_FORMAT = (
    "<"
    "B"       # tractionControl
    "B"       # antiLockBrakes
    "B"       # fuelMix
    "B"       # frontBrakeBias
    "B"       # pitLimiterStatus
    "f"       # fuelInTank (kg)
    "f"       # fuelCapacity (kg)
    "f"       # fuelRemainingLaps
    "H"       # maxRPM
    "H"       # idleRPM
    "B"       # maxGears
    "B"       # drsAllowed
    "H"       # drsActivationDistance
    "B"       # actualTyreCompound
    "B"       # visualTyreCompound
    "B"       # tyresAgeLaps
    "b"       # vehicleFiaFlags
    "f"       # ersStoreEnergy
    "B"       # ersDeployMode
    "f"       # ersHarvestedThisLapMGUK
    "f"       # ersHarvestedThisLapMGUH
    "f"       # ersDeployedThisLap
    "B"       # networkPaused
)
CAR_STATUS_SIZE = struct.calcsize(CAR_STATUS_FORMAT)  # 47 bytes

# Compound ids -> human names (F1 2021)
VISUAL_COMPOUNDS = {16: "Soft", 17: "Medium", 18: "Hard", 7: "Intermediate", 8: "Wet"}
ACTUAL_COMPOUNDS = {16: "C5", 17: "C4", 18: "C3", 19: "C2", 20: "C1",
                    7: "Intermediate", 8: "Wet"}


@dataclass
class CarStatusData:
    fuel_in_tank: float
    fuel_capacity: float
    fuel_remaining_laps: float
    actual_tyre_compound: int
    visual_tyre_compound: int
    tyres_age_laps: int
    fia_flags: int
    ers_store_energy: float
    ers_deploy_mode: int
    ers_deployed_this_lap: float

    @property
    def compound_name(self) -> str:
        return VISUAL_COMPOUNDS.get(self.visual_tyre_compound,
                                    ACTUAL_COMPOUNDS.get(self.actual_tyre_compound, "Unknown"))


def _car_status_from_values(v: tuple) -> CarStatusData:
    return CarStatusData(
        fuel_in_tank=v[5], fuel_capacity=v[6], fuel_remaining_laps=v[7],
        actual_tyre_compound=v[13], visual_tyre_compound=v[14], tyres_age_laps=v[15],
        fia_flags=v[16], ers_store_energy=v[17], ers_deploy_mode=v[18],
        ers_deployed_this_lap=v[21],
    )


def parse_player_car_status(data: bytes, player_index: int) -> CarStatusData:
    v = struct.unpack_from(CAR_STATUS_FORMAT, data, HEADER_SIZE + player_index * CAR_STATUS_SIZE)
    return _car_status_from_values(v)


def parse_all_car_status(data: bytes) -> list[CarStatusData]:
    """Every car's status (tyre compound/age, fuel, ERS) from one Car Status packet."""
    return [_car_status_from_values(struct.unpack_from(CAR_STATUS_FORMAT, data, HEADER_SIZE + i * CAR_STATUS_SIZE))
            for i in range(22)]


# ──────────────────────────────────────────────
# Packet ID 10: Car Damage (tyre wear %, damage)
# ──────────────────────────────────────────────
# Per-car block: 39 bytes; 22 cars x 39 + header(24) = 882 bytes
CAR_DAMAGE_FORMAT = (
    "<"
    "4f"      # tyresWear (%)  (RL, RR, FL, FR)
    "4B"      # tyresDamage (%)
    "4B"      # brakesDamage (%)
    "B"       # frontLeftWingDamage
    "B"       # frontRightWingDamage
    "B"       # rearWingDamage
    "B"       # floorDamage
    "B"       # diffuserDamage
    "B"       # sidepodDamage
    "B"       # drsFault
    "B"       # gearBoxDamage
    "B"       # engineDamage
    "6B"      # engine MGUH/ES/CE/ICE/MGUK/TC wear
)
CAR_DAMAGE_SIZE = struct.calcsize(CAR_DAMAGE_FORMAT)  # 39 bytes


@dataclass
class CarDamageData:
    tyres_wear: tuple          # (RL, RR, FL, FR) percent
    tyres_damage: tuple
    front_left_wing: int
    front_right_wing: int
    rear_wing: int
    floor: int
    gearbox: int
    engine: int

    @property
    def worst_tyre_wear(self) -> float:
        return max(self.tyres_wear)


def _car_damage_from_values(v: tuple) -> CarDamageData:
    return CarDamageData(
        tyres_wear=tuple(v[0:4]), tyres_damage=tuple(v[4:8]),
        front_left_wing=v[12], front_right_wing=v[13], rear_wing=v[14],
        floor=v[15], gearbox=v[19], engine=v[20],
    )


def parse_player_car_damage(data: bytes, player_index: int) -> CarDamageData:
    v = struct.unpack_from(CAR_DAMAGE_FORMAT, data, HEADER_SIZE + player_index * CAR_DAMAGE_SIZE)
    return _car_damage_from_values(v)


def parse_all_car_damage(data: bytes) -> list[CarDamageData]:
    """Every car's damage/tyre-wear from one Car Damage packet."""
    return [_car_damage_from_values(struct.unpack_from(CAR_DAMAGE_FORMAT, data, HEADER_SIZE + i * CAR_DAMAGE_SIZE))
            for i in range(22)]


# ──────────────────────────────────────────────
# Packet ID 4: Participants (driver names, teams, race numbers)
# ──────────────────────────────────────────────
# header(24) + numActiveCars(1) + 22 cars x 56 bytes = 1,257 bytes. Sent every 5 seconds.
PARTICIPANT_FORMAT = (
    "<"
    "B"       # aiControlled (1 = AI, 0 = human)
    "B"       # driverId (255 if network human)
    "B"       # networkId
    "B"       # teamId
    "B"       # myTeam flag
    "B"       # raceNumber
    "B"       # nationality
    "48s"     # name (UTF-8, null terminated)
    "B"       # yourTelemetry (0 = restricted, 1 = public)
)
PARTICIPANT_SIZE = struct.calcsize(PARTICIPANT_FORMAT)  # 56 bytes
PARTICIPANTS_PACKET_SIZE = HEADER_SIZE + 1 + 22 * PARTICIPANT_SIZE  # 1,257 bytes

# Names the game uses when it hides a human player's real name; not worth saying on the radio.
_PLACEHOLDER_NAMES = {"", "player", "driver"}

@dataclass
class ParticipantData:
    ai_controlled: bool
    driver_id: int
    team_id: int
    race_number: int
    nationality: int
    name: str
    telemetry_public: bool

    @property
    def radio_name(self) -> str | None:
        """How an engineer would say this driver on the radio: the surname, or None.

        The game sends AI drivers in upper case ("HAMILTON") and online players as their
        gamertag. All-upper-case words at the end of a name are treated as the surname so
        "Nyck DE VRIES" becomes "De Vries"; otherwise the full name is used as given.
        """
        name = " ".join(self.name.split())
        if name.lower() in _PLACEHOLDER_NAMES:
            return None
        if name.isupper():
            return " ".join(word.capitalize() for word in name.split(" "))
        surname = []
        for word in reversed(name.split(" ")):
            if not (word.isupper() and len(word) > 1):
                break
            surname.insert(0, word.capitalize())
        return " ".join(surname) if surname else name


def _participant_from_values(v: tuple) -> ParticipantData:
    raw = v[7].split(b"\x00", 1)[0]
    return ParticipantData(
        ai_controlled=bool(v[0]), driver_id=v[1], team_id=v[3], race_number=v[5],
        nationality=v[6], name=raw.decode("utf-8", errors="replace").strip(),
        telemetry_public=bool(v[8]),
    )


def parse_participants(data: bytes) -> list[ParticipantData]:
    """Every active car's participant entry, indexed like every other per-car array."""
    active = min(struct.unpack_from("<B", data, HEADER_SIZE)[0], 22)
    return [_participant_from_values(struct.unpack_from(PARTICIPANT_FORMAT, data,
                                                        HEADER_SIZE + 1 + i * PARTICIPANT_SIZE))
            for i in range(active)]


# ──────────────────────────────────────────────
# Packet ID 5: Car Setups (every car's setup; zeros for cars whose setup the game hides)
# ──────────────────────────────────────────────
# Per-car block: 49 bytes; 22 cars x 49 + header(24) = 1,102 bytes. Sent twice a second.
CAR_SETUP_FORMAT = (
    "<"
    "B"       # frontWing
    "B"       # rearWing
    "B"       # onThrottle (differential, %)
    "B"       # offThrottle (%)
    "f"       # frontCamber
    "f"       # rearCamber
    "f"       # frontToe
    "f"       # rearToe
    "B"       # frontSuspension
    "B"       # rearSuspension
    "B"       # frontAntiRollBar
    "B"       # rearAntiRollBar
    "B"       # frontSuspensionHeight
    "B"       # rearSuspensionHeight
    "B"       # brakePressure (%)
    "B"       # brakeBias (%)
    "4f"      # tyre pressures (PSI): RL, RR, FL, FR
    "B"       # ballast
    "f"       # fuelLoad (kg)
)
CAR_SETUP_SIZE = struct.calcsize(CAR_SETUP_FORMAT)  # 49 bytes
CAR_SETUPS_PACKET_SIZE = HEADER_SIZE + 22 * CAR_SETUP_SIZE  # 1,102 bytes


@dataclass
class CarSetupData:
    front_wing: int
    rear_wing: int
    on_throttle: int
    off_throttle: int
    front_camber: float
    rear_camber: float
    front_toe: float
    rear_toe: float
    front_suspension: int
    rear_suspension: int
    front_anti_roll_bar: int
    rear_anti_roll_bar: int
    front_ride_height: int
    rear_ride_height: int
    brake_pressure: int
    brake_bias: int
    tyre_pressures: tuple      # (RL, RR, FL, FR) PSI
    ballast: int
    fuel_load: float

    @property
    def is_hidden(self) -> bool:
        """True when the game sent an all-zero block (another human's setup in online play)."""
        return self.front_wing == 0 and self.rear_wing == 0 and not any(self.tyre_pressures)

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d['tyre_pressures'] = list(self.tyre_pressures)
        return d


def _car_setup_from_values(v: tuple) -> CarSetupData:
    return CarSetupData(*v[0:16], tyre_pressures=tuple(v[16:20]), ballast=v[20], fuel_load=v[21])


def parse_car_setups(data: bytes) -> list[CarSetupData]:
    """Every car's setup from one Car Setups packet, indexed like every other per-car array."""
    return [_car_setup_from_values(struct.unpack_from(CAR_SETUP_FORMAT, data, HEADER_SIZE + i * CAR_SETUP_SIZE))
            for i in range(22)]


# ──────────────────────────────────────────────
# Packet ID 8: Final Classification (sent at the end of the session)
# ──────────────────────────────────────────────
# header(24) + numCars(1) + 22 cars x 37 bytes = 839 bytes. F1 2021 lists each car's stint
# compounds but not the laps they ended on (Session History has those).
FINAL_CLASSIFICATION_FORMAT = (
    "<"
    "B"       # position
    "B"       # numLaps
    "B"       # gridPosition
    "B"       # points
    "B"       # numPitStops
    "B"       # resultStatus (3 = finished, 4 = DSQ, 5 = not classified, 6 = retired)
    "I"       # bestLapTimeInMS
    "d"       # totalRaceTime (s, without penalties)
    "B"       # penaltiesTime (s)
    "B"       # numPenalties
    "B"       # numTyreStints
    "8B"      # tyreStintsActual
    "8B"      # tyreStintsVisual
)
FINAL_CLASSIFICATION_SIZE = struct.calcsize(FINAL_CLASSIFICATION_FORMAT)  # 37 bytes
FINAL_CLASSIFICATION_PACKET_SIZE = HEADER_SIZE + 1 + 22 * FINAL_CLASSIFICATION_SIZE  # 839 bytes


@dataclass
class FinalClassificationData:
    car_index: int
    position: int
    num_laps: int
    grid_position: int
    points: int
    num_pit_stops: int
    result_status: int
    best_lap_time_ms: int
    total_race_time_s: float
    penalties_time_s: int
    num_penalties: int
    tyre_stints_actual: tuple   # one compound id per stint, in order
    tyre_stints_visual: tuple


def parse_final_classification(data: bytes) -> list[FinalClassificationData]:
    """The classified cars (numCars of them), each tagged with its car index."""
    num_cars = min(struct.unpack_from("<B", data, HEADER_SIZE)[0], 22)
    out = []
    for i in range(num_cars):
        v = struct.unpack_from(FINAL_CLASSIFICATION_FORMAT, data, HEADER_SIZE + 1 + i * FINAL_CLASSIFICATION_SIZE)
        stints = min(v[10], 8)
        out.append(FinalClassificationData(
            car_index=i, position=v[0], num_laps=v[1], grid_position=v[2], points=v[3],
            num_pit_stops=v[4], result_status=v[5], best_lap_time_ms=v[6], total_race_time_s=v[7],
            penalties_time_s=v[8], num_penalties=v[9],
            tyre_stints_actual=tuple(v[11:11 + stints]), tyre_stints_visual=tuple(v[19:19 + stints])))
    return out


# ──────────────────────────────────────────────
# Packet ID 11: Session History (one car per packet, cycling through the field)
# ──────────────────────────────────────────────
# header(24) + 7 bytes + 100 laps x 11 bytes + 8 stints x 3 bytes = 1,155 bytes.
SESSION_HISTORY_HEAD_FORMAT = "<7B"   # carIdx, numLaps, numTyreStints, bestLap/S1/S2/S3 lap numbers
LAP_HISTORY_FORMAT = "<IHHHB"         # lapTimeInMS, sector 1/2/3 ms, lapValidBitFlags
LAP_HISTORY_SIZE = struct.calcsize(LAP_HISTORY_FORMAT)        # 11 bytes
TYRE_STINT_HISTORY_FORMAT = "<BBB"    # endLap (255 = current stint), actual, visual compound
TYRE_STINT_HISTORY_SIZE = struct.calcsize(TYRE_STINT_HISTORY_FORMAT)
SESSION_HISTORY_MAX_LAPS = 100
SESSION_HISTORY_MAX_STINTS = 8
SESSION_HISTORY_PACKET_SIZE = (HEADER_SIZE + 7 + SESSION_HISTORY_MAX_LAPS * LAP_HISTORY_SIZE
                               + SESSION_HISTORY_MAX_STINTS * TYRE_STINT_HISTORY_SIZE)  # 1,155 bytes

LAP_VALID = 0x01
SECTOR_VALID = {1: 0x02, 2: 0x04, 3: 0x08}
OPEN_STINT_END_LAP = 255


@dataclass
class LapHistory:
    lap_num: int
    lap_time_ms: int          # 0 while the lap is still being driven
    sector_ms: tuple          # (s1, s2, s3); 0 for a sector not yet completed
    valid_flags: int

    @property
    def valid(self) -> bool:
        return bool(self.valid_flags & LAP_VALID)


@dataclass
class TyreStint:
    stint_num: int            # 0 = the set the car started on
    start_lap: int
    end_lap: int              # last lap on this set; the current lap for the set still on the car
    actual_compound: int
    visual_compound: int
    is_open: bool             # still on the car

    @property
    def laps(self) -> int:
        return max(0, self.end_lap - self.start_lap + 1)

    @property
    def compound_name(self) -> str:
        return VISUAL_COMPOUNDS.get(self.visual_compound, ACTUAL_COMPOUNDS.get(self.actual_compound, "Unknown"))


@dataclass
class SessionHistoryData:
    car_index: int
    num_laps: int             # includes the lap in progress
    best_lap_num: int
    best_sector_lap_nums: tuple
    laps: list                # LapHistory, lap 1 first
    stints: list              # TyreStint, first set first

    def completed_laps(self) -> list:
        return [lap for lap in self.laps if lap.lap_time_ms > 0]


def parse_session_history(data: bytes) -> SessionHistoryData:
    """One car's lap and tyre-stint history.

    Stint start laps are derived: the first set starts on lap 1 and each later set on the lap
    after the previous one's end lap. The set still on the car (end lap 255) runs to the
    car's current lap.
    """
    car, num_laps, num_stints, best, b1, b2, b3 = struct.unpack_from(SESSION_HISTORY_HEAD_FORMAT, data, HEADER_SIZE)
    num_laps = min(num_laps, SESSION_HISTORY_MAX_LAPS)
    num_stints = min(num_stints, SESSION_HISTORY_MAX_STINTS)
    laps = []
    offset = HEADER_SIZE + 7
    for n in range(num_laps):
        t, s1, s2, s3, flags = struct.unpack_from(LAP_HISTORY_FORMAT, data, offset + n * LAP_HISTORY_SIZE)
        laps.append(LapHistory(n + 1, t, (s1, s2, s3), flags))
    offset += SESSION_HISTORY_MAX_LAPS * LAP_HISTORY_SIZE
    stints = []
    start = 1
    for n in range(num_stints):
        end, actual, visual = struct.unpack_from(TYRE_STINT_HISTORY_FORMAT, data, offset + n * TYRE_STINT_HISTORY_SIZE)
        if actual == 0 and visual == 0:
            continue                                  # unused slot
        is_open = end == OPEN_STINT_END_LAP
        end_lap = max(start, num_laps) if is_open else end
        stints.append(TyreStint(len(stints), start, end_lap, actual, visual, is_open))
        if is_open:
            break
        start = end + 1
    return SessionHistoryData(car, num_laps, best, (b1, b2, b3), laps, stints)


# ──────────────────────────────────────────────
# Track ID → Name Mapping (F1 2021)
# ──────────────────────────────────────────────
TRACK_NAMES = {
    0: "Melbourne",
    1: "Paul Ricard",
    2: "Shanghai",
    3: "Bahrain",
    4: "Catalunya",
    5: "Monaco",
    6: "Montreal",
    7: "Silverstone",
    8: "Hockenheim",
    9: "Hungaroring",
    10: "Spa-Francorchamps",
    11: "Monza",
    12: "Singapore",
    13: "Suzuka",
    14: "Abu Dhabi",
    15: "Austin (COTA)",
    16: "Interlagos",
    17: "Red Bull Ring",
    18: "Sochi",
    19: "Mexico City",
    20: "Baku",
    21: "Bahrain Short",
    22: "Silverstone Short",
    23: "Austin Short",
    24: "Suzuka Short",
    25: "Hanoi",
    26: "Zandvoort",
    27: "Imola",
    28: "Portimão",
    29: "Jeddah",
}

# Session type mapping
SESSION_TYPES = {
    0: "Unknown",
    1: "Practice 1",
    2: "Practice 2",
    3: "Practice 3",
    4: "Short Practice",
    5: "Qualifying 1",
    6: "Qualifying 2",
    7: "Qualifying 3",
    8: "Short Qualifying",
    9: "One-Shot Qualifying",
    10: "Race",
    11: "Race 2",
    12: "Race 3",
    13: "Time Trial",
}

