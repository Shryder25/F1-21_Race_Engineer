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

import struct
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# Packet Header (24 bytes)
# ──────────────────────────────────────────────
HEADER_FORMAT = "<HBBBBQfIBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 24 bytes


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


def parse_session_data(data: bytes) -> SessionData:
    """Parse session data (partial — just the fields we need)."""
    values = struct.unpack_from(SESSION_PARTIAL_FORMAT, data, HEADER_SIZE)
    session = SessionData(*values)
    # F1 2021 v1 has fixed arrays: 21 marshal zones and 56 weather samples.
    if len(data) >= 625:
        session.game_paused = data[HEADER_SIZE + 14]
        session.is_spectating = data[HEADER_SIZE + 15]
        session.safety_car_status = data[HEADER_SIZE + 124]
        (session.pit_stop_window_ideal_lap, session.pit_stop_window_latest_lap,
         session.pit_stop_rejoin_position) = struct.unpack_from('<BBB', data, HEADER_SIZE + 589)
    return session


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
