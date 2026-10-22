"""
Synthetic F1 2021 Telemetry Simulator
Sends simulated UDP telemetry packets to 127.0.0.1:20777
Use this to test the AI Engineer without having the game open!
"""

import os
import socket
import struct
import sys
import time

# Safe UTF-8 / replacement handling on Windows CMD
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add parent directory for packet imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 20777

# Packet formats come from packets.py, so the simulator always matches the parser.
from packets import (  # noqa: E402
    CAR_DAMAGE_FORMAT,
    CAR_DAMAGE_SIZE,
    CAR_STATUS_FORMAT,
    CAR_STATUS_SIZE,
    CAR_TELEMETRY_FORMAT,
    CAR_TELEMETRY_SIZE,
    HEADER_FORMAT,
    LAP_DATA_FORMAT,
    LAP_DATA_SIZE,
    PARTICIPANT_FORMAT,
)


def build_header(packet_id: int, session_time: float, frame_id: int, player_idx: int = 0) -> bytes:
    return struct.pack(
        HEADER_FORMAT,
        2021,        # packet_format
        1,           # game_major
        18,          # game_minor
        1,           # packet_version
        packet_id,   # packet_id
        123456789,   # session_uid
        float(session_time),
        frame_id,
        player_idx,
        255          # secondary_player_car_index
    )

def build_session_packet(session_time: float, frame_id: int, track_id: int = 11,
                         session_type: int = 10, track_length: int = 5793, total_laps: int = 53) -> bytes:
    # Monza is track_id = 11
    header = build_header(1, session_time, frame_id)
    # SESSION_PARTIAL_FORMAT: <BbBBHbbB -> weather, trackTemp, airTemp, totalLaps, trackLength, sessionType, trackId, formula
    session_data = struct.pack(
        "<BbBBHbbB",
        0,   # weather (clear)
        35,  # trackTemp (35°C)
        24,  # airTemp (24°C)
        total_laps,  # totalLaps
        track_length, # trackLength (Monza 5793m by default)
        session_type,  # sessionType (10 = Race, 13 = Time Trial, ...)
        track_id, # Monza
        0    # formula F1
    )
    # Pad to standard packet size
    return header + session_data + b"\x00" * (625 - len(header) - len(session_data))

def build_telemetry_packet(session_time: float, frame_id: int, speed: int, throttle: float, brake: float, gear: int, tyre_temp: int) -> bytes:
    header = build_header(6, session_time, frame_id)
    # 22 cars
    cars_data = bytearray()
    for car_idx in range(22):
        if car_idx == 0:
            # Player car
            car_block = struct.pack(
                CAR_TELEMETRY_FORMAT,
                speed,
                throttle,
                0.0,    # steer
                brake,
                0,      # clutch
                gear,
                11500,  # engineRPM
                0,      # drs
                50,     # revLightsPercent
                0,      # revLightsBitValue
                450, 450, 520, 520, # brakesTemperature[4] RL, RR, FL, FR
                tyre_temp, tyre_temp, tyre_temp + 8, tyre_temp + 8, # tyresSurfaceTemperature[4]
                tyre_temp - 5, tyre_temp - 5, tyre_temp, tyre_temp, # tyresInnerTemperature[4]
                105,    # engineTemperature
                22.0, 22.0, 23.5, 23.5, # tyresPressure[4]
                0, 0, 0, 0 # surfaceType[4]
            )
        else:
            car_block = bytes(CAR_TELEMETRY_SIZE)
        cars_data.extend(car_block)
    
    # footer
    footer = b"\x00\x00\x00"
    return header + bytes(cars_data) + footer

def build_lap_packet(session_time: float, frame_id: int, lap_dist: float, lap_time_ms: int, lap_num: int = 1, pos: int = 3) -> bytes:
    header = build_header(2, session_time, frame_id)
    cars_data = bytearray()
    for car_idx in range(22):
        if car_idx == 0:
            car_block = struct.pack(
                LAP_DATA_FORMAT,
                82450,         # lastLapTimeInMS
                lap_time_ms,   # currentLapTimeInMS
                28000,         # sector1TimeInMS
                29000,         # sector2TimeInMS
                lap_dist,      # lapDistance (m)
                float(lap_dist), # totalDistance
                0.0,           # safetyCarDelta
                pos,           # carPosition (P3)
                lap_num,       # currentLapNum
                0,             # pitStatus
                0,             # numPitStops
                0,             # sector
                0,             # currentLapInvalid
                0,             # penalties
                0,             # warnings
                0,             # numUnservedDriveThroughPens
                0,             # numUnservedStopGoPens
                3,             # gridPosition
                4,             # driverStatus (on track)
                2,             # resultStatus (active)
                0,             # pitLaneTimerActive
                0,             # pitLaneTimeInLaneInMS
                0,             # pitStopTimerInMS
                0              # pitStopShouldServePen
            )
        else:
            car_block = bytes(LAP_DATA_SIZE)
        cars_data.extend(car_block)
    return header + bytes(cars_data)

def build_car_status_packet(session_time: float, frame_id: int, fuel: float = 60.0,
                            compound: int = 17, age: int = 0) -> bytes:
    """Car status packet (id 7) for the player car; other cars left zeroed."""
    header = build_header(7, session_time, frame_id)
    car = struct.pack(CAR_STATUS_FORMAT, 0, 0, 1, 58, 0, fuel, 110.0, fuel / 1.6, 13000, 4000,
                      8, 1, 0, compound, compound, age, 0, 2_000_000.0, 1, 0.0, 0.0, 0.0, 0)
    return header + car + b"\x00" * (24 + 22 * CAR_STATUS_SIZE - len(header) - len(car))


def build_car_damage_packet(session_time: float, frame_id: int, wear=(0.0, 0.0, 0.0, 0.0)) -> bytes:
    """Car damage packet (id 10) for the player car; wear is (RL, RR, FL, FR) percent."""
    header = build_header(10, session_time, frame_id)
    car = struct.pack(CAR_DAMAGE_FORMAT, *wear, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    return header + car + b"\x00" * (24 + 22 * CAR_DAMAGE_SIZE - len(header) - len(car))



# A plausible 2021 grid order for the simulator; index 0 is the player.
SIM_DRIVER_NAMES = ["Player", "HAMILTON", "VERSTAPPEN", "BOTTAS", "PEREZ", "NORRIS", "LECLERC",
                    "SAINZ", "RICCIARDO", "GASLY", "ALONSO", "OCON", "VETTEL", "STROLL",
                    "TSUNODA", "RAIKKONEN", "GIOVINAZZI", "RUSSELL", "LATIFI", "SCHUMACHER",
                    "MAZEPIN", "KUBICA"]


def build_participants_packet(session_time: float, frame_id: int, names=None) -> bytes:
    """Participants packet (id 4): one name per car, the player first by default."""
    names = list(SIM_DRIVER_NAMES if names is None else names)[:22]
    header = build_header(4, session_time, frame_id)
    body = bytearray(struct.pack("<B", len(names)))
    for index in range(22):
        name = names[index] if index < len(names) else ""
        body += struct.pack(PARTICIPANT_FORMAT, int(index != 0), index, 0, 0, 0, index + 1, 0,
                            name.encode("utf-8")[:47], 1)
    return header + bytes(body)


def run_simulation(duration_seconds: float = 10.0):
    print("=" * 60)
    print("  F1 2021 TELEMETRY SIMULATOR")
    print(f"  Sending telemetry stream to {TARGET_HOST}:{TARGET_PORT}...")
    print("  Simulating: Monza, Turn 1 Approach + Overheating Tyres")
    print("=" * 60)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    start = time.time()
    frame_id = 1
    
    try:
        # First send session packet (Monza)
        sess_pkt = build_session_packet(0.0, frame_id, track_id=11)
        sock.sendto(sess_pkt, (TARGET_HOST, TARGET_PORT))
        print("-> Sent Session Packet (Monza, Race)")
        sock.sendto(build_participants_packet(0.0, frame_id), (TARGET_HOST, TARGET_PORT))
        time.sleep(0.5)

        # Simulate driving down Monza pit straight towards Turn 1 (braking point at 340m)
        # Starting at lap_distance = 150m, approaching 340m
        lap_dist = 160.0
        lap_time_ms = 4000
        
        while time.time() - start < duration_seconds:
            elapsed = time.time() - start
            frame_id += 1
            
            # Accelerate down straight then brake
            if lap_dist < 320:
                speed = 320
                throttle = 1.0
                brake = 0.0
                gear = 8
            else:
                speed = 140
                throttle = 0.0
                brake = 1.0
                gear = 3
                
            lap_dist += 3.5  # ~300 km/h is ~83 m/s, at 25Hz each step is ~3.3m
            lap_time_ms += 40
            
            # Send overheated tyres (108°C) to trigger setup & temp advisor
            tyre_temp = 108
            
            telem_pkt = build_telemetry_packet(elapsed, frame_id, speed, throttle, brake, gear, tyre_temp)
            lap_pkt = build_lap_packet(elapsed, frame_id, lap_dist, lap_time_ms, lap_num=2, pos=2)
            
            sock.sendto(telem_pkt, (TARGET_HOST, TARGET_PORT))
            sock.sendto(lap_pkt, (TARGET_HOST, TARGET_PORT))
            if frame_id % 10 == 0:   # strategy data arrives at a lower rate than telemetry
                sock.sendto(build_car_status_packet(elapsed, frame_id, fuel=58.0 - elapsed * 0.05,
                                                    age=3), (TARGET_HOST, TARGET_PORT))
                sock.sendto(build_car_damage_packet(elapsed, frame_id, wear=(31, 33, 36, 38)),
                            (TARGET_HOST, TARGET_PORT))
            
            time.sleep(0.04) # ~25 Hz
            
        print("[SUCCESS] Simulation finished successfully.")
    finally:
        sock.close()

if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    run_simulation(dur)
