"""
Synthetic F1 2021 Telemetry Simulator
Sends simulated UDP telemetry packets to 127.0.0.1:20777
Use this to test the AI Engineer without having the game open!
"""

import socket
import struct
import time
import sys
import os

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

# Format definitions matching packets.py
HEADER_FORMAT = "<HBBBBQfIBB"  # 24 bytes
CAR_TELEMETRY_FORMAT = "<HfffBbHBBH4H4B4BH4f4B"  # 60 bytes
LAP_DATA_FORMAT = "<IIf ff f BBBBBB BBBB BB BHHB"  # 43 bytes
# Let's use exact LAP_DATA struct from packets.py:
# "<IIHHfffBBBBBBBBBBBBBHHB" -> wait, let's verify exact format from packets.py
from packets import (
    HEADER_FORMAT, HEADER_SIZE,
    CAR_TELEMETRY_FORMAT, CAR_TELEMETRY_SIZE,
    LAP_DATA_FORMAT, LAP_DATA_SIZE,
    SESSION_PARTIAL_FORMAT
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

def build_session_packet(session_time: float, frame_id: int, track_id: int = 11) -> bytes:
    # Monza is track_id = 11
    header = build_header(1, session_time, frame_id)
    # SESSION_PARTIAL_FORMAT: <BbBBHbbB -> weather, trackTemp, airTemp, totalLaps, trackLength, sessionType, trackId, formula
    session_data = struct.pack(
        "<BbBBHbbB",
        0,   # weather (clear)
        35,  # trackTemp (35°C)
        24,  # airTemp (24°C)
        53,  # totalLaps
        5793, # trackLength (Monza 5793m)
        10,  # sessionType (Race)
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
            
            time.sleep(0.04) # ~25 Hz
            
        print("[SUCCESS] Simulation finished successfully.")
    finally:
        sock.close()

if __name__ == "__main__":
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    run_simulation(dur)
