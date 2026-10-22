import json
import pathlib
import struct
import sys
import time
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))

from packets import (
    HEADER_FORMAT, PacketHeader, CarMotionData, CAR_MOTION_FORMAT, CAR_MOTION_SIZE,
    parse_player_motion, SessionData, CarTelemetryData, LapData, CarDamageData, CarStatusData
)
from track_data import (
    TRACK_DATABASE, get_all_tracks, get_track_setup, get_braking_points,
    build_setup_advisor_prompt, generate_ai_setup_recommendation
)
from voice_engine import VoiceEngine, Priority
from race_radio import RaceRadio
from analysis import AnalysisEngine
from web_hud import TelemetryHUDHandler, update_web_state, get_web_state
from simulate_telemetry import build_header


def build_motion_packet(session_time: float, frame_id: int, x: float, z: float,
                        lat_g: float = 0.0, player_idx: int = 0) -> bytes:
    header = struct.pack(
        HEADER_FORMAT,
        2021, 1, 18, 1, 0, 123456789, float(session_time), frame_id, player_idx, 255
    )
    cars_data = bytearray()
    for car_idx in range(22):
        if car_idx == player_idx:
            car_block = struct.pack(
                CAR_MOTION_FORMAT,
                float(x), 0.0, float(z),
                0.0, 0.0, 0.0,
                0, 0, 0, 0, 0, 0,
                float(lat_g), 0.0, 0.0,
                0.0, 0.0, 0.0
            )
        else:
            car_block = bytes(CAR_MOTION_SIZE)
        cars_data.extend(car_block)
    extra_player = bytes(120)
    return header + bytes(cars_data) + extra_player


class Feature3TrackMappingTests(unittest.TestCase):
    def test_motion_packet_parsing(self):
        packet = build_motion_packet(1.5, 42, 123.45, -678.90, lat_g=2.4)
        motion = parse_player_motion(packet, 0)
        self.assertAlmostEqual(motion.world_position_x, 123.45, places=2)
        self.assertAlmostEqual(motion.world_position_z, -678.90, places=2)
        self.assertAlmostEqual(motion.g_force_lateral, 2.4, places=2)

    def test_analysis_accumulates_track_coordinates(self):
        voice = MagicMock()
        engine = AnalysisEngine(voice)
        session = SessionData(0, 30, 20, 52, 5891, 10, 7, 0)
        engine.update_session(session, session_uid=1)

        # Feed telemetry and motion points around lap
        dummy_telem = CarTelemetryData(
            speed=250, throttle=1.0, steer=0.0, brake=0.0, clutch=0, gear=7, engine_rpm=11000,
            drs=1, rev_lights_percent=50, rev_lights_bit_value=0,
            brakes_temperature=(400, 400, 400, 400),
            tyres_surface_temperature=(95, 95, 95, 95),
            tyres_inner_temperature=(90, 90, 90, 90),
            engine_temperature=95, tyres_pressure=(23, 23, 22, 22), surface_type=(0, 0, 0, 0)
        )
        dummy_lap = LapData(
            last_lap_time_ms=0, current_lap_time_ms=5000, sector1_time_ms=0, sector2_time_ms=0,
            lap_distance=0.0, total_distance=0.0, safety_car_delta=0.0, car_position=1,
            current_lap_num=1, pit_status=0, num_pit_stops=0, sector=0, current_lap_invalid=0,
            penalties=0, warnings=0, num_unserved_drive_through=0, num_unserved_stop_go=0,
            grid_position=1, driver_status=4, result_status=2, pit_lane_timer_active=0,
            pit_lane_time_ms=0, pit_stop_timer_ms=0, pit_stop_should_serve_pen=0
        )
        engine.latest_telemetry = dummy_telem
        engine.latest_lap = dummy_lap

        # Sample at distance 0, 10, 20
        motion1 = parse_player_motion(build_motion_packet(1.0, 1, 10.0, 20.0), 0)
        engine.update_motion(motion1)

        dummy_lap.lap_distance = 15.0
        motion2 = parse_player_motion(build_motion_packet(1.5, 2, 25.0, 35.0), 0)
        engine.update_motion(motion2)

        self.assertGreaterEqual(len(engine.current_lap_map_points), 2)
        self.assertEqual(engine.current_lap_map_points[0], [10.0, 20.0])
        self.assertEqual(engine.current_lap_map_points[1], [25.0, 35.0])


class Feature4RealisticF1RadioProtocolTests(unittest.TestCase):
    def test_cornering_and_braking_silence_filter(self):
        engine = AnalysisEngine(MagicMock())
        dummy_telem = CarTelemetryData(
            speed=200, throttle=1.0, steer=0.0, brake=0.0, clutch=0, gear=6, engine_rpm=10500,
            drs=0, rev_lights_percent=0, rev_lights_bit_value=0,
            brakes_temperature=(400, 400, 400, 400),
            tyres_surface_temperature=(95, 95, 95, 95),
            tyres_inner_temperature=(90, 90, 90, 90),
            engine_temperature=95, tyres_pressure=(23, 23, 22, 22), surface_type=(0, 0, 0, 0)
        )
        engine.latest_telemetry = dummy_telem

        # On straight (no brake, no steer, no G): not busy
        self.assertFalse(engine.is_driver_busy())

        # Heavy braking: busy
        dummy_telem.brake = 0.6
        self.assertTrue(engine.is_driver_busy())
        dummy_telem.brake = 0.0
        self.assertFalse(engine.is_driver_busy())

        # Steering through a slow or medium-load corner: the engineer may talk
        dummy_telem.steer = 0.35
        engine.latest_motion = parse_player_motion(build_motion_packet(1.0, 1, 0, 0, lat_g=2.1), 0)
        self.assertFalse(engine.is_driver_busy())

        # High-load corner (fast, high lateral G): busy
        dummy_telem.speed = 260
        engine.latest_motion = parse_player_motion(build_motion_packet(1.0, 1, 0, 0, lat_g=4.2), 0)
        self.assertTrue(engine.is_driver_busy())

    def test_priority_interrupts_break_silence(self):
        spoken = []
        radio = RaceRadio(lambda msg, prio, **opts: spoken.append((msg, prio, opts.get('key'))))
        radio.reset(SessionData(0, 30, 20, 52, 5891, 10, 7, 0))

        # Sudden Safety Car deployment -> Priority.CRITICAL interrupt
        radio.update_safety_car(1)
        self.assertEqual(len(spoken), 1)
        self.assertIn("Safety Car", spoken[-1][0])
        self.assertEqual(spoken[-1][1], Priority.CRITICAL)

        # Sudden Rain transition -> Priority.CRITICAL interrupt
        radio.update_weather_alert(4)
        self.assertEqual(len(spoken), 2)
        self.assertIn("Rain", spoken[-1][0])
        self.assertEqual(spoken[-1][1], Priority.CRITICAL)

        # Critical tyre wear (>75%) -> Priority.CRITICAL interrupt
        radio.update_damage_alert(78.5, 0)
        self.assertEqual(len(spoken), 3)
        self.assertIn("seventy-eight", spoken[-1][0])
        self.assertEqual(spoken[-1][1], Priority.CRITICAL)


class Feature5TrackDatabaseAndSetupAdvisorTests(unittest.TestCase):
    REQUIRED_CIRCUITS = [
        "Albert Park", "Bahrain", "Imola", "Portimão", "Shanghai", "Barcelona",
        "Monaco", "Baku", "Canada", "Paul Ricard", "Red Bull Ring", "Silverstone",
        "Hungaroring", "Spa", "Zandvoort", "Monza", "Sochi", "Marina Bay",
        "Suzuka", "COTA", "Mexico City", "Interlagos", "Jeddah", "Abu Dhabi"
    ]

    def test_all_24_circuits_present_in_database(self):
        tracks = get_all_tracks()
        self.assertEqual(len(tracks), 24)
        all_names = set()
        for t in tracks:
            all_names.add(t["short_name"])
            for a in t.get("aliases", []):
                all_names.add(a)
        for circuit in self.REQUIRED_CIRCUITS:
            self.assertIn(circuit, all_names, f"Missing required circuit: {circuit}")

    def test_setup_components_present_for_every_track(self):
        for tid, t in TRACK_DATABASE.items():
            setup = t["setup"]
            # 1. Aerodynamics
            self.assertIn("front_wing", setup["aerodynamics"])
            self.assertIn("rear_wing", setup["aerodynamics"])
            # 2. Transmission
            self.assertIn("diff_on_throttle", setup["transmission"])
            self.assertIn("diff_off_throttle", setup["transmission"])
            # 3. Suspension Geometry
            self.assertIn("front_camber", setup["suspension_geometry"])
            self.assertIn("rear_camber", setup["suspension_geometry"])
            self.assertIn("front_toe", setup["suspension_geometry"])
            self.assertIn("rear_toe", setup["suspension_geometry"])
            # 4. Suspension
            self.assertIn("front_suspension", setup["suspension"])
            self.assertIn("rear_suspension", setup["suspension"])
            self.assertIn("front_anti_roll", setup["suspension"])
            self.assertIn("rear_anti_roll", setup["suspension"])
            self.assertIn("front_ride_height", setup["suspension"])
            self.assertIn("rear_ride_height", setup["suspension"])
            # 5. Brakes
            self.assertIn("brake_pressure", setup["brakes"])
            self.assertIn("brake_bias", setup["brakes"])
            # 6. Tyres
            self.assertIn("front_right_pressure", setup["tyres"])
            self.assertIn("rear_left_pressure", setup["tyres"])

    def test_monza_low_downforce_and_monaco_high_downforce(self):
        monza = TRACK_DATABASE[11]["setup"]["aerodynamics"]
        monaco = TRACK_DATABASE[5]["setup"]["aerodynamics"]
        self.assertLess(monza["front_wing"], 4)
        self.assertGreater(monaco["front_wing"], 8)

    def test_setup_advisor_prompt_and_fallback_generation(self):
        sys_p, user_p = build_setup_advisor_prompt(7, weather="dry")
        self.assertIn("Formula 1 Chief Race Engineer", sys_p)
        self.assertIn("Silverstone", user_p)

        rec = generate_ai_setup_recommendation(11, weather="dry")
        self.assertEqual(rec["track_name"], "Autodromo Nazionale Monza")
        self.assertIn("Aerodynamics", rec["ai_consultation"])


class Feature6WebHUDStructureTests(unittest.TestCase):
    def test_template_and_static_files_exist(self):
        root = pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'
        self.assertTrue((root / 'templates' / 'index.html').is_file())
        self.assertTrue((root / 'static' / 'style.css').is_file())
        self.assertTrue((root / 'static' / 'app.js').is_file())

    def test_web_state_snapshot_includes_all_hud_fields(self):
        update_web_state({
            'car_x': 100.5, 'car_z': -250.2, 'g_force_lat': 1.85,
            'track_map': [[0, 0], [10, 10]], 'braking_zones': [[5, 5, 0.8]],
            'ers_zones': [[10, 10]], 'tyre_wear': [12.0, 14.0, 11.5, 13.0],
            'ers_percent': 85, 'safety_car': 'NONE'
        })
        state = get_web_state()
        self.assertEqual(state['car_x'], 100.5)
        self.assertEqual(state['car_z'], -250.2)
        self.assertEqual(state['g_force_lat'], 1.85)
        self.assertEqual(state['ers_percent'], 85)
        self.assertEqual(len(state['track_map']), 2)


if __name__ == '__main__':
    unittest.main()
