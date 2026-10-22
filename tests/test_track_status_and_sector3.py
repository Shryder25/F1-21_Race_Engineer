"""HUD regressions: sector 3 never showed, and the track status was stuck on "track clear".

- The game never sends S3 live; it is lap time - S1 - S2 at the line. The engine worked it
  out but threw it away when the next lap started, so the HUD's S3 box stayed empty.
- The pill only knew SC/VSC; yellow and red flags (marshal zones) and the player's own
  flag (Car Status vehicleFiaFlags) were never read.
"""

import pathlib
import struct
import sys
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))

import track_status
import web_hud
from analysis import AnalysisEngine
from packets import (
    HEADER_SIZE,
    CarStatusData,
    SessionData,
    parse_player_lap_data,
    parse_player_telemetry,
    parse_session_data,
)
from simulate_telemetry import build_lap_packet, build_session_packet, build_telemetry_packet
from test_analysis import SilentVoice


def session_packet(zones=(), safety_car=0):
    data = bytearray(build_session_packet(0.0, 1, track_length=5000))
    data[HEADER_SIZE + 18] = len(zones)
    for i, (start, flag) in enumerate(zones):
        struct.pack_into('<fb', data, HEADER_SIZE + 19 + i * 5, start, flag)
    data[HEADER_SIZE + 124] = safety_car
    return bytes(data)


def status(fia_flags=0):
    return CarStatusData(
        fuel_in_tank=50.0,
        fuel_capacity=100.0,
        fuel_remaining_laps=20.0,
        actual_tyre_compound=16,
        visual_tyre_compound=16,
        tyres_age_laps=2,
        fia_flags=fia_flags,
        ers_store_energy=2000000.0,
        ers_deploy_mode=1,
        ers_deployed_this_lap=0.0,
    )


class MarshalZoneParsingTests(unittest.TestCase):
    def test_zones_and_flags_are_read_from_the_session_packet(self):
        session = parse_session_data(session_packet([(0.0, 0), (0.25, 3), (0.6, 1)], safety_car=2))
        self.assertEqual(
            [(round(s, 2), f) for s, f in session.marshal_zones], [(0.0, 0), (0.25, 3), (0.6, 1)]
        )
        self.assertEqual(session.safety_car_status, 2)
        self.assertEqual(session.track_length, 5000)

    def test_only_the_zones_in_use_are_kept(self):
        self.assertEqual(parse_session_data(session_packet([])).marshal_zones, ())


class TrackStatusTests(unittest.TestCase):
    def describe(self, zones=(), safety_car=0, player_flag=0, sector_starts=None):
        return track_status.describe(
            parse_session_data(session_packet(zones, safety_car)), player_flag, sector_starts
        )

    def test_clear(self):
        self.assertEqual(self.describe([(0.0, 0), (0.5, 1)])['label'], 'TRACK CLEAR')
        self.assertEqual(track_status.describe(None)['state'], 'CLEAR')

    def test_yellow_zone_names_its_sector_once_the_sector_lines_are_known(self):
        result = self.describe([(0.1, 0), (0.45, 3), (0.9, 3)], sector_starts={2: 1600.0, 3: 3500.0})
        self.assertEqual(result['state'], 'YELLOW')
        self.assertEqual(result['label'], 'YELLOW FLAG · S2 S3')
        self.assertEqual(result['yellow_zones'], 2)
        self.assertEqual(self.describe([(0.45, 3)])['label'], 'YELLOW FLAG')

    def test_player_flag_counts(self):
        self.assertEqual(self.describe(player_flag=3)['state'], 'YELLOW')
        self.assertEqual(self.describe(player_flag=2)['label'], 'BLUE FLAG')
        self.assertEqual(self.describe(player_flag=4)['state'], 'RED')

    def test_most_serious_wins(self):
        self.assertEqual(self.describe([(0.2, 3)], safety_car=1)['label'], 'SAFETY CAR DEPLOYED')
        self.assertEqual(self.describe([(0.2, 3)], safety_car=2)['label'], 'VIRTUAL SAFETY CAR')
        self.assertEqual(self.describe([(0.2, 4)], safety_car=1)['label'], 'RED FLAG')
        self.assertEqual(self.describe(safety_car=3)['label'], 'FORMATION LAP')


class EngineStreamsStatusAndSector3Tests(unittest.TestCase):
    def setUp(self):
        self.engine = AnalysisEngine(SilentVoice())
        self.engine.update_session(SessionData(0, 30, 20, 20, 1000, 10, 11, 0))
        self.engine.update_car_status(status())
        self.telemetry = parse_player_telemetry(build_telemetry_packet(1, 1, 100, 1, 0, 4, 95), 0)
        self.lap = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 3), 0)

    def drive_lap(self, lap_num, s1=15000, s2=16000, lap_ms=45000):
        for distance in range(50, 1000, 50):  # the new lap starts at 10 m, past the line
            self.engine.update_telemetry(
                self.telemetry,
                replace(
                    self.lap,
                    lap_distance=distance,
                    current_lap_time_ms=100 + distance * 40,
                    current_lap_num=lap_num,
                    sector=0 if distance < 300 else 1 if distance < 700 else 2,
                    sector1_time_ms=s1 if distance >= 300 else 0,
                    sector2_time_ms=s2 if distance >= 700 else 0,
                ),
            )
        self.engine.update_telemetry(
            self.telemetry,
            replace(
                self.lap,
                lap_distance=10,
                current_lap_time_ms=500,
                current_lap_num=lap_num + 1,
                last_lap_time_ms=lap_ms,
                sector=0,
                sector1_time_ms=0,
                sector2_time_ms=0,
            ),
        )

    def test_sector_3_reaches_the_hud_after_the_line(self):
        self.drive_lap(1)
        state = web_hud.get_web_state(now=0)
        self.assertEqual(state['current_sectors_ms'], {})
        self.assertEqual(state['last_lap_sectors_ms'], {'1': 15000, '2': 16000, '3': 14000})  # JSON keys
        self.assertEqual(state['personal_sectors_ms']['3'], 14000)
        self.assertEqual(state['last_lap_sector_refs_ms'], {})  # first lap: nothing to beat

    def test_last_lap_is_compared_with_the_bests_before_it(self):
        self.drive_lap(1)
        self.drive_lap(2, s1=14800, s2=16100, lap_ms=44500)  # S3 = 13600
        state = web_hud.get_web_state(now=0)
        self.assertEqual(state['last_lap_sectors_ms']['3'], 13600)
        self.assertEqual(state['last_lap_sector_refs_ms']['3'], 14000)
        self.assertEqual(state['personal_sectors_ms']['3'], 13600)

    def test_sector_lines_are_learned_and_name_the_yellow_sector(self):
        self.drive_lap(1)
        self.assertEqual(self.engine.sector_starts, {2: 300, 3: 700})
        self.engine.session = parse_session_data(session_packet([(0.05, 0), (0.5, 3)]))
        self.engine.session.track_length = 1000
        self.drive_lap(2)
        self.assertEqual(web_hud.get_web_state(now=0)['track_status']['label'], 'YELLOW FLAG · S2')

    def test_track_status_follows_the_game(self):
        self.drive_lap(1)
        self.assertEqual(web_hud.get_web_state(now=0)['track_status']['state'], 'CLEAR')
        self.engine.update_car_status(status(fia_flags=3))
        self.drive_lap(2)
        self.assertEqual(web_hud.get_web_state(now=0)['track_status']['state'], 'YELLOW')
        self.engine.session.safety_car_status = 1
        self.drive_lap(3)
        state = web_hud.get_web_state(now=0)
        self.assertEqual(state['track_status']['label'], 'SAFETY CAR DEPLOYED')
        self.assertEqual(state['safety_car'], 'SAFETY CAR')

    def test_new_session_clears_last_lap_and_sector_lines(self):
        self.drive_lap(1)
        self.engine.update_session(SessionData(0, 30, 20, 20, 1200, 10, 12, 0))
        self.assertEqual(self.engine.last_lap_sectors, {})
        self.assertEqual(self.engine.sector_starts, {})
        self.assertEqual(web_hud.get_web_state(now=0)['last_lap_sectors_ms'], {})


if __name__ == '__main__':
    unittest.main()
