"""Pit wall: full-field timing tower (gaps at timing loops), gap-to-leader history, tyre
stints, strategy odds, and the HUD state store behind the WebSocket push."""

import base64
import json
import math
import os
import pathlib
import socket
import struct
import sys
import time
import unittest
from dataclasses import replace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))

import pit_wall
import web_hud
from analysis import AnalysisEngine
from packets import CarStatusData, SessionData, SessionHistoryData, TyreStint, parse_player_lap_data
from simulate_telemetry import build_lap_packet
from test_analysis import SilentVoice

TRACK = 1000.0
SOFT, MEDIUM, HARD = 16, 17, 18


def lap(position, distance, last_ms=0, pit_status=0, stops=0, result=2):
    base = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)
    lap_num = int(distance // TRACK) + 1 if distance >= 0 else 1
    return replace(base, car_position=position, current_lap_num=lap_num,
                   lap_distance=distance % TRACK if distance >= 0 else distance,
                   last_lap_time_ms=last_ms, pit_status=pit_status, num_pit_stops=stops,
                   result_status=result)


def status(compound=MEDIUM, age=3):
    return CarStatusData(50.0, 110.0, 30.0, compound, compound, age, 0, 2_000_000.0, 1, 0.0)


def run_field(wall, speeds, offsets, until, step=0.25, statuses=None):
    """Drive cars at constant speeds (m/s); offsets are seconds behind the start."""
    t = 0.0
    while t <= until + 1e-9:
        dists = [v * (t - o) for v, o in zip(speeds, offsets, strict=False)]
        order = sorted(range(len(dists)), key=lambda i: -dists[i])
        laps = [None] * len(dists)
        for pos, i in enumerate(order, start=1):
            laps[i] = lap(pos, dists[i])
        wall.update(laps, t, 0, statuses)
        t += step
    return laps


class TimingTowerTests(unittest.TestCase):
    def test_gap_and_interval_at_timing_loops(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        run_field(wall, [50.0, 50.0, 50.0], [0.0, 2.0, 3.5], until=30.0)
        rows = wall.rows
        self.assertEqual([r['index'] for r in rows], [0, 1, 2])
        self.assertEqual(rows[0]['gap_s'], 0.0)
        self.assertAlmostEqual(rows[1]['gap_s'], 2.0, places=2)
        self.assertAlmostEqual(rows[1]['interval_s'], 2.0, places=2)
        self.assertAlmostEqual(rows[2]['gap_s'], 3.5, places=2)
        self.assertAlmostEqual(rows[2]['interval_s'], 1.5, places=2)
        self.assertIsNone(rows[0]['interval_s'])

    def test_lapped_car_and_memory_stays_bounded(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        run_field(wall, [50.0, 25.0], [0.0, 0.0], until=60.0)
        slow = wall.rows[1]
        self.assertEqual(slow['laps_down'], 1)
        self.assertGreater(slow['gap_s'], 29.0)          # gap is still a time, measured a lap apart
        # Loops every car has passed are forgotten.
        self.assertGreaterEqual(min(wall.first_crossing), wall.oldest_loop)
        self.assertLess(len(wall.first_crossing), 40)

    def test_gap_history_records_each_line_crossing(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        run_field(wall, [50.0, 50.0], [0.0, 2.0], until=65.0)
        chart = wall.gap_chart({0: 'Verstappen', 1: 'Hamilton'}, {0: 2, 1: 0}, player_index=1)
        self.assertEqual([c['name'] for c in chart], ['Verstappen', 'Hamilton'])
        self.assertEqual(chart[0]['points'][:2], [[1, 0.0], [2, 0.0]])
        self.assertEqual([p[0] for p in chart[1]['points']], [1, 2, 3])
        self.assertTrue(all(abs(p[1] - 2.0) < 0.05 for p in chart[1]['points']))
        self.assertTrue(chart[1]['is_player'])

    def test_flashback_trims_history_after_the_restored_lap(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        run_field(wall, [50.0, 50.0], [0.0, 2.0], until=65.0)
        revision = wall.history_revision
        wall.update([lap(1, 1500.0), lap(2, 1400.0)], 30.0, 0)   # back to lap 2
        self.assertGreater(wall.history_revision, revision)
        self.assertEqual(sorted(wall.gap_history[0]), [1])
        self.assertEqual(wall.crossings, {})
        self.assertEqual(wall.last_session_time, 30.0)

    def test_grid_before_the_line_is_not_timed(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        wall.update([lap(1, -20.0), lap(2, -40.0)], 0.0, 0)
        wall.update([lap(1, -10.0), lap(2, -30.0)], 0.5, 0)
        self.assertEqual(wall.crossings, {})
        self.assertIsNone(wall.rows[1]['gap_s'])

    def test_retired_cars_go_to_the_bottom(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        laps = [lap(1, 500.0, result=6), lap(2, 400.0), lap(3, 300.0)]
        wall.update(laps, 1.0, 1)
        self.assertEqual([r['index'] for r in wall.rows], [1, 2, 0])
        self.assertEqual(wall.rows[-1]['status'], 'DNF')
        self.assertTrue(wall.rows[0]['is_player'])

    def test_qualifying_gaps_come_from_best_laps(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK, race=False)
        laps = [lap(2, 100.0, last_ms=91_000), lap(1, 200.0, last_ms=90_250), lap(3, 300.0)]
        wall.update(laps, 5.0, 0, [status(SOFT, 2), status(SOFT, 4), status(MEDIUM, 1)])
        rows = wall.rows
        self.assertEqual([r['index'] for r in rows], [1, 0, 2])
        self.assertEqual(rows[0]['gap_s'], 0.0)
        self.assertAlmostEqual(rows[1]['gap_s'], 0.75)
        self.assertAlmostEqual(rows[1]['interval_s'], 0.75)
        self.assertIsNone(rows[2]['gap_s'])
        self.assertTrue(rows[0]['fastest_lap'])
        self.assertEqual((rows[1]['compound'], rows[1]['tyre_age']), ('S', 2))

    def test_compound_letters(self):
        self.assertEqual(pit_wall.compound_letter('Soft'), 'S')
        self.assertEqual(pit_wall.compound_letter('Intermediate'), 'I')
        self.assertEqual(pit_wall.compound_letter('Wet'), 'W')
        self.assertEqual(pit_wall.compound_letter('C3'), None)
        self.assertEqual(pit_wall.compound_letter(None), None)


class StintTests(unittest.TestCase):
    def test_session_history_stints_win_over_status_tracking(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        history = SessionHistoryData(0, 9, 3, (1, 1, 1), [], [
            TyreStint(0, 1, 4, SOFT, SOFT, False), TyreStint(1, 5, 9, HARD, HARD, True)])
        wall.update([lap(1, 8500.0), lap(2, 8400.0)], 1.0, 0, [status(HARD), status(MEDIUM)],
                    histories={0: history})
        stints = wall.stints()
        self.assertEqual(stints[0]['stints'], [
            {'compound': 'S', 'start_lap': 1, 'end_lap': 4, 'open': False},
            {'compound': 'H', 'start_lap': 5, 'end_lap': 9, 'open': True}])
        # No history for car 1: built from its Car Status compound.
        self.assertEqual(stints[1]['stints'], [{'compound': 'M', 'start_lap': 9, 'end_lap': 9, 'open': True}])

    def test_status_fallback_splits_on_a_stop(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        for n in range(1, 8):
            stops = 1 if n >= 4 else 0
            compound = HARD if n >= 4 else MEDIUM
            wall.update([lap(1, (n - 1) * TRACK + 500.0, stops=stops)], float(n), 0, [status(compound)])
        sets = wall.stints()[0]['stints']
        self.assertEqual([(s['compound'], s['start_lap'], s['end_lap'], s['open']) for s in sets],
                         [('M', 1, 4, False), ('H', 5, 7, True)])

    def test_revision_moves_only_when_stints_change(self):
        wall = pit_wall.PitWall()
        wall.reset(TRACK)
        wall.update([lap(1, 500.0)], 1.0, 0, [status()])
        revision = wall.stint_revision
        wall.update([lap(1, 600.0)], 2.0, 0, [status()])
        self.assertEqual(wall.stint_revision, revision)
        wall.update([lap(1, 1600.0)], 22.0, 0, [status()])
        self.assertEqual(wall.stint_revision, revision + 1)


class EngineWiringTests(unittest.TestCase):
    RACE = SessionData(0, 30, 20, 52, 5891, 10, 7, 0)

    def test_timing_tower_and_charts_reach_the_hud(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(replace(self.RACE, track_length=1000), 1, 0)
        engine.update_participants([])
        captured = {}
        for t in range(0, 50):
            d0, d1 = 50.0 * t, 50.0 * (t - 2)
            engine.update_field([lap(1, d0), lap(2, d1)], 1, float(t))
            captured.update(engine.pit_wall_update())
        self.assertEqual(len(captured['timing_tower']), 2)
        self.assertTrue(captured['timing_tower'][1]['is_player'])
        self.assertAlmostEqual(captured['timing_tower'][1]['gap_s'], 2.0, places=1)
        self.assertEqual([c['index'] for c in captured['gap_history']], [0, 1])
        self.assertIn('tyre_stints', captured)
        # Unchanged chart data is not sent again.
        self.assertNotIn('gap_history', engine.pit_wall_update())

    def test_new_session_clears_the_pit_wall(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(self.RACE, 1, 0)
        engine.update_field([lap(1, 100.0), lap(2, 50.0)], 0, 1.0)
        web_hud.update_web_state({'timing_tower': engine.pit_wall.rows})
        engine.update_session(replace(self.RACE, track_id=11), 2, 0)
        self.assertEqual(engine.pit_wall.rows, [])
        state = web_hud.get_web_state(now=0)
        self.assertEqual(state['timing_tower'], [])
        self.assertEqual(state['gap_history'], [])
        self.assertIsNone(state['strategy_odds'])

    def test_strategy_odds_from_the_race_simulation(self):
        engine = AnalysisEngine(SilentVoice(), race_simulation='sync', sim_runs=60)
        engine.update_session(self.RACE, 1, 0)
        laps = [replace(lap(i + 1, 19 * 5891.0 + 100), current_lap_num=20, last_lap_time_ms=90000)
                for i in range(6)]
        engine.update_car_status(status(age=12), [status(age=12 + i) for i in range(6)])
        engine.update_field(laps, 2, 1000.0)
        engine.latest_lap = laps[2]
        self.assertIsNone(engine.strategy_odds())
        engine.update_race_simulation()
        odds = engine.strategy_odds()
        self.assertIsNotNone(odds)
        self.assertEqual(odds['runs'], 60)
        self.assertEqual(odds['lap'], 20)
        self.assertEqual(len(odds['position_probs']), 6)
        self.assertAlmostEqual(sum(odds['position_probs']), 1.0, places=2)
        self.assertGreaterEqual(odds['p_points'], odds['p_podium'])
        self.assertGreaterEqual(odds['p_podium'], odds['p_win'])
        self.assertEqual(odds['plans'][0]['label'], odds['recommended'])
        self.assertEqual(odds['history'], [{'lap': 20, 'p_win': odds['p_win'],
                                            'p_podium': odds['p_podium'], 'p_points': odds['p_points']}])
        self.assertIs(engine.strategy_odds(), odds)        # same result: cached, not rebuilt
        json.dumps(odds, allow_nan=False)


class StateStoreTests(unittest.TestCase):
    def setUp(self):
        self.saved = web_hud.get_web_state(now=0)
        self.addCleanup(web_hud.replace_web_state, self.saved)

    def test_delta_messages_carry_only_changed_keys(self):
        version, full = web_hud.web_state_message(None, now=0)
        message = json.loads(full)
        self.assertEqual(message['type'], 'full')
        self.assertIn('speed', message['state'])
        self.assertNotIn('last_update_monotonic', message['state'])
        web_hud.update_web_state({'speed': 123, 'gear': message['state']['gear']})
        version2, delta = web_hud.web_state_message(version, now=0)
        delta = json.loads(delta)
        self.assertEqual(delta['type'], 'delta')
        self.assertEqual(delta['state'], {'speed': 123})       # gear did not change
        self.assertGreater(version2, version)
        _, empty = web_hud.web_state_message(version2, now=0)
        self.assertEqual(json.loads(empty)['state'], {})

    def test_values_are_snapshotted_when_set(self):
        trail = [[1.0, 2.0]]
        web_hud.update_web_state({'track_map': trail})
        trail.append([3.0, 4.0])                                 # the engine keeps mutating its list
        self.assertEqual(web_hud.get_web_state(now=0)['track_map'], [[1.0, 2.0]])
        self.assertEqual(json.loads(web_hud.web_state_json(now=0))['track_map'], [[1.0, 2.0]])

    def test_non_finite_numbers_become_null(self):
        web_hud.update_web_state({'gap_ahead_s': float('nan'), 'opponent_threats': [{'gap_s': math.inf}]})
        state = json.loads(web_hud.web_state_json(now=0))
        self.assertIsNone(state['gap_ahead_s'])
        self.assertIsNone(state['opponent_threats'][0]['gap_s'])

    def test_connection_status_is_live(self):
        web_hud.update_web_state({'connected': True, 'last_update_monotonic': 10.0})
        _, message = web_hud.web_state_message(None, now=10.5)
        self.assertEqual(json.loads(message)['status'], {'connected': True, 'data_age_seconds': 0.5})
        self.assertFalse(json.loads(web_hud.web_state_json(now=20.0))['connected'])

    def test_voice_log_goes_through_the_store(self):
        version, _ = web_hud.web_state_message(None, now=0)
        web_hud.add_voice_log('Box box')
        _, delta = web_hud.web_state_message(version, now=0)
        self.assertEqual(json.loads(delta)['state']['voice_log'][0]['message'], 'Box box')

    def test_wait_returns_on_change(self):
        version, _ = web_hud.web_state_message(None, now=0)
        started = time.monotonic()
        self.assertEqual(web_hud.wait_for_web_state(version, 0.05), version)
        self.assertGreaterEqual(time.monotonic() - started, 0.04)
        web_hud.update_web_state({'speed': 77})
        self.assertGreater(web_hud.wait_for_web_state(version, 5.0), version)


class WebSocketTests(unittest.TestCase):
    def setUp(self):
        self.saved = web_hud.get_web_state(now=0)
        self.addCleanup(web_hud.replace_web_state, self.saved)
        self.server = web_hud.start_web_hud_server(port=0)
        self.assertIsNotNone(self.server)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def connect(self):
        sock = socket.create_connection(('127.0.0.1', self.port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        sock.sendall((f'GET /ws HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\n'
                      f'Connection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n'
                      f'Sec-WebSocket-Version: 13\r\n\r\n').encode())
        reader = sock.makefile('rb')
        head = b''
        while not head.endswith(b'\r\n\r\n'):
            head += reader.read(1)
        self.assertIn(b' 101 ', head.split(b'\r\n')[0])
        self.assertIn(web_hud.ws_accept_key(key).encode(), head)
        return sock, reader

    @staticmethod
    def read_message(reader):
        b0, b1 = reader.read(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack('!H', reader.read(2))[0]
        elif n == 127:
            n = struct.unpack('!Q', reader.read(8))[0]
        return b0 & 0x0F, reader.read(n)

    @staticmethod
    def client_frame(payload: bytes, opcode=0x1):
        mask = os.urandom(4)
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        return struct.pack('!BB', 0x80 | opcode, 0x80 | len(payload)) + mask + masked

    def test_full_state_then_pushed_deltas(self):
        sock, reader = self.connect()
        try:
            opcode, payload = self.read_message(reader)
            self.assertEqual(opcode, 0x1)
            first = json.loads(payload)
            self.assertEqual(first['type'], 'full')
            self.assertIn('timing_tower', first['state'])
            web_hud.update_web_state({'speed': 311, 'timing_tower': [{'index': 0, 'position': 1}]})
            deadline = time.monotonic() + 5
            seen = {}
            while time.monotonic() < deadline and 'speed' not in seen:
                _, payload = self.read_message(reader)
                message = json.loads(payload)
                self.assertEqual(message['type'], 'delta')
                seen.update(message['state'])
            self.assertEqual(seen['speed'], 311)
            self.assertEqual(seen['timing_tower'], [{'index': 0, 'position': 1}])
            # Ping is answered with a pong carrying the same payload.
            sock.sendall(self.client_frame(b'hi', 0x9))
            for _ in range(50):
                opcode, payload = self.read_message(reader)
                if opcode == 0xA:
                    break
            self.assertEqual((opcode, payload), (0xA, b'hi'))
            sock.sendall(self.client_frame(b'', 0x8))
        finally:
            reader.close()
            sock.close()

    def test_heartbeat_when_nothing_changes(self):
        sock, reader = self.connect()
        try:
            self.read_message(reader)
            started = time.monotonic()
            _, payload = self.read_message(reader)
            message = json.loads(payload)
            self.assertEqual(message['state'], {})
            self.assertIn('connected', message['status'])
            self.assertLess(time.monotonic() - started, web_hud.WS_HEARTBEAT_S + 1.0)
        finally:
            reader.close()
            sock.close()

    def test_plain_get_on_ws_path_is_rejected(self):
        sock = socket.create_connection(('127.0.0.1', self.port), timeout=5)
        try:
            sock.sendall(b'GET /ws HTTP/1.1\r\nHost: localhost\r\n\r\n')
            self.assertIn(b' 400 ', sock.recv(200).split(b'\r\n')[0])
        finally:
            sock.close()

    def test_polling_endpoint_still_serves_the_whole_state(self):
        import urllib.request
        web_hud.update_web_state({'speed': 199})
        with urllib.request.urlopen(f'http://127.0.0.1:{self.port}/api/telemetry', timeout=5) as res:
            state = json.loads(res.read())
        self.assertEqual(state['speed'], 199)
        self.assertIn('data_age_seconds', state)
        self.assertNotIn('last_update_monotonic', state)

    def test_frame_encoding_lengths(self):
        self.assertEqual(web_hud.ws_frame(b'x' * 10)[1], 10)
        self.assertEqual(web_hud.ws_frame(b'x' * 300)[1], 126)
        self.assertEqual(web_hud.ws_frame(b'x' * 70000)[1], 127)


if __name__ == '__main__':
    unittest.main()
