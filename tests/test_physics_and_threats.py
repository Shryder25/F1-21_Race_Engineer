"""Tests for the Box-Box-derived physics tire/fuel model and opponent-threat scoring."""

import json
import pathlib
import sys
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import physics_models as pm
from ai_client import AIClient, AIUnavailable
from analysis import AnalysisEngine
from engineer_brain import EngineerBrain, rule_based_answer
from packets import SessionData, parse_all_car_damage, parse_all_car_status, parse_player_car_status
from race_radio import RaceRadio
from simulate_telemetry import (
    build_car_damage_packet,
    build_car_status_packet,
    build_lap_packet,
)
from test_analysis import SilentVoice
from voice_commands import match_command


class TireModelTests(unittest.TestCase):
    def test_fresh_tyre_is_healthy(self):
        r = pm.evaluate_tire('Medium', 2, (6, 5, 6, 5), (88, 87, 89, 86))
        self.assertGreater(r.health_score, 85)
        self.assertGreater(r.remaining_laps, 10)
        self.assertEqual(r.overheat_risk, pm.OverheatRisk.NONE)

    def test_old_worn_hot_tyre_is_critical(self):
        r = pm.evaluate_tire('Soft', 22, (80, 78, 82, 79), (118, 117, 119, 116))
        self.assertLess(r.health_score, 20)
        self.assertEqual(r.remaining_laps, 0.0)
        self.assertEqual(r.overheat_risk, pm.OverheatRisk.CRITICAL)

    def test_unknown_compound_falls_back_to_medium(self):
        known = pm.evaluate_tire('Medium', 5, None, None)
        unknown = pm.evaluate_tire('Bogus', 5, None, None)
        self.assertEqual(known.deg_rate_s_per_lap, unknown.deg_rate_s_per_lap)

    def test_missing_wear_and_temp_still_produces_a_result(self):
        r = pm.evaluate_tire(None, None, None, None)
        self.assertEqual(r.compound, 'MEDIUM')
        self.assertGreaterEqual(r.health_score, 0)

    def test_degradation_increases_monotonically_with_age(self):
        young = pm.evaluate_tire('Hard', 3, (10, 10, 10, 10), None)
        old = pm.evaluate_tire('Hard', 20, (10, 10, 10, 10), None)
        self.assertLess(young.performance_loss_s, old.performance_loss_s)
        self.assertGreater(young.remaining_laps, old.remaining_laps)


class FuelModelTests(unittest.TestCase):
    def test_no_fuel_reading_returns_unknowns(self):
        r = pm.evaluate_fuel(None, 1.9, 20)
        self.assertEqual(r.fuel_kg, 0.0)
        self.assertIsNone(r.margin_laps)

    def test_plenty_of_fuel_allows_push(self):
        r = pm.evaluate_fuel(60.0, 1.8, 10)
        self.assertTrue(r.can_push)
        self.assertFalse(r.must_save)

    def test_tight_fuel_forces_save(self):
        r = pm.evaluate_fuel(15.0, 1.8, 10)
        self.assertTrue(r.must_save)
        self.assertFalse(r.can_push)

    def test_no_known_burn_rate_gives_no_margin(self):
        r = pm.evaluate_fuel(50.0, None, 10)
        self.assertIsNone(r.margin_laps)
        self.assertIsNone(r.can_push)


class OpponentThreatTests(unittest.TestCase):
    def test_close_faster_car_behind_is_high_threat(self):
        t = pm.evaluate_opponent_threat(5, 3, gap_s=-0.6, pace_delta_s=-0.3)
        self.assertEqual(t.threat_level, 'HIGH')
        self.assertIn('DRS range', t.factors[0])
        self.assertIsNotNone(t.laps_to_catch)

    def test_distant_car_is_low_threat(self):
        t = pm.evaluate_opponent_threat(6, 5, gap_s=8.0, pace_delta_s=0.05)
        self.assertEqual(t.threat_level, 'LOW')

    def test_car_ahead_pulling_away_has_no_catch_estimate(self):
        t = pm.evaluate_opponent_threat(2, 1, gap_s=1.8, pace_delta_s=0.2)
        self.assertIsNone(t.laps_to_catch)

    def test_zero_pace_delta_handled_without_division_error(self):
        t = pm.evaluate_opponent_threat(3, 2, gap_s=-1.0, pace_delta_s=0.0)
        self.assertIsNone(t.laps_to_catch)


class AllCarPacketTests(unittest.TestCase):
    def test_parse_all_car_status_and_damage_cover_every_car(self):
        status = parse_all_car_status(build_car_status_packet(0, 1, 55.5, 17, 4))
        self.assertEqual(len(status), 22)
        self.assertAlmostEqual(status[0].fuel_in_tank, 55.5)
        damage = parse_all_car_damage(build_car_damage_packet(0, 1, (10, 20, 30, 40)))
        self.assertEqual(len(damage), 22)
        self.assertEqual(damage[0].tyres_wear, (10.0, 20.0, 30.0, 40.0))

    def test_all_car_status_matches_single_car_parse(self):
        packet = build_car_status_packet(0, 1, 42.0, 18, 7)
        single = parse_player_car_status(packet, 0)
        all_cars = parse_all_car_status(packet)
        self.assertEqual(single, all_cars[0])


class RaceRadioThreatTests(unittest.TestCase):
    def setUp(self):
        self.spoken = []
        self.radio = RaceRadio(lambda msg, prio, **opts: self.spoken.append((msg, prio)))
        self.radio.reset(SessionData(0, 30, 20, 53, 5793, 10, 11, 0))

    def _lap(self, position, distance, lap_num=3, last_lap_ms=90000):
        return replace(self.base_lap, car_position=position, total_distance=distance,
                       current_lap_num=lap_num, last_lap_time_ms=last_lap_ms)

    def _drive(self, positions_and_offsets, last_laps_ms=None, frames=((0, 0), (10, 10), (11, 20))):
        """Replay several frames of a tight field so every nearby car's trajectory brackets
        the player's distance (what RaceRadio's crossing-time lookup requires), the same
        pattern test_race_radio.py's own gap test uses.
        """
        from packets import parse_player_lap_data
        base = parse_player_lap_data(build_lap_packet(1, 1, 0, 100, 1, 2), 0)
        last_laps_ms = last_laps_ms or {}
        for now, offset in frames:
            laps = [replace(base, car_position=99, total_distance=-1000.0, result_status=0)] * 22
            laps = list(laps)
            for index, (position, distance) in positions_and_offsets.items():
                laps[index] = replace(base, car_position=position, total_distance=distance + offset,
                                      result_status=2, pit_status=0, current_lap_num=3,
                                      last_lap_time_ms=last_laps_ms.get(index, 0))
            self.radio.update_field(laps, player_index=0, session_time=now)

    def test_opponent_threats_are_scored_for_nearby_positions(self):
        # Player (idx 0) P3 at distance ~20; idx 1 just ahead (P2), idx 2 just behind (P4),
        # idx 7 far down the order (P8) should be excluded by the position window.
        self._drive({0: (3, 20), 1: (2, 30), 2: (4, 10), 7: (8, -500)})
        threats = self.radio.opponent_threats
        indices = {t.car_index for t in threats}
        self.assertTrue(indices <= {1, 2})
        self.assertNotIn(7, indices)
        self.assertTrue(len(threats) >= 1)

    def test_pace_delta_uses_last_lap_times(self):
        self._drive({0: (3, 20), 1: (2, 30)}, last_laps_ms={0: 90000, 1: 89500})
        by_index = {t.car_index: t for t in self.radio.opponent_threats}
        self.assertIn(1, by_index)
        self.assertAlmostEqual(by_index[1].pace_delta_s, -0.5, places=2)


class AnalysisContextThreatTests(unittest.TestCase):
    def test_tire_health_and_threats_appear_in_build_context(self):
        engine = AnalysisEngine(SilentVoice())
        engine.update_session(SessionData(0, 30, 20, 53, 5793, 10, 11, 0), 1, 0)
        engine.latest_status = parse_player_car_status(build_car_status_packet(0, 1, 55.0, 17, 10), 0)
        engine.latest_damage = parse_all_car_damage(build_car_damage_packet(0, 1, (40, 41, 39, 42)))[0]
        engine.radio.opponent_threats = [
            pm.evaluate_opponent_threat(1, 2, gap_s=-0.9, pace_delta_s=-0.2)]
        ctx = engine.build_context()
        self.assertIsNotNone(ctx['tire_health'])
        self.assertEqual(ctx['tire_health']['limiting_factor'] in ('wear', 'age', 'performance', 'cliff'), True)
        self.assertEqual(len(ctx['opponent_threats']), 1)
        self.assertEqual(ctx['opponent_threats'][0]['threat_level'], 'HIGH')

    def test_no_status_means_no_tire_health(self):
        engine = AnalysisEngine(SilentVoice())
        self.assertIsNone(engine.tire_health())
        self.assertEqual(engine.opponent_threats(), [])


class ThreatsVoiceCommandTests(unittest.TestCase):
    def test_threats_phrase_matches_command(self):
        for phrase in ('any threats', 'who is attacking me', 'is anyone closing in'):
            self.assertEqual(match_command(phrase), 'threats', phrase)
        # "who's behind me" should still be the plain gap query, not threats
        self.assertEqual(match_command('whats behind me'), 'gap_behind')

    def test_rule_based_threats_answer(self):
        ctx = {'session_mode': 'race', 'opponent_threats': [
            {'threat_level': 'HIGH', 'gap_s': -0.8, 'factors': ['gap 0.8s (DRS range)']}]}
        answer = rule_based_answer('threats', ctx)
        self.assertIn('behind', answer)
        self.assertIn('0.8', answer)

    def test_no_threats_answer(self):
        ctx = {'session_mode': 'race', 'opponent_threats': None}
        self.assertEqual(rule_based_answer('threats', ctx), 'No immediate threats.')

    def test_threats_blocked_outside_race(self):
        ctx = {'session_mode': 'time_trial', 'opponent_threats': None}
        self.assertIn('time trial', rule_based_answer('threats', ctx))


class FakeJSONHandler(BaseHTTPRequestHandler):
    reply = '{"voice": "Gap ahead two point four.", "advice": "Tyres are fine, keep pushing."}'
    accept_json_mode = True
    status = 200

    def do_POST(self):
        length = int(self.headers['Content-Length'])
        body = json.loads(self.rfile.read(length))
        if body.get('response_format') and not FakeJSONHandler.accept_json_mode:
            self.send_response(400)
            self.end_headers()
            return
        self.send_response(self.status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'choices': [{'message': {'content': self.reply}}]}).encode())

    def log_message(self, *args):
        pass


class AIClientJSONModeTests(unittest.TestCase):
    def setUp(self):
        FakeJSONHandler.reply = '{"voice": "Gap ahead two point four.", "advice": "Tyres are fine, keep pushing."}'
        FakeJSONHandler.accept_json_mode = True
        FakeJSONHandler.status = 200
        self.server = HTTPServer(('127.0.0.1', 0), FakeJSONHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.client = AIClient(f'http://127.0.0.1:{self.server.server_port}/v1', 'm', timeout=5)

    def test_chat_json_parses_structured_reply(self):
        result = self.client.chat_json('sys', 'user')
        self.assertEqual(result['voice'], 'Gap ahead two point four.')
        self.assertEqual(result['advice'], 'Tyres are fine, keep pushing.')

    def test_chat_json_retries_without_response_format_if_rejected(self):
        FakeJSONHandler.accept_json_mode = False
        result = self.client.chat_json('sys', 'user')
        self.assertEqual(result['voice'], 'Gap ahead two point four.')

    def test_chat_json_extracts_object_wrapped_in_prose(self):
        FakeJSONHandler.reply = ('<think>reasoning here</think>Sure, here you go: '
                                 '{"voice": "Box this lap.", "advice": ""}')
        result = self.client.chat_json('sys', 'user')
        self.assertEqual(result['voice'], 'Box this lap.')

    def test_chat_json_raises_on_unparsable_reply(self):
        FakeJSONHandler.reply = 'not json at all, sorry'
        with self.assertRaises(AIUnavailable):
            self.client.chat_json('sys', 'user')

    def test_chat_json_raises_on_server_error(self):
        FakeJSONHandler.status = 500
        with self.assertRaises(AIUnavailable):
            self.client.chat_json('sys', 'user')


class BrainJSONPathTests(unittest.TestCase):
    CONTEXT = {'session_mode': 'race', 'gap_ahead_s': 2.43}

    def test_brain_prefers_json_voice_field_and_records_advice(self):
        class JSONClient:
            def chat_json(self, system, user):
                return {'voice': 'Gap ahead is two point four.', 'advice': 'Tyres look good.'}

            def chat(self, system, user):
                raise AssertionError('should not fall back to plain chat when JSON succeeds')

        brain = EngineerBrain(lambda: self.CONTEXT, JSONClient())
        self.assertEqual(brain.answer('gap_ahead'), 'Gap ahead is two point four.')
        self.assertEqual(brain.last_source, 'ai')
        self.assertEqual(brain.last_advice, 'Tyres look good.')

    def test_brain_falls_back_to_plain_chat_when_json_mode_fails(self):
        class PartialClient:
            def chat_json(self, system, user):
                raise AIUnavailable('server rejected response_format')

            def chat(self, system, user):
                return 'Gap ahead is two point four.'

        brain = EngineerBrain(lambda: self.CONTEXT, PartialClient())
        self.assertEqual(brain.answer('gap_ahead'), 'Gap ahead is two point four.')
        self.assertEqual(brain.last_source, 'ai')
        self.assertEqual(brain.last_advice, '')

    def test_brain_falls_back_to_rules_when_everything_fails(self):
        class DeadClient:
            def chat_json(self, system, user):
                raise AIUnavailable('down')

            def chat(self, system, user):
                raise AIUnavailable('down')

        brain = EngineerBrain(lambda: self.CONTEXT, DeadClient())
        self.assertEqual(brain.answer('gap_ahead'), 'Gap ahead is 2.4.')
        self.assertEqual(brain.last_source, 'fallback')


if __name__ == '__main__':
    unittest.main()
