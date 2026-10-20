import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
from ai_client import AIClient, AIUnavailable, strip_reasoning
from analysis import AnalysisEngine, PROACTIVE_KEYS
from engineer_brain import (EngineerBrain, build_prompt, clean_radio_text, format_context,
                            rule_based_answer)
from history import TelemetryHistory
from packets import SessionData, parse_player_car_damage, parse_player_car_status
from simulate_telemetry import (build_car_damage_packet, build_car_status_packet,
                                build_session_packet)
from test_analysis import SilentVoice
from voice_commands import ACK_TEXT, CommandDispatcher, VoiceCommandListener, match_command
from voice_engine import Priority


class PacketTests(unittest.TestCase):
    def test_car_status_and_damage_round_trip(self):
        status = parse_player_car_status(build_car_status_packet(0, 1, 55.5, 17, 4), 0)
        self.assertAlmostEqual(status.fuel_in_tank, 55.5)
        self.assertEqual((status.actual_tyre_compound, status.tyres_age_laps), (17, 4))
        self.assertEqual(status.compound_name, 'Medium')
        damage = parse_player_car_damage(build_car_damage_packet(0, 1, (10, 20, 30, 40)), 0)
        self.assertEqual(damage.tyres_wear, (10.0, 20.0, 30.0, 40.0))
        self.assertEqual(damage.worst_tyre_wear, 40.0)


class HistoryStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.history = TelemetryHistory(pathlib.Path(self.temp.name) / 'h.sqlite3')
        self.addCleanup(self.history.close)
        self.session = SessionData(0, 30, 20, 53, 5793, 10, 11, 0)

    def stats(self, lap, age, wear, fuel_start):
        return {'lap_num': lap, 'lap_time_ms': 90000, 'eligible': True, 'compound': 17,
                'tyre_age_laps': age, 'wear': (wear - 1, wear - 1, wear, wear - 2),
                'fuel_start': fuel_start, 'fuel_end': fuel_start - 1.9,
                'sectors': {1: 30000, 2: 31000, 3: 29000}, 'sector_deltas': {1: -120, 2: 40}}

    def test_lap_stats_are_logged_and_queried_by_track_and_compound(self):
        old = self.history.start_session(self.session, 1)
        new = self.history.start_session(self.session, 2)
        for lap in range(1, 4):
            self.history.record_lap_stats(old, self.stats(lap, lap, lap * 2.0, 100 - 1.9 * lap))
        self.history.record_lap_stats(new, self.stats(1, 1, 2.0, 100))
        self.history.record_lap_stats(old, self.stats(2, 2, 4.5, 96))   # re-logging replaces, not duplicates
        samples = self.history.tyre_wear_samples(self.session, 17, exclude_capture=new)
        self.assertEqual(sorted(samples), [(1, 2.0), (2, 4.5), (3, 6.0)])
        self.assertEqual(self.history.tyre_wear_samples(self.session, 18), [])
        self.assertEqual(len(self.history.fuel_use_samples(self.session, exclude_capture=new)), 3)
        row = self.history.connection.execute(
            'SELECT s1_delta_ms, s3_delta_ms FROM lap_stats WHERE capture_id=? AND lap_num=1', (old,)).fetchone()
        self.assertEqual(row, (-120, None))


class AnalysisIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.history = TelemetryHistory(pathlib.Path(self.temp.name) / 'h.sqlite3')
        self.addCleanup(self.history.close)
        self.voice = SilentVoice()
        self.engine = AnalysisEngine(self.voice, history=self.history)
        self.engine.update_session(SessionData(0, 30, 20, 53, 5793, 10, 11, 0), 77, 0)

    def feed_laps(self, count, rate=2.5, start_fuel=100.0):
        for lap in range(1, count + 1):
            self.engine.update_car_status(parse_player_car_status(
                build_car_status_packet(0, 1, start_fuel - 1.9 * lap, 17, lap), 0))
            self.engine.update_car_damage(parse_player_car_damage(
                build_car_damage_packet(0, 1, (rate * lap,) * 4), 0))
            self.engine.lap_start_fuel = start_fuel - 1.9 * (lap - 1)
            self.engine._record_lap_stats(lap, 90000 + 80 * lap, True, {1: 30000, 2: 31000, 3: 29000})

    def test_laps_feed_the_ml_pit_prediction_and_prompt_context(self):
        self.feed_laps(8)
        lap = type('L', (), {'current_lap_num': 9, 'car_position': 3, 'last_lap_time_ms': 90640})()
        self.engine.latest_lap = lap
        prediction = self.engine.pit_prediction()
        # 20% worn at 2.5%/lap: 60% in 16 laps (lap 25), give or take the model's range
        self.assertIn(prediction.ideal_lap, (24, 25))
        soonest, _expected, latest = prediction.ideal_lap_range
        self.assertLessEqual(soonest, 25)
        self.assertGreaterEqual(latest, 25)
        self.assertEqual(prediction.basis, 'prior')           # no fitted models: this stint only
        fuel = self.engine.fuel_prediction()
        self.assertAlmostEqual(fuel['per_lap_kg'], 1.9, places=1)
        self.assertAlmostEqual(fuel['laps_of_fuel'], (100 - 1.9 * 8) / 1.9, delta=0.5)
        low, mid, high = fuel['per_lap_kg_range']
        self.assertLess(low, 1.9)
        self.assertGreater(high, 1.9)
        context = self.engine.build_context()
        self.assertEqual(context['compound'], 'Medium')
        self.assertEqual(context['pit_prediction']['ideal_lap'], prediction.ideal_lap)
        text = format_context(context)
        self.assertIn('ML pit prediction', text)
        self.assertIn(f'ideal stop lap {prediction.ideal_lap} (range [', text)
        self.assertIn('Tyre model ranges', text)
        rows = self.history.connection.execute('SELECT COUNT(*) FROM lap_stats').fetchone()[0]
        self.assertEqual(rows, 8)

    def test_history_from_earlier_sessions_improves_a_fresh_stint(self):
        self.feed_laps(8)
        self.engine.update_session(SessionData(0, 30, 20, 53, 5793, 10, 11, 0), 78, 0)  # new session
        self.engine.update_car_status(parse_player_car_status(build_car_status_packet(0, 1, 100, 17, 1), 0))
        self.engine.latest_lap = type('L', (), {'current_lap_num': 2})()
        self.assertIsNotNone(self.engine.pit_prediction())

    def test_on_demand_mode_stays_silent_except_critical_events(self):
        quiet_voice = SilentVoice()
        quiet = AnalysisEngine(quiet_voice, on_demand=True)
        quiet.speak('Minus 0.3 to your best.', Priority.LOW, key='live_delta')
        quiet.speak('Braking now.', Priority.CRITICAL, key='braking')
        self.assertEqual(quiet_voice.messages, [])
        quiet.speak('Critical! Front Left overheating.', Priority.CRITICAL, key='tyre_temperature')
        self.assertEqual(len(quiet_voice.messages), 1)
        self.assertIn('tyre_temperature', PROACTIVE_KEYS)

    def test_default_mode_keeps_existing_callouts(self):
        self.voice.messages.clear()   # drop the session welcome
        self.engine.speak('Lap complete.', Priority.TIMING, key='lap_time')
        self.assertEqual(len(self.voice.messages), 1)


class FakeAIHandler(BaseHTTPRequestHandler):
    reply = '<think>Gap is 2.4, tyres fine.</think>Gap ahead is two point four, keep pushing.'
    status = 200
    seen = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        FakeAIHandler.seen.append((self.path, self.headers.get('Authorization'), body))
        self.send_response(self.status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'choices': [{'message': {'content': self.reply}}]}).encode())

    def log_message(self, *args):
        pass


class AIClientTests(unittest.TestCase):
    def setUp(self):
        FakeAIHandler.seen = []
        FakeAIHandler.status = 200
        self.server = HTTPServer(('127.0.0.1', 0), FakeAIHandler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.url = f'http://127.0.0.1:{self.server.server_port}/v1'

    def test_strip_reasoning(self):
        self.assertEqual(strip_reasoning('<think>hmm\nmore</think>\nBox this lap.'), 'Box this lap.')
        self.assertEqual(strip_reasoning('<think>never finished'), '')
        self.assertEqual(strip_reasoning('Plain answer'), 'Plain answer')

    def test_chat_posts_openai_format_and_strips_thinking(self):
        client = AIClient(self.url, 'deepseek-r1:8b', 'secret', timeout=5)
        answer = client.chat('sys', 'user')
        self.assertEqual(answer, 'Gap ahead is two point four, keep pushing.')
        path, auth, body = FakeAIHandler.seen[0]
        self.assertEqual((path, auth, body['model']), ('/v1/chat/completions', 'Bearer secret', 'deepseek-r1:8b'))
        self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])

    def test_errors_raise_ai_unavailable(self):
        FakeAIHandler.status = 500
        with self.assertRaises(AIUnavailable):
            AIClient(self.url, 'm', timeout=2).chat('s', 'u')
        with self.assertRaises(AIUnavailable):
            AIClient('http://127.0.0.1:9/v1', 'm', timeout=1).chat('s', 'u')


class BrainTests(unittest.TestCase):
    CONTEXT = {'track': 'Monza', 'gap_ahead_s': 2.43, 'gap_behind_s': None, 'weather': 'light rain',
               'track_temp_c': 28, 'air_temp_c': 21,
               'tyre_surface_c': {'FL': 98, 'FR': 101, 'RL': 95, 'RR': 96},
               'tyre_wear_pct': {'FL': 31.0, 'FR': 33.0, 'RL': 28.0, 'RR': 29.0},
               'pit_prediction': {'wear_now_pct': 33.0, 'wear_per_lap_pct': 2.5, 'ideal_lap': 18,
                                  'latest_lap': 24, 'finish_without_stop': False, 'samples': 9},
               'last_lap_ms': 83456, 'delta_to_best_ms': 350, 'pace_trend_s_per_lap': 0.12}

    def test_prompt_contains_live_stats_and_ml_prediction(self):
        system, user = build_prompt('pit_window', self.CONTEXT)
        self.assertIn('two short sentences', system)
        self.assertIn('ideal stop lap 18', user)
        self.assertIn('Gap to car ahead (s): 2.4', user)
        self.assertNotIn('Gap to car behind', user)       # missing data is omitted, not invented
        self.assertIn('When is my pit window?', user)

    def test_fallback_answers_without_ai(self):
        self.assertEqual(rule_based_answer('gap_ahead', self.CONTEXT), 'Gap ahead is 2.4.')
        self.assertEqual(rule_based_answer('gap_behind', self.CONTEXT), 'No gap behind reading right now.')
        self.assertIn('lap 18', rule_based_answer('pit_window', self.CONTEXT))
        self.assertIn('Fronts 98 and 101', rule_based_answer('tyre_temps', self.CONTEXT))
        self.assertIn('Light rain', rule_based_answer('weather', self.CONTEXT))
        self.assertIn('1:23.456', rule_based_answer('pace', self.CONTEXT))

    def test_brain_uses_ai_then_falls_back_when_it_fails(self):
        class Good:
            def chat(self, system, user):
                return '**Gap ahead is two point four.** Tyres are good, keep pushing.'

        class Down:
            def chat(self, system, user):
                raise AIUnavailable('timeout')

        brain = EngineerBrain(lambda: self.CONTEXT, Good())
        self.assertEqual(brain.answer('gap_ahead'), 'Gap ahead is two point four. Tyres are good, keep pushing.')
        self.assertEqual(brain.last_source, 'ai')
        brain = EngineerBrain(lambda: self.CONTEXT, Down())
        self.assertEqual(brain.answer('gap_ahead'), 'Gap ahead is 2.4.')
        self.assertEqual(brain.last_source, 'fallback')

    def test_long_answers_are_trimmed_to_radio_length(self):
        text = ' '.join(['Sentence number %d is fairly long indeed.' % i for i in range(20)])
        trimmed = clean_radio_text(text, max_words=20)
        self.assertLessEqual(len(trimmed.split()), 20)
        self.assertTrue(trimmed.endswith('.'))


class CommandMatchingTests(unittest.TestCase):
    def test_all_supported_commands(self):
        cases = {
            'Weather report': 'weather', 'what is the weather doing': 'weather',
            'Tire temps': 'tyre_temps', 'tyre temperatures please': 'tyre_temps',
            'Pit window': 'pit_window', 'when do we box': 'pit_window',
            'Gap to car ahead': 'gap_ahead', 'gap to the car in front': 'gap_ahead',
            'Gap to car behind': 'gap_behind', "how far is the car behind": 'gap_behind',
            'Pace update': 'pace', 'Engineer, give me a pace update.': 'pace',
            'what is the pit window for these tyres': 'pit_window',
        }
        for phrase, expected in cases.items():
            self.assertEqual(match_command(phrase), expected, phrase)

    def test_unknown_or_empty_speech_is_ignored(self):
        for phrase in ('', 'hello there', 'engineer', 'play some music'):
            self.assertIsNone(match_command(phrase), phrase)


class RecordingVoice:
    is_running = True

    def __init__(self):
        self.said = []
        self.lock = threading.Lock()

    def say(self, message, priority, **kwargs):
        with self.lock:
            self.said.append((message, priority, kwargs))
        return True


class SlowBrain:
    def __init__(self, delay=0.4):
        self.delay = delay
        self.commands = []

    def answer(self, command):
        self.commands.append(command)
        time.sleep(self.delay)
        return f'Answer for {command}.'


class DispatcherTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=3):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_we_are_checking_is_spoken_before_the_slow_ai_answers(self):
        voice, brain = RecordingVoice(), SlowBrain(delay=0.5)
        dispatcher = CommandDispatcher(voice, brain)
        self.addCleanup(dispatcher.shutdown)
        started = time.monotonic()
        self.assertEqual(dispatcher.handle_text('Engineer, gap to car ahead'), 'gap_ahead')
        self.assertLess(time.monotonic() - started, 0.2)             # handle_text does not block on the AI
        self.assertEqual(voice.said[0][0], ACK_TEXT)                  # cue is queued immediately
        self.assertEqual(len(voice.said), 1)
        self.assertTrue(self.wait_for(lambda: len(voice.said) == 2))
        self.assertEqual(voice.said[1][0], 'Answer for gap_ahead.')
        # cue is CRITICAL (instant, ignores the corner filter); the answer is lower priority,
        # so it never interrupts the cue that is still being spoken
        self.assertEqual([item[1] for item in voice.said], [Priority.CRITICAL, Priority.STRATEGY])
        self.assertNotEqual(voice.said[0][2]['key'], voice.said[1][2]['key'])

    def test_duplicates_are_debounced_but_different_commands_are_answered(self):
        voice, brain = RecordingVoice(), SlowBrain(delay=0.05)
        dispatcher = CommandDispatcher(voice, brain, debounce_s=5)
        self.addCleanup(dispatcher.shutdown)
        dispatcher.handle_text('pace update')
        dispatcher.handle_text('pace update')
        dispatcher.handle_text('weather report')
        self.assertTrue(self.wait_for(lambda: len(voice.said) == 4))
        self.assertEqual(sorted(brain.commands), ['pace', 'weather'])

    def test_unknown_speech_does_nothing_and_brain_errors_are_spoken(self):
        voice = RecordingVoice()

        class Broken:
            def answer(self, command):
                raise RuntimeError('boom')

        dispatcher = CommandDispatcher(voice, Broken())
        self.addCleanup(dispatcher.shutdown)
        self.assertIsNone(dispatcher.handle_text('nice tune for a song'))
        self.assertEqual(voice.said, [])
        dispatcher.handle_text('tyre temps')
        self.assertTrue(self.wait_for(lambda: len(voice.said) == 2))
        self.assertIn('Sorry', voice.said[1][0])

    def test_muted_voice_logs_the_response(self):
        voice = RecordingVoice()
        voice.is_running = False
        voice.say = lambda *a, **k: False
        logged = []
        brain = SlowBrain(delay=0)
        dispatcher = CommandDispatcher(voice, brain, on_log=lambda m, status: logged.append((m, status)))
        self.addCleanup(dispatcher.shutdown)
        dispatcher.handle_text('pit window')
        self.assertTrue(self.wait_for(lambda: len(logged) == 2))
        self.assertEqual(logged[0], (ACK_TEXT, 'muted'))


class ListenerTests(unittest.TestCase):
    def test_transcripts_are_dispatched_and_engineer_speech_is_ignored(self):
        heard = []

        class Dispatcher:
            def handle_text(self, text):
                heard.append(text)

        class Voice:
            is_busy = False

        voice = Voice()
        listener = VoiceCommandListener(Dispatcher(), voice=voice)
        listener.transcribe = lambda audio: audio
        listener._on_audio(None, 'weather report')
        voice.is_busy = True
        listener._on_audio(None, 'we are checking')
        self.assertEqual(heard, ['weather report'])

    def test_unavailable_microphone_disables_commands_gracefully(self):
        def broken_microphone(device_index=None):
            raise OSError('no microphone')

        listener = VoiceCommandListener(object(), microphone_factory=broken_microphone,
                                        recognizer_factory=lambda: type('R', (), {})())
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            self.skipTest('SpeechRecognition not installed; start() returns False before the mic is touched')
        self.assertFalse(listener.start())


class StreamRoutingTests(unittest.TestCase):
    def test_status_and_damage_packets_reach_the_analysis_engine(self):
        from telemetry_stream import TelemetryStream
        engine = AnalysisEngine(SilentVoice())
        stream = TelemetryStream(engine)
        stream.handle(build_session_packet(0, 1, 11), now=0)
        self.assertFalse(stream.handle(build_car_status_packet(0.1, 2, 57.0, 16, 3), now=1))
        self.assertFalse(stream.handle(build_car_damage_packet(0.1, 2, (20, 21, 24, 22)), now=1))
        self.assertEqual(engine.latest_status.tyres_age_laps, 3)
        self.assertEqual(engine.latest_status.compound_name, 'Soft')
        self.assertEqual(engine.latest_damage.worst_tyre_wear, 24.0)
        bad = bytearray(build_car_damage_packet(0.2, 3, (20, 21, 24, 22)))
        bad[24:28] = b'\x00\x00\xc8\x43'      # 400.0% tyre wear is impossible
        self.assertFalse(stream.handle(bytes(bad), now=1))
        self.assertEqual(engine.latest_damage.worst_tyre_wear, 24.0)


class SetupConsultTests(unittest.TestCase):
    def test_setup_consult_uses_deepseek_when_available_and_falls_back_otherwise(self):
        from track_data import generate_ai_setup_recommendation

        class Client:
            seen = None

            def chat(self, system, user, max_tokens=600):
                Client.seen = (system, user, max_tokens)
                return 'Run low drag: wings 1 and 3. Short gearing is not needed at Monza.'

        class Down:
            def chat(self, system, user, max_tokens=600):
                raise AIUnavailable('offline')

        result = generate_ai_setup_recommendation(11, 'dry', 'understeer', Client())
        self.assertIn('low drag', result['ai_consultation'])
        self.assertIn('Monza', Client.seen[1])
        self.assertIn('understeer', Client.seen[1])
        self.assertGreaterEqual(Client.seen[2], 1000)      # room for R1 reasoning plus the answer
        fallback = generate_ai_setup_recommendation(11, 'dry', 'balanced', Down())
        self.assertIn('Engineer Setup Briefing', fallback['ai_consultation'])
        self.assertIn('ai_consultation_error', fallback)

    def test_hud_server_handles_requests_concurrently(self):
        import socketserver
        import web_hud
        server = web_hud.start_web_hud_server(port=0)
        self.assertIsNotNone(server)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.assertIsInstance(server, socketserver.ThreadingMixIn)


if __name__ == '__main__':
    unittest.main()
