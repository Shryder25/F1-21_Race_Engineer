"""Radio upgrades: game-button push-to-talk, offline Whisper, intent matching,
the calls-per-lap budget and no talking in high-load corners."""

import json
import pathlib
import struct
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import game_buttons
import intents
import push_to_talk as ptt
import radio_discipline
import speech_to_text as stt
import voice_engine
from analysis import AnalysisEngine
from packets import parse_buttons_event
from simulate_telemetry import build_header
from telemetry_stream import TelemetryStream
from voice_commands import match_command
from voice_engine import Priority, VoiceEngine


def buttons_packet(mask: int, size: int = 36) -> bytes:
    data = build_header(3, 1.0, 10) + b'BUTN' + struct.pack('<I', mask)
    return data + bytes(size - len(data))


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ──────────────────────────────────────────────
# 1. Push-to-talk from the game's Buttons event
# ──────────────────────────────────────────────

class GameButtonParsingTests(unittest.TestCase):
    def test_buttons_event_is_parsed(self):
        self.assertEqual(parse_buttons_event(buttons_packet(0x0400)), 0x0400)
        self.assertIsNone(parse_buttons_event(build_header(3, 1.0, 1) + b'SSTA' + bytes(8)))

    def test_button_names_aliases_and_masks(self):
        self.assertEqual(game_buttons.parse_game_button('R1'), 0x0400)
        self.assertEqual(game_buttons.parse_game_button('rb'), 0x0400)
        self.assertEqual(game_buttons.parse_game_button('dpad-up'), 0x0040)
        self.assertEqual(game_buttons.parse_game_button('0x2000'), 0x2000)
        self.assertEqual(game_buttons.button_label(0x0400), 'r1')
        self.assertEqual(game_buttons.button_label(0x0600), 'l1+r1')
        for bad in ('banana', '0', '0x100000'):
            with self.assertRaises(ValueError):
                game_buttons.parse_game_button(bad)

    def test_stream_forwards_buttons_before_any_session(self):
        seen = []
        stream = TelemetryStream(SimpleNamespace())
        stream.on_buttons = seen.append
        self.assertFalse(stream.handle(buttons_packet(0x0400)))
        self.assertFalse(stream.handle(buttons_packet(0, size=40)))   # the other patch's padding
        self.assertEqual(seen, [0x0400, 0])

    def test_stream_ignores_buttons_without_a_handler_and_survives_a_bad_one(self):
        stream = TelemetryStream(SimpleNamespace())
        self.assertFalse(stream.handle(buttons_packet(1)))

        def boom(_mask):
            raise RuntimeError('broken')
        stream.on_buttons = boom
        self.assertFalse(stream.handle(buttons_packet(1)))

    def test_finder_picks_a_new_press_and_ignores_buttons_already_held(self):
        feed = iter([b'not buttons', None, buttons_packet(0x0001), buttons_packet(0x0001),
                     buttons_packet(0x0401)])
        found = game_buttons.find_game_button(lambda: next(feed, None), parse_buttons_event,
                                              out=lambda *_: None)
        self.assertEqual(found, 0x0400)

    def test_finder_times_out_with_a_hint_when_the_game_sends_nothing(self):
        lines, clock = [], iter(range(100))
        found = game_buttons.find_game_button(lambda: None, parse_buttons_event, timeout_s=3,
                                              out=lines.append, clock=lambda: next(clock))
        self.assertIsNone(found)
        self.assertIn('UDP Telemetry', lines[-1])


class GameButtonWatcherTests(unittest.TestCase):
    def make(self, mode=ptt.HOLD, **options):
        events, now = [], [0.0]
        watcher = ptt.GameButtonWatcher(0x0400, lambda: events.append('open'),
                                        lambda: events.append('close'), mode=mode,
                                        clock=lambda: now[0], **options)
        return watcher, events, now

    def test_hold_talks_while_the_bit_is_set(self):
        watcher, events, now = self.make()
        watcher.handle_status(0x0001)          # another button: nothing
        now[0] = 0.1
        watcher.handle_status(0x0401)
        now[0] = 1.0
        watcher.handle_status(0x0001)
        self.assertEqual(events, ['open', 'close'])

    def test_toggle_opens_and_closes_on_presses(self):
        watcher, events, now = self.make(ptt.TOGGLE)
        for t, mask in ((0.1, 0x400), (0.3, 0), (1.0, 0x400), (1.2, 0)):
            now[0] = t
            watcher.handle_status(mask)
        self.assertEqual(events, ['open', 'close'])

    def test_lost_release_packet_is_closed_by_the_ticker(self):
        watcher, events, now = self.make(max_open_s=2.0)
        now[0] = 0.1
        watcher.handle_status(0x400)
        now[0] = 2.5
        watcher.tick()                          # no event arrived, the safety close fires
        self.assertEqual(events, ['open', 'close'])

    def test_thread_starts_and_stops_cleanly(self):
        events = []
        watcher = ptt.GameButtonWatcher(0x400, lambda: events.append('open'),
                                        lambda: events.append('close'), tick_s=0.01)
        self.assertTrue(watcher.start())
        watcher.handle_status(0x400)
        watcher.stop()
        self.assertEqual(events, ['open', 'close'])   # stop closes an open radio


class GameButtonSettingsTests(unittest.TestCase):
    def test_saved_game_button_is_used_and_kept_next_to_a_wheel_button(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'push_to_talk.json'
            ptt.save_game_button(0x0400, ptt.HOLD, path)
            settings = ptt.resolve_settings(config_path=path)
            self.assertEqual((settings.binding, settings.game_button), (None, 0x0400))
            ptt.save_config(ptt.Binding('button', 5), 'T300', ptt.HOLD, path)
            saved = json.loads(path.read_text())
            self.assertEqual((saved['button'], saved['game_button']), ('5', 'r1'))
            both = ptt.resolve_settings(config_path=path)
            self.assertEqual((both.binding.label, both.game_button), ('5', 0x0400))
            override = ptt.resolve_settings(config_path=path, game_button='l1')
            self.assertEqual(override.game_button, 0x0200)
            with self.assertRaises(ValueError):
                ptt.resolve_settings(config_path=path, game_button='banana')

    def test_bad_saved_game_button_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'push_to_talk.json'
            path.write_text('{"game_button": "banana"}')
            self.assertIsNone(ptt.resolve_settings(config_path=path))

    def test_start_with_only_a_game_button_drives_the_recorder(self):
        class Recorder:
            recording = stopped = False

            def start(self):
                return True

            def begin(self):
                self.recording = True

            def end(self):
                self.recording = False
                return None

            def stop(self):
                self.stopped = True

        recorder = Recorder()
        radio = ptt.start(ptt.Settings(None, game_button=0x0400), object(), recorder=recorder,
                          transcriber=lambda audio: None)
        self.assertIsNotNone(radio)
        self.assertIsNone(radio.watcher)
        radio.on_game_buttons(0x0400)
        self.assertTrue(recorder.recording)
        time.sleep(0.06)                       # past the contact-bounce window
        radio.on_game_buttons(0)
        self.assertFalse(recorder.recording)
        radio.stop()
        self.assertTrue(recorder.stopped)

    def test_run_game_button_finder_saves_the_choice(self):
        packets = [buttons_packet(0), buttons_packet(0x2000)]

        class Sock:
            closed = False

            def settimeout(self, _):
                pass

            def recvfrom(self, _size):
                if packets:
                    return packets.pop(0), ('127.0.0.1', 1)
                raise TimeoutError

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'ptt.json'
            sock = Sock()
            with patch('builtins.print'):
                code = ptt.run_game_button_finder(20777, ptt.TOGGLE, path, socket_factory=lambda: sock)
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(path.read_text()), {'game_button': 'l3', 'mode': 'toggle'})
            self.assertTrue(sock.closed)


class TwoButtonsTests(unittest.TestCase):
    def test_second_button_does_not_restart_or_cut_the_recording(self):
        class Recorder:
            recording = False
            begins = 0

            def begin(self):
                self.recording = True
                self.begins += 1

            def end(self):
                self.recording = False
                return None

        recorder = Recorder()
        radio = ptt.PushToTalk(object(), recorder, lambda audio: None)
        self.addCleanup(radio.shutdown)
        radio.on_start()
        radio.on_start()
        self.assertEqual(recorder.begins, 1)
        radio.on_stop()
        radio.on_stop()                        # the other button's release: nothing to do
        self.assertFalse(recorder.recording)


# ──────────────────────────────────────────────
# 2. Offline Whisper speech to text
# ──────────────────────────────────────────────

class FakeAudio:
    def __init__(self, samples=1600):
        self.samples = samples

    def get_raw_data(self, convert_rate=None, convert_width=None):
        assert (convert_rate, convert_width) == (16000, 2)
        return struct.pack(f'<{self.samples}h', *([1000] * self.samples))


class FakeWhisperModel:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append((samples, options))
        return self.result


class WhisperTests(unittest.TestCase):
    def test_transcribes_offline_with_the_radio_prompt(self):
        model = FakeWhisperModel({'text': ' How are the rears? ',
                                  'segments': [{'no_speech_prob': 0.01, 'avg_logprob': -0.2}]})
        loads = []
        whisper = stt.WhisperTranscriber('base.en', loader=lambda name: loads.append(name) or model)
        self.assertEqual(whisper(FakeAudio()), 'How are the rears?')
        self.assertEqual(whisper(FakeAudio()), 'How are the rears?')
        self.assertEqual(loads, ['base.en'])            # the model loads once
        samples, options = model.calls[0]
        self.assertEqual(samples.dtype.name, 'float32')
        self.assertAlmostEqual(float(samples[0]), 1000 / 32768, places=5)
        self.assertIn('undercut', options['initial_prompt'])
        self.assertFalse(options['fp16'])
        self.assertNotIn('language', options)           # .en models are English only

    def test_multilingual_model_is_told_the_language(self):
        model = FakeWhisperModel({'text': 'gap ahead'})
        whisper = stt.WhisperTranscriber('base', 'en-GB', loader=lambda name: model)
        whisper(FakeAudio())
        self.assertEqual(model.calls[0][1]['language'], 'en')

    def test_silence_and_hallucinations_are_dropped(self):
        noise = {'text': 'Thank you.', 'segments': [{'no_speech_prob': 0.1, 'avg_logprob': -0.3}]}
        self.assertIsNone(stt.WhisperTranscriber(loader=lambda n: FakeWhisperModel(noise))(FakeAudio()))
        quiet = {'text': 'box', 'segments': [{'no_speech_prob': 0.9, 'avg_logprob': -1.4}]}
        self.assertIsNone(stt.WhisperTranscriber(loader=lambda n: FakeWhisperModel(quiet))(FakeAudio()))
        self.assertIsNone(stt.WhisperTranscriber(loader=lambda n: FakeWhisperModel({'text': ''}))(
            FakeAudio()))
        self.assertIsNone(stt.WhisperTranscriber(loader=lambda n: FakeWhisperModel({'text': 'x'}))(
            FakeAudio(samples=0)))

    def test_falls_back_to_google_when_whisper_is_not_installed(self):
        recognizer = SimpleNamespace(recognize_google=lambda audio, language: 'gap behind')
        with patch('builtins.print'):
            transcriber = stt.build_transcriber('whisper', available=lambda: False,
                                                recognizer_factory=lambda: recognizer)
        self.assertEqual(transcriber(FakeAudio()), 'gap behind')

    def test_falls_back_when_the_model_cannot_load(self):
        recognizer = SimpleNamespace(recognize_google=lambda audio, language: 'pit window')

        def broken(name):
            raise RuntimeError('no torch')
        with patch('builtins.print'):
            transcriber = stt.build_transcriber('whisper', loader=broken,
                                                recognizer_factory=lambda: recognizer)
            self.assertEqual(transcriber(FakeAudio()), 'pit window')
            self.assertEqual(transcriber(FakeAudio()), 'pit window')

    def test_preload_loads_in_the_background(self):
        loaded = threading.Event()
        model = FakeWhisperModel({'text': 'hi'})
        whisper = stt.WhisperTranscriber(loader=lambda name: loaded.set() or model)
        stt.FallbackTranscriber(whisper, lambda audio: None, 'Google').preload()
        self.assertTrue(loaded.wait(2))

    def test_google_and_vosk_engines_are_unchanged(self):
        recognizer = SimpleNamespace(recognize_vosk=lambda audio: '{"text": "weather"}')
        transcriber = stt.build_transcriber('vosk', recognizer_factory=lambda: recognizer)
        self.assertEqual(transcriber(FakeAudio()), 'weather')


# ──────────────────────────────────────────────
# 3. Intent matching
# ──────────────────────────────────────────────

class IntentMatchingTests(unittest.TestCase):
    def test_natural_phrasing(self):
        cases = {
            'how are my tyres holding up?': 'tyre_temps',
            'are the rears getting hot': 'tyre_temps',
            'what are the tyres like': 'tyre_temps',
            'whos that behind me': 'gap_behind',
            'how far ahead is the leader': 'gap_ahead',
            'what lap do I box': 'pit_window',
            'should I stay out': 'pit_window',
            'should we pit before the car ahead': 'pit_window',
            'is Verstappen coming in': 'rivals',
            'am I under pressure': 'threats',
            'is the car behind a threat': 'threats',
            'can we undercut them': 'strategy',
            'how is my pace looking': 'pace',
            'will it stop raining': 'weather',
        }
        for phrase, expected in cases.items():
            self.assertEqual(match_command(phrase), expected, phrase)

    def test_misheard_words_still_match(self):
        for phrase, expected in (('how is the temprature on the fronts', 'tyre_temps'),
                                 ('whats the forcast', 'weather'),
                                 ('whats the strategie', 'strategy')):
            self.assertEqual(match_command(phrase), expected, phrase)

    def test_non_commands_are_not_matched(self):
        for phrase in ('', 'hello there', 'engineer', 'play some music', 'how much fuel do we have',
                       'hey engineer'):
            self.assertIsNone(match_command(phrase), phrase)

    def test_match_details_explain_the_choice(self):
        result = intents.match_intent('Engineer, when will they box?')
        self.assertEqual(result.intent, 'rivals')
        self.assertIn(intents.RIVAL, result.concepts)
        self.assertGreater(result.scores['rivals'], result.scores['pit_window'])


# ──────────────────────────────────────────────
# 4. Calls-per-lap budget
# ──────────────────────────────────────────────

class RecordingVoice:
    is_running = True

    def __init__(self):
        self.calls = []

    def say(self, message, priority=Priority.NORMAL, **options):
        self.calls.append(message)
        return True


class CallBudgetTests(unittest.TestCase):
    def test_budget_rules(self):
        budget = radio_discipline.CallBudget(2)
        budget.new_lap(3)
        for _ in range(2):
            self.assertTrue(budget.allows('traffic_gaps', Priority.RACE))
            budget.spend('traffic_gaps', Priority.RACE)
        self.assertFalse(budget.allows('sector_1', Priority.TIMING))
        self.assertTrue(budget.allows('safety_car', Priority.CRITICAL))
        self.assertTrue(budget.allows('pit_window', Priority.STRATEGY))
        self.assertTrue(budget.allows('drs_behind', Priority.RACE))
        budget.new_lap(3)                      # same lap: nothing changes
        self.assertEqual(budget.remaining, 0)
        budget.new_lap(4)
        self.assertEqual(budget.remaining, 2)
        unlimited = radio_discipline.CallBudget(0)
        self.assertTrue(all(unlimited.allows('x', Priority.LOW) for _ in range(50)))

    def test_engine_drops_proactive_calls_over_budget_but_never_urgent_ones(self):
        voice = RecordingVoice()
        engine = AnalysisEngine(voice, on_demand=False, calls_per_lap=2)
        engine.call_budget.new_lap(5)
        for n in range(4):
            engine.speak(f'Gap {n}.', Priority.RACE, key='traffic_gaps', ttl=5)
        engine.speak('Safety Car, Safety Car.', Priority.CRITICAL, key='safety_car', ttl=5)
        engine.speak('Box this lap.', Priority.STRATEGY, key='pit_window', ttl=5)
        self.assertEqual(voice.calls, ['Gap 0.', 'Gap 1.', 'Safety Car, Safety Car.', 'Box this lap.'])
        engine.call_budget.new_lap(6)
        engine.speak('Gap 9.', Priority.RACE, key='traffic_gaps', ttl=5)
        self.assertEqual(voice.calls[-1], 'Gap 9.')

    def test_calls_silenced_by_quiet_mode_do_not_use_the_budget(self):
        voice = RecordingVoice()
        engine = AnalysisEngine(voice, on_demand=True, calls_per_lap=1)
        engine.speak('Sector one.', Priority.TIMING, key='sector_1')
        self.assertEqual(engine.call_budget.used, 0)


# ──────────────────────────────────────────────
# 5. No talking in high-load corners
# ──────────────────────────────────────────────

class DriverLoadGateTests(unittest.TestCase):
    def test_high_load_corner_holds_until_the_straight(self):
        gate = radio_discipline.DriverLoadGate()
        gate.update(300, 0.2, 0.0, now=0.0)
        self.assertFalse(gate.busy(now=0.0))
        gate.update(150, 3.8, 0.0, now=1.0)          # slow corner: talking is fine
        self.assertFalse(gate.busy(now=1.0))
        gate.update(250, 4.1, 0.0, now=2.0)          # Copse: busy
        self.assertTrue(gate.busy(now=2.0))
        gate.update(240, 2.5, 0.0, now=2.3)          # still turning
        self.assertTrue(gate.busy(now=2.3))
        gate.update(260, 0.8, 0.0, now=2.5)          # straightening up...
        self.assertTrue(gate.busy(now=2.5))
        gate.update(270, 0.5, 0.0, now=3.0)          # ...on the straight long enough
        self.assertFalse(gate.busy(now=3.0))

    def test_s_bend_counts_as_one_corner(self):
        gate = radio_discipline.DriverLoadGate()
        gate.update(260, 4.0, now=0.0)
        gate.update(250, 1.0, now=0.2)               # flick between Maggots and Becketts
        gate.update(240, -3.9, now=0.4)
        gate.update(240, 1.0, now=0.7)
        self.assertTrue(gate.busy(now=0.7))

    def test_heavy_braking_and_stale_data(self):
        gate = radio_discipline.DriverLoadGate()
        gate.update(310, 0.3, 0.9, now=0.0)
        self.assertTrue(gate.busy(now=0.0))
        self.assertFalse(gate.busy(now=5.0))         # no telemetry for a while: don't hold
        gate.update(200, 0.0, 0.2, now=6.0)
        self.assertFalse(gate.busy(now=6.0))

    def test_thresholds_are_configurable(self):
        gate = radio_discipline.DriverLoadGate(radio_discipline.LoadThresholds(
            corner_speed_kph=120, corner_g=2.5))
        gate.update(150, 3.0, now=0.0)
        self.assertTrue(gate.busy(now=0.0))


class FakeSpeech:
    def __init__(self):
        self.started = []
        self.callbacks = {}
        self.current = None

    def setProperty(self, *args):
        pass

    def getProperty(self, name):
        return []

    def connect(self, event, callback):
        self.callbacks[event] = callback

    def startLoop(self, use_driver):
        pass

    def say(self, message, name=None):
        self.current = message
        self.started.append(message)
        self.callbacks['started-utterance'](name)

    def iterate(self):
        self.current = None

    def isBusy(self):
        return self.current is not None

    def stop(self):
        self.current = None

    def endLoop(self):
        pass


class CornerHoldVoiceTests(unittest.TestCase):
    def test_calls_wait_for_the_straight_but_critical_calls_do_not(self):
        driver = FakeSpeech()
        with patch.object(voice_engine, 'pyttsx3', SimpleNamespace(init=lambda: driver)):
            voice = VoiceEngine()
            voice.start()
        self.addCleanup(voice.shutdown)
        busy = [True]
        voice.set_silence_filter(lambda: busy[0])
        voice.say('Gap ahead one point two.', Priority.RACE, key='gap', ttl=10)
        voice.say('Safety Car, Safety Car.', Priority.CRITICAL, key='sc', ttl=10)
        self.assertTrue(wait_for(lambda: 'Safety Car, Safety Car.' in driver.started))
        time.sleep(0.2)
        self.assertNotIn('Gap ahead one point two.', driver.started)   # held mid-corner
        busy[0] = False
        self.assertTrue(wait_for(lambda: 'Gap ahead one point two.' in driver.started))


if __name__ == '__main__':
    unittest.main()
