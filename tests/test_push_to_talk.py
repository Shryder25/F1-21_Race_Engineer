"""Push-to-talk radio button: button handling, recording, free-form questions."""

import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import push_to_talk as ptt
from engineer_brain import UNKNOWN_QUESTION_REPLY, EngineerBrain
from voice_commands import ACK_TEXT, QUESTION, CommandDispatcher, VoiceCommandListener
from voice_engine import Priority


def wait_for(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class FakeJoystick:
    def __init__(self, name='Thrustmaster T300RS', buttons=12, hats=1):
        self.name = name
        self.buttons = [False] * buttons
        self.hats = [(0, 0)] * hats

    def get_name(self):
        return self.name

    def get_numbuttons(self):
        return len(self.buttons)

    def get_numhats(self):
        return len(self.hats)

    def get_button(self, i):
        return self.buttons[i]

    def get_hat(self, i):
        return self.hats[i]


class FakeBackend:
    def __init__(self, joysticks):
        self.joysticks = joysticks
        self.changed = False
        self.closed = False

    def poll(self):
        changed, self.changed = self.changed, False
        return changed

    def devices(self):
        return list(self.joysticks)

    def close(self):
        self.closed = True


class RecordingVoice:
    is_running = True

    def __init__(self):
        self.said = []
        self.lock = threading.Lock()

    def say(self, message, priority, **kwargs):
        with self.lock:
            self.said.append((message, priority, kwargs))
        return True


class BindingTests(unittest.TestCase):
    def test_parse_buttons_and_hats(self):
        self.assertEqual(ptt.parse_binding('5'), ptt.Binding('button', 5))
        self.assertEqual(ptt.parse_binding(' HAT0:Up '), ptt.Binding('hat', 0, 'up'))
        self.assertEqual(ptt.parse_binding('hat1:left').label, 'hat1:left')
        for bad in ('', 'x', 'hat0:sideways', 'hat:up', '-1'):
            with self.assertRaises(ValueError):
                ptt.parse_binding(bad)

    def test_pressed_reads_the_right_input(self):
        joy = FakeJoystick()
        button, up = ptt.Binding('button', 3), ptt.Binding('hat', 0, 'up')
        self.assertFalse(button.is_pressed(joy))
        joy.buttons[3] = True
        self.assertTrue(button.is_pressed(joy))
        joy.hats[0] = (1, 1)            # diagonal up-right still counts as up
        self.assertTrue(up.is_pressed(joy))
        joy.hats[0] = (0, -1)
        self.assertFalse(up.is_pressed(joy))
        self.assertFalse(ptt.Binding('button', 40).is_pressed(joy))   # out of range is never pressed
        self.assertFalse(ptt.Binding('hat', 3, 'up').exists_on(joy))


class WatcherStateTests(unittest.TestCase):
    def make(self, mode=ptt.HOLD, **kwargs):
        events = []
        watcher = ptt.ButtonWatcher(ptt.Binding('button', 0), lambda: events.append('start'),
                                    lambda: events.append('stop'), mode=mode, **kwargs)
        return watcher, events

    def test_hold_talks_while_held(self):
        watcher, events = self.make()
        watcher.feed(False, 0.0)
        watcher.feed(True, 1.0)
        watcher.feed(True, 1.5)
        self.assertTrue(watcher.is_open)
        watcher.feed(False, 2.0)
        self.assertEqual(events, ['start', 'stop'])

    def test_toggle_opens_and_closes_on_presses(self):
        watcher, events = self.make(ptt.TOGGLE)
        watcher.feed(True, 1.0)
        watcher.feed(False, 1.2)
        self.assertTrue(watcher.is_open)
        watcher.feed(True, 3.0)
        watcher.feed(False, 3.2)
        self.assertEqual(events, ['start', 'stop'])
        self.assertFalse(watcher.is_open)

    def test_contact_bounce_is_ignored(self):
        watcher, events = self.make(debounce_s=0.05)
        watcher.feed(True, 1.0)
        watcher.feed(False, 1.01)       # bounce
        watcher.feed(True, 1.02)
        self.assertEqual(events, ['start'])
        watcher.feed(False, 2.0)
        self.assertEqual(events, ['start', 'stop'])

    def test_radio_left_open_closes_itself_and_needs_a_release(self):
        watcher, events = self.make(max_open_s=10)
        watcher.feed(True, 0.0)
        watcher.feed(True, 10.5)        # held too long
        self.assertEqual(events, ['start', 'stop'])
        watcher.feed(True, 11.0)
        self.assertEqual(events, ['start', 'stop'])   # still held: does not reopen
        watcher.feed(False, 12.0)
        watcher.feed(True, 13.0)
        self.assertEqual(events, ['start', 'stop', 'start'])

    def test_callback_errors_do_not_break_the_watcher(self):
        watcher = ptt.ButtonWatcher(ptt.Binding('button', 0), lambda: 1 / 0, lambda: None)
        watcher.feed(True, 1.0)
        self.assertTrue(watcher.is_open)

    def test_bad_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.make('sometimes')


class WatcherThreadTests(unittest.TestCase):
    def test_polls_the_named_device_and_handles_unplugging(self):
        pad, wheel = FakeJoystick('Xbox Controller'), FakeJoystick('Logitech G29')
        backend = FakeBackend([pad, wheel])
        events = []
        watcher = ptt.ButtonWatcher(ptt.Binding('button', 4), lambda: events.append('start'),
                                    lambda: events.append('stop'), device='g29',
                                    backend_factory=lambda: backend, poll_hz=500, rescan_s=0.01)
        self.assertTrue(watcher.start())
        self.addCleanup(watcher.stop)
        self.assertTrue(wait_for(lambda: watcher.device_name == 'Logitech G29'))
        pad.buttons[4] = True            # other controller: ignored
        time.sleep(0.05)
        self.assertEqual(events, [])
        wheel.buttons[4] = True
        self.assertTrue(wait_for(lambda: events == ['start']))
        backend.joysticks = []           # unplugged while talking: radio closes
        backend.changed = True
        self.assertTrue(wait_for(lambda: events == ['start', 'stop']))
        watcher.stop()
        self.assertTrue(backend.closed)

    def test_missing_pygame_reports_false(self):
        def broken():
            raise ImportError('No module named pygame')
        watcher = ptt.ButtonWatcher(ptt.Binding('button', 0), lambda: None, lambda: None,
                                    backend_factory=broken)
        self.assertFalse(watcher.start())


class FakeAudio:
    def __init__(self, raw, rate, width):
        self.raw, self.rate, self.width = raw, rate, width


class RecorderTests(unittest.TestCase):
    def make(self):
        recorder = ptt.MicRecorder(audio_factory=FakeAudio, min_s=0.3)
        recorder.sample_rate, recorder.sample_width, recorder.chunk = 1000, 2, 100
        recorder._preroll = ptt.deque(maxlen=2)
        return recorder

    def test_records_between_begin_and_end_with_preroll(self):
        recorder = self.make()
        for i in range(5):
            recorder.feed(bytes([i]) * 200)          # only the last two chunks are kept
        recorder.begin()
        self.assertTrue(recorder.recording)
        recorder.feed(b'\x09' * 400)
        audio = recorder.end()
        self.assertFalse(recorder.recording)
        self.assertEqual(audio.raw, b'\x03' * 200 + b'\x04' * 200 + b'\x09' * 400)
        self.assertEqual((audio.rate, audio.width), (1000, 2))

    def test_a_quick_tap_is_ignored(self):
        recorder = self.make()
        recorder.begin()
        recorder.feed(b'\x00' * 100)                 # 0.05 s
        self.assertIsNone(recorder.end())
        self.assertIsNone(recorder.end())            # end without begin is safe

    def test_no_microphone_returns_false(self):
        try:
            import speech_recognition  # noqa: F401
        except ImportError:
            self.skipTest('SpeechRecognition not installed')

        def broken(device_index=None):
            raise OSError('no microphone')
        self.assertFalse(ptt.MicRecorder(microphone_factory=broken).start())


class FakeRecorder:
    def __init__(self, audio='clip'):
        self.audio = audio
        self.recording = False

    def begin(self):
        self.recording = True

    def end(self):
        self.recording = False
        return self.audio


class PushToTalkTests(unittest.TestCase):
    def test_release_transcribes_and_sends_a_free_form_question(self):
        sent = []

        class Dispatcher:
            def handle_text(self, text, *, free_form=False):
                sent.append((text, free_form))

        logs = []
        radio = ptt.PushToTalk(Dispatcher(), FakeRecorder(), lambda audio: 'how are the rears',
                               on_log=lambda m, status: logs.append((m, status)))
        self.addCleanup(radio.shutdown)
        radio.on_start()
        self.assertTrue(radio.recently_active())
        radio.on_stop()
        self.assertTrue(wait_for(lambda: sent))
        self.assertEqual(sent, [('how are the rears', True)])
        self.assertEqual(logs, [('Driver: how are the rears', 'heard')])

    def test_unreadable_radio_asks_to_say_again(self):
        voice = RecordingVoice()
        radio = ptt.PushToTalk(object(), FakeRecorder(), lambda audio: None, voice=voice)
        self.addCleanup(radio.shutdown)
        radio.on_start()
        radio.on_stop()
        self.assertTrue(wait_for(lambda: voice.said))
        self.assertEqual(voice.said[0][0], ptt.SAY_AGAIN_TEXT)

    def test_too_short_press_does_nothing(self):
        called = []
        radio = ptt.PushToTalk(object(), FakeRecorder(audio=None), called.append)
        self.addCleanup(radio.shutdown)
        radio.on_start()
        radio.on_stop()
        time.sleep(0.05)
        self.assertEqual(called, [])

    def test_hands_free_listener_stands_down_while_the_button_is_used(self):
        heard = []

        class Dispatcher:
            def handle_text(self, text):
                heard.append(text)

        busy = [True]
        listener = VoiceCommandListener(Dispatcher(), ignore_when=lambda: busy[0])
        listener.transcribe = lambda audio: audio
        listener._on_audio(None, 'pit window')
        busy[0] = False
        listener._on_audio(None, 'weather report')
        self.assertEqual(heard, ['weather report'])


class FakeClient:
    def __init__(self, reply=None):
        self.reply = reply
        self.prompts = []

    def chat_json(self, system, user):
        self.prompts.append(user)
        if self.reply is None:
            from ai_client import AIUnavailable
            raise AIUnavailable('offline')
        return {'voice': self.reply, 'advice': ''}

    def chat(self, system, user):
        from ai_client import AIUnavailable
        raise AIUnavailable('offline')


RACE_CTX = {'session_mode': 'race', 'gap_ahead_s': 1.24, 'car_ahead_name': 'Hamilton'}


class FreeFormQuestionTests(unittest.TestCase):
    def test_the_ai_gets_the_drivers_own_words(self):
        client = FakeClient('Rears are fine, ninety-two. Keep pushing.')
        brain = EngineerBrain(lambda: RACE_CTX, client)
        answer = brain.answer_question('Can I keep pushing on these "rears"?')
        self.assertEqual(answer, 'Rears are fine, ninety-two. Keep pushing.')
        self.assertIn("Driver asks: \"Can I keep pushing on these 'rears'?\"", client.prompts[0])
        self.assertIn('Hamilton', client.prompts[0])
        self.assertEqual(brain.last_source, 'ai')

    def test_offline_falls_back_to_rules_when_the_question_matches_a_command(self):
        brain = EngineerBrain(lambda: RACE_CTX, FakeClient(None))
        self.assertEqual(brain.answer_question("what's the gap to the guy ahead", 'gap_ahead'),
                         'Gap to Hamilton ahead is 1.2.')
        self.assertEqual(brain.answer_question('is my front wing damaged'), UNKNOWN_QUESTION_REPLY)
        self.assertEqual(brain.last_source, 'fallback')

    def test_session_mode_still_answers_instantly(self):
        client = FakeClient('should not be used')
        brain = EngineerBrain(lambda: {'session_mode': 'time_trial'}, client)
        self.assertEqual(brain.answer_question('who is behind me', 'gap_behind'),
                         'No other cars in time trial.')
        self.assertEqual(client.prompts, [])


class SlowBrain:
    def __init__(self):
        self.calls = []

    def answer(self, command):
        self.calls.append(('command', command))
        return f'Answer for {command}.'

    def answer_question(self, question, command=None):
        self.calls.append(('question', question, command))
        return 'Free answer.'


class DispatcherFreeFormTests(unittest.TestCase):
    def test_free_form_questions_are_acknowledged_and_answered(self):
        voice, brain = RecordingVoice(), SlowBrain()
        dispatcher = CommandDispatcher(voice, brain)
        self.addCleanup(dispatcher.shutdown)
        self.assertEqual(dispatcher.handle_text('is it going to rain soon', free_form=True), 'weather')
        self.assertEqual(dispatcher.handle_text('box this lap or stay out', free_form=True), 'pit_window')
        self.assertEqual(dispatcher.handle_text('how much fuel have I got', free_form=True), QUESTION)
        self.assertTrue(wait_for(lambda: len(voice.said) == 6))
        self.assertEqual(voice.said[0][0], ACK_TEXT)
        self.assertEqual(voice.said[0][1], Priority.CRITICAL)
        self.assertIn(('question', 'how much fuel have I got', None), brain.calls)
        self.assertIn(('question', 'is it going to rain soon', 'weather'), brain.calls)
        self.assertFalse(any(call[0] == 'command' for call in brain.calls))

    def test_hands_free_mic_still_ignores_unknown_speech(self):
        voice, brain = RecordingVoice(), SlowBrain()
        dispatcher = CommandDispatcher(voice, brain)
        self.addCleanup(dispatcher.shutdown)
        self.assertIsNone(dispatcher.handle_text('how much fuel have I got'))
        self.assertIsNone(dispatcher.handle_text('   ', free_form=True))
        self.assertEqual(voice.said, [])

    def test_repeated_question_is_debounced(self):
        voice, brain = RecordingVoice(), SlowBrain()
        dispatcher = CommandDispatcher(voice, brain, debounce_s=5)
        self.addCleanup(dispatcher.shutdown)
        dispatcher.handle_text('how much fuel', free_form=True)
        dispatcher.handle_text('How much fuel?', free_form=True)
        self.assertTrue(wait_for(lambda: len(voice.said) == 2))
        time.sleep(0.05)
        self.assertEqual(len(brain.calls), 1)


class FinderAndSettingsTests(unittest.TestCase):
    def test_finder_reports_a_new_press_and_ignores_buttons_already_held(self):
        wheel = FakeJoystick('Fanatec CSL DD')
        wheel.buttons[0] = True                     # e.g. a shifter that is always "on"
        backend = FakeBackend([FakeJoystick('Xbox Controller'), wheel])
        ticks = iter(range(1000))
        clock = lambda: next(ticks) * 0.01          # noqa: E731

        def sleep(_s):
            if clock() > 0.5:
                wheel.buttons[7] = True
        lines = []
        found = ptt.find_button(backend, out=lines.append, clock=clock, sleep=sleep)
        self.assertEqual(found, (1, 'Fanatec CSL DD', ptt.Binding('button', 7)))
        self.assertIn('Fanatec CSL DD', '\n'.join(lines))

    def test_finder_detects_dpad_and_handles_no_controller(self):
        pad = FakeJoystick('Xbox Controller')
        backend = FakeBackend([pad])

        calls = []

        def sleep(_s):                              # the first sleep is before the baseline
            calls.append(_s)
            if len(calls) > 1:
                pad.hats[0] = (-1, 0)
        found = ptt.find_button(backend, out=lambda *_: None, sleep=sleep)
        self.assertEqual(found[2], ptt.Binding('hat', 0, 'left'))
        lines = []
        self.assertIsNone(ptt.find_button(FakeBackend([]), out=lines.append))
        self.assertIn('No wheel or controller', lines[0])

    def test_finder_times_out(self):
        ticks = iter(range(10000))
        self.assertIsNone(ptt.find_button(FakeBackend([FakeJoystick()]), timeout_s=1,
                                          out=lambda *_: None, clock=lambda: next(ticks) * 0.1,
                                          sleep=lambda _s: None))

    def test_saved_choice_is_used_and_command_line_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'push_to_talk.json'
            self.assertIsNone(ptt.resolve_settings(config_path=path))
            ptt.save_config(ptt.Binding('hat', 0, 'up'), 'Logitech G29', ptt.TOGGLE, path)
            self.assertEqual(json.loads(path.read_text())['button'], 'hat0:up')
            saved = ptt.resolve_settings(config_path=path)
            self.assertEqual((saved.binding.label, saved.device, saved.mode),
                             ('hat0:up', 'Logitech G29', ptt.TOGGLE))
            override = ptt.resolve_settings('3', None, ptt.HOLD, config_path=path)
            self.assertEqual((override.binding.label, override.device, override.mode),
                             ('3', None, ptt.HOLD))
            path.write_text('{"button": "banana"}')
            self.assertIsNone(ptt.resolve_settings(config_path=path))
            with self.assertRaises(ValueError):
                ptt.resolve_settings('banana', config_path=path)

    def test_run_finder_saves_the_choice(self):
        wheel = FakeJoystick('Thrustmaster T300RS')
        wheel.buttons[2] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / 'ptt.json'

            class Backend(FakeBackend):
                calls = 0

                def devices(self):
                    return self.joysticks

            backend = Backend([wheel])
            original = ptt._snapshot
            first = [True]

            def snapshot(joystick):                 # baseline sees nothing pressed
                if first[0]:
                    first[0] = False
                    return (False,) * len(joystick.buttons), tuple(joystick.hats)
                return original(joystick)
            ptt._snapshot = snapshot
            self.addCleanup(setattr, ptt, '_snapshot', original)
            self.assertEqual(ptt.run_button_finder(ptt.HOLD, path, lambda: backend), 0)
            self.assertEqual(json.loads(path.read_text()),
                             {'button': '2', 'device': 'Thrustmaster T300RS', 'mode': 'hold'})
            self.assertTrue(backend.closed)


class StartTests(unittest.TestCase):
    def test_start_wires_button_to_recorder_and_stops_cleanly(self):
        wheel = FakeJoystick()
        backend = FakeBackend([wheel])

        class Recorder(FakeRecorder):
            started = stopped = False

            def start(self):
                self.started = True
                return True

            def stop(self):
                self.stopped = True

        recorder = Recorder()
        settings = ptt.Settings(ptt.Binding('button', 1))
        radio = ptt.start(settings, object(), backend_factory=lambda: backend,
                          recorder=recorder, recognizer=object())
        self.assertIsNotNone(radio)
        wheel.buttons[1] = True
        self.assertTrue(wait_for(lambda: recorder.recording))
        self.assertTrue(radio.recently_active())
        radio.stop()
        self.assertTrue(recorder.stopped)

    def test_start_gives_up_when_the_mic_is_missing(self):
        class NoMic:
            def start(self):
                return False
        self.assertIsNone(ptt.start(ptt.Settings(ptt.Binding('button', 1)), object(),
                                    recorder=NoMic(), recognizer=object()))


if __name__ == '__main__':
    unittest.main()
