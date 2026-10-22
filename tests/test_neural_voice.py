import threading
import time
import unittest

import numpy as np

from neural_voice import NeuralSpeechDriver
from radio_fx import radio_filter
from test_analysis import SilentVoice  # noqa: F401  (puts ai_engineer/ on the import path)
from voice_engine import Priority, VoiceEngine


class RadioFilterTests(unittest.TestCase):
    def _energy_at(self, signal, sr, freq):
        spectrum = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(signal.size, 1 / sr)
        return spectrum[np.argmin(np.abs(freqs - freq))]

    def test_cuts_low_and_high_frequencies_keeps_voice_band(self):
        sr = 24000
        t = np.arange(sr) / sr
        clean = (np.sin(2 * np.pi * 80 * t) + np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 8000 * t)) / 3
        out = radio_filter(clean, sr, beeps=False, noise_level=0, drive=0.01)
        self.assertGreater(self._energy_at(out, sr, 1000), 20 * self._energy_at(out, sr, 80))
        self.assertGreater(self._energy_at(out, sr, 1000), 20 * self._energy_at(out, sr, 8000))

    def test_output_is_bounded_float32_with_beeps_added(self):
        sr = 24000
        out = radio_filter(np.random.default_rng(0).normal(0, 3, sr), sr, seed=1)
        self.assertEqual(out.dtype, np.float32)
        self.assertLessEqual(float(np.max(np.abs(out))), 1.0)
        self.assertGreater(out.size, sr)
        self.assertEqual(radio_filter([], sr).size, 0)


class FakePlayer:
    def __init__(self):
        self.played = []
        self.stops = 0
        self._stop = threading.Event()
        self.hold = False

    def play(self, samples, sample_rate):
        self.played.append((len(samples), sample_rate))
        self._stop.clear()
        if self.hold:
            self._stop.wait(timeout=2)

    def stop(self):
        self.stops += 1
        self._stop.set()


def fake_synth(calls):
    def synthesize(text):
        calls.append(text)
        return np.zeros(2400, dtype=np.float32), 24000

    return synthesize


def wait_for(condition, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


class NeuralDriverTests(unittest.TestCase):
    def test_voice_engine_speaks_through_driver_and_caches_repeats(self):
        calls, player, spoken = [], FakePlayer(), []
        engine = VoiceEngine(
            on_spoken=spoken.append, driver_factory=lambda: NeuralSpeechDriver(fake_synth(calls), player)
        )
        engine.start()
        try:
            self.assertTrue(engine.is_running)
            engine.say('We are checking.', Priority.CRITICAL, key='a', cooldown=0)
            self.assertTrue(wait_for(lambda: len(player.played) == 1 and not engine.is_busy))
            engine.say('We are checking.', Priority.CRITICAL, key='b', cooldown=0)
            self.assertTrue(wait_for(lambda: len(player.played) == 2))
            self.assertEqual(calls, ['We are checking.'])  # second time came from cache
            self.assertEqual(spoken, ['We are checking.'] * 2)
        finally:
            engine.shutdown()

    def test_urgent_call_interrupts_playback(self):
        calls, player = [], FakePlayer()
        player.hold = True
        engine = VoiceEngine(driver_factory=lambda: NeuralSpeechDriver(fake_synth(calls), player))
        engine.start()
        try:
            engine.say('Gap ahead two point four.', Priority.LOW, key='gap')
            self.assertTrue(wait_for(lambda: len(player.played) == 1))
            engine.say('Box, box.', Priority.CRITICAL, key='box', cooldown=0)
            self.assertTrue(wait_for(lambda: len(player.played) == 2))
            self.assertGreaterEqual(player.stops, 1)
        finally:
            player.hold = False
            engine.shutdown()

    def test_synthesis_failure_does_not_kill_the_radio(self):
        player = FakePlayer()

        def broken(text):
            raise RuntimeError('model crashed')

        engine = VoiceEngine(driver_factory=lambda: NeuralSpeechDriver(broken, player))
        engine.start()
        try:
            engine.say('Hello.', Priority.CRITICAL, key='x', cooldown=0)
            self.assertTrue(wait_for(lambda: not engine.is_busy))
            self.assertTrue(engine.is_running)
        finally:
            engine.shutdown()


if __name__ == '__main__':
    unittest.main()


class KokoroDownloadTests(unittest.TestCase):
    def test_downloads_missing_files_once(self):
        import io
        import tempfile
        from pathlib import Path

        import neural_voice

        fetched = []

        class Response(io.BytesIO):
            headers = {'Content-Length': '6'}

        def opener(url, timeout):
            fetched.append(url.rsplit('/', 1)[-1])
            return Response(b'abcdef')

        with tempfile.TemporaryDirectory() as tmp:
            self.assertTrue(neural_voice.download_model_files(tmp, lambda _: None, opener=opener))
            self.assertEqual(sorted(fetched), ['kokoro-v1.0.onnx', 'voices-v1.0.bin'])
            self.assertEqual((Path(tmp) / 'voices-v1.0.bin').read_bytes(), b'abcdef')
            self.assertTrue(neural_voice.download_model_files(tmp, lambda _: None, opener=opener))
            self.assertEqual(len(fetched), 2)  # already there, nothing fetched again

    def test_failed_download_leaves_no_partial_file(self):
        import tempfile
        import urllib.error
        from pathlib import Path

        import neural_voice

        def opener(url, timeout):
            raise urllib.error.URLError('offline')

        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(neural_voice.download_model_files(tmp, lambda _: None, opener=opener))
            self.assertEqual(list(Path(tmp).iterdir()), [])
