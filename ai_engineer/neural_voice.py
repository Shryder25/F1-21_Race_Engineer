"""Neural text-to-speech for the race radio (Kokoro), with pyttsx3 as the fallback.

``NeuralSpeechDriver`` has the same small interface as a pyttsx3 engine, so
``VoiceEngine`` keeps all its queueing, priority and interrupt logic unchanged and
just gets a better-sounding voice. Speech is synthesized and played on a helper
thread so the radio queue never blocks; ``stop()`` cuts playback mid-sentence when a
more urgent call arrives.

Kokoro setup (free, offline, Apache 2.0): ``pip install kokoro-onnx sounddevice``.
That's all: on the first start the engineer downloads the two model files (~350 MB) into
./models/kokoro and uses Kokoro from then on. Without the packages it uses the system voice.
"""

from __future__ import annotations

import os
import threading
import urllib.error
import urllib.request
from collections import OrderedDict
from pathlib import Path

import numpy as np

from log_setup import get_logger
from radio_fx import radio_filter

log = get_logger('voice')

DEFAULT_KOKORO_DIR = Path(__file__).resolve().parent.parent / 'models' / 'kokoro'
DEFAULT_KOKORO_VOICE = 'bm_george'  # British male; bm_lewis / bm_daniel / bm_fable also work
KOKORO_MODEL_FILE = 'kokoro-v1.0.onnx'
KOKORO_VOICES_FILE = 'voices-v1.0.bin'
KOKORO_DOWNLOAD_URL = 'https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/'
SETUP_HINT = (
    'For the natural engineer voice run: pip install kokoro-onnx sounddevice '
    '(model files download automatically on the next start).'
)


def missing_packages() -> list[str]:
    """pip packages Kokoro needs that aren't installed."""
    missing = []
    for module, package in (('kokoro_onnx', 'kokoro-onnx'), ('sounddevice', 'sounddevice')):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
        except Exception as error:  # installed but can't load, e.g. no PortAudio audio library
            missing.append(f'{package} (installed but failed to load: {error})')
    return missing


def download_model_files(
    model_dir: Path = DEFAULT_KOKORO_DIR,
    on_status=print,
    base_url: str = KOKORO_DOWNLOAD_URL,
    opener=urllib.request.urlopen,
) -> bool:
    """Fetch any missing Kokoro model file. Partial downloads never count as present."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in (KOKORO_MODEL_FILE, KOKORO_VOICES_FILE):
        target = model_dir / name
        if target.exists():
            continue
        partial = target.with_suffix(target.suffix + '.part')
        on_status(f'[Voice] Downloading {name} for the Kokoro voice (first run only)...')
        try:
            with opener(base_url + name, timeout=60) as response, open(partial, 'wb') as out:
                total = int(response.headers.get('Content-Length') or 0)
                done, shown = 0, -25
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if total and done * 100 // total >= shown + 25:
                        shown = done * 100 // total
                        on_status(f'[Voice] {name}: {shown}%')
            partial.replace(target)
        except (urllib.error.URLError, OSError) as error:
            on_status(f'[Voice] Could not download {name}: {error}')
            partial.unlink(missing_ok=True)
            return False
    return True


# Fixed calls worth having ready before they're needed: neural synthesis takes about a
# second on CPU, and these are the ones where a delay matters most.
PREWARM_CALLS = (
    'We are checking.',
    'Safety Car, Safety Car. Keep the delta positive, no overtaking.',
    'VSC, VSC. Virtual Safety Car deployed, keep the delta positive.',
    'VSC ending, VSC ending. Green flag, push now.',
    'Box, box. Box this lap, last lap of the window.',
    'We can see front wing damage. Box this lap for a new nose. Box, box.',
)


class SoundDevicePlayer:
    """Blocking playback that another thread can cut short."""

    def __init__(self):
        import sounddevice

        self._sd = sounddevice

    def play(self, samples, sample_rate):
        self._sd.play(samples, sample_rate)
        self._sd.wait()

    def stop(self):
        self._sd.stop()


def default_tts_threads() -> int:
    """CPU threads for Kokoro: a quarter of the PC's threads (1 to 4).

    By default ONNX Runtime uses every core for each line it speaks, and that burst steals
    CPU from the game for a moment: a frame drop every time the engineer talks.
    """
    return max(1, min(4, (os.cpu_count() or 4) // 4))


def kokoro_synthesizer(
    model_dir: Path = DEFAULT_KOKORO_DIR,
    voice: str = DEFAULT_KOKORO_VOICE,
    speed: float = 1.1,
    lang: str = 'en-gb',
    threads: int | None = None,
):
    """Return ``text -> (samples, sample_rate)`` backed by Kokoro. Raises if unavailable."""
    from kokoro_onnx import Kokoro

    model_dir = Path(model_dir)
    model, voices = model_dir / KOKORO_MODEL_FILE, model_dir / KOKORO_VOICES_FILE
    if not model.exists() or not voices.exists():
        raise FileNotFoundError(
            f'Kokoro model files not found in {model_dir} (need {KOKORO_MODEL_FILE} and {KOKORO_VOICES_FILE})'
        )
    threads = threads or default_tts_threads()
    kokoro = None
    if hasattr(Kokoro, 'from_session'):
        try:
            import onnxruntime

            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            session = onnxruntime.InferenceSession(
                str(model), sess_options=options, providers=['CPUExecutionProvider']
            )
            kokoro = Kokoro.from_session(session, str(voices))
        except Exception as exc:  # older kokoro-onnx: fall back to its own defaults
            log.warning(f'Could not limit Kokoro to {threads} threads ({exc}).')
    if kokoro is None:
        kokoro = Kokoro(str(model), str(voices))

    def synthesize(text):
        return kokoro.create(text, voice=voice, speed=speed, lang=lang)

    return synthesize


class NeuralSpeechDriver:
    """pyttsx3-shaped driver: say/iterate/isBusy/stop around a synthesize function."""

    CACHE_SIZE = 64  # common calls ("We are checking.", "Box, box...") replay instantly

    def __init__(self, synthesize, player=None, *, radio_effect: bool = True):
        self._synthesize = synthesize
        self._player = player if player is not None else SoundDevicePlayer()
        self._radio_effect = radio_effect
        self._callbacks = {}
        self._lock = threading.Lock()
        self._current = None  # utterance name while synthesizing or playing
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()

    # pyttsx3 surface used by VoiceEngine ------------------------------------
    def setProperty(self, name, value):
        pass  # rate/volume/voice belong to the synthesizer settings

    def getProperty(self, name):
        return [] if name == 'voices' else None

    def connect(self, event, callback):
        self._callbacks[event] = callback

    def startLoop(self, use_driver=False):
        pass

    def endLoop(self):
        self.stop()

    def iterate(self):
        pass

    def isBusy(self):
        with self._lock:
            return self._current is not None

    def say(self, text, name=None):
        token = object()
        with self._lock:
            self._current = (token, name)
        threading.Thread(
            target=self._speak, args=(token, text, name), name='NeuralSpeech', daemon=True
        ).start()

    def stop(self):
        with self._lock:
            playing = self._current is not None
            self._current = None
        if playing:
            try:
                self._player.stop()
            except Exception as exc:
                log.warning(f'Could not stop playback: {exc}')

    def prewarm(self, texts):
        """Synthesize fixed calls in the background so they play with no delay later."""

        def run():
            for text in texts:
                try:
                    self._audio_for(text)
                except Exception as exc:
                    log.warning(f'Prewarm failed for "{text}": {exc}')
                    return

        threading.Thread(target=run, name='NeuralPrewarm', daemon=True).start()

    # internals --------------------------------------------------------------
    def _is_current(self, token):
        with self._lock:
            return self._current is not None and self._current[0] is token

    def _audio_for(self, text):
        with self._lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached
        samples, sample_rate = self._synthesize(text)
        if self._radio_effect:
            samples = radio_filter(samples, sample_rate)
        audio = (np.asarray(samples, dtype=np.float32), int(sample_rate))
        with self._lock:
            self._cache[text] = audio
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return audio

    def _speak(self, token, text, name):
        try:
            samples, sample_rate = self._audio_for(text)
            if not self._is_current(token):
                return  # interrupted while synthesizing
            callback = self._callbacks.get('started-utterance')
            if callback:
                callback(name)
            self._player.play(samples, sample_rate)
        except Exception as exc:
            log.warning(f'Could not speak "{text}": {exc}')
        finally:
            with self._lock:
                if self._current is not None and self._current[0] is token:
                    self._current = None


def build_kokoro_driver_factory(
    model_dir=DEFAULT_KOKORO_DIR,
    voice=DEFAULT_KOKORO_VOICE,
    speed=1.1,
    radio_effect=True,
    download=True,
    threads=None,
):
    """Check Kokoro can load now, and return a factory for VoiceEngine; None if it can't.

    With ``download`` the model files are fetched on first use when the packages are there.
    """
    missing = missing_packages()
    if missing:
        log.warning(
            f'Kokoro voice not installed (missing: {", ".join(missing)}). '
            f'Using the system voice. {SETUP_HINT}'
        )
        return None
    if download and not download_model_files(model_dir):
        log.info(f'Using the system voice until the Kokoro files are in {model_dir}.')
        return None
    try:
        synthesize = kokoro_synthesizer(model_dir, voice, speed, threads=threads)
        player = SoundDevicePlayer()
    except Exception as exc:
        log.warning(f'Kokoro unavailable ({exc}). Using the system voice instead.')
        return None

    def factory():
        driver = NeuralSpeechDriver(synthesize, player, radio_effect=radio_effect)
        driver.prewarm(PREWARM_CALLS)
        return driver

    return factory
