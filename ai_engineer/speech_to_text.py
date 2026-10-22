"""Speech to text for the radio: offline Whisper first, the old engines as a fallback.

``--stt-engine whisper`` (the default) runs OpenAI's open-source Whisper model locally
(``pip install openai-whisper``), so the radio works with no internet and no API key.
The model is downloaded once (``base.en`` is about 140 MB, ``tiny.en`` about 75 MB) to
the Whisper cache folder and then loads from disk. If Whisper is not installed, or the
model can't load, the engineer falls back to the previous SpeechRecognition engine
(Google), so nothing stops working.

Whisper is given a short prompt with racing words (box, undercut, inters, DRS...) so
radio jargon is spelled the way the command matcher expects.
"""

from __future__ import annotations

import threading

DEFAULT_WHISPER_MODEL = 'base.en'
WHISPER_MODELS = ('tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small')

# Biases Whisper towards racing vocabulary; it is not text the driver has to say.
RADIO_PROMPT = ('Team radio, Formula 1. Box box, pit window, undercut, overcut, tyre temps, '
                'softs, mediums, hards, inters, wets, DRS, Safety Car, VSC, gap ahead, gap behind, pace.')

# Whisper invents these on silence or wind noise; never treat them as a request.
HALLUCINATIONS = frozenset({
    'thank you', 'thanks for watching', 'thank you for watching', 'you', 'bye', 'okay',
    'please subscribe', 'subtitles by the amara org community',
})
NO_SPEECH_PROB = 0.6     # Whisper's own "this was silence" score above which a clip is dropped
MIN_AVG_LOGPROB = -1.0   # ... combined with a low-confidence decode


def whisper_available() -> bool:
    try:
        import whisper  # noqa: F401
    except Exception:    # ImportError, or torch failing to load
        return False
    return True


def audio_to_float32(audio):
    """SpeechRecognition ``AudioData`` -> 16 kHz mono float32 numpy array for Whisper."""
    import numpy as np
    raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _clean(text: str) -> str:
    return ' '.join(''.join(c if c.isalnum() or c == ' ' else ' ' for c in text.lower()).split())


class WhisperTranscriber:
    """Callable(audio) -> text or None, running a local Whisper model.

    The model loads once, in the background if ``preload()`` is called at startup, so
    the first radio call isn't slowed down by loading it. Safe to call from several
    threads (the push-to-talk worker and the hands-free listener).
    """

    def __init__(self, model_name: str = DEFAULT_WHISPER_MODEL, language: str = 'en-GB', *,
                 loader=None, prompt: str = RADIO_PROMPT):
        self.model_name = model_name
        self.language = (language or 'en').split('-')[0].lower()
        self.prompt = prompt
        self._loader = loader
        self._model = None
        self._error = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()      # one decode at a time; Whisper isn't thread-safe

    def _load(self):
        with self._load_lock:
            if self._model is None and self._error is None:
                try:
                    if self._loader is not None:
                        self._model = self._loader(self.model_name)
                    else:
                        import whisper
                        print(f'[STT] Loading Whisper {self.model_name} (first run downloads it)...')
                        self._model = whisper.load_model(self.model_name)
                        print(f'[STT] Whisper {self.model_name} ready, speech recognition is offline.')
                except Exception as error:
                    self._error = error
                    print(f'[STT] Whisper could not load ({error}).')
            return self._model

    def preload(self):
        threading.Thread(target=self._load, name='WhisperLoad', daemon=True).start()

    @property
    def failed(self) -> bool:
        return self._error is not None

    def __call__(self, audio) -> str | None:
        model = self._load()
        if model is None:
            return None
        samples = audio_to_float32(audio)
        if samples.size == 0:
            return None
        options = {'temperature': 0.0, 'condition_on_previous_text': False,
                   'initial_prompt': self.prompt, 'fp16': False}
        if not self.model_name.endswith('.en'):
            options['language'] = self.language
        with self._run_lock:
            result = model.transcribe(samples, **options)
        return self._accept(result)

    @staticmethod
    def _accept(result) -> str | None:
        text = str(result.get('text') or '').strip()
        segments = result.get('segments') or []
        if segments and all(s.get('no_speech_prob', 0) > NO_SPEECH_PROB
                            and s.get('avg_logprob', 0) < MIN_AVG_LOGPROB for s in segments):
            return None                         # it was noise
        if not _clean(text) or _clean(text) in HALLUCINATIONS:
            return None
        return text


class FallbackTranscriber:
    """Try Whisper; if it can't load, use the SpeechRecognition engine for every clip after."""

    def __init__(self, primary, fallback, fallback_name: str):
        self.primary = primary
        self.fallback = fallback
        self.fallback_name = fallback_name
        self._warned = False

    def preload(self):
        if hasattr(self.primary, 'preload'):
            self.primary.preload()

    def __call__(self, audio):
        if not getattr(self.primary, 'failed', False):
            text = self.primary(audio)
            if not getattr(self.primary, 'failed', False):
                return text
        if not self._warned:
            self._warned = True
            print(f'[STT] Using {self.fallback_name} speech recognition instead.')
        return self.fallback(audio)


def _speech_recognition_transcriber(engine: str, language: str, recognizer_factory=None):
    from voice_commands import transcribe
    state = {}
    lock = threading.Lock()

    def run(audio):
        with lock:
            if 'recognizer' not in state:
                if recognizer_factory is None:
                    import speech_recognition as sr
                    state['recognizer'] = sr.Recognizer()
                else:
                    state['recognizer'] = recognizer_factory()
        return transcribe(state['recognizer'], audio, engine, language)
    return run


def build_transcriber(engine: str = 'whisper', language: str = 'en-GB', *,
                      whisper_model: str = DEFAULT_WHISPER_MODEL, available=whisper_available,
                      loader=None, recognizer_factory=None):
    """The radio's speech-to-text callable for ``--stt-engine``.

    ``whisper`` uses local Whisper when installed, else Google through SpeechRecognition
    (with a console note saying so). ``google`` and ``vosk`` are unchanged.
    """
    if engine != 'whisper':
        return _speech_recognition_transcriber(engine, language, recognizer_factory)
    fallback = _speech_recognition_transcriber('google', language, recognizer_factory)
    if loader is None and not available():
        print('[STT] Offline Whisper not installed (pip install openai-whisper); '
              'using Google speech recognition, which needs internet.')
        return fallback
    return FallbackTranscriber(WhisperTranscriber(whisper_model, language, loader=loader),
                               fallback, 'Google')
