"""On-demand voice commands: speech-to-text, command matching and the latency cue.

When a command is recognised the engineer immediately says "We are checking"
(a plain TTS string, so it needs no audio file and plays in well under a second),
while the answer is generated in a background thread and spoken when ready.

Speech recognition uses the optional ``SpeechRecognition`` package (plus
``PyAudio`` for the microphone). Engines:
  * ``google`` - online, no setup (default)
  * ``vosk``   - fully offline; needs ``pip install vosk`` and a model unpacked to ./model
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from voice_engine import Priority

ACK_TEXT = 'We are checking.'

COMMANDS = ('weather', 'tyre_temps', 'pit_window', 'gap_ahead', 'gap_behind', 'pace', 'threats', 'strategy',
            'rivals')

# "When will Hamilton box?", "what are the others doing?", "rival strategy": questions about
# other cars' stops. The driver's own stop ("when do we box") stays a pit-window question.
_RIVAL_WORDS = re.compile(r'\b(rivals?|others|other cars|everyone|the field|opponents?|car ahead|car behind|they|they\'re)\b')
_STOP_WORDS = re.compile(r'\b(pit|pitting|box|boxing|stop|stopping|strategy|strategies|tyres?|doing)\b')
_NAMED_STOP = re.compile(r'\b(?:when (?:is|will|does)|is|will) (?!(?:i|we|me|my|us|it|the|there|this|that)\b)[a-z]+ '
                         r'(?:going to )?(?:pit|box|stop)')

_WAKE_WORDS = re.compile(r'^(?:hey |ok |okay )?(?:engineer|race engineer|box box)[, ]+', re.IGNORECASE)


def normalise(text: str) -> str:
    text = text.lower().replace('tire', 'tyre')
    text = re.sub(r"[^a-z0-9' ]+", ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def match_command(text: str) -> str | None:
    """Map a transcript to one of COMMANDS, or None if it is not a known request."""
    t = _WAKE_WORDS.sub('', normalise(text)) if text else ''
    t = normalise(t)
    if not t:
        return None
    if (_RIVAL_WORDS.search(t) and _STOP_WORDS.search(t)) or _NAMED_STOP.search(t):
        return 'rivals'
    if re.search(r'\b(threat|threats|attacking|pressure|catching|closing)\b', t):
        return 'threats'
    if re.search(r'\b(undercut|overcut|strategy|plan|how many stops|one stop|two stop|which tyres?|what tyres?|compound)\b', t):
        return 'strategy'
    if 'behind' in t or 'chasing' in t:
        return 'gap_behind'
    if re.search(r'\b(pit|box|stop|window)\b', t):
        return 'pit_window'
    if re.search(r'\b(gap|ahead|front)\b', t):
        return 'gap_ahead'
    if re.search(r'\b(tyre|tyres|temps|temperature|temperatures)\b', t):
        return 'tyre_temps'
    if re.search(r'\b(weather|rain|forecast|raining)\b', t):
        return 'weather'
    if re.search(r'\b(pace|lap time|how am i doing|how are we doing)\b', t):
        return 'pace'
    return None


class CommandDispatcher:
    """Acknowledge instantly, then answer from the brain in the background."""

    def __init__(self, voice, brain, *, on_log=None, ack_text: str = ACK_TEXT,
                 debounce_s: float = 2.0, max_workers: int = 2, answer_ttl: float = 30.0):
        self.voice = voice
        self.brain = brain
        self.on_log = on_log
        self.ack_text = ack_text
        self.debounce_s = debounce_s
        self.answer_ttl = answer_ttl
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='EngineerAnswer')
        self._lock = threading.Lock()
        self._last: tuple[str, float] | None = None
        self._sequence = 0

    def _speak(self, message, key, ttl, priority=Priority.STRATEGY):
        queued = self.voice.say(message, priority, key=key, ttl=ttl, cooldown=0)
        if not queued and not getattr(self.voice, 'is_running', False) and self.on_log:
            self.on_log(message, status='muted')

    def handle_text(self, text: str) -> str | None:
        """Process a transcript. Returns the command id if one was recognised."""
        command = match_command(text)
        if command is None:
            return None
        now = time.monotonic()
        with self._lock:
            if self._last and self._last[0] == command and now - self._last[1] < self.debounce_s:
                return command  # duplicate recognition of the same request
            self._last = (command, now)
            self._sequence += 1
            sequence = self._sequence
        # CRITICAL so the cue is instant: it skips the braking/cornering silence filter, and the
        # STRATEGY-priority answer behind it can never interrupt it.
        self._speak(self.ack_text, f'cmd_ack:{sequence}', 6, Priority.CRITICAL)
        self._executor.submit(self._answer, command, sequence)         # AI works in the background
        return command

    def _answer(self, command: str, sequence: int):
        try:
            message = self.brain.answer(command)
        except Exception as error:  # never let a bad answer kill the worker silently
            print(f'[Engineer] Could not answer {command}: {error}')
            message = 'Sorry, I could not get that. Try again.'
        self._speak(message, f'cmd_answer:{sequence}', self.answer_ttl)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


class VoiceCommandListener:
    """Background microphone listener feeding transcripts to a dispatcher."""

    def __init__(self, dispatcher: CommandDispatcher, *, engine: str = 'google',
                 device_index: int | None = None, voice=None, language: str = 'en-GB',
                 recognizer_factory=None, microphone_factory=None):
        self.dispatcher = dispatcher
        self.engine = engine
        self.device_index = device_index
        self.voice = voice          # used to ignore the engineer's own voice
        self.language = language
        self._recognizer_factory = recognizer_factory
        self._microphone_factory = microphone_factory
        self._recognizer = None
        self._stop = None

    def transcribe(self, audio) -> str | None:
        try:
            if self.engine == 'vosk':
                import json
                return json.loads(self._recognizer.recognize_vosk(audio)).get('text') or None
            return self._recognizer.recognize_google(audio, language=self.language)
        except Exception as error:   # unintelligible audio / network error: just ignore the clip
            if type(error).__name__ != 'UnknownValueError':
                print(f'[Voice] Speech recognition error: {error}')
            return None

    def _on_audio(self, _recognizer, audio):
        if self.voice is not None and getattr(self.voice, 'is_busy', False):
            return  # do not transcribe the engineer talking
        text = self.transcribe(audio)
        if text:
            print(f'[Voice] Heard: "{text}"')
            self.dispatcher.handle_text(text)

    def start(self) -> bool:
        try:
            import speech_recognition as sr
        except ImportError:
            print('[Voice] Voice commands disabled: pip install SpeechRecognition PyAudio')
            return False
        try:
            self._recognizer = (self._recognizer_factory or sr.Recognizer)()
            self._recognizer.dynamic_energy_threshold = True
            microphone = (self._microphone_factory or sr.Microphone)(device_index=self.device_index)
            with microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self._stop = self._recognizer.listen_in_background(microphone, self._on_audio,
                                                               phrase_time_limit=6)
        except Exception as error:   # no microphone / PyAudio missing
            print(f'[Voice] Microphone unavailable, voice commands disabled: {error}')
            return False
        print('[Voice] Listening for commands: weather, tyre temps, pit window, '
              'gap ahead, gap behind, pace update.')
        return True

    def stop(self):
        if self._stop:
            self._stop(wait_for_stop=False)
            self._stop = None
