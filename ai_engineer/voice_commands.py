"""On-demand voice commands: speech-to-text, command matching and the latency cue.

When a command is recognised the engineer immediately says "We are checking"
(a plain TTS string, so it needs no audio file and plays in well under a second),
while the answer is generated in a background thread and spoken when ready.

Speech recognition (speech_to_text.py) uses the optional ``SpeechRecognition`` package
(plus ``PyAudio`` for the microphone). Engines:
  * ``whisper`` - offline OpenAI Whisper model (default; ``pip install openai-whisper``),
                  falls back to ``google`` if it isn't installed
  * ``google``  - online, no setup
  * ``vosk``    - fully offline; needs ``pip install vosk`` and a model unpacked to ./model

Commands are recognised by intent scoring (intents.py), not fixed phrases.
"""

from __future__ import annotations

import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import intents
from intents import normalise  # noqa: F401  (kept importable from here)
from log_setup import get_logger
from voice_engine import Priority

log = get_logger('voice')

ACK_TEXT = 'We are checking.'
QUESTION = 'question'  # handle_text() result for a free-form question
MAX_QUESTION_CHARS = 300

COMMANDS = (
    'weather',
    'tyre_temps',
    'pit_window',
    'gap_ahead',
    'gap_behind',
    'pace',
    'threats',
    'strategy',
    'rivals',
)


def match_command(text: str) -> str | None:
    """Map a transcript to one of COMMANDS, or None if it is not a known request.

    Uses intent scoring (intents.py), not fixed regexes, so natural phrasing works:
    "how are my tyres holding up?" is ``tyre_temps``, "is Verstappen coming in?" is ``rivals``.
    """
    return intents.match(text) if text else None


class CommandDispatcher:
    """Acknowledge instantly, then answer from the brain in the background."""

    def __init__(
        self,
        voice,
        brain,
        *,
        on_log=None,
        ack_text: str = ACK_TEXT,
        debounce_s: float = 2.0,
        max_workers: int = 2,
        answer_ttl: float = 30.0,
    ):
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

    def handle_text(self, text: str, *, free_form: bool = False) -> str | None:
        """Process a transcript. Returns the command id, ``QUESTION`` or None.

        The always-on microphone hears everything, so by default only the fixed commands are
        answered. ``free_form=True`` is for deliberate requests (push-to-talk button, the HUD
        text box): any question is sent to the brain, which answers it from the live data.
        """
        command = match_command(text)
        question = None
        if free_form and hasattr(self.brain, 'answer_question'):
            question = re.sub(r'\s+', ' ', str(text or '')).strip()[:MAX_QUESTION_CHARS]
            if not normalise(question):
                return None
            debounce_key = command or f'question:{normalise(question)}'
        elif command is None:
            return None
        else:
            debounce_key = command
        now = time.monotonic()
        with self._lock:
            if self._last and self._last[0] == debounce_key and now - self._last[1] < self.debounce_s:
                return command or QUESTION  # duplicate recognition of the same request
            self._last = (debounce_key, now)
            self._sequence += 1
            sequence = self._sequence
        # CRITICAL so the cue is instant: it skips the braking/cornering silence filter, and the
        # STRATEGY-priority answer behind it can never interrupt it.
        self._speak(self.ack_text, f'cmd_ack:{sequence}', 6, Priority.CRITICAL)
        self._executor.submit(self._answer, command, sequence, question)  # AI works in the background
        return command or QUESTION

    def _answer(self, command: str | None, sequence: int, question: str | None = None):
        try:
            if question is not None:
                message = self.brain.answer_question(question, command)
            else:
                message = self.brain.answer(command)
        except Exception as error:  # never let a bad answer kill the worker silently
            log.warning(f'Could not answer {question or command}: {error}')
            message = 'Sorry, I could not get that. Try again.'
        self._speak(message, f'cmd_answer:{sequence}', self.answer_ttl)

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


def transcribe(recognizer, audio, engine: str = 'google', language: str = 'en-GB') -> str | None:
    """Speech to text with SpeechRecognition. None for unintelligible audio or an error."""
    try:
        if engine == 'vosk':
            import json

            return json.loads(recognizer.recognize_vosk(audio)).get('text') or None
        return recognizer.recognize_google(audio, language=language) or None
    except Exception as error:  # unintelligible audio / network error: just ignore the clip
        if type(error).__name__ != 'UnknownValueError':
            log.warning(f'Speech recognition error: {error}')
        return None


class VoiceCommandListener:
    """Background microphone listener feeding transcripts to a dispatcher."""

    def __init__(
        self,
        dispatcher: CommandDispatcher,
        *,
        engine: str = 'google',
        device_index: int | None = None,
        voice=None,
        language: str = 'en-GB',
        recognizer_factory=None,
        microphone_factory=None,
        ignore_when=None,
        transcriber=None,
    ):
        self.dispatcher = dispatcher
        self.transcriber = transcriber  # callable(audio) -> text; overrides ``engine`` (Whisper)
        self.ignore_when = ignore_when  # callable; True skips a clip (push-to-talk in use)
        self.engine = engine
        self.device_index = device_index
        self.voice = voice  # used to ignore the engineer's own voice
        self.language = language
        self._recognizer_factory = recognizer_factory
        self._microphone_factory = microphone_factory
        self._recognizer = None
        self._stop = None

    def transcribe(self, audio) -> str | None:
        if self.transcriber is not None:
            try:
                return self.transcriber(audio)
            except Exception as error:
                log.warning(f'Speech recognition error: {error}')
                return None
        return transcribe(self._recognizer, audio, self.engine, self.language)

    def _on_audio(self, _recognizer, audio):
        if self.voice is not None and getattr(self.voice, 'is_busy', False):
            return  # do not transcribe the engineer talking
        if self.ignore_when is not None and self.ignore_when():
            return  # the push-to-talk button is handling this
        text = self.transcribe(audio)
        if text:
            log.info(f'Heard: "{text}"')
            self.dispatcher.handle_text(text)

    def start(self) -> bool:
        try:
            import speech_recognition as sr
        except ImportError:
            log.warning('Voice commands disabled: pip install SpeechRecognition PyAudio')
            return False
        try:
            self._recognizer = (self._recognizer_factory or sr.Recognizer)()
            self._recognizer.dynamic_energy_threshold = True
            microphone = (self._microphone_factory or sr.Microphone)(device_index=self.device_index)
            with microphone as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
            self._stop = self._recognizer.listen_in_background(
                microphone, self._on_audio, phrase_time_limit=6
            )
        except Exception as error:  # no microphone / PyAudio missing
            log.warning(f'Microphone unavailable, voice commands disabled: {error}')
            return False
        log.info(
            'Listening for commands: weather, tyre temps, pit window, gap ahead, '
            'gap behind, pace, threats, strategy, rival stops (any natural wording).'
        )
        return True

    def stop(self):
        if self._stop:
            self._stop(wait_for_stop=False)
            self._stop = None
