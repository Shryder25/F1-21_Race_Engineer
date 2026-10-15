"""Nonblocking race radio with expiring, interruptible priority callouts.

All TTS operations belong to one worker, including stop: Windows speech drivers
require their COM event loop and control calls to stay on the owning thread.
"""

import threading
import time
from dataclasses import dataclass
from enum import IntEnum

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class Priority(IntEnum):
    CRITICAL = 0
    STRATEGY = 1
    HIGH = STRATEGY
    RACE = 2
    TIMING = 3
    NORMAL = 4
    LOW = 5


@dataclass
class _Callout:
    message: str
    priority: Priority
    sequence: int
    key: str
    expires: float
    cooldown: float

    @property
    def order(self):
        return self.priority, self.sequence


class VoiceEngine:
    """Bounded radio queue; urgent callouts interrupt lower priority speech."""

    _MAX_PENDING = 64

    def __init__(self, rate: int = 175, volume: float = 0.9,
                 cooldown_seconds: float = 30.0, voice_index: int = 0,
                 on_spoken=None):
        self._rate = rate
        self._volume = volume
        self._cooldown = cooldown_seconds
        self._voice_index = voice_index
        self._on_spoken = on_spoken
        self._condition = threading.Condition()
        self._lifecycle = threading.RLock()
        self._pending: list[_Callout] = []
        self._active: _Callout | None = None
        self._last_said: dict[str, tuple[float, float]] = {}
        self._sequence = 0
        self._clear_requested = False
        self._stop_requested = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._engine_ready = threading.Event()

    def start(self):
        """Start one speech worker; initialization failure leaves voice disabled."""
        with self._lifecycle:
            if self._thread and self._thread.is_alive():
                return
            with self._condition:
                self._stop_requested = False
                self._clear_requested = False
                self._engine_ready.clear()
                self._thread = threading.Thread(target=self._worker, name='VoiceEngine', daemon=True)
                self._thread.start()
            if not self._engine_ready.wait(timeout=5):
                self.shutdown()

    def _prune(self, now):
        self._pending[:] = [item for item in self._pending if item.expires > now]
        self._last_said = {key: record for key, record in self._last_said.items()
                           if record[1] > now}

    def say(self, message: str, priority: Priority = Priority.NORMAL, *,
            key: str | None = None, ttl: float = 15, cooldown: float | None = None) -> bool:
        """Queue current information, replacing a pending callout of the same topic.

        ``True`` means accepted, not spoken. Expired, replaced, interrupted, or
        cleared callouts never return to the queue. Cooldown starts at speech.
        """
        if not message or ttl <= 0:
            return False
        topic = message if key is None else key
        interval = self._cooldown if cooldown is None else cooldown
        now = time.monotonic()
        with self._condition:
            if not self._running or self._stop_requested:
                return False
            self._prune(now)
            if self._active is not None and self._active.key == topic:
                return False
            last = self._last_said.get(topic)
            if last and now - last[0] < interval:
                return False
            self._pending[:] = [item for item in self._pending if item.key != topic]
            if len(self._pending) >= self._MAX_PENDING:
                lowest = max(self._pending, key=lambda item: item.order)
                if priority >= lowest.priority:
                    return False
                self._pending.remove(lowest)
            self._sequence += 1
            self._pending.append(_Callout(message, Priority(priority), self._sequence,
                                          topic, now + ttl, max(0, interval)))
            self._condition.notify_all()
            return True

    def say_immediate(self, message: str):
        """Queue a critical callout without the previous spoken-message cooldown."""
        return self.say(message, Priority.CRITICAL, cooldown=0)

    def clear(self):
        """Cancel pending speech and request worker-thread cancellation of speech."""
        with self._condition:
            self._pending.clear()
            self._clear_requested = True
            self._condition.notify_all()

    def _started(self, name):
        with self._condition:
            item = self._active
            if item is None or name != str(item.sequence):
                return
            now = time.monotonic()
            self._last_said[item.key] = (now, now + item.cooldown)
        if self._on_spoken:
            try:
                self._on_spoken(item.message)
            except Exception as exc:
                print(f'[VoiceEngine] Speech callback error: {exc}')

    def _worker(self):
        engine = None
        loop_started = False
        try:
            if pyttsx3 is None:
                print('[VoiceEngine] pyttsx3 unavailable. Voice disabled.')
                return
            engine = pyttsx3.init()
            engine.setProperty('rate', self._rate)
            engine.setProperty('volume', self._volume)
            voices = engine.getProperty('voices')
            if voices and 0 <= self._voice_index < len(voices):
                engine.setProperty('voice', voices[self._voice_index].id)
            engine.connect('started-utterance', self._started)
            engine.startLoop(False)
            loop_started = True
            engine.iterate()
            with self._condition:
                self._running = not self._stop_requested
                self._engine_ready.set()

            while True:
                with self._condition:
                    if self._stop_requested:
                        break
                    now = time.monotonic()
                    self._prune(now)
                    next_item = min(self._pending, key=lambda item: item.order, default=None)
                    interrupt = (self._active is not None and next_item is not None
                                 and next_item.priority <= Priority.TIMING
                                 and next_item.priority < self._active.priority)
                    if self._clear_requested or interrupt:
                        engine.stop()
                        self._active = None
                        self._clear_requested = False
                    if self._active is None and next_item is not None:
                        self._pending.remove(next_item)
                        if next_item.expires > time.monotonic():
                            self._active = next_item
                            engine.say(next_item.message, name=str(next_item.sequence))
                engine.iterate()
                with self._condition:
                    if self._active is not None and not engine.isBusy():
                        self._active = None
                    if self._active is not None or not self._pending:
                        self._condition.wait(timeout=0.01 if self._active else 0.05)
        except Exception as exc:
            print(f'[VoiceEngine] Voice disabled: {exc}')
        finally:
            if engine is not None:
                try:
                    engine.stop()
                    if loop_started:
                        engine.endLoop()
                except Exception as exc:
                    print(f'[VoiceEngine] Speech cleanup error: {exc}')
            with self._condition:
                self._running = False
                self._active = None
                self._pending.clear()
                self._engine_ready.set()

    def shutdown(self):
        """Cancel speech and join the worker before releasing its thread handle."""
        with self._lifecycle:
            with self._condition:
                self._stop_requested = True
                self._running = False
                self._pending.clear()
                self._condition.notify_all()
                thread = self._thread
            if thread is not None and thread is not threading.current_thread():
                thread.join()
                self._thread = None

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._running and self._thread is not None and self._thread.is_alive()
