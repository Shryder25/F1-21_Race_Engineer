"""Push-to-talk on a steering wheel or controller button, like a real team radio.

Hold the button (or press it once in ``toggle`` mode), ask anything, let go. The engineer
says "We are checking", sends your words plus the live telemetry to the AI brain and
speaks the answer. Questions are free-form: "how are the rears holding up?", "can I
make it to the end on these?", "who's that behind me?" all work, not just the fixed
voice commands.

Pieces (each can be swapped out in tests):
  * ``ButtonWatcher``  reads the wheel/controller with pygame in its own thread and
                       turns button presses into start/stop of the radio.
  * ``MicRecorder``    keeps the microphone open with a short pre-roll buffer, so the
                       first word is not clipped when you press and talk straight away.
  * ``PushToTalk``     glues them together: record, transcribe, dispatch.
  * ``find_button``    "press the button you want" helper behind
                       ``python main.py --find-ptt-button``. It saves the choice to
                       ``push_to_talk.json`` so you only do it once.

Needs ``pip install pygame SpeechRecognition PyAudio``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from voice_engine import Priority
from log_setup import get_logger

log = get_logger('ptt')

HOLD, TOGGLE = 'hold', 'toggle'
MODES = (HOLD, TOGGLE)
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / 'push_to_talk.json'
SAY_AGAIN_TEXT = "Say again, didn't get that."
HAT_DIRECTIONS = {'up': (0, 1), 'down': (0, -1), 'left': (-1, 0), 'right': (1, 0)}


# ──────────────────────────────────────────────
# Button binding
# ──────────────────────────────────────────────

@dataclass(frozen=True)
class Binding:
    """A button (``kind='button'``) or a D-pad/hat direction (``kind='hat'``)."""
    kind: str
    index: int
    direction: str | None = None

    @property
    def label(self) -> str:
        return str(self.index) if self.kind == 'button' else f'hat{self.index}:{self.direction}'

    def is_pressed(self, joystick) -> bool:
        if self.kind == 'button':
            return self.index < joystick.get_numbuttons() and bool(joystick.get_button(self.index))
        if self.index >= joystick.get_numhats():
            return False
        x, y = joystick.get_hat(self.index)
        want_x, want_y = HAT_DIRECTIONS[self.direction]
        return (want_x != 0 and x == want_x) or (want_y != 0 and y == want_y)

    def exists_on(self, joystick) -> bool:
        count = joystick.get_numbuttons() if self.kind == 'button' else joystick.get_numhats()
        return self.index < count


def parse_binding(text) -> Binding:
    """``'5'`` is button 5; ``'hat0:up'`` is up on the first D-pad/hat."""
    value = str(text).strip().lower()
    if value.isdigit():
        return Binding('button', int(value))
    if value.startswith('hat') and ':' in value:
        index, direction = value[3:].split(':', 1)
        if index.isdigit() and direction in HAT_DIRECTIONS:
            return Binding('hat', int(index), direction)
    raise ValueError(f'Unknown push-to-talk button {text!r}: use a number like 5, or hat0:up')


# ──────────────────────────────────────────────
# Saved settings (written by --find-ptt-button)
# ──────────────────────────────────────────────

def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or 'button' not in data:
        return None
    try:
        parse_binding(data['button'])
    except ValueError:
        log.warning(f'Ignoring {path}: bad button {data.get("button")!r}')
        return None
    return data


def save_config(binding: Binding, device: str | None, mode: str = HOLD,
                path: Path = DEFAULT_CONFIG_PATH) -> Path:
    path = Path(path)
    path.write_text(json.dumps({'button': binding.label, 'device': device, 'mode': mode},
                               indent=2) + '\n', encoding='utf-8')
    return path


# ──────────────────────────────────────────────
# Joystick access (pygame)
# ──────────────────────────────────────────────

class PygameJoysticks:
    """Minimal pygame wrapper. Create and use it from one thread only."""

    def __init__(self):
        # Keep reading the wheel while F1 2021 has focus; without this SDL drops input
        # as soon as the game window is in front.
        os.environ.setdefault('SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS', '1')
        os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
        import pygame
        self.pygame = pygame
        try:
            pygame.display.init()       # the event pump needs it; no window is opened
        except pygame.error:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
            pygame.display.init()
        pygame.joystick.init()
        self._open = {}
        self._device_events = {getattr(pygame, 'JOYDEVICEADDED', -1),
                               getattr(pygame, 'JOYDEVICEREMOVED', -1)}

    def poll(self) -> bool:
        """Read new input. True when a controller was plugged in or removed."""
        changed = False
        for event in self.pygame.event.get():
            if event.type in self._device_events:
                changed = True
        return changed

    def devices(self) -> list:
        """Open joysticks, refreshed. Each has get_name/get_button/get_hat/get_numbuttons/get_numhats."""
        current = {}
        for index in range(self.pygame.joystick.get_count()):
            joystick = self.pygame.joystick.Joystick(index)
            joystick.init()
            current[index] = joystick
        self._open = current
        return [current[i] for i in sorted(current)]

    def close(self):
        try:
            self.pygame.joystick.quit()
            self.pygame.display.quit()
        except Exception:
            pass


def _device_matches(device, index: int, joystick) -> bool:
    if device is None:
        return True
    text = str(device).strip()
    if text.isdigit():
        return index == int(text)
    return text.lower() in joystick.get_name().lower()


def describe_devices(joysticks) -> list[str]:
    return [f'  [{i}] {j.get_name()}  ({j.get_numbuttons()} buttons, {j.get_numhats()} hats)'
            for i, j in enumerate(joysticks)]


# ──────────────────────────────────────────────
# Button watcher
# ──────────────────────────────────────────────

class ButtonWatcher:
    """Polls the bound button and calls ``on_start`` / ``on_stop`` for the radio.

    ``hold``: talk while the button is held. ``toggle``: press to open, press to close.
    A radio left open is closed after ``max_open_s`` so a forgotten toggle can't record forever.
    """

    def __init__(self, binding: Binding, on_start, on_stop, *, device=None, mode: str = HOLD,
                 backend_factory=PygameJoysticks, poll_hz: float = 100, debounce_s: float = 0.05,
                 max_open_s: float = 15.0, rescan_s: float = 2.0):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        self.binding = binding
        self.on_start = on_start
        self.on_stop = on_stop
        self.device = device
        self.mode = mode
        self.backend_factory = backend_factory
        self.poll_interval = 1.0 / poll_hz
        self.debounce_s = debounce_s
        self.max_open_s = max_open_s
        self.rescan_s = rescan_s
        self.is_open = False
        self.device_name = None
        self._pressed = False
        self._last_edge = None
        self._opened_at = None
        self._wait_release = False
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._error = None
        self._thread = None

    # Pure state machine, driven by the poll loop (and directly by tests).
    def feed(self, pressed: bool, now: float):
        if self.is_open and now - self._opened_at >= self.max_open_s:
            self._close()
            self._wait_release = pressed
        if pressed == self._pressed:
            return
        if self._last_edge is not None and now - self._last_edge < self.debounce_s:
            return                                  # contact bounce
        self._pressed = pressed
        self._last_edge = now
        if self._wait_release:
            self._wait_release = pressed
            return
        if self.mode == HOLD:
            if pressed and not self.is_open:
                self._open(now)
            elif not pressed and self.is_open:
                self._close()
        elif pressed:                               # toggle: act on the press only
            if self.is_open:
                self._close()
            else:
                self._open(now)

    def _open(self, now):
        self.is_open = True
        self._opened_at = now
        self._safe(self.on_start)

    def _close(self):
        self.is_open = False
        self._safe(self.on_stop)

    @staticmethod
    def _safe(callback):
        try:
            callback()
        except Exception as error:
            log.warning(f'Radio callback failed: {error}')

    def _find(self, backend):
        for index, joystick in enumerate(backend.devices()):
            if _device_matches(self.device, index, joystick) and self.binding.exists_on(joystick):
                return joystick
        return None

    def _run(self):
        try:
            backend = self.backend_factory()
        except Exception as error:      # pygame missing or no input subsystem
            self._error = error
            self._ready.set()
            return
        self._ready.set()
        joystick, next_scan = None, 0.0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                try:
                    changed = backend.poll()
                    if changed or (joystick is None and now >= next_scan):
                        found = self._find(backend)
                        if found is not None and found is not joystick:
                            self.device_name = found.get_name()
                            log.info(f'Radio button {self.binding.label} on {self.device_name}')
                        elif found is None and joystick is not None:
                            log.info('Controller disconnected, waiting for it to come back.')
                        joystick, next_scan = found, now + self.rescan_s
                    pressed = joystick is not None and self.binding.is_pressed(joystick)
                except Exception as error:   # unplugged mid-read
                    if joystick is not None:
                        log.warning(f'Lost the controller ({error}).')
                    joystick, pressed, next_scan = None, False, now + self.rescan_s
                self.feed(pressed, now)
                self._stop.wait(self.poll_interval)
        finally:
            if self.is_open:
                self._close()
            backend.close()

    def start(self, timeout: float = 5.0) -> bool:
        self._thread = threading.Thread(target=self._run, name='PushToTalkButton', daemon=True)
        self._thread.start()
        self._ready.wait(timeout)
        if self._error is not None:
            log.warning(f'Controller input unavailable ({self._error}). pip install pygame')
            return False
        return True

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)


# ──────────────────────────────────────────────
# Microphone recorder
# ──────────────────────────────────────────────

class MicRecorder:
    """Records between ``begin()`` and ``end()``.

    The microphone stream stays open so pressing the button records instantly. Audio
    outside a press is only kept for ``preroll_s`` in memory and then thrown away.
    """

    def __init__(self, device_index: int | None = None, *, preroll_s: float = 0.3,
                 max_s: float = 15.0, min_s: float = 0.3, microphone_factory=None,
                 audio_factory=None):
        self.device_index = device_index
        self.preroll_s = preroll_s
        self.max_s = max_s
        self.min_s = min_s
        self._microphone_factory = microphone_factory
        self._audio_factory = audio_factory
        self._lock = threading.Lock()
        self._frames = None
        self._preroll = deque()
        self._stop = threading.Event()
        self._thread = None
        self._mic = None
        self.sample_rate = 16000
        self.sample_width = 2
        self.chunk = 1024

    def start(self) -> bool:
        try:
            import speech_recognition as sr
        except ImportError:
            log.warning('Push-to-talk needs: pip install SpeechRecognition PyAudio')
            return False
        try:
            self._mic = (self._microphone_factory or sr.Microphone)(device_index=self.device_index)
            source = self._mic.__enter__()
        except Exception as error:      # no microphone / PyAudio missing
            log.warning(f'Microphone unavailable: {error}')
            return False
        self._audio_factory = self._audio_factory or sr.AudioData
        self.sample_rate, self.sample_width = source.SAMPLE_RATE, source.SAMPLE_WIDTH
        self.chunk = source.CHUNK
        self._preroll = deque(maxlen=max(1, int(self.preroll_s * self.sample_rate / self.chunk)))
        self._thread = threading.Thread(target=self._read_loop, args=(source,),
                                        name='PushToTalkMic', daemon=True)
        self._thread.start()
        return True

    def _read_loop(self, source):
        max_chunks = int(self.max_s * self.sample_rate / self.chunk)
        try:
            while not self._stop.is_set():
                data = source.stream.read(source.CHUNK)
                with self._lock:
                    if self._frames is None:
                        self._preroll.append(data)
                    elif len(self._frames) < max_chunks:
                        self._frames.append(data)
        except Exception as error:
            if not self._stop.is_set():
                log.warning(f'Microphone stopped: {error}')

    def feed(self, data: bytes):
        """Push one chunk by hand (tests); same path as the read loop."""
        with self._lock:
            if self._frames is None:
                self._preroll.append(data)
            else:
                self._frames.append(data)

    @property
    def recording(self) -> bool:
        return self._frames is not None

    def begin(self):
        with self._lock:
            self._frames = list(self._preroll)
            self._preroll.clear()

    def end(self):
        """The recorded clip as SpeechRecognition ``AudioData``, or None for a tap that was too short."""
        with self._lock:
            frames, self._frames = self._frames, None
        if not frames:
            return None
        raw = b''.join(frames)
        if len(raw) / (self.sample_rate * self.sample_width) < self.min_s:
            return None
        return self._audio_factory(raw, self.sample_rate, self.sample_width)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        if self._mic is not None:
            try:
                self._mic.__exit__(None, None, None)
            except Exception:
                pass
            self._mic = None


# ──────────────────────────────────────────────
# Glue
# ──────────────────────────────────────────────

class PushToTalk:
    """Button down: record. Button up: transcribe in the background and ask the engineer."""

    def __init__(self, dispatcher, recorder, transcriber, *, voice=None, on_log=None):
        self.dispatcher = dispatcher
        self.recorder = recorder
        self.transcriber = transcriber        # callable(audio) -> text or None
        self.voice = voice
        self.on_log = on_log
        self.last_active = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='PushToTalkSTT')

    def on_start(self):
        self.last_active = time.monotonic()
        self.recorder.begin()
        log.info('Radio open, listening...')

    def on_stop(self):
        self.last_active = time.monotonic()
        audio = self.recorder.end()
        if audio is None:
            log.info('Radio closed (too short, ignored).')
            return
        log.info('Radio closed, transcribing...')
        self._executor.submit(self._process, audio)

    def _process(self, audio):
        try:
            text = self.transcriber(audio)
        except Exception as error:
            log.warning(f'Speech recognition failed: {error}')
            text = None
        self.last_active = time.monotonic()
        if not text:
            if self.voice is not None:
                self.voice.say(SAY_AGAIN_TEXT, Priority.CRITICAL, key='ptt_say_again', ttl=5, cooldown=0)
            if self.on_log:
                self.on_log('(radio unreadable)', status='heard')
            return
        log.info(f'Driver: "{text}"')
        if self.on_log:
            self.on_log(f'Driver: {text}', status='heard')
        self.dispatcher.handle_text(text, free_form=True)

    def recently_active(self, window_s: float = 3.0) -> bool:
        """True while the radio is open or was just used (lets the hands-free mic stand down)."""
        if self.recorder.recording:
            return True
        return self.last_active is not None and time.monotonic() - self.last_active < window_s

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)


@dataclass
class Settings:
    binding: Binding
    device: str | None = None
    mode: str = HOLD


def resolve_settings(button=None, device=None, mode=None,
                     config_path: Path = DEFAULT_CONFIG_PATH) -> Settings | None:
    """Command-line values win over the saved push_to_talk.json. None when no button is set."""
    saved = load_config(config_path) or {}
    button = button if button is not None else saved.get('button')
    if button is None:
        return None
    binding = parse_binding(button)
    same_as_saved = 'button' in saved and parse_binding(saved['button']) == binding
    saved_device = saved.get('device') if same_as_saved else None  # a new button may be on any device
    mode = mode or saved.get('mode') or HOLD
    if mode not in MODES:
        mode = HOLD
    return Settings(binding, device if device is not None else saved_device, mode)


class RadioButton:
    """What ``start`` returns: the running watcher, recorder and glue."""

    def __init__(self, watcher, recorder, ptt):
        self.watcher, self.recorder, self.ptt = watcher, recorder, ptt

    def recently_active(self, window_s: float = 3.0) -> bool:
        return self.ptt.recently_active(window_s)

    def stop(self):
        self.watcher.stop()
        self.recorder.stop()
        self.ptt.shutdown()


def start(settings: Settings, dispatcher, *, voice=None, engine: str = 'google',
          mic_index: int | None = None, language: str = 'en-GB', on_log=None,
          backend_factory=PygameJoysticks, recorder=None, recognizer=None) -> RadioButton | None:
    """Start the radio button. None (with a console message) if the mic or controller is unavailable."""
    from voice_commands import transcribe
    recorder = recorder or MicRecorder(mic_index)
    if not recorder.start():
        return None
    if recognizer is None:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
    ptt = PushToTalk(dispatcher, recorder,
                     lambda audio: transcribe(recognizer, audio, engine, language),
                     voice=voice, on_log=on_log)
    watcher = ButtonWatcher(settings.binding, ptt.on_start, ptt.on_stop, device=settings.device,
                            mode=settings.mode, backend_factory=backend_factory)
    if not watcher.start():
        recorder.stop()
        ptt.shutdown()
        return None
    where = f' on "{settings.device}"' if settings.device else ''
    action = 'Hold' if settings.mode == HOLD else 'Press'
    log.info(f'{action} button {settings.binding.label}{where} to talk to the engineer. '
          'Ask anything.')
    return RadioButton(watcher, recorder, ptt)


# ──────────────────────────────────────────────
# "Press the button you want" helper
# ──────────────────────────────────────────────

def _snapshot(joystick):
    buttons = tuple(bool(joystick.get_button(i)) for i in range(joystick.get_numbuttons()))
    hats = tuple(tuple(joystick.get_hat(i)) for i in range(joystick.get_numhats()))
    return buttons, hats


def _hat_direction(value):
    for name, (x, y) in HAT_DIRECTIONS.items():
        if (x and value[0] == x) or (y and value[1] == y):
            return name
    return None


def find_button(backend, *, timeout_s: float = 30.0, out=print, clock=time.monotonic,
                sleep=time.sleep):
    """Wait for a button press on any controller. Returns (device index, device name, Binding) or None."""
    backend.poll()
    joysticks = backend.devices()
    if not joysticks:
        out('No wheel or controller found. Plug it in (and close apps that grab it exclusively), '
            'then run this again.')
        return None
    out('Controllers found:')
    for line in describe_devices(joysticks):
        out(line)
    out(f'\nPress and release the button you want as the radio button (within {timeout_s:.0f} s)...')
    sleep(0.2)
    backend.poll()
    baseline = [_snapshot(j) for j in joysticks]   # ignores buttons that are always on (shifters, mode switches)
    deadline = clock() + timeout_s
    while clock() < deadline:
        if backend.poll():
            out('Controller list changed; run this again.')
            return None
        for index, joystick in enumerate(joysticks):
            buttons, hats = _snapshot(joystick)
            base_buttons, base_hats = baseline[index]
            for button, down in enumerate(buttons):
                if down and not base_buttons[button]:
                    return index, joystick.get_name(), Binding('button', button)
            for hat, value in enumerate(hats):
                if value != base_hats[hat]:
                    direction = _hat_direction(value)
                    if direction:
                        return index, joystick.get_name(), Binding('hat', hat, direction)
        sleep(0.01)
    out('No button press seen.')
    return None


def run_button_finder(mode: str = HOLD, config_path: Path = DEFAULT_CONFIG_PATH,
                      backend_factory=PygameJoysticks) -> int:
    """CLI entry for ``python main.py --find-ptt-button``. Returns a process exit code."""
    try:
        backend = backend_factory()
    except Exception as error:
        print(f'Controller input unavailable ({error}). Install it with: pip install pygame')
        return 1
    try:
        found = find_button(backend)
    finally:
        backend.close()
    if not found:
        return 1
    _index, name, binding = found
    path = save_config(binding, name, mode, config_path)
    print(f'\nRadio button: {binding.label} on "{name}" ({mode} to talk).')
    print(f'Saved to {path}. Start the engineer as usual (python main.py) and it will use it.')
    print(f'Or pass it by hand: python main.py --ptt-button {binding.label} --ptt-device "{name}"')
    return 0


if __name__ == '__main__':
    import sys
    raise SystemExit(run_button_finder(TOGGLE if '--toggle' in sys.argv else HOLD))
