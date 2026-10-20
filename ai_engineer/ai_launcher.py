"""Start the local DeepSeek model (Ollama) automatically when the engineer starts.

When the AI points at a local Ollama server (the default, http://localhost:11434/v1),
``LocalAILauncher`` does in the background, so the engineer starts instantly:
  1. starts ``ollama serve`` if Ollama is installed but not running,
  2. downloads the model if it isn't there yet (first run only, a few GB),
  3. loads it into memory so the first question doesn't wait for a cold start,
  4. keeps it loaded while the engineer runs (Ollama unloads idle models after 5 minutes).
Until it's ready, questions are answered from the built-in rules as before.
On exit it stops the Ollama server only if it started it. Standard library only.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

LOCAL_HOSTS = {'localhost', '127.0.0.1', '::1', '0.0.0.0'}
OLLAMA_PORT = 11434
KEEP_ALIVE = '15m'           # how long Ollama keeps the model loaded after each ping
PING_EVERY_S = 240           # re-ping well inside Ollama's default 5 minute unload
START_WAIT_S = 30


def ollama_root(base_url: str) -> str | None:
    """'http://localhost:11434/v1' -> 'http://localhost:11434'; None if not a local Ollama."""
    parsed = urlparse(base_url)
    if parsed.hostname not in LOCAL_HOSTS:
        return None
    if (parsed.port or 80) != OLLAMA_PORT:
        return None
    return f'{parsed.scheme}://{parsed.hostname}:{OLLAMA_PORT}'


def find_ollama() -> str | None:
    """The ollama executable: on PATH, or the default install location on Windows/macOS."""
    found = shutil.which('ollama')
    if found:
        return found
    candidates = []
    if sys.platform == 'win32':
        local = os.environ.get('LOCALAPPDATA')
        if local:
            candidates.append(Path(local) / 'Programs' / 'Ollama' / 'ollama.exe')
    elif sys.platform == 'darwin':
        candidates.append(Path('/Applications/Ollama.app/Contents/Resources/ollama'))
        candidates.append(Path('/usr/local/bin/ollama'))
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def model_installed(names, model: str) -> bool:
    """Ollama lists 'deepseek-r1:8b'; a bare 'deepseek-r1' means 'deepseek-r1:latest'."""
    wanted = model if ':' in model else f'{model}:latest'
    return any(name == model or name == wanted for name in names)


class LocalAILauncher:
    def __init__(self, base_url: str, model: str, *, on_status=None, on_ready=None,
                 executable_finder=find_ollama, popen=subprocess.Popen,
                 start_wait_s: float = START_WAIT_S, ping_every_s: float = PING_EVERY_S):
        self.root = ollama_root(base_url)
        self.model = model
        self.on_status = on_status or (lambda text: print(f'[AI] {text}'))
        self.on_ready = on_ready
        self._find = executable_finder
        self._popen = popen
        self._start_wait_s = start_wait_s
        self._ping_every_s = ping_every_s
        self._process = None
        self._stop = threading.Event()
        self._thread = None
        self.ready = threading.Event()
        self.status = 'not started'

    @property
    def applies(self) -> bool:
        return self.root is not None

    # ── HTTP helpers ─────────────────────────────────────────
    def _request(self, path, body=None, timeout=5.0):
        data = None if body is None else json.dumps(body).encode('utf-8')
        request = urllib.request.Request(f'{self.root}{path}', data=data,
                                         method='GET' if body is None else 'POST',
                                         headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(request, timeout=timeout)

    def installed_models(self):
        """Model names on the server, or None if the server isn't answering."""
        try:
            with self._request('/api/tags', timeout=2.0) as response:
                payload = json.loads(response.read().decode('utf-8'))
            return [m.get('name') or m.get('model') for m in payload.get('models', [])]
        except (urllib.error.URLError, OSError, ValueError):
            return None

    # ── steps ────────────────────────────────────────────────
    def _say(self, text):
        self.status = text
        try:
            self.on_status(text)
        except Exception:
            pass

    def _start_server(self):
        exe = self._find()
        if not exe:
            self._say('Ollama is not installed, so DeepSeek is off. Install it from '
                      'https://ollama.com/download (answers come from built-in rules meanwhile).')
            return False
        self._say('Starting Ollama...')
        flags = 0
        if sys.platform == 'win32':
            flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        env = dict(os.environ)
        env.setdefault('OLLAMA_KEEP_ALIVE', KEEP_ALIVE)
        try:
            self._process = self._popen([exe, 'serve'], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, env=env, creationflags=flags)
        except OSError as error:
            self._say(f'Could not start Ollama: {error}')
            return False
        deadline = time.monotonic() + self._start_wait_s
        while time.monotonic() < deadline and not self._stop.is_set():
            if self.installed_models() is not None:
                return True
            time.sleep(0.5)
        self._say('Ollama did not start in time; answering from built-in rules.')
        return False

    def _pull(self):
        self._say(f'Downloading {self.model} (first run only, this can take a while)...')
        last_shown = -10
        try:
            with self._request('/api/pull', {'model': self.model, 'stream': True},
                               timeout=60.0) as response:
                for raw in response:
                    if self._stop.is_set():
                        return False
                    try:
                        line = json.loads(raw.decode('utf-8'))
                    except ValueError:
                        continue
                    if line.get('error'):
                        self._say(f"Download failed: {line['error']}")
                        return False
                    total, done = line.get('total'), line.get('completed')
                    if total and done:
                        percent = int(done * 100 / total)
                        if percent >= last_shown + 10:
                            last_shown = percent
                            self._say(f'Downloading {self.model}: {percent}%')
        except (urllib.error.URLError, OSError) as error:
            self._say(f'Download failed: {error}')
            return False
        return True

    def _load(self, timeout=180.0):
        """Load the model into memory (an empty prompt loads without generating)."""
        try:
            with self._request('/api/generate', {'model': self.model, 'prompt': '',
                                                 'keep_alive': KEEP_ALIVE, 'stream': False},
                               timeout=timeout) as response:
                response.read()
            return True
        except (urllib.error.URLError, OSError) as error:
            self._say(f'Could not load {self.model}: {error}')
            return False

    def _run(self):
        models = self.installed_models()
        if models is None:
            if not self._start_server():
                return
            models = self.installed_models() or []
        if not model_installed(models, self.model):
            if not self._pull():
                return
        if self._stop.is_set():
            return
        self._say(f'Loading {self.model}...')
        if not self._load():
            return
        self.ready.set()
        self._say(f'DeepSeek ready ({self.model}). Ask me anything.')
        if self.on_ready:
            try:
                self.on_ready()
            except Exception:
                pass
        while not self._stop.wait(self._ping_every_s):
            self._load(timeout=30.0)      # keeps the model in memory between questions

    # ── lifecycle ────────────────────────────────────────────
    def start(self):
        """Begin in the background. Returns False when the AI isn't a local Ollama server."""
        if not self.applies:
            return False
        self._thread = threading.Thread(target=self._run, name='LocalAI', daemon=True)
        self._thread.start()
        return True

    def shutdown(self):
        self._stop.set()
        if self._process is not None:
            try:
                self._process.terminate()
            except OSError:
                pass
            self._process = None
