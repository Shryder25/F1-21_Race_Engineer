"""``python main.py --ai-test``: check the AI end to end without the game and print what happens.

It starts Ollama and loads the model the same way the engineer does, asks one question through
the same code the radio uses, and prints the model's raw reply, how long it took, and the reason
if the answer was thrown away. Nothing needs the game, a microphone or the voice.
"""

from __future__ import annotations

import atexit
import json
import time
import urllib.error
import urllib.request

from ai_launcher import LocalAILauncher, model_installed
from engineer_brain import EngineerBrain

DEFAULT_TEST_QUESTION = 'Should I push now or look after the tyres?'
SAMPLE_CONTEXT = {
    'session_mode': 'race',
    'lap': 12,
    'total_laps': 30,
    'position': 5,
    'tyre_compound': 'Medium',
    'tyre_age_laps': 11,
    'tyre_wear_avg_pct': 31,
}
LOAD_WAIT_S = 600  # first run may download the model


def _get_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode('utf-8'))


def run_ai_test(client, *, load_options=None, autostart=True, question=None, out=print) -> int:
    """Return 0 when the AI answered, 1 when it didn't (the printout says why)."""
    question = question or DEFAULT_TEST_QUESTION
    out(f'AI test: model {client.model} at {client.base_url}, timeout {client.timeout:g} s')
    if client.ollama_root:
        if autostart:
            launcher = LocalAILauncher(
                client.base_url, client.model, load_options=load_options, on_status=out
            )
            launcher.start()
            atexit.register(launcher.shutdown)
            deadline = time.monotonic() + LOAD_WAIT_S
            while not launcher.ready.wait(0.5):
                thread = launcher._thread
                if time.monotonic() > deadline or thread is None or not thread.is_alive():
                    out(f'FAILED: the model did not load ({launcher.status}).')
                    return 1
        try:
            version = _get_json(f'{client.ollama_root}/api/version').get('version', '?')
            models = [
                m.get('name') or m.get('model') for m in _get_json(f'{client.ollama_root}/api/tags')['models']
            ]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as error:
            out(f'FAILED: Ollama is not answering at {client.ollama_root} ({error}). Is it running?')
            return 1
        out(f'Ollama {version}; installed models: {", ".join(models) or "none"}')
        if not model_installed(models, client.model):
            out(
                f'FAILED: {client.model} is not installed. Run "ollama pull {client.model}", '
                f'or start the engineer with --ai-model and one of the names above.'
            )
            return 1
    brain = EngineerBrain(lambda: dict(SAMPLE_CONTEXT), client)
    out(f'Asking: "{question}"')
    started = time.monotonic()
    answer = brain.answer_question(question)
    took = time.monotonic() - started
    raw = getattr(client, 'last_reply', '')
    if raw:
        out(f'Raw reply: {raw[:400]!r}')
    placement = client.model_placement() if hasattr(client, 'model_placement') else None
    if placement:
        out(f'Model is {placement}.')
    if brain.last_source == 'ai':
        out(f'OK in {took:.1f} s. The engineer would say: "{answer}"')
        return 0
    out(f'FAILED after {took:.1f} s: {brain.last_failure or "no answer"}')
    out(f'The engineer would say: "{answer}"')
    return 1
