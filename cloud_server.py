"""
F1 2021 AI Race Engineer — cloud entry point
============================================
Runs the engineer on a cloud host (Railway, Fly.io, any Docker host) with telemetry fed by
tools/cloud_relay.py on the game PC. Same engine and HUD as `python main.py`; the only
differences are where packets come from and that there is no voice, microphone or console.

    CLOUD_MODE=1 RELAY_SECRET=... python cloud_server.py [any main.py flag]

Environment:
    RELAY_SECRET    required. Long random string, also given to the relay on your PC.
    CONTROL_SECRET  optional. Open https://<host>/control?token=<it> once to unlock the HUD
                    controls (typed questions, map reset, AI setup) in that browser.
                    Unset = everyone, you included, gets a view-only HUD.
    PORT            set by the host; the HUD and /relay are both served on it.
    DATA_DIR        where laps, track maps and fitted models are kept (default /data, mount a
                    volume there so they survive a redeploy).
    F1_AI_BASE_URL  OpenAI-compatible endpoint for the AI brain, e.g. https://api.deepseek.com
                    or https://api.groq.com/openai/v1. Unset = rule-based answers only.
    F1_AI_MODEL     model name at that endpoint (e.g. deepseek-chat).
    F1_AI_API_KEY   API key for that endpoint.
"""

import os
import signal
import sys
from pathlib import Path

ai_engineer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ai_engineer")
sys.path.insert(0, ai_engineer_dir)


def cloud_argv(argv: list[str], environ=os.environ) -> list[str]:
    """main.py flags for a headless server, before any the caller passed (theirs win)."""
    flags = ['--no-voice', '--no-mic', '--no-ptt', '--no-ai-autostart']
    flags += ['--history-dir', str(Path(environ.get('DATA_DIR') or '/data') / 'telemetry-history')]
    flags += ['--log-format', environ.get('LOG_FORMAT', 'text')]
    if not environ.get('F1_AI_BASE_URL'):
        flags.append('--no-ai')  # no local Ollama in the cloud: rules only until an API is set
    return flags + argv


def _stop(signum, frame):
    # The host sends SIGTERM on redeploy/stop: shut down like Ctrl+C so laps are saved.
    raise KeyboardInterrupt


if __name__ == "__main__":
    os.environ['CLOUD_MODE'] = '1'
    signal.signal(signal.SIGTERM, _stop)
    sys.argv = [sys.argv[0], *cloud_argv(sys.argv[1:])]
    from main import main

    main()
