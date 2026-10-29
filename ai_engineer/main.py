"""
F1 2021 AI Race Engineer — Main Entry Point
=============================================
UDP telemetry listener that feeds data into the analysis engine
and delivers feedback through the voice engine.

Usage:
    python main.py [--port 20777] [--no-voice] [--log-level DEBUG] [--log-format json]
    python main.py --config my-settings.toml     # thresholds etc. (default: ../config.toml)
    python main.py --record race.f1rec.gz        # also save every datagram for replay
    python main.py --replay race.f1rec.gz [--replay-speed 1|10|max]   # run from a recording
    python main.py --find-ptt-button             # pick the wheel/controller radio button once
    python main.py --log-channels channels/      # log every sample per channel to Parquet
    python main.py --export-motec                # newest file in channels/ -> .ld + .ldx for MoTeC i2

Game Setup (F1 2021 Settings → Telemetry):
    UDP Telemetry:  ON
    UDP Broadcast:  ON
    UDP IP Address: 127.0.0.1
    UDP Port:       20777
    UDP Format:     2021
"""

import argparse
import atexit
import os
import socket
import sys
import time
from pathlib import Path

# Safe UTF-8 / replacement handling on Windows CMD
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add the ai_engineer directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import webbrowser

import auto_refit
import push_to_talk
import radio_discipline
import strategy_replan
from ai_client import AIClient, ollama_options
from ai_launcher import LocalAILauncher
from analysis import AnalysisEngine
from engineer_brain import EngineerBrain
from fitted_models import ModelStore
from history import TelemetryHistory
from log_setup import FORMATS, LEVELS, get_logger
from log_setup import configure as configure_logging
from packets import (
    TRACK_NAMES,
    CarTelemetryData,
    LapData,
    SessionData,
)
from pipeline.config import ConfigError, EngineConfig, apply_overrides, load_config
from radio_examples import RadioExamples
from speech_to_text import DEFAULT_WHISPER_MODEL, WHISPER_MODELS, build_transcriber
from telemetry_stream import TelemetryStream
from track_maps import TrackMapStore
from udp_recording import MAX_DATAGRAM, RecordingWriter, ReplayFinished, ReplaySource, parse_speed
from voice_commands import CommandDispatcher, VoiceCommandListener
from voice_engine import Priority, VoiceEngine
from web_hud import (
    add_voice_log,
    set_command_handler,
    set_reset_map_handler,
    set_setup_ai_client,
    start_web_hud_server,
)

log = get_logger('main')


# ──────────────────────────────────────────────
# Console Display
# ──────────────────────────────────────────────


def print_telemetry_banner(port: int = 20777):
    """Print the startup banner (console UI on stdout, like the live dashboard; not a log)."""
    print("=" * 60)
    print("  === F1 2021 AI RACE ENGINEER ===")
    print("=" * 60)
    print("  Listening for telemetry data...")
    print("  Make sure F1 2021 has UDP Telemetry enabled:")
    print("    Settings -> Telemetry -> UDP Telemetry: ON")
    print(f"    UDP IP: 127.0.0.1 | Port: {port}")
    print("=" * 60)
    print()


def format_tyre_temps(telemetry: CarTelemetryData) -> str:
    """Format tyre temperatures for console display."""
    t = telemetry.tyres_surface_temperature
    return f"  FL: {t[2]:3d} C  |  FR: {t[3]:3d} C\n  RL: {t[0]:3d} C  |  RR: {t[1]:3d} C"


def print_live_dashboard(
    telemetry: CarTelemetryData, lap: LapData, session: SessionData | None, delta_ms: int | None
):
    """Print a compact live dashboard to the console."""
    # Build the display
    speed = telemetry.speed
    throttle = int(telemetry.throttle * 100)
    brake = int(telemetry.brake * 100)
    gear = telemetry.gear if telemetry.gear >= 0 else "R"
    rpm = telemetry.engine_rpm
    drs = "DRS" if telemetry.drs else "   "

    lap_time_ms = lap.current_lap_time_ms
    lap_secs = lap_time_ms / 1000.0
    lap_mins = int(lap_secs // 60)
    lap_rem = lap_secs % 60

    position = lap.car_position
    lap_num = lap.current_lap_num

    # Track info
    track_name = ""
    if session:
        track_name = TRACK_NAMES.get(session.track_id, "Unknown")

    # Delta
    if delta_ms is not None:
        delta_sign = "+" if delta_ms > 0 else "-"
        delta_str = f"{delta_sign}{abs(delta_ms) / 1000:.3f}s"
    else:
        delta_str = "---"

    # Throttle/Brake bars (ASCII safe for any Windows codepage)
    throttle_bar = "#" * (throttle // 5) + "-" * (20 - throttle // 5)
    brake_bar = "#" * (brake // 5) + "-" * (20 - brake // 5)

    # Clear previous output and redraw (move cursor up)
    output = (
        f"\r\033[K"
        f"  {track_name} | P{position} | Lap {lap_num} | "
        f"{lap_mins}:{lap_rem:06.3f} | Delta: {delta_str}\n\033[K"
        f"  Speed: {speed:3d} km/h | Gear: {gear} | RPM: {rpm:5d} | {drs}\n\033[K"
        f"  Throttle: [{throttle_bar}] {throttle:3d}%\n\033[K"
        f"  Brake:    [{brake_bar}] {brake:3d}%\n\033[K"
        f"  Tyres (Surface C):\n\033[K"
        f"{format_tyre_temps(telemetry)}\n\033[K"
        f"  Engine: {telemetry.engine_temperature} C | "
        f"Pressures: FL {telemetry.tyres_pressure[2]:.1f} FR {telemetry.tyres_pressure[3]:.1f} "
        f"RL {telemetry.tyres_pressure[0]:.1f} RR {telemetry.tyres_pressure[1]:.1f}"
    )

    # Move cursor up to overwrite previous frame
    sys.stdout.write(f"\033[9A{output}")
    sys.stdout.flush()


# ──────────────────────────────────────────────
# Main UDP Listener
# ──────────────────────────────────────────────


def export_motec(args) -> int:
    """--export-motec: write a channel-store file out for MoTeC i2 and return an exit code."""
    try:
        from motec import export_session

        result = export_session(
            args.export_motec, args.motec_out, rate_hz=args.motec_rate, fmt=args.motec_format
        )
    except (OSError, ValueError, RuntimeError) as e:
        log.error(f"MoTeC export failed: {e}")
        return 1
    log.info(
        f"MoTeC: wrote {result['path']} ({result['channels']} channels at {result['rate_hz']} Hz, "
        f"{result['laps']} laps)"
    )
    if result.get('ldx'):
        log.info(f"MoTeC: wrote {result['ldx']} (lap markers; keep it next to the .ld)")
    return 0


def run_ai_test_from(args, cfg: EngineConfig) -> int:
    from ai_check import run_ai_test

    load_options = ollama_options(args.ai_context, 0 if args.ai_cpu else args.ai_gpu_layers, args.ai_threads)
    client = AIClient(
        cfg.ai.base_url or None,
        cfg.ai.model or None,
        args.ai_api_key,
        cfg.ai.timeout_s,
        load_options=load_options,
    )
    return run_ai_test(
        client,
        load_options=load_options,
        autostart=cfg.ai.autostart,
        question=args.ai_test,
        examples=load_radio_examples(args),
    )


def load_radio_examples(args):
    """Real engineer radio lines for the AI prompt, or None (off, or no file built yet)."""
    if args.no_radio_examples:
        return None
    return RadioExamples.load(args.radio_examples)


def settings_from(args) -> EngineConfig:
    """config.toml (or --config FILE), with every command-line flag the user gave on top."""
    config = load_config(args.config, required=args.config is not None)
    return apply_overrides(
        config,
        {
            'network.udp_port': args.port,
            'voice.rate_wpm': args.voice_rate,
            'voice.tts': args.tts,
            'voice.kokoro_voice': args.kokoro_voice,
            'voice.kokoro_speed': args.kokoro_speed,
            'voice.radio_fx': False if args.no_radio_fx else None,
            'simulation.enabled': False if args.no_race_sim else None,
            'simulation.runs': args.sim_runs,
            'simulation.plan_b_margin_s': args.plan_b_margin,
            'simulation.plan_b_places': args.plan_b_places,
            'ai.base_url': args.ai_base_url,
            'ai.model': args.ai_model,
            'ai.timeout_s': args.ai_timeout,
            'ai.autostart': False if args.no_ai_autostart else None,
            'radio.on_demand': False if args.chatty else None,
            'radio.calls_per_lap': args.calls_per_lap,
            'radio.braking_calls': True if args.braking_calls else None,
            'radio.race_briefing': False if args.no_race_briefing else None,
            'radio.corner_speed_kph': args.corner_speed,
            'radio.corner_g': args.corner_g,
            'logging.level': args.log_level or ('DEBUG' if args.debug else None),
            'logging.format': args.log_format,
            'logging.file': args.log_file,
        },
    )


def build_parser() -> argparse.ArgumentParser:
    defaults = EngineConfig()
    parser = argparse.ArgumentParser(
        description="F1 2021 AI Race Engineer",
        epilog='Defaults come from config.toml at the repo root; a flag overrides the file.',
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        metavar='FILE',
        help='Settings file (default: config.toml at the repo root, if present)',
    )
    parser.add_argument(
        "--port", type=int, default=None, help=f"UDP port to listen on (default: {defaults.network.udp_port})"
    )
    parser.add_argument("--no-voice", action="store_true", help="Disable voice output (console only)")
    parser.add_argument("--debug", action="store_true", help="Debug output (same as --log-level DEBUG)")
    parser.add_argument(
        '--log-level',
        type=str.upper,
        choices=LEVELS,
        default=None,
        help=f'Log detail (default: {defaults.logging.level})',
    )
    parser.add_argument(
        '--log-format',
        choices=FORMATS,
        default=None,
        help='text: readable terminal lines (default); json: one JSON object per line',
    )
    parser.add_argument(
        '--log-file', default=None, metavar='FILE', help='Also write every log record to FILE as JSON lines'
    )
    parser.add_argument(
        "--voice-rate",
        type=int,
        default=None,
        help=f"TTS speech rate in WPM (default: {defaults.voice.rate_wpm})",
    )
    parser.add_argument("--voice-index", type=int, default=0, help="System voice index (default: 0)")
    parser.add_argument(
        '--history-dir',
        type=Path,
        default=Path(__file__).resolve().parent.parent / 'telemetry-history',
        help='Directory for track/session telemetry history',
    )
    parser.add_argument(
        '--models',
        type=Path,
        default=None,
        metavar='FILE',
        help='Tyre/fuel models fitted by tools/fit_models.py '
        '(default: <history-dir>/fitted_models.json; built-in estimates if missing)',
    )
    parser.add_argument(
        '--no-auto-refit',
        action='store_true',
        help="Don't refit the tyre/fuel models from your laps when the engineer closes",
    )
    parser.add_argument(
        '--no-race-sim',
        action='store_true',
        help='Turn off the Monte Carlo race simulator (strategy win chances)',
    )
    parser.add_argument(
        '--sim-runs',
        type=int,
        default=None,
        metavar='N',
        help=f'Races the Monte Carlo simulator runs per update (default {defaults.simulation.runs})',
    )
    parser.add_argument(
        '--plan-b-margin',
        type=float,
        default=None,
        metavar='SECONDS',
        help='Only call Plan B when it beats the current plan by this much race time '
        f'(default {defaults.simulation.plan_b_margin_s:g})',
    )
    parser.add_argument(
        '--plan-b-places',
        type=float,
        default=None,
        metavar='PLACES',
        help='... or by this many finishing places on average '
        f'(default {defaults.simulation.plan_b_places:g})',
    )
    tts = parser.add_argument_group('engineer voice')
    tts.add_argument(
        '--tts',
        choices=('kokoro', 'system'),
        default=None,
        help='Voice: kokoro (default; free offline neural voice, falls back to the system '
        'voice if not installed) or system (pyttsx3)',
    )
    tts.add_argument(
        '--kokoro-dir',
        type=Path,
        default=None,
        help='Folder with kokoro-v1.0.onnx and voices-v1.0.bin (default: ./models/kokoro)',
    )
    tts.add_argument(
        '--kokoro-voice',
        default=None,
        help='Kokoro voice, e.g. bm_george, bm_lewis, bm_daniel (default: bm_george)',
    )
    tts.add_argument(
        '--kokoro-speed',
        type=float,
        default=None,
        help='Kokoro speaking speed (default: 1.1, a touch brisk like real radio)',
    )
    tts.add_argument(
        '--tts-threads',
        type=int,
        default=None,
        metavar='N',
        help='CPU threads Kokoro may use (default: a quarter of your CPU threads, 1-4). '
        'Lower = fewer frame drops while it speaks, higher = voice ready sooner',
    )
    tts.add_argument(
        '--no-radio-fx',
        action='store_true',
        help='Kokoro only: clean studio voice instead of the team-radio effect',
    )
    ai = parser.add_argument_group('AI engineer (DeepSeek via an OpenAI-compatible API)')
    ai.add_argument(
        '--ai-base-url',
        default=None,
        help='API base URL (default: $F1_AI_BASE_URL or Ollama http://localhost:11434/v1)',
    )
    ai.add_argument('--ai-model', default=None, help='Model name (default: $F1_AI_MODEL or deepseek-r1:8b)')
    ai.add_argument(
        '--ai-api-key',
        default=None,
        help='API key for cloud providers such as Groq (default: $F1_AI_API_KEY)',
    )
    ai.add_argument(
        '--ai-timeout',
        type=float,
        default=None,
        help='Seconds to wait for the model before answering from rules (default: 30; '
        'DeepSeek R1 thinks before it answers)',
    )
    ai.add_argument(
        '--ai-test',
        nargs='?',
        const='',
        default=None,
        metavar='QUESTION',
        help='Check the AI without the game: start it, ask one question (or QUESTION), print the '
        'raw reply, the time it took and why an answer was thrown away, then exit',
    )
    ai.add_argument(
        '--ai-for-commands',
        action='store_true',
        help='Send the 9 fixed questions (gap ahead, tyre temps...) to the AI too. By default '
        'they are answered instantly from the live data and the AI handles everything else',
    )
    ai.add_argument(
        '--ai-full-context',
        action='store_true',
        help='Send the AI all the data for every question instead of just what the question '
        'is about (slower)',
    )
    ai.add_argument(
        '--radio-examples',
        type=Path,
        default=None,
        metavar='FILE',
        help='Real F1 radio lines shown to the AI as examples of how an engineer talks '
        '(default: data/openf1/examples_train.jsonl, built by tools/openf1_radio.py; '
        'skipped when the file is missing)',
    )
    ai.add_argument(
        '--no-radio-examples',
        action='store_true',
        help='Do not show the AI real F1 radio lines',
    )
    ai.add_argument(
        '--ai-context',
        type=int,
        default=4096,
        metavar='TOKENS',
        help='Context window asked of local Ollama (default: 4096, enough for a full race; '
        'bigger uses more graphics memory)',
    )
    ai.add_argument(
        '--ai-gpu-layers',
        type=int,
        default=None,
        metavar='N',
        help='Local Ollama: how many model layers go on the graphics card (default: Ollama '
        'decides, usually all). Fewer leaves more of the GPU to the game',
    )
    ai.add_argument(
        '--ai-cpu',
        action='store_true',
        help='Local Ollama: run the model on the CPU only (same as --ai-gpu-layers 0). '
        'No frame drops from the AI, but slower answers',
    )
    ai.add_argument(
        '--ai-threads',
        type=int,
        default=None,
        metavar='N',
        help='Local Ollama: CPU threads for the model (default: Ollama decides)',
    )
    ai.add_argument(
        '--ai-normal-priority',
        action='store_true',
        help=argparse.SUPPRESS,  # normal priority is the default now; kept so old commands work
    )
    ai.add_argument(
        '--ai-low-priority',
        action='store_true',
        help="Start Ollama below the game's priority. Fewer frame drops on a busy CPU, but in a "
        'race the AI can then be starved and answer too late',
    )
    ai.add_argument(
        '--no-ai-autostart',
        action='store_true',
        help="Don't start Ollama / download / preload the local DeepSeek model automatically",
    )
    ai.add_argument('--no-ai', action='store_true', help='Answer commands from rules only')
    cmd = parser.add_argument_group('voice commands')
    cmd.add_argument('--no-mic', action='store_true', help='Disable microphone voice commands')
    cmd.add_argument(
        '--stt-engine',
        choices=('whisper', 'google', 'vosk'),
        default='whisper',
        help='Speech-to-text engine (default: whisper, offline; falls back to google '
        'if openai-whisper is not installed; vosk is also offline)',
    )
    cmd.add_argument(
        '--whisper-model',
        choices=WHISPER_MODELS,
        default=DEFAULT_WHISPER_MODEL,
        help=f'Whisper model size (default: {DEFAULT_WHISPER_MODEL}; tiny.en is quicker, '
        'small.en more accurate but slower)',
    )
    cmd.add_argument('--mic-index', type=int, default=None, help='Microphone device index')
    cmd.add_argument(
        '--chatty',
        action='store_true',
        help='Race info every lap too: threats, rival tyres and stops, Safety Car chance, gaps, '
        'sector times (no driving tips unless --braking-calls)',
    )
    cmd.add_argument(
        '--braking-calls',
        action='store_true',
        help='Call out every braking point ("brake for turn 3, 100 metres"). Off by default',
    )
    cmd.add_argument(
        '--no-race-briefing',
        action='store_true',
        help='With --chatty: no per-lap race-info call (threats, rival tyres, Safety Car chance)',
    )
    disc = parser.add_argument_group('radio discipline (config.toml [radio])')
    disc.add_argument(
        '--calls-per-lap',
        type=int,
        default=None,
        metavar='N',
        help='Most proactive callouts per lap; urgent calls (Safety Car, flags, box, pit window, '
        f'strategy, DRS attack) always get through (default {defaults.radio.calls_per_lap}, 0 = no limit)',
    )
    disc.add_argument(
        '--corner-speed',
        type=float,
        default=None,
        metavar='KPH',
        help=f'No talking in corners taken faster than this... (default {defaults.radio.corner_speed_kph:g})',
    )
    disc.add_argument(
        '--corner-g',
        type=float,
        default=None,
        metavar='G',
        help=f'...with at least this much lateral G (default {defaults.radio.corner_g:g})',
    )
    ptt = parser.add_argument_group('push-to-talk radio button (wheel or controller)')
    ptt.add_argument(
        '--find-ptt-button',
        action='store_true',
        help='Press the wheel/controller button you want as the radio button; saves it to push_to_talk.json and exits',
    )
    ptt.add_argument(
        '--ptt-button',
        default=None,
        metavar='BUTTON',
        help='Radio button: a button number like 5, or a D-pad direction like hat0:up (default: saved choice)',
    )
    ptt.add_argument(
        '--ptt-device',
        default=None,
        metavar='DEVICE',
        help='Controller index or part of its name, e.g. 0 or "T300" (default: saved choice, else any)',
    )
    ptt.add_argument(
        '--ptt-mode',
        choices=push_to_talk.MODES,
        default=None,
        help='hold: talk while held (default). toggle: press to talk, press again to send',
    )
    ptt.add_argument(
        '--find-game-button',
        action='store_true',
        help="Press the wheel button you want; reads it from the game's own UDP button "
        'feed (no pygame needed), saves it to push_to_talk.json and exits',
    )
    ptt.add_argument(
        '--ptt-game-button',
        default=None,
        metavar='BUTTON',
        help="Radio button read from the game's UDP Buttons event: r1, l1, cross, "
        'dpad_up, r3, ... (default: saved choice)',
    )
    ptt.add_argument('--no-ptt', action='store_true', help='Ignore the saved push-to-talk button')
    ptt.add_argument(
        '--always-listen',
        action='store_true',
        help='Keep the hands-free wake-word microphone on as well as the radio button',
    )
    rec = parser.add_argument_group('recording and replay (format: docs/udp-recording-format.md)')
    rec.add_argument(
        '--record',
        type=Path,
        default=None,
        metavar='FILE',
        help='Save every incoming UDP datagram with its arrival time to FILE '
        '(a name ending .gz is compressed)',
    )
    rec.add_argument(
        '--replay',
        type=Path,
        default=None,
        metavar='FILE',
        help='Read telemetry from a recording instead of the game',
    )
    rec.add_argument(
        '--replay-speed',
        type=parse_speed,
        default=1.0,
        metavar='SPEED',
        help='Replay speed: 1 (real time, default), 10, or max (as fast as possible)',
    )
    ch = parser.add_argument_group('channel data store and MoTeC export (docs/channel-store.md)')
    ch.add_argument(
        '--log-channels',
        type=Path,
        default=None,
        metavar='DIR',
        nargs='?',
        const=Path('channels'),
        help='Log every telemetry sample to DIR/<session>.parquet (default channels/), keyed '
        'by lap distance (needs pyarrow). Works live and with --replay',
    )
    ch.add_argument(
        '--export-motec',
        type=Path,
        default=None,
        metavar='FILE_OR_DIR',
        nargs='?',
        const=Path('channels'),
        help='Export a channel file to MoTeC i2 (.ld + .ldx) and exit. Given a folder, '
        'or nothing, it uses the newest file in that folder (default channels/)',
    )
    ch.add_argument(
        '--motec-out',
        type=Path,
        default=None,
        metavar='FILE',
        help='Output path for --export-motec (default: next to the Parquet file)',
    )
    ch.add_argument(
        '--motec-format',
        choices=('ld', 'csv'),
        default='ld',
        help='ld (MoTeC i2 log, default) or csv (MoTeC-style CSV)',
    )
    ch.add_argument(
        '--motec-rate',
        type=int,
        default=None,
        metavar='HZ',
        help='Sample rate for the export (default: the rate the game sent)',
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    try:
        cfg = settings_from(args)
        configure_logging(cfg.logging.level, cfg.logging.format, cfg.logging.file or None)
    except (ConfigError, ValueError, OSError) as error:
        parser.error(str(error))
    if args.record and args.replay:
        parser.error('--record and --replay cannot be used together')
    if args.export_motec:
        sys.exit(export_motec(args))
    if args.find_ptt_button:
        sys.exit(push_to_talk.run_button_finder(args.ptt_mode or push_to_talk.HOLD))
    if args.find_game_button:
        sys.exit(
            push_to_talk.run_game_button_finder(cfg.network.udp_port, args.ptt_mode or push_to_talk.HOLD)
        )
    if args.ai_test is not None:
        sys.exit(run_ai_test_from(args, cfg))
    ptt_settings = None
    if not args.no_ptt and not args.no_mic and not args.no_voice:
        try:
            ptt_settings = push_to_talk.resolve_settings(
                args.ptt_button, args.ptt_device, args.ptt_mode, game_button=args.ptt_game_button
            )
        except ValueError as error:
            parser.error(str(error))

    # ─── Initialize Voice Engine ───
    driver_factory = None
    if cfg.voice.tts == 'kokoro' and not args.no_voice:
        from neural_voice import DEFAULT_KOKORO_DIR, build_kokoro_driver_factory

        driver_factory = build_kokoro_driver_factory(
            args.kokoro_dir or DEFAULT_KOKORO_DIR,
            cfg.voice.kokoro_voice,
            cfg.voice.kokoro_speed,
            radio_effect=cfg.voice.radio_fx,
            threads=args.tts_threads,
        )
        if driver_factory:
            log.info(f'Engineer voice: Kokoro ({cfg.voice.kokoro_voice})')
    voice = VoiceEngine(
        driver_factory=driver_factory,
        rate=cfg.voice.rate_wpm,
        volume=0.9,
        cooldown_seconds=cfg.voice.cooldown_s,
        voice_index=args.voice_index,
        on_spoken=add_voice_log,
    )

    if not args.no_voice:
        voice.start()
        time.sleep(0.5)
        voice.say(
            'Radio check. Engineer standing by, waiting for the car.', Priority.LOW, key='startup', ttl=10
        )
    else:
        log.info("Voice output disabled (--no-voice)")

    # ─── Initialize Analysis Engine ───
    history = TelemetryHistory(args.history_dir / 'telemetry.sqlite3')
    models_path = args.models or args.history_dir / 'fitted_models.json'
    fitted = ModelStore.load_or_empty(models_path)
    if fitted.wear or fitted.fuel:
        log.info(
            f"Loaded {len(fitted.wear)} tyre and {len(fitted.fuel)} fuel fits from {models_path}"
            + (' (simulator data only)' if fitted.simulated_only else '')
        )
    else:
        log.info(
            'No fitted tyre/fuel models yet; using built-in estimates. '
            'Fit them with: python tools/fit_models.py recordings/*.f1rec.gz'
        )
    sim = cfg.simulation
    analysis = AnalysisEngine(
        voice,
        history=history,
        on_demand=cfg.radio.on_demand,
        fitted=fitted,
        race_simulation='async' if sim.enabled else 'off',
        sim_runs=sim.runs,
        replan=strategy_replan.ReplanConfig(margin_s=sim.plan_b_margin_s, margin_positions=sim.plan_b_places),
        track_store=TrackMapStore(args.history_dir / 'track_maps.json'),
        config=cfg,
        calls_per_lap=cfg.radio.calls_per_lap,
        load_thresholds=radio_discipline.LoadThresholds(
            corner_speed_kph=cfg.radio.corner_speed_kph,
            corner_g=cfg.radio.corner_g,
            straight_g=cfg.radio.straight_g,
            straight_s=cfg.radio.straight_s,
            brake=cfg.radio.busy_brake,
        ),
    )
    stream = TelemetryStream(analysis)
    voice.set_silence_filter(analysis.is_driver_busy)

    # ─── AI brain + on-demand voice commands ───
    load_options = ollama_options(args.ai_context, 0 if args.ai_cpu else args.ai_gpu_layers, args.ai_threads)
    client = (
        None
        if args.no_ai
        else AIClient(
            cfg.ai.base_url or None,
            cfg.ai.model or None,
            args.ai_api_key,
            cfg.ai.timeout_s,
            load_options=load_options,
        )
    )
    brain = EngineerBrain(
        analysis.build_context,
        client,
        rules_first=not args.ai_for_commands,
        focused=not args.ai_full_context,
        examples=load_radio_examples(args),
    )
    dispatcher = CommandDispatcher(voice, brain, on_log=add_voice_log)
    # Typed HUD questions are deliberate, so they may be free-form like the radio button.
    set_command_handler(lambda text: dispatcher.handle_text(text, free_form=True))
    set_setup_ai_client(client)
    set_reset_map_handler(analysis.reset_map)
    listener = None
    radio = None
    transcriber = None
    if not args.no_mic and not args.no_voice:
        # One shared speech-to-text (offline Whisper by default) for both microphones.
        transcriber = build_transcriber(args.stt_engine, whisper_model=args.whisper_model)
        if hasattr(transcriber, 'preload'):
            transcriber.preload()  # load Whisper now, not on the first radio call
    if ptt_settings:
        radio = push_to_talk.start(
            ptt_settings,
            dispatcher,
            voice=voice,
            engine=args.stt_engine,
            mic_index=args.mic_index,
            on_log=add_voice_log,
            transcriber=transcriber,
        )
        if radio is not None:
            stream.on_buttons = radio.on_game_buttons  # the game's BUTN event drives the radio
    if not args.no_mic and not args.no_voice and (radio is None or args.always_listen):
        listener = VoiceCommandListener(
            dispatcher,
            engine=args.stt_engine,
            device_index=args.mic_index,
            voice=voice,
            ignore_when=radio.recently_active if radio else None,
            transcriber=transcriber,
        )
        listener.start()
    if client:
        log.info(f"Engineer brain: {client.model} @ {client.base_url} (falls back to rules if unreachable)")
        if cfg.ai.autostart:
            # Local Ollama: start it, fetch the model on first run, preload it, keep it warm.
            launcher = LocalAILauncher(
                client.base_url,
                client.model,
                load_options=load_options,
                low_priority=args.ai_low_priority and not args.ai_normal_priority,
                on_ready=lambda: voice.say(
                    'Pit wall is connected. Ask me anything.', Priority.LOW, key='ai_ready', ttl=15
                ),
            )
            if launcher.start():
                atexit.register(launcher.shutdown)

    channel_logger = None
    if args.log_channels:
        try:
            from channel_store import ChannelLogger

            channel_logger = ChannelLogger(
                args.log_channels, on_file=lambda path: log.info(f"Channels saved to {path}")
            )
            log.info(f"Logging every channel sample to {args.log_channels}/")
        except RuntimeError as e:  # pyarrow missing
            log.warning(f"Channel logging disabled: {e}")

    # ─── Set up UDP Socket (or a recording standing in for it) ───
    recorder = None
    if args.replay:
        try:
            sock = ReplaySource(args.replay, args.replay_speed)
        except (OSError, ValueError) as e:
            log.error(f"Cannot open recording {args.replay}: {e}")
            voice.shutdown()
            history.close()
            sys.exit(1)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Non-blocking with 1s timeout for clean shutdown

        try:
            sock.bind(("0.0.0.0", cfg.network.udp_port))
        except OSError as e:
            log.error(
                f"Cannot bind to UDP port {cfg.network.udp_port}: {e}. "
                "Is another application using this port?"
            )
            voice.shutdown()
            history.close()
            sock.close()
            sys.exit(1)
        if args.record:
            try:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                recorder = RecordingWriter(args.record, metadata={'port': cfg.network.udp_port})
            except OSError as e:  # includes FileExistsError: never overwrite a recording
                log.error(f"Cannot create recording {args.record}: {e}")
                voice.shutdown()
                history.close()
                sock.close()
                sys.exit(1)

    # Enable Virtual Terminal Processing on Windows cmd.exe
    if sys.platform == "win32":
        try:
            os.system("")
        except Exception:
            pass

    # Start Visual Telemetry Web HUD Server (Port 5000)
    hud_port = cfg.network.hud_port
    web_server = start_web_hud_server(port=hud_port)
    try:
        webbrowser.open(f"http://localhost:{hud_port}")
    except Exception:
        pass

    print_telemetry_banner(cfg.network.udp_port)
    if args.replay:
        speed = 'max speed' if args.replay_speed is None else f'{args.replay_speed:g}x'
        log.info(f"Replay: reading telemetry from {args.replay} at {speed}")
    else:
        log.info(f"Bound to UDP port {cfg.network.udp_port}")
    if recorder:
        log.info(f"Recording every datagram to {args.record}")
    log.info(f"Live Telemetry Dashboard running at: http://localhost:{hud_port}")
    log.info(
        "Waiting for F1 2021 telemetry data... (Launch F1 2021, start Time Trial/Grand Prix/Race, "
        "and make sure UDP Telemetry is ON)"
    )

    # ─── State ───
    latest_telemetry: CarTelemetryData | None = None
    latest_lap: LapData | None = None
    latest_session: SessionData | None = None
    packets_received = 0
    last_display_time = 0.0
    display_interval = 0.05  # 20 FPS console update
    first_packet_logged = False
    spinner_chars = ["|", "/", "-", "\\"]
    spinner_idx = 0
    last_wait_tick = 0.0

    try:
        while True:
            try:
                data, addr = sock.recvfrom(MAX_DATAGRAM)
            except ReplayFinished as done:
                log.info(f"Replay: {done}")
                break
            except TimeoutError:
                stream.check_timeout()
                if not first_packet_logged:
                    now = time.time()
                    if now - last_wait_tick >= 0.5:
                        last_wait_tick = now
                        spin = spinner_chars[spinner_idx % len(spinner_chars)]
                        spinner_idx += 1
                        sys.stdout.write(
                            f"\r  [{spin}] Listening on 0.0.0.0:{cfg.network.udp_port}... "
                            "waiting for car on track..."
                        )
                        sys.stdout.flush()
                continue
            except ConnectionResetError:
                continue

            packets_received += 1
            if recorder:
                recorder.write(data)
            if channel_logger:
                channel_logger.handle(data)
            if not stream.handle(data):
                if stream.last_error:
                    log.debug(f'Ignored packet: {stream.last_error}')
                continue
            latest_session = stream.latest_session
            latest_telemetry = stream.latest_telemetry
            latest_lap = stream.latest_lap

            if not first_packet_logged:
                first_packet_logged = True
                sys.stdout.write("\r" + " " * 75 + "\r")
                log.info('Valid F1 2021 telemetry stream detected!')
                # Prepare space for 9 lines of dashboard
                for _ in range(9):
                    print()

            # ─── Run Analysis ───
            if latest_telemetry and latest_lap:
                # ─── Console Display ───
                now = time.time()
                if now - last_display_time >= display_interval:
                    last_display_time = now
                    try:
                        print_live_dashboard(
                            latest_telemetry,
                            latest_lap,
                            latest_session,
                            analysis.delta.last_reported_delta_ms,
                        )
                    except Exception:
                        pass  # Don't crash on display errors

    except KeyboardInterrupt:
        log.info("Shutting down...")
        voice.say_immediate("That's it from us. Good work today, see you in the garage.")
        time.sleep(2)
    finally:
        sock.close()
        if recorder:
            recorder.close()
            log.info(f"Saved {recorder.count} datagrams to {args.record}")
        if channel_logger:
            try:
                channel_logger.close()
            except Exception as e:
                log.warning(f"Could not finish the channel file: {e}")
        if listener:
            listener.stop()
        if radio:
            radio.stop()
        dispatcher.shutdown()
        if not args.no_auto_refit:
            # Learn from this session: refit tyre/fuel/pit models from every lap driven so far.
            try:
                auto_refit.refit(history, models_path, extra_recordings=[args.record] if args.record else ())
            except Exception as error:  # never let learning break shutdown
                log.warning(f'Model refit skipped: {error}')
        history.close()
        if web_server:
            web_server.shutdown()
            web_server.server_close()
        voice.shutdown()
        log.info(f"Received {packets_received} packets total.")
        log.info("AI Race Engineer offline.")


if __name__ == "__main__":
    main()
