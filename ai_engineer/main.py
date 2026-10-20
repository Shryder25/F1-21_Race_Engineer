"""
F1 2021 AI Race Engineer — Main Entry Point
=============================================
UDP telemetry listener that feeds data into the analysis engine
and delivers feedback through the voice engine.

Usage:
    python main.py [--port 20777] [--no-voice] [--debug]
    python main.py --record race.f1rec.gz        # also save every datagram for replay
    python main.py --replay race.f1rec.gz [--replay-speed 1|10|max]   # run from a recording
    python main.py --find-ptt-button             # pick the wheel/controller radio button once

Game Setup (F1 2021 Settings → Telemetry):
    UDP Telemetry:  ON
    UDP Broadcast:  ON
    UDP IP Address: 127.0.0.1
    UDP Port:       20777
    UDP Format:     2021
"""

import socket
import sys
import time
import argparse
import os
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

from packets import (
    CarTelemetryData, LapData, SessionData, TRACK_NAMES,
)
from voice_engine import VoiceEngine, Priority
from analysis import AnalysisEngine
from history import TelemetryHistory
from fitted_models import ModelStore
from telemetry_stream import TelemetryStream
from web_hud import (start_web_hud_server, add_voice_log, set_command_handler,
                     set_setup_ai_client, set_reset_map_handler)
from ai_client import AIClient
from engineer_brain import EngineerBrain
from voice_commands import CommandDispatcher, VoiceCommandListener
import push_to_talk
from udp_recording import (MAX_DATAGRAM, RecordingWriter, ReplayFinished, ReplaySource,
                           parse_speed)
import webbrowser


# ──────────────────────────────────────────────
# Console Display
# ──────────────────────────────────────────────

def print_telemetry_banner():
    """Print the startup banner."""
    print("=" * 60)
    print("  === F1 2021 AI RACE ENGINEER ===")
    print("=" * 60)
    print("  Listening for telemetry data...")
    print("  Make sure F1 2021 has UDP Telemetry enabled:")
    print("    Settings -> Telemetry -> UDP Telemetry: ON")
    print("    UDP IP: 127.0.0.1 | Port: 20777")
    print("=" * 60)
    print()


def format_tyre_temps(telemetry: CarTelemetryData) -> str:
    """Format tyre temperatures for console display."""
    t = telemetry.tyres_surface_temperature
    return (
        f"  FL: {t[2]:3d} C  |  FR: {t[3]:3d} C\n"
        f"  RL: {t[0]:3d} C  |  RR: {t[1]:3d} C"
    )


def print_live_dashboard(telemetry: CarTelemetryData, lap: LapData,
                         session: SessionData | None, delta_ms: int | None):
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
        delta_str = f"{delta_sign}{abs(delta_ms)/1000:.3f}s"
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

def main():
    parser = argparse.ArgumentParser(description="F1 2021 AI Race Engineer")
    parser.add_argument("--port", type=int, default=20777,
                        help="UDP port to listen on (default: 20777)")
    parser.add_argument("--no-voice", action="store_true",
                        help="Disable voice output (console only)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")
    parser.add_argument("--voice-rate", type=int, default=175,
                        help="TTS speech rate in WPM (default: 175)")
    parser.add_argument("--voice-index", type=int, default=0,
                        help="System voice index (default: 0)")
    parser.add_argument('--history-dir', type=Path,
                        default=Path(__file__).resolve().parent.parent / 'telemetry-history',
                        help='Directory for track/session telemetry history')
    parser.add_argument('--models', type=Path, default=None, metavar='FILE',
                        help='Tyre/fuel models fitted by tools/fit_models.py '
                             '(default: <history-dir>/fitted_models.json; built-in estimates if missing)')
    parser.add_argument('--no-race-sim', action='store_true',
                        help='Turn off the Monte Carlo race simulator (strategy win chances)')
    parser.add_argument('--sim-runs', type=int, default=300, metavar='N',
                        help='Races the Monte Carlo simulator runs per update (default 300)')
    tts = parser.add_argument_group('engineer voice')
    tts.add_argument('--tts', choices=('system', 'kokoro'), default='system',
                     help='Voice: system (pyttsx3) or kokoro (free offline neural voice, see neural_voice.py)')
    tts.add_argument('--kokoro-dir', type=Path, default=None,
                     help='Folder with kokoro-v1.0.onnx and voices-v1.0.bin (default: ./models/kokoro)')
    tts.add_argument('--kokoro-voice', default='bm_george',
                     help='Kokoro voice, e.g. bm_george, bm_lewis, bm_daniel (default: bm_george)')
    tts.add_argument('--kokoro-speed', type=float, default=1.1,
                     help='Kokoro speaking speed (default: 1.1, a touch brisk like real radio)')
    tts.add_argument('--no-radio-fx', action='store_true',
                     help='Kokoro only: clean studio voice instead of the team-radio effect')
    ai = parser.add_argument_group('AI engineer (DeepSeek via an OpenAI-compatible API)')
    ai.add_argument('--ai-base-url', default=None,
                    help='API base URL (default: $F1_AI_BASE_URL or Ollama http://localhost:11434/v1)')
    ai.add_argument('--ai-model', default=None,
                    help='Model name (default: $F1_AI_MODEL or deepseek-r1:8b)')
    ai.add_argument('--ai-api-key', default=None,
                    help='API key for cloud providers such as Groq (default: $F1_AI_API_KEY)')
    ai.add_argument('--ai-timeout', type=float, default=15.0,
                    help='Seconds to wait for the model before answering from rules (default: 15)')
    ai.add_argument('--no-ai', action='store_true', help='Answer commands from rules only')
    cmd = parser.add_argument_group('voice commands')
    cmd.add_argument('--no-mic', action='store_true', help='Disable microphone voice commands')
    cmd.add_argument('--stt-engine', choices=('google', 'vosk'), default='google',
                     help='Speech-to-text engine (default: google; vosk is offline)')
    cmd.add_argument('--mic-index', type=int, default=None, help='Microphone device index')
    cmd.add_argument('--chatty', action='store_true',
                     help='Keep the old constant callouts instead of speaking only when asked')
    ptt = parser.add_argument_group('push-to-talk radio button (wheel or controller)')
    ptt.add_argument('--find-ptt-button', action='store_true',
                     help='Press the wheel/controller button you want as the radio button; saves it to push_to_talk.json and exits')
    ptt.add_argument('--ptt-button', default=None, metavar='BUTTON',
                     help='Radio button: a button number like 5, or a D-pad direction like hat0:up (default: saved choice)')
    ptt.add_argument('--ptt-device', default=None, metavar='DEVICE',
                     help='Controller index or part of its name, e.g. 0 or "T300" (default: saved choice, else any)')
    ptt.add_argument('--ptt-mode', choices=push_to_talk.MODES, default=None,
                     help='hold: talk while held (default). toggle: press to talk, press again to send')
    ptt.add_argument('--no-ptt', action='store_true', help='Ignore the saved push-to-talk button')
    ptt.add_argument('--always-listen', action='store_true',
                     help='Keep the hands-free wake-word microphone on as well as the radio button')
    rec = parser.add_argument_group('recording and replay (format: docs/udp-recording-format.md)')
    rec.add_argument('--record', type=Path, default=None, metavar='FILE',
                     help='Save every incoming UDP datagram with its arrival time to FILE '
                          '(a name ending .gz is compressed)')
    rec.add_argument('--replay', type=Path, default=None, metavar='FILE',
                     help='Read telemetry from a recording instead of the game')
    rec.add_argument('--replay-speed', type=parse_speed, default=1.0, metavar='SPEED',
                     help='Replay speed: 1 (real time, default), 10, or max (as fast as possible)')
    args = parser.parse_args()
    if args.record and args.replay:
        parser.error('--record and --replay cannot be used together')
    if args.find_ptt_button:
        sys.exit(push_to_talk.run_button_finder(args.ptt_mode or push_to_talk.HOLD))
    ptt_settings = None
    if not args.no_ptt and not args.no_mic and not args.no_voice:
        try:
            ptt_settings = push_to_talk.resolve_settings(args.ptt_button, args.ptt_device, args.ptt_mode)
        except ValueError as error:
            parser.error(str(error))

    # ─── Initialize Voice Engine ───
    driver_factory = None
    if args.tts == 'kokoro' and not args.no_voice:
        from neural_voice import DEFAULT_KOKORO_DIR, build_kokoro_driver_factory
        driver_factory = build_kokoro_driver_factory(
            args.kokoro_dir or DEFAULT_KOKORO_DIR, args.kokoro_voice, args.kokoro_speed,
            radio_effect=not args.no_radio_fx)
        if driver_factory:
            print(f'[INFO] Engineer voice: Kokoro ({args.kokoro_voice})')
    voice = VoiceEngine(
        driver_factory=driver_factory,
        rate=args.voice_rate,
        volume=0.9,
        cooldown_seconds=30.0,
        voice_index=args.voice_index,
        on_spoken=add_voice_log,
    )

    if not args.no_voice:
        voice.start()
        time.sleep(0.5)
        voice.say('Radio check. Engineer standing by, waiting for the car.', Priority.LOW, key='startup', ttl=10)
    else:
        print("[INFO] Voice output disabled (--no-voice)")

    # ─── Initialize Analysis Engine ───
    history = TelemetryHistory(args.history_dir / 'telemetry.sqlite3')
    models_path = args.models or args.history_dir / 'fitted_models.json'
    fitted = ModelStore.load_or_empty(models_path)
    if fitted.wear or fitted.fuel:
        print(f"[Models] Loaded {len(fitted.wear)} tyre and {len(fitted.fuel)} fuel fits from {models_path}"
              + (' (simulator data only)' if fitted.simulated_only else ''))
    else:
        print('[Models] No fitted tyre/fuel models yet; using built-in estimates. '
              'Fit them with: python tools/fit_models.py recordings/*.f1rec.gz')
    analysis = AnalysisEngine(voice, history=history, on_demand=not args.chatty, fitted=fitted,
                              race_simulation='off' if args.no_race_sim else 'async',
                              sim_runs=args.sim_runs)
    stream = TelemetryStream(analysis)
    voice.set_silence_filter(analysis.is_driver_busy)

    # ─── AI brain + on-demand voice commands ───
    client = None if args.no_ai else AIClient(args.ai_base_url, args.ai_model,
                                              args.ai_api_key, args.ai_timeout)
    brain = EngineerBrain(analysis.build_context, client)
    dispatcher = CommandDispatcher(voice, brain, on_log=add_voice_log)
    # Typed HUD questions are deliberate, so they may be free-form like the radio button.
    set_command_handler(lambda text: dispatcher.handle_text(text, free_form=True))
    set_setup_ai_client(client)
    set_reset_map_handler(analysis.reset_map)
    listener = None
    radio = None
    if ptt_settings:
        radio = push_to_talk.start(ptt_settings, dispatcher, voice=voice, engine=args.stt_engine,
                                   mic_index=args.mic_index, on_log=add_voice_log)
    if not args.no_mic and not args.no_voice and (radio is None or args.always_listen):
        listener = VoiceCommandListener(dispatcher, engine=args.stt_engine,
                                        device_index=args.mic_index, voice=voice,
                                        ignore_when=radio.recently_active if radio else None)
        listener.start()
    if client:
        print(f"[AI] Engineer brain: {client.model} @ {client.base_url} "
              f"(falls back to rules if unreachable)")

    # ─── Set up UDP Socket (or a recording standing in for it) ───
    recorder = None
    if args.replay:
        try:
            sock = ReplaySource(args.replay, args.replay_speed)
        except (OSError, ValueError) as e:
            print(f"[ERROR] Cannot open recording {args.replay}: {e}")
            voice.shutdown()
            history.close()
            sys.exit(1)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)  # Non-blocking with 1s timeout for clean shutdown

        try:
            sock.bind(("0.0.0.0", args.port))
        except OSError as e:
            print(f"[ERROR] Cannot bind to port {args.port}: {e}")
            print("  Is another application using this port?")
            voice.shutdown()
            history.close()
            sock.close()
            sys.exit(1)
        if args.record:
            try:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                recorder = RecordingWriter(args.record, metadata={'port': args.port})
            except OSError as e:  # includes FileExistsError: never overwrite a recording
                print(f"[ERROR] Cannot create recording {args.record}: {e}")
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
    web_server = start_web_hud_server(port=5000)
    try:
        webbrowser.open("http://localhost:5000")
    except Exception:
        pass

    print_telemetry_banner()
    if args.replay:
        speed = 'max speed' if args.replay_speed is None else f'{args.replay_speed:g}x'
        print(f"[REPLAY] Reading telemetry from {args.replay} at {speed}")
    else:
        print(f"[OK] Bound to UDP port {args.port}")
    if recorder:
        print(f"[REC] Recording every datagram to {args.record}")
    print("[HUD] [*] Live Telemetry Dashboard running at: http://localhost:5000")
    print("[STATUS] [*] Waiting for F1 2021 telemetry data...")
    print("         (Launch F1 2021, start Time Trial/Grand Prix/Race, and make sure UDP Telemetry is ON)")
    print()

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
                print(f"\n\n[REPLAY] {done}")
                break
            except socket.timeout:
                stream.check_timeout()
                if not first_packet_logged:
                    now = time.time()
                    if now - last_wait_tick >= 0.5:
                        last_wait_tick = now
                        spin = spinner_chars[spinner_idx % len(spinner_chars)]
                        spinner_idx += 1
                        sys.stdout.write(f"\r  [{spin}] Listening on 0.0.0.0:{args.port}... waiting for car on track...")
                        sys.stdout.flush()
                continue
            except ConnectionResetError:
                continue

            packets_received += 1
            if recorder:
                recorder.write(data)
            if not stream.handle(data):
                if args.debug and stream.last_error:
                    print(f'[DEBUG] Ignored packet: {stream.last_error}')
                continue
            latest_session = stream.latest_session
            latest_telemetry = stream.latest_telemetry
            latest_lap = stream.latest_lap

            if not first_packet_logged:
                first_packet_logged = True
                sys.stdout.write("\r" + " " * 75 + "\r")
                print('[CONNECTED] >> Valid F1 2021 telemetry stream detected!')
                print()
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
                            analysis.delta.last_reported_delta_ms
                        )
                    except Exception:
                        pass  # Don't crash on display errors

    except KeyboardInterrupt:
        print("\n\n[INFO] Shutting down...")
        voice.say_immediate("That's it from us. Good work today, see you in the garage.")
        time.sleep(2)
    finally:
        sock.close()
        if recorder:
            recorder.close()
            print(f"[REC] Saved {recorder.count} datagrams to {args.record}")
        if listener:
            listener.stop()
        if radio:
            radio.stop()
        dispatcher.shutdown()
        history.close()
        if web_server:
            web_server.shutdown()
            web_server.server_close()
        voice.shutdown()
        print(f"[INFO] Received {packets_received} packets total.")
        print("[INFO] AI Race Engineer offline.")


if __name__ == "__main__":
    main()
