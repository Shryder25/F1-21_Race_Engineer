"""
F1 2021 AI Race Engineer — Main Entry Point
=============================================
UDP telemetry listener that feeds data into the analysis engine
and delivers feedback through the voice engine.

Usage:
    python main.py [--port 20777] [--no-voice] [--debug]

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
from telemetry_stream import TelemetryStream
from web_hud import start_web_hud_server, add_voice_log, set_command_handler, set_setup_ai_client
from ai_client import AIClient
from engineer_brain import EngineerBrain
from voice_commands import CommandDispatcher, VoiceCommandListener
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
    args = parser.parse_args()

    # ─── Initialize Voice Engine ───
    voice = VoiceEngine(
        rate=args.voice_rate,
        volume=0.9,
        cooldown_seconds=30.0,
        voice_index=args.voice_index,
        on_spoken=add_voice_log,
    )

    if not args.no_voice:
        voice.start()
        time.sleep(0.5)
        voice.say('Engine online. Waiting for telemetry.', Priority.LOW, key='startup', ttl=10)
    else:
        print("[INFO] Voice output disabled (--no-voice)")

    # ─── Initialize Analysis Engine ───
    history = TelemetryHistory(args.history_dir / 'telemetry.sqlite3')
    analysis = AnalysisEngine(voice, history=history, on_demand=not args.chatty)
    stream = TelemetryStream(analysis)
    voice.set_silence_filter(analysis.is_driver_busy)

    # ─── AI brain + on-demand voice commands ───
    client = None if args.no_ai else AIClient(args.ai_base_url, args.ai_model,
                                              args.ai_api_key, args.ai_timeout)
    brain = EngineerBrain(analysis.build_context, client)
    dispatcher = CommandDispatcher(voice, brain, on_log=add_voice_log)
    set_command_handler(dispatcher.handle_text)
    set_setup_ai_client(client)
    listener = None
    if not args.no_mic and not args.no_voice:
        listener = VoiceCommandListener(dispatcher, engine=args.stt_engine,
                                        device_index=args.mic_index, voice=voice)
        listener.start()
    if client:
        print(f"[AI] Engineer brain: {client.model} @ {client.base_url} "
              f"(falls back to rules if unreachable)")

    # ─── Set up UDP Socket ───
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
    print(f"[OK] Bound to UDP port {args.port}")
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
                data, addr = sock.recvfrom(2048)
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
        voice.say_immediate("Engineer signing off. Good session.")
        time.sleep(2)
    finally:
        sock.close()
        if listener:
            listener.stop()
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
