# F1-21 Race Engineer

An AI race engineer for the F1 2021 game. It reads the game's live UDP telemetry, works out tyre, fuel and race strategy, and talks to you over a team-radio voice the way a real pit wall would: short calls at the right moment, and answers when you ask.

## Features

- **Live telemetry** for all 22 cars: laps, sectors, gaps, tyres, fuel, damage, flags and weather.
- **DeepSeek AI engineer** running locally through Ollama, with rule-based answers as a fallback so a question is never ignored.
- **Kokoro voice** (offline neural TTS with a team-radio filter), falling back to the system voice.
- **Offline Whisper speech-to-text** for questions in your own words.
- **Push-to-talk on your wheel** via the game's own button feed, or any controller button.
- **Tyre, fuel and strategy models** fitted to your own recorded laps, with ranges (box lap window, fuel margin).
- **Monte Carlo race simulator** that plays out the rest of the race for each plan and gives the odds, with live Plan A / Plan B re-planning.
- **Web HUD** at `http://localhost:5000`: timing, track map, strategy, race report and the PIT WALL tab.
- **MoTeC i2 export** from a Parquet channel log of about 80 channels.

## Requirements

- Python 3.11+
- F1 2021 (PC)
- [Ollama](https://ollama.com/download), installed once. The engineer starts it and pulls the model for you.
- Python packages:

  ```bash
  pip install -r ai_engineer/requirements.txt
  ```

## Quick start

1. In F1 2021, open **Settings → Telemetry Settings** and set:
   - UDP Telemetry: **On**
   - UDP IP Address: **127.0.0.1**
   - UDP Port: **20777**
   - UDP Format: **2021**
2. Start the engineer:

   ```bash
   python main.py
   ```

   Or with recording, channel logging and all radio calls on:

   ```bash
   python main.py --record recordings/race1.f1rec.gz --log-channels channels/ --tts kokoro --chatty --calls-per-lap 6 --ai-model qwen2.5:3b
   ```

3. Wait for **"Pit wall is connected"**, then go on track. The HUD opens in your browser.

**First run** downloads a few things once: the Ollama model (about 2 GB for `qwen2.5:3b`, about 5 GB for the default `deepseek-r1:8b`), the Whisper speech model (about 140 MB) and the Kokoro voice (about 350 MB).

| Flag | What it does |
|---|---|
| `--record FILE` | Save every UDP packet for replay and backtesting |
| `--log-channels DIR` | Log channel data to Parquet (for MoTeC export) |
| `--tts kokoro` | Kokoro voice (default; `system` for the OS voice) |
| *(default)* | Urgent calls (Safety Car, flags, damage, box/strategy) plus one end-of-lap pace call: lap time, where you lost time (sector and corner, with why) and race pace. No tyre temperature warnings |
| `--chatty` | Adds race info: gaps, sector times, and one call a lap on threats, rival tyres and stops, or Safety Car chance. No braking tips (`--braking-calls` turns them back on) |
| `--calls-per-lap N` | Max proactive callouts per lap (default 4, 0 = no limit) |
| `--race-calls-per-lap N` | Tighter limit in a race (default 2) |
| `--ai-model NAME` | Ollama model; `qwen2.5:3b` is lighter if you get frame drops |
| `--ai-test` | Check the AI without the game: asks one question and prints the reply, the time it took, or why it failed |

`python main.py --help` lists every option.
`ollama rm deepseek-r1:8b` run this command if you have installed deepseek r1:8b to uninstall it and then run `--ai-model qwen2.5:3b` in main.py to install the new model. Also use `--ai-cpu` to keep the AI of your GPU completely to reduce frame drops.

## Wheel button setup

Run this once and press the button you want to use for the radio:

```bash
python main.py --find-game-button
```

After that, hold the button, ask your question, and let go. To use a controller button read through pygame instead, run `python main.py --find-ptt-button`. More in [docs/push-to-talk.md](docs/push-to-talk.md).

## Voice commands

Ask in your own words; these nine are answered instantly from live data:

| Command | Example |
|---|---|
| Weather | "weather report" |
| Tyre temps | "how are the tyre temps?" |
| Pit window | "when's the pit window?" |
| Gap ahead | "gap to the car ahead" |
| Gap behind | "gap to the car behind" |
| Pace | "pace update" |
| Threats | "any threats?" |
| Strategy | "what's the strategy?", "what are our chances?" |
| Rival stops | "when will Hamilton box?" |

Anything else goes to the AI.

## Key commands

```bash
# Record a session while you drive
python main.py --record recordings/race1.f1rec.gz

# Replay it through the engineer without the game (1, 10 or max)
python main.py --replay recordings/race1.f1rec.gz --replay-speed 10

# Score predictions against what happened
python tools/backtest.py recordings/*.f1rec.gz

# Fit tyre and fuel models to your recordings
python tools/fit_models.py recordings/*.f1rec.gz

# Export the newest channel log to MoTeC i2 (.ld + .ldx)
python main.py --export-motec

# Pick a push-to-talk button (game feed or controller)
python main.py --find-game-button
python main.py --find-ptt-button
```

## HUD

Open `http://localhost:5000` (it opens automatically). The **PIT WALL** tab has the full-field timing tower, gap chart, tyre stints and the simulator's win / podium / points odds. Hover the status light: **Live push** means the HUD is getting data over a WebSocket; **Polling** means the socket dropped and it is reconnecting on its own.

## Architecture

Entry point: `main.py`, which runs `ai_engineer/main.py`.

| Module (`ai_engineer/`) | Role |
|---|---|
| `packets.py` | Parses the F1 2021 UDP packet formats. |
| `telemetry_stream.py` | Validates packets and pairs them per game frame. |
| `analysis.py` | The engine (`AnalysisEngine`): wires the pipeline stages together and runs them on every frame. |
| `pipeline/` | The engine's stages, each only calling the ones below it: `state` (typed per-session values), `radio` (when to speak, tyre and braking calls), `models` (wear, degradation, fuel), `strategy` (plan, undercut, rival stops, race sim), `track_map`, `outputs` (HUD, history, AI prompt), `laps` (delta, sectors, corners), `ingest` (one handler per packet) and `config` (loads `config.toml`). |
| `log_setup.py` | Logging: readable terminal lines or JSON lines. |
| `physics_models.py` | Instant physics tyre, fuel and opponent-threat model (no history needed). |
| `strategy_ml.py`, `fitted_models.py` | Tyre wear and fuel models learned from recorded laps. |
| `race_strategy.py` | Stop count, compounds, undercut and overcut. |
| `rival_strategy.py` | Predicts each rival's next stop lap and tyre. |
| `race_sim.py` | Monte Carlo race simulator. |
| `strategy_replan.py`, `track_wetness.py` | Live Plan A / Plan B re-planning and the wet/drying track model. |
| `race_radio.py`, `radio_discipline.py` | Gap, pace and threat tracking, and when the engineer is allowed to speak. |
| `engineer_brain.py`, `ai_client.py`, `ai_launcher.py` | Builds AI prompts, talks to Ollama or a cloud API, and auto-starts Ollama. |
| `voice_commands.py`, `intents.py` | Matches spoken questions to commands and answers them. |
| `speech_to_text.py`, `push_to_talk.py`, `game_buttons.py` | Whisper STT and push-to-talk from the wheel or controller. |
| `voice_engine.py`, `neural_voice.py`, `radio_fx.py` | Text-to-speech (Kokoro or system) and the radio filter. |
| `web_hud.py`, `pit_wall.py`, `templates/`, `static/` | The web HUD and pit wall timing. |
| `history.py` | SQLite store of sessions, laps and stints. |
| `channel_store.py`, `motec.py` | Parquet channel log and MoTeC i2 export. |
| `udp_recording.py` | Record and replay of raw UDP sessions. |
| `track_data.py`, `track_profiles.py`, `track_analysis.py` | 24-circuit database, real-race track profiles and setup advice. |

Tools in `tools/`: `backtest.py`, `fit_models.py`, `channels.py`, `golden_sessions.py`, `build_real_race_calibration.py`, `openf1_radio.py` (real F1 radio + race situation dataset for training the engineer model; run it on your PC, F1's audio server blocks Colab). `training/make_sft.py` + `training/finetune_engineer_colab.ipynb` fine-tune Qwen 2.5 1.5B on that dataset (free Colab T4) into an Ollama model, `--ai-model f1-engineer`. Once `data/openf1/examples_train.jsonl` exists, the AI is shown 3-4 real engineer lines from similar moments with every answer (`--no-radio-examples` turns it off).

## Settings and logs

- **Settings**: `config.toml` at the repo root holds every tuneable threshold (tyre temperatures, calls per lap, when the driver counts as busy, rival-call gaps, simulator runs, voice, AI timeout, ports), each one commented. The shipped values are the defaults. Command-line flags override the file; `--config FILE` uses another one. A misspelt key is an error at startup.
- **Logs**: `--log-level DEBUG|INFO|WARNING|ERROR` (`--debug` = DEBUG), `--log-format json` for one JSON object per line, `--log-file FILE` to also keep them as JSON lines. Logs go to stderr; the live console dashboard stays on stdout.
- **As a package**: `pip install -e ".[dev]"` gives an `f1-race-engineer` command (same flags as `python main.py`) plus ruff and mypy.

## Tests and checks

All of these run in CI as part of the `tests` check:

```bash
ruff check .            # lint
ruff format .           # format (CI runs ruff format --check)
mypy                    # strict type check of ai_engineer/pipeline
cd tests && python -m unittest discover -v
```

## More docs

- [docs/features.md](docs/features.md): detailed notes on every feature
- [docs/push-to-talk.md](docs/push-to-talk.md), [docs/ai-performance.md](docs/ai-performance.md)
- [docs/fitted-models.md](docs/fitted-models.md), [docs/race-simulator.md](docs/race-simulator.md)
- [docs/channel-store.md](docs/channel-store.md), [docs/udp-recording-format.md](docs/udp-recording-format.md)
