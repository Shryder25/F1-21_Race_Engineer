# F1-21 Race Engineer (AI-Powered)

An intelligent, real-time virtual race engineer for the F1 2021 game. This project ingests live UDP telemetry data from the game to act as a highly realistic pit-wall strategist. Moving beyond simple static scripts, it leverages machine learning and Large Language Models (LLMs) to analyze lap data, predict pit windows, and communicate through an authentic Formula 1 radio protocol.

---

## 🧠 AI Engineer, Learning Engine & Voice Commands

**Learning engine.** Every completed lap is logged to the local SQLite database (`telemetry-history/telemetry.sqlite3`, table `lap_stats`): tyre wear per tyre, tyre compound and age, fuel at lap start/end, sector times and sector deltas to your personal best. Tyre wear and fuel come from the game's Car Status and Car Damage UDP packets.

**Tyre and fuel models fitted to the game, with ranges (`ai_engineer/fitted_models.py`, `strategy_ml.py`).** Tyre wear, tyre fall-off and fuel burn are fitted per actual compound (C1-C5, inter, wet) and per track from your recorded stints (`python tools/fit_models.py recordings/*.f1rec.gz`), and every prediction is a range: *box around lap 19, window lap 17 to 21*; *fuel margin +0.6 laps (-0.4 to +1.5)*. A track with no data falls back to the compound pooled over other tracks (scaled by tyre severity), then to a wide built-in estimate; live, the model narrows with this stint's own laps and earlier sessions in the history database. Race strategy is re-planned at both ends of the range to show whether the stop count holds. Details, fallbacks and the file format: [`docs/fitted-models.md`](docs/fitted-models.md).

**Monte Carlo race simulator (`ai_engineer/race_sim.py`).** Once a lap in a race, the engineer plays the rest of the race 300 times for the whole field, lap by lap, for each stop plan the strategy model offers: Safety Car and VSC chances per track (lap 1 weighted up), a pit-loss distribution with the odd slow stop, dirty air, being held up and track-dependent overtaking, rivals' own stop plans, and tyre fall-off/fuel drawn from the fitted models' ranges. Every plan sees the same random races, so the comparison is the plan, not the dice. The result is odds: *"Two-stop comes out best 60% of the time, one-stop 30%. Expected finish P4, P3 to P6."* It is in the strategy voice answer ("what are our chances?"), the HUD strategy card and the AI prompt. Off with `--no-race-sim`; `--sim-runs N` sets the races per update. Model, constants and what real recordings would sharpen: [`docs/race-simulator.md`](docs/race-simulator.md).

**Physics tyre/fuel/threat model (`ai_engineer/physics_models.py`).** Alongside the learned wear curve, every frame is also scored by a physics model adapted from the [Box-Box](https://github.com/ploverbyHB/Box-Box) race engineer project: a thermal grip curve per compound (ideal temperature window, overheat/cold penalties), a wear-cliff and competitive-loss budget that gives a 0-100 tyre health score and remaining-laps estimate from the very first lap (no history needed, unlike the learned curve), a fuel-margin check, and an opponent-threat score for nearby cars (gap, DRS range, pace delta, laps to catch) built from the game's per-car lap and car-status/damage packets. Ask *"any threats"* / *"who's attacking me"* for the nearest one.

**DeepSeek brain (`ai_client.py`, `engineer_brain.py`).** Live telemetry, the ML predictions and the physics model are placed in a system prompt that demands short F1-radio replies, and the model is asked for structured JSON (`{"voice": "...", "advice": "..."}`, also following Box-Box's approach) rather than trusting it to self-format free text; a server that cannot do JSON mode falls back to plain text automatically. Any OpenAI-compatible endpoint works:

| Provider | Flags |
|---|---|
| Ollama (default, local) | `ollama pull deepseek-r1:8b`, then just run the engineer |
| Groq | `--ai-base-url https://api.groq.com/openai/v1 --ai-model deepseek-r1-distill-llama-70b --ai-api-key <key>` |
| DeepSeek cloud | `--ai-base-url https://api.deepseek.com/v1 --ai-model deepseek-chat --ai-api-key <key>` |

The same settings can come from `F1_AI_BASE_URL`, `F1_AI_MODEL` and `F1_AI_API_KEY`. If the model is offline or slower than `--ai-timeout` (15 s), the engineer answers from the same data using built-in rules, so a command is never ignored. `--no-ai` forces rules only.

**Track Database & DeepSeek track analyst (`track_analysis.py`, `track_profiles.py`).** The HUD's Track Database tab no longer just echoes stored values. Each of the 24 circuits has a real-race profile (Pirelli severity ratings for abrasion, traction, braking, lateral load and tyre stress, plus real race history: pit-lane loss, Safety Car and rain frequency, track temperature, overtaking difficulty). From that the analysis model computes, for the chosen conditions and race distance (25/50/100%):
- tyre degradation per compound in seconds per lap of tyre age and laps before the cliff, with a Low / Medium / High / Very High verdict that differs by track (Monaco and Baku Low, Silverstone, Suzuka and Bahrain Very High) and by weather (inters and full wets in the rain);
- the fastest pit strategy and the best alternative with a different stop count (two dry compounds enforced, track position valued where passing is hard);
- a setup that really changes in the wet (more wing, open diff, softer car, higher ride, lower brake pressure, rearward bias) and with the handling preference.

DeepSeek then analyses the circuit automatically using all of that data and returns its own degradation verdict, strategy, setup numbers (validated and clamped to the game's ranges), per-category setup notes, key corners and a radio briefing. Answers are cached per track/conditions; "Re-analyse with DeepSeek" asks again. If DeepSeek is offline the page shows the data model's answer and says so. The track analyst uses the same `--ai-*` settings with a 90 s timeout, since a full analysis is longer than a radio reply.

*Calibrating from real races.* `tools/build_real_race_calibration.py` loads real Grands Prix through [FastF1](https://github.com/theOehrly/Fast-F1), the data source [f1-race-replay](https://github.com/IAmTomShaw/f1-race-replay) replays, and measures fuel-corrected tyre degradation per compound (Theil-Sen slope per stint, the method of f1-race-replay's Bayesian tyre model), stint lengths, pit loss, stop counts, Safety Car and rain frequency. It writes `ai_engineer/data/real_race_calibration.json`, which then overrides the baseline ratings for those tracks:
```
pip install fastf1
python tools/build_real_race_calibration.py --years 2021 2022 2023
```

**Voice commands (`voice_commands.py`).** The engineer no longer chats constantly. Ask for what you need: *"weather report"*, *"tyre temps"*, *"pit window"*, *"gap to car ahead"*, *"gap to car behind"*, *"pace update"*, *"any threats"* (optionally prefixed with "engineer"). The moment a command is recognised the engineer says **"We are checking"** while the AI works in the background, then speaks the answer. Only critical tyre overheating still interrupts on its own — and only when the temperature is genuinely climbing fast (rate-of-change, not a single static-threshold reading), with a 40-second cooldown so one real overheat event doesn't repeat every frame while the tyre stays hot; `--chatty` restores the old callouts.

- Speech-to-text: `pip install SpeechRecognition PyAudio`. Default engine is Google (online); `--stt-engine vosk` runs offline (`pip install vosk`, unpack a Vosk model to `./model`). `--mic-index N` picks a microphone, `--no-mic` disables it.
- No microphone? Send text to the engineer: `curl -X POST localhost:5000/api/command -d "{\"text\": \"pit window\"}"`.
- Test without the game: `test_simulation.bat` now also sends tyre wear and fuel packets.

---

## 🎞️ Recording & Replay (for backtesting)

Every UDP datagram the game sends can be saved with its arrival time and played back later, so a session can be re-run through the engineer without the game (format spec: [`docs/udp-recording-format.md`](docs/udp-recording-format.md)).

```bash
python main.py --record recordings/silverstone.f1rec.gz        # drive as usual, also record
python ai_engineer/record_udp.py                               # record only (no engineer running)
python main.py --replay recordings/silverstone.f1rec.gz --replay-speed 10   # 1, 10 or max
python ai_engineer/replay_udp.py recordings/silverstone.f1rec.gz --speed 1  # send over UDP to a running engineer
python ai_engineer/replay_udp.py recordings/silverstone.f1rec.gz --info     # what is in the file
```

`1` replays in real time with the recorded gaps between datagrams, `10` ten times faster, `max` without waiting. `main.py --replay` never drops datagrams; UDP replay at `max` can, if the receiver falls behind. In code, `udp_recording.replay_into(stream, path)` feeds a `TelemetryStream` directly using the recorded clock.

### Golden-session regression tests

`tests/golden/sessions/` holds recorded sessions (race, qualifying, wet, Safety Car) with the radio calls, voice-command answers and key numbers the engineer produced on them. The test suite replays them and fails if behaviour changes. Four sessions come from a deterministic full-field simulator; four more slots wait for real recordings from the game. See [`tests/golden/README.md`](tests/golden/README.md) for how to record one and how to accept an intended change (`python tools/golden_sessions.py bless NAME`).

### Backtest report: predictions vs what happened

`tools/backtest.py` replays recorded races through the engineer and scores its predictions against what the recording shows really happened, per race and across all races (mean and median absolute error, bias, worst miss):

* **Pit lap**: the next stop predicted by the ML wear model (60% and 75% wear laps) and the race-strategy plan, plus the game's own pit window as a baseline, vs the lap you pitted on; also how often each said "no stop" when you did pit, or called a stop that never came.
* **Tyre wear**: worst-tyre wear the fitted wear model predicts 1, 3 and 5 laps ahead vs the wear the game reported, and how often the truth landed inside the model's 80% range.
* **Lap time**: the engineer's race pace vs the next lap (excluding lap 1, in/out laps and Safety Car laps).
* **Gaps**: the live gap to the cars ahead and behind vs the true gap, and the gap 1 and 3 laps later forecast from the last-lap pace difference.

```bash
python tools/backtest.py recordings/*.f1rec.gz --out reports/backtest.md --json reports/backtest.json
python tools/backtest.py --simulated          # built-in simulated races only (pipeline check)
```

### Fitting the tyre and fuel models

```bash
python tools/fit_models.py recordings/*.f1rec.gz --history telemetry-history/telemetry.sqlite3 --report reports/models.md
python tools/fit_models.py --simulated        # noisy simulator stints with known answers (pipeline check)
```

Saves `telemetry-history/fitted_models.json`, which `main.py` loads at start-up (`--models FILE` for another). The report lists every fit with its range and a hold-out check of how often the ranges catch the truth.

Simulated races (the golden sessions plus two longer races generated into `recordings/simulated/`) only show the pipeline works; real error numbers need your own recordings (`python main.py --record ...`). Simple baselines (previous lap time, "gap stays the same", last lap's wear rate) are scored alongside, so the report shows whether the engineer beats a rule of thumb.

---

## 🚀 Core Features

* **DeepSeek AI Integration:** Uses local or cloud-based DeepSeek LLMs to process complex telemetry statistics and translate them into natural, strategic pit-wall advice.
* **Predictive Machine Learning:** Logs session data into a local SQLite database (`history.py`) and utilizes `scikit-learn` to calculate predictive tire degradation curves and optimal pit windows[cite: 1].
* **Voice-Activated Radio Command:** Trigger the engineer using voice commands (e.g., "Gap to car behind", "Weather report"). Includes a real-time "We are checking" latency audio cue while the AI processes the response.
* **Realistic F1 Radio Protocol:** Enforces strict communication rules. The engineer will not interrupt during heavy braking or high-G cornering, matching real-world F1 etiquette.
* **Live Track Mapping & Telemetry Visualizer:** Translates X and Z axis telemetry into a live track map on the HUD, recorded continuously from the first metre of lap 1 (not only after a lap completes). A "Reset Map" button clears the recorded trail on the server as well as on screen, so a spin or a dirty out-lap doesn't stay on the map for the rest of the session.
* **Race Report Table:** A lap-by-lap table (lap time, sector 1/2/3, speed trap, tyre compound) on its own HUD tab, built from the same per-lap stats the learning engine already records plus the game's Speed Trap ('SPTP') event packets.
* **Comprehensive Setup Advisor:** Includes a track database (`track_data.py`) covering all 24 circuits in F1 2021[cite: 1]. Suggests aerodynamic, transmission, and suspension setups customized to specific tracks.
* **F1-TV Style Web HUD:** A professional, dark-mode web dashboard (`web_hud.py`) displaying live timing towers, color-coded sector deltas, tire wear percentages, and individual tire temperatures[cite: 1].

---

## 📂 Repository Architecture

The project is structured into distinct modules to handle stream ingestion, data analysis, and user interfaces[cite: 1].

* `main.py` — The core execution loop initializing the telemetry listener and AI services[cite: 1].
* `run_engineer.bat` & `test_simulation.bat` — Windows batch scripts for quick launching and testing the application[cite: 1].
* **`ai_engineer/`**
  * `analysis.py` — Contains data analytics and machine learning logic for predicting race strategies[cite: 1].
  * `history.py` — Manages historical race logs and database interactions[cite: 1].
  * `packets.py` & `telemetry_stream.py` — Handle decoding and real-time processing of F1 2021 UDP binary packets[cite: 1].
  * `race_radio.py` — Manages speech-to-text input, F1 radio timing logic, and text-to-speech output[cite: 1].
  * `simulate_telemetry.py` — Enables offline testing by simulating live telemetry without needing the game running[cite: 1].
  * `track_data.py` — Stores track-specific metadata, DRS zones, and AI setup recommendations[cite: 1].
  * `voice_engine.py` — Core voice interface engine[cite: 1].
  * `web_hud.py` — The backend service serving the local F1-TV style web dashboard[cite: 1].
* **`tests/`**
  * Contains unit and integration tests (`test_analysis.py`, `test_hud.py`, `test_race_radio.py`, `test_stream.py`, etc.) to ensure system stability[cite: 1].
  * Includes frontend UI tests like `hud_client_test.js`[cite: 1].

---

## ⚙️ Installation Prerequisites

1. **Python 3.10+** installed on your system.
2. **UDP Telemetry Configuration**: In the F1 2021 game settings, enable UDP Telemetry, set the port to `20777`, set the send rate to 20Hz, and choose the 2021 format.
3. **Local LLM Setup (Optional but Recommended)**: Install Ollama and pull a DeepSeek model for zero-latency, offline AI processing:
   ```bash
   ollama run deepseek-coder-v2-lite

   🛠️ Setup Instructions
Clone the repository and navigate to the project directory:

Bash
git clone [https://github.com/Shryder25/F1-21_Race_Engineer.git](https://github.com/Shryder25/F1-21_Race_Engineer.git)
cd F1-21_Race_Engineer
Install the required dependencies:

Bash
pip install -r ai_engineer/requirements.txt
pip install scikit-learn pandas SpeechRecognition pyttsx3 sqlite3
Run the application using the provided batch file:

Bash
run_engineer.bat
Open the local Web HUD in your browser to monitor telemetry visually.

🎧 Usage Guide
During a Session: The application listens silently to your telemetry, speaking only when a critical event occurs (e.g., sudden weather change, critical tire wear) or when prompted.

Voice Commands: Press your mapped radio button or use the wake word followed by a command:

"Engineer, what is my gap to the car behind?"

"Give me a pace update."

"What is the pit window for these softs?"

HUD Interaction: Access the HUD on a secondary monitor or mobile device to view the Live Track Map, sector timings, and access the Track Setup Advisor before hitting the track.
