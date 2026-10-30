# Feature reference

Detailed notes on every part of the engineer. For setup and the quick start, see the [README](../README.md).


---

## 🧠 AI Engineer, Learning Engine & Voice Commands

**Learning engine.** Every completed lap is logged to the local SQLite database (`telemetry-history/telemetry.sqlite3`, table `lap_stats`): tyre wear per tyre, tyre compound and age, fuel at lap start/end, sector times and sector deltas to your personal best. Tyre wear and fuel come from the game's Car Status and Car Damage UDP packets.

**Every car's stints and lap times.** The Session History packet (id 11) gives every car's lap times, sectors and lap validity, and every set of tyres with the lap it came off on. They are logged to the same database (tables `car_laps` and `car_stints`: compound, first and last lap, laps on the set, in- and out-laps), along with the Final Classification at the end of the session (`final_classification`: position, grid, stops, best lap, compounds used) and each car's visible setup from the Car Setups packet (`car_setups`, as JSON). `tools/fit_models.py` uses other cars' clean laps for the tyre fall-off fit (`--player-only` to turn that off), and `tools/backtest.py` lists every car's stints per race and checks their pit laps against the lap data. Tyre age for other cars is counted from the start of each stint, as if every set were new.

**Tyre and fuel models fitted to the game, with ranges (`ai_engineer/fitted_models.py`, `strategy_ml.py`).** Tyre wear, tyre fall-off and fuel burn are fitted per actual compound (C1-C5, inter, wet) and per track from your recorded stints (`python tools/fit_models.py recordings/*.f1rec.gz`), and every prediction is a range: *box around lap 19, window lap 17 to 21*; *fuel margin +0.6 laps (-0.4 to +1.5)*. A track with no data falls back to the compound pooled over other tracks (scaled by tyre severity), then to a wide built-in estimate; live, the model narrows with this stint's own laps and earlier sessions in the history database. Race strategy is re-planned at both ends of the range to show whether the stop count holds. Details, fallbacks and the file format: [`docs/fitted-models.md`](docs/fitted-models.md).

**Rival strategy model (`ai_engineer/rival_strategy.py`).** For every rival the engineer keeps a running prediction of their next stop lap (with an 80% window) and which tyre they will fit, from their tyre age, their own measured tyre wear (when the game shares it), the track's typical strategy and the game's pit window, and it learns from the race itself: the wear at which rivals have been boxing, how much longer or shorter they run than planned, and which tyres they have been fitting. Rain brings a slick runner's stop forward onto inters. The predictions feed strategy calls before a rival boxes, not just after: the undercut maths counts the laps the car ahead is expected to stay out, a car behind that is about to box with a working undercut gets a pre-emptive *"box this lap to stay ahead"*, and a car alongside that is due gets a heads-up (*"Hamilton ahead likely to box lap 22 for hards"*). Ask *"when will Hamilton box"* or *"what are the others doing"*; the HUD strategy card lists nearby rivals' predicted stops. The race simulator can sample from the same predictions (`sample_stop_lap`, `sample_next_compound`).

**Monte Carlo race simulator (`ai_engineer/race_sim.py`).** Once a lap in a race, the engineer plays the rest of the race 300 times for the whole field, lap by lap, for each stop plan the strategy model offers: Safety Car and VSC chances per track (lap 1 weighted up), a pit-loss distribution with the odd slow stop, dirty air, being held up and track-dependent overtaking, rivals' own stop plans, and tyre fall-off/fuel drawn from the fitted models' ranges. Every plan sees the same random races, so the comparison is the plan, not the dice. The result is odds: *"Two-stop comes out best 60% of the time, one-stop 30%. Expected finish P4, P3 to P6."* It is in the strategy voice answer ("what are our chances?"), the HUD strategy card and the AI prompt. Off with `--no-race-sim`; `--sim-runs N` sets the races per update. Model, constants and what real recordings would sharpen: [`docs/race-simulator.md`](docs/race-simulator.md).

**Live re-planning, Plan A / Plan B (`ai_engineer/strategy_replan.py`).** The simulator re-runs every lap and straight away on a Safety Car, VSC, rain or a nearby rival's stop, always against the plan we are actually on (Plan A). The engineer only calls *"Plan B, Plan B. Box this lap for hards."* when the new plan beats Plan A by a set margin (3 s or half a place on average, ahead in 60% of the simulated races; `--plan-b-margin`, `--plan-b-places`), and on routine laps only after two re-runs agree, so it does not flip-flop. The call plays in quiet mode too and shows on the HUD strategy card. Wet races use a track-wetness model (`ai_engineer/track_wetness.py`): rain soaks the track, it dries with track temperature and sun, the game's own weather forecast sets the rain, and the field's lap times on slicks vs inters vs wets keep the estimate honest. The simulator then finds the right lap to switch tyres (*"Track's ready for slicks. Plan B, Plan B. Box this lap for mediums."*), and the weather answer and HUD say how wet the track is and when slicks will be quicker.

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

- Speech-to-text: `pip install SpeechRecognition PyAudio openai-whisper`. Default engine is offline Whisper (`--whisper-model base.en`, falls back to Google if Whisper isn't installed); `--stt-engine google` or `--stt-engine vosk` pick the others. Commands are matched by intent, so natural phrasing works ("how are my tyres holding up?"). `--mic-index N` picks a microphone, `--no-mic` disables it.
- **Race radio is not qualifying radio.** In a race the engineer gives gaps either side with their trend once a lap at the line, race pace every 3 laps (lap 1 and in/out laps never count), and the plan. No "X off your best" lap times, sector deltas or live deltas: that stays in practice and qualifying. Corner coaching in a race only comes when the same corner loses time three laps running (beyond normal tyre drift), at most every 3 laps. Lap 1, a DRS fight (a car within 1 s either side) and the last two laps are quiet: urgent calls only. The Safety Car / VSC call carries the box-or-stay-out decision in the same line. DRS calls are information (gap, who, their tyres), not driving instructions.
- **Radio discipline.** At most 4 proactive callouts per lap, 2 in a race (`--calls-per-lap N`, `--race-calls-per-lap N`, 0 = no limit); urgent calls (Safety Car, flags, penalties, damage, box/pit window, strategy, a car attacking with DRS) always get through. Nothing except critical calls is said in a high-load corner (`--corner-speed 200 --corner-g 3.5`) or under heavy braking; held calls are said on the next straight.
- **Push-to-talk on your wheel or controller (`push_to_talk.py`).** Run `python main.py --find-ptt-button` once and press the button you want; after that, hold it, ask anything in your own words ("can I make it to the end on these?"), let go, and the engineer answers from the live data. Or read the wheel button from the game's own UDP button feed with `python main.py --find-game-button` (no pygame needed). Setup and button tips: [docs/push-to-talk.md](docs/push-to-talk.md).
- No microphone? Send text to the engineer: `curl -X POST localhost:5000/api/command -d "{\"text\": \"pit window\"}"`. Typed text can be any question, like the radio button.
- Test without the game: `test_simulation.bat` now also sends tyre wear and fuel packets.

**Fast answers without frame drops** ([docs/ai-performance.md](docs/ai-performance.md)). The 9 fixed questions (and short versions of them asked over push-to-talk) are answered instantly from the live data. Everything else goes to the AI with only the data that question is about, the reply is streamed and spoken as soon as the radio line is complete, and Ollama is started below the game's priority. If you still get frame drops, try `--ai-model qwen2.5:3b` (a much lighter model) or `--ai-cpu` (keeps the AI off the graphics card).

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

## 📈 Channel data store & MoTeC i2 export

Every telemetry sample of your car (about 80 channels: speed, throttle, brake, steering, gear, RPM, tyre and brake temperatures, tyre pressures, suspension, wheel speed and slip, G-forces, fuel, ERS, tyre wear and more) can be logged to a Parquet file, one row per sample with the lap number and lap distance on every row, so laps can be overlaid metre by metre. The same file exports to MoTeC i2 (`.ld` log + `.ldx` lap markers), or to a MoTeC-style CSV. Needs `pyarrow` (in `ai_engineer/requirements.txt`). Details and limitations: [`docs/channel-store.md`](docs/channel-store.md).

```bash
python main.py --log-channels channels/                      # log while you drive (or with --replay)
python tools/channels.py build recordings/race.f1rec.gz       # or rebuild from a raw recording
python tools/channels.py laps channels/<session>.parquet      # lap list: times, clean / pit / invalid
python tools/channels.py compare channels/<session>.parquet 4 7 --channels speed throttle brake
python main.py --export-motec                                 # newest file in channels/ -> .ld + .ldx for MoTeC i2
```

### Golden-session regression tests

`tests/golden/sessions/` holds recorded sessions (race, qualifying, wet, Safety Car) with the radio calls, voice-command answers and key numbers the engineer produced on them. The test suite replays them and fails if behaviour changes. Four sessions come from a deterministic full-field simulator; four more slots wait for real recordings from the game. See [`tests/golden/README.md`](tests/golden/README.md) for how to record one and how to accept an intended change (`python tools/golden_sessions.py bless NAME`).

### Backtest report: predictions vs what happened

`tools/backtest.py` replays recorded races through the engineer and scores its predictions against what the recording shows really happened, per race and across all races (mean and median absolute error, bias, worst miss):

* **Pit lap**: the next stop predicted by the ML wear model (60% and 75% wear laps) and the race-strategy plan, plus the game's own pit window as a baseline, vs the lap you pitted on; also how often each said "no stop" when you did pit, or called a stop that never came.
* **Tyre wear**: worst-tyre wear the fitted wear model predicts 1, 3 and 5 laps ahead vs the wear the game reported, and how often the truth landed inside the model's 80% range.
* **Lap time**: the engineer's race pace vs the next lap (excluding lap 1, in/out laps and Safety Car laps).
* **Gaps**: the live gap to the cars ahead and behind vs the true gap, and the gap 1 and 3 laps later forecast from the last-lap pace difference.
* **Rival stops**: every rival's predicted next stop lap vs the lap that car really pitted on (against a plan-from-tyre-age-only baseline), and their predicted next tyre vs the one they fitted (against the "soft to medium, medium to hard" rule).

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
* **Predictive Machine Learning:** Logs session data into a local SQLite database (`history.py`) and fits tyre degradation curves (numpy) to predict pit windows.
* **Voice-Activated Radio Command:** Trigger the engineer using voice commands (e.g., "Gap to car behind", "Weather report"). Includes a real-time "We are checking" latency audio cue while the AI processes the response.
* **Realistic F1 Radio Protocol:** Enforces strict communication rules. The engineer will not interrupt during heavy braking or high-G cornering, matching real-world F1 etiquette.
* **Live Track Mapping & Telemetry Visualizer:** Translates X and Z axis telemetry into a live track map on the HUD. The first clean lap on a circuit is saved as that circuit's outline (and the first trip down the pit lane as its pit lane) in `telemetry-history/track_maps.json`, so every later session shows the whole track and pit lane straight away. The circuit is mapped from every car on track, not just yours, so in a race or practice with AI cars the full track appears within a minute or two (any car's pit stop maps the pit lane). The map is drawn as a wide road with edge lines, sized to always fit the panel, with every car shown inside it in its team colour and your car ringed in white. F1 2021 flashbacks are followed: the map cuts the lap trail back to where you rewound to and shows a FLASHBACK badge. "Reset Map" forgets the saved map for that track so a bad one can be recorded again.
* **Pit Stop Call Sequence:** A stop is called the way a real pit wall runs it (`pit_sequence.py`): (1) the box call itself (undercut, cover, Plan B, last lap of the window, wing damage, worn-out tyres); (2) a confirm about 700 m before pit entry ("Box, box. Pit entry coming up, confirm box."); (3) "Stay out, stay out" if the call no longer makes sense before the entry line: the car being undercut boxes first, a car ahead stopping opens the overcut, or a Plan B drops the stop. Stops for damage, worn-out tyres or to cover a car that has already boxed are never taken back; (4) an out-lap brief once back on track: places gained or lost in the stop, the cars directly ahead and behind with gap, compound and tyre age ("yet to stop" if they haven't), and what to do ("Get the tyres in, but keep Norris behind"). Pit entry is learned per circuit from where the field turns in and saved in `track_maps.json` (`pit_entry_m`); until a car has been seen pitting it assumes 400 m before the line. The calls use the keys `pit_confirm`, `pit_abort` and `pit_brief`, which break quiet mode and skip the per-lap budget.
* **Race Report Table:** A lap-by-lap table (lap time, sector 1/2/3, speed trap, tyre compound) on its own HUD tab, built from the same per-lap stats the learning engine already records plus the game's Speed Trap ('SPTP') event packets.
* **Comprehensive Setup Advisor:** Includes a track database (`track_data.py`) covering all 24 circuits in F1 2021. Suggests aerodynamic, transmission, and suspension setups customized to specific tracks.
* **F1-TV Style Web HUD:** A professional, dark-mode web dashboard (`web_hud.py`) displaying live timing towers, color-coded sector deltas, tire wear percentages, and individual tire temperatures.
* **Pit Wall Tab:** A full-field timing tower (gap to leader, interval, last/best lap, tyre and age, stops, in-pit flag), a gap-to-leader race chart lap by lap, a tyre stint chart for every car with our planned stints dashed in, and the race simulator's odds (win / podium / points, which plan comes out best, finishing-position spread, SC/VSC/rain chances, and how the odds moved over the race). Gaps are measured at timing loops every ~100 m like real F1 timing (`pit_wall.py`).
* **Live Push:** The HUD gets its data over a WebSocket (`ws://localhost:5000/ws`): the server sends only what changed, up to 30 times a second, instead of the browser polling every 150 ms. If the socket drops, the HUD falls back to polling `/api/telemetry` and reconnects on its own.

