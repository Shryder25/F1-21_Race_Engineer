# Golden-session regression tests

A golden session is a recorded F1 2021 session (raw UDP, `.f1rec.gz`, the recorder's
format: `docs/udp-recording-format.md`) stored next to what the engineer said and
computed when it was replayed (`expected.json`). `tests/test_golden_sessions.py` replays
every session on each test run and fails if the engineer now behaves differently on the
same input.

What is compared:

| Part | What it holds | Tolerance |
|---|---|---|
| `calls` | Every radio call the engineer queued, in order: time, topic key, priority, whether it breaks on-demand silence (`proactive`), exact text | text and order exact, time ±0.25 s |
| `laps` | Key numbers at the end of each lap (position, lap times, gaps, fuel, tyre wear, ML pit window, tyre health, strategy plan, top threat) plus the answer to every voice command | numbers ±0.011, text exact |
| `final`, `answers` | The same at the end of the session | same |
| `race_report`, `race_control_feed` | The HUD race report table and race-control feed | same |

The replay is fully offline and deterministic: no sockets, no real voice, a fresh history
database, and a simulated clock driven by the recording's own arrival times, so
cooldowns behave as they did live (`harness.py`). The engineer runs in `--chatty` mode so
every call it would make is captured; `proactive: true` marks the ones the driver also
hears in the default on-demand mode. Answers come from the rule-based brain (no LLM).

## Sessions in the store

```
python tools/golden_sessions.py list
```

| Slot | Kind | Source |
|---|---|---|
| `race_dry_red_bull_ring` | race | synthetic |
| `qualifying_silverstone` | qualifying | synthetic |
| `race_wet_hungaroring` | wet | synthetic |
| `race_safety_car_zandvoort` | Safety Car | synthetic |
| `real_race`, `real_qualifying`, `real_wet`, `real_safety_car` | one of each | **needs a real recording** (skipped until then) |

The synthetic sessions come from `synth.py` + `scenarios.py`: a deterministic 20-car
simulator (speed profile per track, tyre wear and deg, fuel, pit stops, following and
overtaking, Safety Car bunching) with scripted moments (rain, crash, flags, damage,
penalties). Its packets use the struct formats from `packets.py`. It is not the game: car
physics are a simple model and only the player's telemetry and motion blocks are filled.
That is why the four `real_*` slots exist.

## When a golden test fails

1. Read the failure: it shows a diff of the radio calls and each number that moved.
   `python tools/golden_sessions.py check NAME` prints the same thing.
2. If the change is a bug, fix the code.
3. If the change is intended (a reworded call, a better prediction), accept it:
   `python tools/golden_sessions.py bless NAME`, then commit `expected.json`. Its git diff
   is the review record of how the engineer's behaviour changed.

`python tools/golden_sessions.py show NAME` prints the transcript like a radio log
(`--all` includes braking callouts).

## Recording a real session (on your PC, with the game)

This needs F1 2021 running, which the cloud sandbox can't do. One session per slot.

1. In F1 2021: Settings → Telemetry Settings → UDP Telemetry **On**, UDP Broadcast Mode
   Off, UDP IP Address `127.0.0.1`, UDP Port `20777`, UDP Send Rate **20Hz** (smaller
   files than 60Hz, plenty for the engineer), UDP Format **2021**.
2. Before the session starts, in the repo folder run one of:
   ```
   python ai_engineer/record_udp.py --out recordings/real_race.f1rec.gz --note "Austria, 5 laps, 1 stop"
   python main.py --record recordings/real_race.f1rec.gz      # engineer running at the same time
   ```
3. Drive the session, then stop the recorder with Ctrl+C after the chequered flag (or
   after the in-lap in qualifying).
4. Store it in its slot and bless it:
   ```
   python tools/golden_sessions.py add real_race --recording recordings/real_race.f1rec.gz --note "Austria, 5 laps, 1 stop"
   ```
   If the file is over ~20 MB (60 Hz captures), add `--every 3` to keep one frame in
   three of the per-frame packets.
5. Read the transcript (`python tools/golden_sessions.py show real_race`). Every call
   should be one a real engineer would make at that moment; if one is wrong, that is a bug
   to fix before blessing again. Then commit the slot folder and `manifest.json`.

What makes each slot useful:

| Slot | Record |
|---|---|
| `real_race` | A dry race of 5+ laps (25% distance is fine) with at least one pit stop, ideally with cars close ahead and behind. |
| `real_qualifying` | A qualifying session: out lap, two or more flying laps (one of them a PB), in lap. Run wide once to get a lap deleted. |
| `real_wet` | A race or practice with changing weather (Dynamic weather, pick a forecast with rain), including a switch to intermediates. |
| `real_safety_car` | A race with Safety Car on and AI damage on until a Safety Car or VSC comes out; pit under it if you can. |

## Regenerating the synthetic sessions

Only needed after changing `synth.py` or `scenarios.py`:

```
python tools/golden_sessions.py generate --bless
```

The writer is reproducible (fixed gzip header and start time), so regenerating with no
changes gives byte-identical files.
