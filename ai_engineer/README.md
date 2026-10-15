# F1 2021 AI Race Engineer

> An intelligent real-time race engineer that reads your F1 2021 telemetry and talks to you like a real pit wall.

## Quick Start

### 1. Install Dependencies
```bash
cd ai_engineer
pip install -r requirements.txt
```

### 2. Configure F1 2021
In-game: **Settings → Telemetry**
| Setting | Value |
|---------|-------|
| UDP Telemetry | **ON** |
| UDP Broadcast | **ON** |
| UDP IP Address | `127.0.0.1` |
| UDP Port | `20777` |
| UDP Format | `2021` |

### 3. Run the Engineer
```bash
python main.py
```

Requires Python 3.10 or later. Start a session in F1 2021 after launching the engineer.
The browser HUD opens at `http://localhost:5000`.

## Command Line Options
```
python main.py [--port 20777] [--no-voice] [--debug] [--voice-rate 175] [--voice-index 0]
```

| Flag | Description |
|------|-------------|
| `--port` | UDP port (default: 20777) |
| `--no-voice` | Disable speech; keep the console and browser HUD |
| `--debug` | Show packet parsing errors |
| `--voice-rate` | Speech speed in WPM (default: 175) |
| `--voice-index` | Select system voice by index |
| `--history-dir` | Directory for saved telemetry (default: `telemetry-history` in the project root) |

Completed lap recordings are saved locally in `telemetry-history/telemetry.sqlite3`.
Each recording belongs to its own track and session capture. Invalid or incomplete
recordings are archived but excluded from references. Returning to a track loads
its fastest eligible recorded lap with the same track length, formula, and weather
category (dry, intermediate, or wet). A new session clears live timing and corner
state. These are sampled lap recordings, not a lossless archive of raw UDP packets.

## What It Does

### 🗣️ Voice Callouts
- **Braking points** — "Brake for Turn 1, 120 metres" 
- **Tyre temperatures** — "Front tyres overheating at 108 degrees"
- **Lap deltas** — "You're 0.3 seconds faster than your best"
- **Lap times** — "Personal best! 1 minute 29.432"
- **Pit windows** — Uses the game's current ideal/latest strategy laps in race sessions
- **Traffic gaps** — Ahead/behind timing derived from recorded crossings at matching race distances
- **Personal sectors** — Completed sector times compared with eligible personal sector records
- **Race pace** — Average of the latest three clean, fully recorded race laps

Radio priority, highest first:

1. Braking and critical tyre alerts
2. Pit-window calls
3. Driver gaps and race pace
4. Sector and completed-lap times
5. Routine tyre alerts
6. Live-delta coaching and session introductions

Urgent calls interrupt lower-priority speech. The queue is bounded, replaces older
pending updates of the same topic, and expires time-sensitive instructions. Radio
logs show speech when it starts; in `--no-voice` mode they identify muted messages.
Paused games, spectating, session changes, and telemetry loss cancel queued speech.
Driver gaps remain unavailable until sufficient crossing history exists; gaps are
not inferred by dividing a distance difference by current speed.

Reference laps exclude invalid laps, pit/in/out laps, neutralized laps, partial
recordings, large missing sections, and rewound recordings. Corner timing currently
supports Silverstone. Its measured deltas require a recorded reference; approximate
track-guide speeds and braking points are labelled separately.

### 📊 Live Console Dashboard
- Speed, gear, RPM, DRS status
- Throttle & brake bars
- Tyre surface temperatures (all 4 corners)
- Tyre pressures & engine temperature
- Current lap time and delta to best

The browser HUD also shows distance-aligned speed traces, corner diagnostics,
race information, and a radio log. It polls at about 7 Hz and reports stale
telemetry after two seconds without a valid matched frame. Chart.js loads from a
CDN; if it is unavailable, the remaining gauges and radio log still update.

### 🏁 Tracks with Braking Data
Bahrain, Silverstone, Monza, Spa, Catalunya, Monaco, Abu Dhabi, Red Bull Ring,
Montreal, Baku, Zandvoort, Imola, Jeddah, Interlagos, Suzuka, COTA, Mexico City,
Singapore, Hungaroring, Sochi, Portimão

## Architecture

```
ai_engineer/
├── main.py          # UDP listener + console dashboard
├── packets.py       # F1 2021 packet struct definitions  
├── telemetry_stream.py # Datagram validation, matching frames, freshness
├── analysis.py      # Tyre/braking/delta/corner analysis
├── history.py       # Track/session SQLite lap archive
├── race_radio.py    # Pit strategy, measured gaps, sectors, race pace
├── voice_engine.py  # Expiring priority queue and interruptible TTS
├── web_hud.py       # Browser dashboard and telemetry API
├── track_data.py    # Braking points + setup advice per track
└── requirements.txt # pip dependencies
```

**Zero game files are modified.** The engineer is a standalone application that reads the game's UDP telemetry broadcast.

## Validation

From the project root, run `python -B -m unittest discover -s tests -v`.
The suite includes synthetic UDP-to-HTTP integration and fake speech-driver tests.
Node.js enables the browser-script regression test. Real-game telemetry and audible
Windows speech should also be checked on the gaming machine.

Packet layout reference: [F1 2021 protocol documentation mirror](https://github.com/martinezruben/official-f1-2021-udp#data-output-from-f1-2021).
