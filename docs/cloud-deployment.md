# Cloud deployment (permanent HUD URL)

Run the engineer on a cloud host so the HUD has a fixed `https://` address anyone can open,
with telemetry fed from your game PC. Nothing on your PC has to accept incoming connections.

```
F1 2021 ──UDP──> tools/cloud_relay.py (your PC) ──WebSocket, outbound──> cloud_server.py ──HTTPS──> any browser
```

* **Relay** (`tools/cloud_relay.py`): reads the game's UDP packets, packs ~50 ms of them into
  one compressed message and sends it to `wss://<your-app>/relay`. Reconnects by itself.
  Standard library only, nothing to install.
* **Cloud server** (`cloud_server.py`): the same engine and HUD as `python main.py`, with
  packets coming from the relay instead of UDP. No voice, microphone or console. Laps, track
  maps and fitted models are kept in `/data`, and the models are refit when it restarts.

Local mode (`python main.py`) is unchanged.

## Environment variables

| Variable | Needed | What it is |
|---|---|---|
| `RELAY_SECRET` | yes | Random string, 16+ characters. The relay must send the same one. Make one with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `CONTROL_SECRET` | no | Unlocks the HUD controls (typed questions, Reset Map, AI setup consult) in your browser: open `https://<your-app>/control?token=<CONTROL_SECRET>` once. Unset = the HUD is view-only for everyone. |
| `F1_AI_BASE_URL` | no | OpenAI-compatible API for the AI brain, e.g. `https://api.deepseek.com` or `https://api.groq.com/openai/v1`. Unset = rule-based answers only (there is no Ollama in the cloud). |
| `F1_AI_MODEL` | with the above | e.g. `deepseek-chat` |
| `F1_AI_API_KEY` | with the above | that provider's API key |
| `PORT` | set by the host | HUD and `/relay` share it (Dockerfile default 8080) |
| `DATA_DIR` | no | default `/data`; mount a volume there |

Without `CONTROL_SECRET` a stranger with the link can watch but can't send questions to the
engineer, which would spend your AI API credit.

## Railway

1. railway.app → New Project → Deploy from GitHub repo → this repo. It finds `railway.toml`
   and builds the `Dockerfile`.
2. Service → Variables: add `RELAY_SECRET` (and the optional ones above).
3. Service → Settings → Networking → Generate Domain. That's your permanent URL.
4. Optional but recommended: add a Volume mounted at `/data` so laps and models survive redeploys.

## Fly.io

```
fly launch --no-deploy            # keep the fly.toml in the repo; pick your own app name
fly volumes create engineer_data --size 1
fly secrets set RELAY_SECRET=... CONTROL_SECRET=... F1_AI_BASE_URL=... F1_AI_MODEL=... F1_AI_API_KEY=...
fly deploy
```

The URL is `https://<app-name>.fly.dev`. `fly.toml` keeps one machine always on, since the
relay holds a long-lived connection and the HUD state is in memory.

## On your PC

Game: Settings → Telemetry → UDP Telemetry ON, IP `127.0.0.1`, port `20777`, format 2021.
A UDP send rate of 20 or 30 Hz uses about half the upload of 60 Hz.

```
set RELAY_URL=https://your-app.up.railway.app
set RELAY_SECRET=<same secret>
python tools/cloud_relay.py
```

It prints `connected to the cloud engineer`, then the packet rate and upload every 10 s.

**Engineer in your headset too.** The game sends to one port only, so let the relay pass the
packets on to a local engineer:

```
python tools/cloud_relay.py --also-forward 127.0.0.1:20778
python main.py --port 20778
```

## Check it

* `https://<your-app>/healthz` → `ok`
* The HUD at `https://<your-app>/` shows live data once you're on track. `/api/telemetry` has
  a `relay` key that says whether the relay is connected.
* Server logs show `Relay connected from ...`. `Relay rejected (bad secret)` means the two
  `RELAY_SECRET` values differ.

## Limits and next steps

* One instance only. The HUD state is an in-memory dict in `web_hud.py`; to run several
  workers it would move to Redis (`update_web_state()` writes it, `web_state_message()`
  reads it; see the note in `ai_engineer/cloud_link.py`).
* One relay feeds the server at a time; a reconnecting relay takes over from the old link.
* Upload is roughly 10–40 KB/s after compression, depending on the game's send rate.
