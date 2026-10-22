# AI answers: speed, frame rate and "I don't have that"

The engineer's AI (DeepSeek through Ollama) runs on the same PC as the game. Each answer
briefly uses the graphics card and CPU, and that is what shows up as a frame drop. This page
covers what the engineer does about it and the settings you can change.

## What happens by default

| | What it does | Why |
|---|---|---|
| Instant answers | The 9 fixed questions (gap ahead/behind, tyre temps, pit window, weather, pace, threats, strategy, rivals) and short questions that match one of them ("gap ahead?") are answered from the live data, without the AI. | No wait and no load on the graphics card. `--ai-for-commands` sends them to the AI again. |
| Focused prompt | The AI gets the core data (lap, position, gaps, lap times, tyre set) plus only the topics your question is about: car (battery, damage, brakes, DRS), tyres, fuel, strategy, the field (running order, threats, rival stops), pace, weather or rules (penalties, flags). A question that fits no topic gets everything. Naming a driver adds the running order. | A shorter prompt is read faster. `--ai-full-context` always sends everything. |
| Streaming | The reply is streamed, and the engineer starts speaking as soon as the radio line is complete. The connection is then closed, so the model stops working. | The answer arrives sooner and the graphics card is freed sooner. |
| Context window 4096 | Every request asks Ollama for a 4096-token window. The preload and keep-warm pings ask for the same, so the model is never reloaded between questions. | Ollama's default window (2048 on many versions) is smaller than a full race's data, and Ollama silently cuts the start of a prompt that doesn't fit. The model then really doesn't have the data. Change it with `--ai-context`. |
| Low priority | When the engineer starts Ollama itself, it runs below normal priority, so the game's frames come first. | `--ai-normal-priority` turns this off. If Ollama is already running (the Windows tray app starts it at login), quit it from the tray before starting the engineer so the engineer starts it at low priority. |
| Voice threads | Kokoro uses a quarter of your CPU threads (1 to 4) instead of all of them. | Each spoken line used to be a short burst on every core. `--tts-threads N` changes it. |

## More data for the AI

On top of tyres, fuel, gaps, pace and strategy, the AI now also gets:

- **ERS battery**: charge %, deploy mode, and how much was used this lap.
- **Car damage and wear** for the wings, floor, gearbox and engine, and tyre damage.
- **Brake temperatures, tyre pressures, tyre carcass temperatures, engine temperature** and whether DRS is open.
- **The current lap**: sector, running time, and whether the lap is invalid.
- **Penalties and warnings**: time penalties, track-limit warnings, unserved drive-throughs and stop-gos, stops made and grid slot.
- **Race control**: the last 5 race-control messages.
- **The running order**: every car's gap, interval, last and best lap, tyre and age, and stops.

Things the game doesn't send, such as other cars' damage or setups, still get an "I don't have that" answer.

## If it's still slow or the game still drops frames

Try these in order:

1. **A lighter model.** `deepseek-r1:8b` is about 5 GB and shares the graphics card with the game. A 3-billion-parameter model answers 2 to 3 times faster in about 2 GB:
   `python main.py --ai-model qwen2.5:3b` (or `llama3.2:3b`). The engineer downloads it on the first run.
2. **Keep the AI off the graphics card.** `python main.py --ai-cpu` runs the model on the CPU only: no frame drops from the AI, but answers take longer. A middle ground is `--ai-gpu-layers 16` (part on the card, part on the CPU).
3. **Cap the AI's CPU threads** if it runs on the CPU: `--ai-threads 4`.
4. **Use a cloud model.** Groq or DeepSeek cloud (see the table above) use none of your PC: `--ai-base-url https://api.groq.com/openai/v1 --ai-model llama-3.1-8b-instant --ai-api-key <key>`.

The console prints `[Engineer] AI too slow (...)` when an answer took longer than `--ai-timeout` (default 30 s) and the engineer answered from rules instead.
