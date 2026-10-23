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
| Normal priority | Ollama runs at normal priority. | It used to start below the game's priority, but in a race the game keeps the CPU busy and the AI then got so little time that answers missed the timeout. `--ai-low-priority` brings the old behaviour back. |
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

The console prints `AI (...) answered in N s` for every answer, and `AI (...) too slow` when an answer took longer than `--ai-timeout` (default 30 s); the engineer then says "the AI took too long" on the radio.

## "AI took too long" in a race, but `--ai-test` is quick

`--ai-test` runs with the game closed, so the AI has the whole graphics card and CPU. In a race it shares them with F1 2021. The "too slow" console line says where the model was loaded at that moment, for example `The model is 40% on the graphics card, 60% on the processor`.

- **Start F1 2021 first, then the engineer.** The engineer loads the model when it starts. If the game starts afterwards and needs the graphics memory, the graphics driver can move part of the model into normal RAM and answers get many times slower. Loaded after the game, the model gets only the memory that's really free.
- **Don't use `--ai-cpu` in a race.** With the game using most of the CPU, a CPU-only model is too slow on most PCs.
- **Graphics memory.** On a card with 4 to 6 GB (for example an RTX 3050), set the game's textures to Medium, use `qwen2.5:3b` or `qwen2.5:1.5b`, and if the console shows the model mostly on the processor, try `--ai-gpu-layers 20`.
- **Still too slow:** a cloud model (see above) uses none of your PC.
