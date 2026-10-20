# Push-to-talk radio button (wheel or controller)

Hold a button on your wheel or controller, ask the engineer anything, let go. The
engineer says "We are checking", then answers from the live telemetry over the radio.
You can ask in your own words, not just the fixed commands:

- "How are the rears holding up?"
- "Can I make it to the end on these tyres?"
- "Who's behind me and is he quicker?"
- "Should I box this lap or stay out?"
- "Is it going to rain?"

Free-form questions need the AI brain (DeepSeek / Ollama / Groq, see the main README).
If the AI is offline, questions that match a fixed command (gaps, tyres, pace, pit
window, weather, strategy, threats) still get a rule-based answer; anything else gets
a short "can't answer that without the AI" reply.

## 1. Install

```
cd ai_engineer
pip install -r requirements.txt      # includes pygame, SpeechRecognition and PyAudio
```

On Windows, if `PyAudio` fails to install, run `pip install pipwin` then `pipwin install pyaudio`,
or use Python 3.11/3.12 where prebuilt wheels exist.

## 2. Pick your radio button (once)

Plug in the wheel or controller, then from the project folder run:

```
python main.py --find-ptt-button
```

It lists every controller it can see, then waits for you to press a button:

```
Controllers found:
  [0] Thrustmaster T300RS Racing wheel  (13 buttons, 1 hats)
  [1] Xbox Controller  (11 buttons, 1 hats)

Press and release the button you want as the radio button (within 30 s)...

Radio button: 7 on "Thrustmaster T300RS Racing wheel" (hold to talk).
Saved to ...\F1-21_Race_Engineer\push_to_talk.json.
```

Buttons and D-pad (hat) directions both work. The choice is saved to `push_to_talk.json`
in the project folder, so from now on `python main.py` (or `run_engineer.bat`) uses it
automatically. Run `--find-ptt-button` again any time to change it.

Prefer press-once instead of hold? Use `python main.py --find-ptt-button --ptt-mode toggle`
(press to open the radio, press again to send). The radio closes itself after 15 seconds.

### Choosing a good button

The game still sees the button too, so pick one F1 2021 does nothing with, or unbind it in
**Settings → Controls** in the game. Good choices:

- **Wheel:** a spare face button or rotary push. Many wheels have one or two buttons that
  are unassigned by default.
- **Xbox/PlayStation pad:** clicking a stick (L3/R3) if you do not use it for look-back, or
  a D-pad direction you have unbound. Avoid the buttons used for MFD/radio menus unless you
  unbind them.

## 3. Drive

```
python main.py
```

You will see `[PTT] Hold button 7 on "Thrustmaster..." to talk to the engineer.` Hold the
button, talk, release. The console and the HUD radio log show what was heard.

## Options

| Flag | What it does |
|------|--------------|
| `--find-ptt-button` | Press the button you want; saves it and exits |
| `--ptt-button 7` / `--ptt-button hat0:up` | Use this button without the saved file |
| `--ptt-device "T300"` | Only listen on the controller whose name contains this (or its index, e.g. `0`) |
| `--ptt-mode hold\|toggle` | Hold to talk (default) or press to start / press to send |
| `--no-ptt` | Ignore the saved button for this run |
| `--always-listen` | Keep the hands-free "engineer, ..." microphone on as well |
| `--mic-index N` | Which microphone to record from |
| `--stt-engine vosk` | Offline speech recognition instead of Google |

When a radio button is set, the always-on wake-word microphone is switched off so the same
question is not answered twice. `--always-listen` turns it back on; it then ignores speech
while you are using the button.

## Troubleshooting

- **"No wheel or controller found"**: check Windows *Set up USB game controllers*
  (`joy.cpl`) shows it. Close software that takes exclusive control (some Steam Input
  settings do this; disable Steam Input for the game or the terminal).
- **Button works in `--find-ptt-button` but not while the game is in front**: the engineer
  reads controllers in the background (`SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS=1`). If your
  wheel software blocks that, try running the terminal as the same user/elevation as the game.
- **Wrong controller**: pass `--ptt-device` with part of the name shown by `--find-ptt-button`.
- **"Say again, didn't get that"**: speech was not recognised. Talk after pressing, keep the
  button held until you finish, and check `--mic-index`. A tap shorter than 0.3 s is ignored.
- **Unplugged mid-session**: the engineer notices and picks the controller back up when it
  is plugged in again.

The microphone stream stays open while the engineer runs so the start of your sentence is
not cut off, but only the last 0.3 s is kept in memory outside a button press; nothing is
recorded or saved unless you hold the button.
