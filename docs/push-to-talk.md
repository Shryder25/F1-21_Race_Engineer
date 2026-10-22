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
a short "can't answer that without the AI" reply. If the AI is on but didn't answer, you
hear "the AI took too long" or "the AI didn't give me an answer" instead, and the console
says why. Run `python main.py --ai-test` (add `--ai-model` if you use one) to check it
without the game.

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

### Or: read the button from the game itself (no pygame)

F1 2021 sends its own button feed over UDP (the "Buttons" event, `BUTN`). The engineer can
use that instead of reading the wheel directly, which avoids wheel software or Steam Input
blocking a second program. With the game running and UDP Telemetry on:

```
python main.py --find-game-button
```

Press the wheel button you want within 30 seconds. The game reports wheel buttons as the pad
button they are mapped to, so you will see something like `Radio button: R1 from the game`.
It is saved to `push_to_talk.json` next to any pygame button, and both work. Or pass it by
hand: `python main.py --ptt-game-button r1` (names: cross, triangle, circle, square, l1, r1,
l2, r2, l3, r3, dpad_up/down/left/right, options, rs_up/down/left/right, special; Xbox names
a, b, x, y, lb, rb, lt, rt also work).

The same button still does its normal job in the game, so pick one you don't use, or unbind
it. If the game ever drops the "released" packet, the radio closes itself after 15 seconds.

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
| `--find-game-button` | Press a wheel button; reads it from the game's UDP feed, saves it and exits |
| `--ptt-game-button r1` | Use this game button (from the UDP Buttons event) without the saved file |
| `--ptt-button 7` / `--ptt-button hat0:up` | Use this button without the saved file |
| `--ptt-device "T300"` | Only listen on the controller whose name contains this (or its index, e.g. `0`) |
| `--ptt-mode hold\|toggle` | Hold to talk (default) or press to start / press to send |
| `--no-ptt` | Ignore the saved button for this run |
| `--always-listen` | Keep the hands-free "engineer, ..." microphone on as well |
| `--mic-index N` | Which microphone to record from |
| `--stt-engine whisper\|google\|vosk` | Speech recognition. Default `whisper`, offline (see below) |
| `--whisper-model tiny.en` | Smaller, quicker Whisper model (default `base.en`) |

## Offline speech recognition (Whisper)

The radio turns your voice into text with OpenAI's open-source Whisper model running on your
own PC, so it works with no internet and no account. `pip install -r requirements.txt`
installs it (`openai-whisper`, which brings PyTorch, about 1 GB on disk). The first start
downloads the model once (`base.en` is about 140 MB, `tiny.en` about 75 MB); it then loads in
the background while the engineer starts. If Whisper is not installed or can't load, the
engineer says so in the console and uses Google speech recognition (online) as before.

`base.en` takes roughly half a second to a second per question on a modern CPU; use
`--whisper-model tiny.en` if that feels slow, or `small.en` if words are often misheard.

## How questions are understood

Fixed commands (tyre temps, gaps, pit window, pace, weather, threats, strategy, rival stops)
are recognised by meaning, not exact words: "how are my tyres holding up?", "are the rears
getting hot?" and "tyre temps" are all the tyre question, and a slightly misheard word
("temprature") still matches. Anything else goes to the AI as a free-form question.

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
- **"The AI took too long"**: the model didn't answer within `--ai-timeout` (30 s). Common
  with `--ai-cpu` while the game is running. Try a longer `--ai-timeout`, drop `--ai-cpu`, or
  a lighter model.
- **"The AI didn't give me an answer"**: the console line `AI (...) gave no usable answer`
  says why (model not installed, empty reply...). `python main.py --ai-test` shows the raw reply.
- **Unplugged mid-session**: the engineer notices and picks the controller back up when it
  is plugged in again.

The microphone stream stays open while the engineer runs so the start of your sentence is
not cut off, but only the last 0.3 s is kept in memory outside a button press; nothing is
recorded or saved unless you hold the button.
