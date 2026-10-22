"""The F1 2021 game's own button feed (the 'BUTN' UDP event) for push-to-talk.

F1 2021 sends a Buttons event whenever the controller state changes: a 32-bit mask of
which buttons are held right now. Wheel buttons arrive as the pad button the game has
them mapped to, so a wheel button bound to "R1" shows up as the R1 bit. Reading the
radio button from the game means no pygame, no second program fighting the wheel driver
for input, and it keeps working while the game window has focus.

Find which bit your button sends with ``python main.py --find-game-button``.
"""

from __future__ import annotations

import time

# Button flags from the F1 2021 UDP spec (appendix "Button flags").
BUTTON_FLAGS = {
    'cross': 0x00001,       # Cross or A
    'triangle': 0x00002,    # Triangle or Y
    'circle': 0x00004,      # Circle or B
    'square': 0x00008,      # Square or X
    'dpad_left': 0x00010,
    'dpad_right': 0x00020,
    'dpad_up': 0x00040,
    'dpad_down': 0x00080,
    'options': 0x00100,     # Options or Menu
    'l1': 0x00200,          # L1 or LB
    'r1': 0x00400,          # R1 or RB
    'l2': 0x00800,          # L2 or LT
    'r2': 0x01000,          # R2 or RT
    'l3': 0x02000,          # left stick click
    'r3': 0x04000,          # right stick click
    'rs_left': 0x08000,     # right stick directions (F1 2021 added these)
    'rs_right': 0x10000,
    'rs_up': 0x20000,
    'rs_down': 0x40000,
    'special': 0x80000,
}
ALIASES = {
    'a': 'cross', 'y': 'triangle', 'b': 'circle', 'x': 'square',
    'lb': 'l1', 'rb': 'r1', 'lt': 'l2', 'rt': 'r2', 'menu': 'options',
    'ls': 'l3', 'rs': 'r3', 'left_stick': 'l3', 'right_stick': 'r3',
    'up': 'dpad_up', 'down': 'dpad_down', 'left': 'dpad_left', 'right': 'dpad_right',
}
_NAMES_BY_FLAG = {flag: name for name, flag in BUTTON_FLAGS.items()}
ALL_FLAGS = sum(BUTTON_FLAGS.values())


def parse_game_button(text) -> int:
    """``'r1'``, ``'RB'``, ``'dpad-up'`` or a raw mask like ``'0x400'`` -> the bit mask."""
    value = str(text).strip().lower().replace('-', '_').replace(' ', '_')
    name = ALIASES.get(value, value)
    if name in BUTTON_FLAGS:
        return BUTTON_FLAGS[name]
    try:
        mask = int(value, 0)
    except ValueError:
        mask = 0
    if mask > 0 and mask & ALL_FLAGS == mask:
        return mask
    raise ValueError(f'Unknown game button {text!r}: use a name like r1, l1, cross, dpad_up '
                     f'(choices: {", ".join(BUTTON_FLAGS)})')


def button_label(mask: int) -> str:
    """'r1', or 'l1+r1' for a combination, or the hex mask for unknown bits."""
    names = [name for flag, name in _NAMES_BY_FLAG.items() if mask & flag]
    return '+'.join(names) if names and sum(BUTTON_FLAGS[n] for n in names) == mask else hex(mask)


def find_game_button(receive, parse, *, timeout_s: float = 30.0, out=print,
                     clock=time.monotonic) -> int | None:
    """Wait for a newly pressed button in the game's Buttons events.

    ``receive()`` returns the next datagram (or None on a socket timeout); ``parse(data)``
    returns the button mask for a Buttons event, None for anything else. Buttons already
    held when the first event arrives are ignored. Returns the new button's flag or None.
    """
    out(f'Press and release the wheel button you want as the radio button (within {timeout_s:.0f} s).')
    out('F1 2021 must be running with UDP Telemetry on, and you need to be in a session or menu.')
    deadline = clock() + timeout_s
    baseline = None
    seen_event = False
    while clock() < deadline:
        data = receive()
        if data is None:
            continue
        status = parse(data)
        if status is None:
            continue
        if not seen_event:
            seen_event = True
            out('Game button feed found.')
        if baseline is None:
            baseline = status
        new = status & ~baseline
        if new:
            return new & -new                      # lowest new bit, one button
        baseline &= status                         # a released button may be chosen later
    if not seen_event:
        out('No button events from the game. Check Settings > Telemetry: UDP Telemetry On, '
            'UDP Format 2021, and the same port as the engineer.')
    else:
        out('No new button press seen.')
    return None
