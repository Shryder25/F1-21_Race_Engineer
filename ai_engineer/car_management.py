"""Car management radio calls: fuel, ERS, brakes, engine temperature and damage.

The game sends all of this every frame, and a real engineer manages it for the driver:

  * Fuel      the margin at the flag, in laps, checked once a lap in a race. "Fuel is tight,
              lift and coast" / "fuel is marginal, lift slightly" are urgent; "fuel is good,
              you can push" is chatty only. The fuel mix is used in the wording.
  * ERS       battery low at the line ("harvest this lap"), or full all lap and barely used
              ("use it"). Race only, chatty only.
  * Brakes    a brake disc spiking in the same corner several laps running. Chatty only.
              It names the corner and the brake, never a braking point (radio-modes rule).
  * Engine    engine temperature held high: "temps are up" (chatty, mentions the dirty air
              when following closely), or critical (urgent).
  * Damage    a puncture (tyre damage jumping at once), and engine or gearbox damage when it
              jumps or passes the warning and critical levels (all urgent). DRS fault (chatty).

Every threshold is in config.toml's [car] section and is an estimate until checked against
a real recording. Calls go through the engine's speak(), so the quiet/chatty gating, the
per-lap budget and the busy-driver filter all apply.

The game's ``fuelRemainingLaps`` is documented as "the value on the MFD". Depending on the
game build that is either laps of fuel left or the margin itself, so the first reading with
plenty of race left decides which (``fuel_reading``) for the rest of the session.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import radio_phrases as say
from voice_engine import Priority

WHEELS = ('rear left', 'rear right', 'front left', 'front right')  # telemetry tuple order
FUEL_MIX_NAMES = {0: 'lean', 1: 'standard', 2: 'rich', 3: 'max'}
ERS_STORE_MAX_J = 4_000_000.0
LEAN_MIX = 0
ERS_HOT_MODES = (2, 3)  # hotlap, overtake

Speak = Callable[..., Any]


def laps_words(laps: float) -> str:
    """A fuel margin as said on the radio: "half a lap", "one point two laps"."""
    laps = round(abs(laps), 1)
    if laps < 0.15:
        return 'nothing'
    if abs(laps - 0.5) < 0.05:
        return 'half a lap'
    whole, tenth = int(laps), int(round((laps - int(laps)) * 10))
    if tenth == 10:
        whole, tenth = whole + 1, 0
    if tenth == 0:
        return f'{say.number_words(whole)} lap' + ('' if whole == 1 else 's')
    return f'{say.number_words(whole) if whole else "nought"} point {say.number_words(tenth)} laps'


def fuel_band(margin: float, cfg: Any) -> str:
    """'save', 'lift', 'ok' or 'good' for a fuel margin in laps."""
    if margin < cfg.fuel_save_laps:
        return 'save'
    if margin < cfg.fuel_lift_laps:
        return 'lift'
    if margin >= cfg.fuel_good_laps:
        return 'good'
    return 'ok'


def corner_at(corners: Sequence[dict[str, Any]], distance: float) -> str | None:
    for corner in corners:
        if corner.get('start', 0) <= distance <= corner.get('end', -1):
            name = corner.get('name')
            return str(name) if name else None
    return None


@dataclass
class CarManagementState:
    """Per-session state; ``CarManager.reset()`` replaces it with a fresh one."""

    lap_num: int = 0
    # Fuel
    fuel_reading: str | None = None  # 'absolute' (laps of fuel) or 'delta' (the margin itself)
    fuel_margin: float | None = None
    fuel_band: str | None = None
    fuel_called_band: str | None = None
    fuel_called_lap: int = -99
    fuel_good_said: bool = False  # "fuel is good" is said once a race
    # ERS
    ers_lap_min_pct: float = 100.0
    ers_calls: dict[str, int] = field(default_factory=dict)  # topic -> lap last called
    # Brakes: peak disc temp per (corner, wheel) this lap, hot-lap streaks, last call lap
    brake_peaks: dict[tuple[str, int], int] = field(default_factory=dict)
    brake_streaks: dict[tuple[str, int], int] = field(default_factory=dict)
    brake_called: dict[str, int] = field(default_factory=dict)
    hot_brakes: list[str] = field(default_factory=list)  # for the HUD / AI: "front left, Turn 4"
    # Engine temperature
    engine_hot_since: float | None = None
    engine_critical_since: float | None = None
    engine_called_at: float = -1e9
    engine_critical_called_at: float = -1e9
    # Damage: the last damage sample seen, and what was already called
    last_tyre_damage: tuple[int, ...] | None = None
    punctures: set[int] = field(default_factory=set)
    last_parts: dict[str, int] = field(default_factory=dict)
    drs_fault: bool = False


class CarManager:
    """Fuel, ERS, brake, engine and damage calls, from the frame data the engine already has."""

    def __init__(self, speak: Speak, config: Any, chat_ok: Callable[[], bool] | None = None) -> None:
        self.speak = speak
        self.config = config  # pipeline.config.CarConfig
        # Chatty-only calls (fuel good, ERS, brakes, engine temps up) ask this first, so they
        # leave room in the lap's call budget for race information. Urgent calls never ask.
        self.chat_ok = chat_ok if chat_ok is not None else (lambda: True)
        self.state = CarManagementState()

    def reset(self) -> None:
        self.state = CarManagementState()

    # ── Per frame ──

    def update(
        self,
        now: float,
        lap: Any,
        telemetry: Any,
        status: Any = None,
        damage: Any = None,
        *,
        race: bool = False,
        total_laps: int = 0,
        corners: Sequence[dict[str, Any]] = (),
        gap_ahead_s: float | None = None,
    ) -> None:
        """One telemetry frame. Lap-end checks (fuel, ERS, brakes) run when the lap changes."""
        state = self.state
        lap_num = int(getattr(lap, 'current_lap_num', 0) or 0)
        in_pit = bool(getattr(lap, 'pit_status', 0))
        if lap_num != state.lap_num:
            if state.lap_num > 0 and lap_num == state.lap_num + 1:
                self._lap_end(lap_num, status, race=race, total_laps=total_laps, in_pit=in_pit)
            state.lap_num = lap_num
            state.brake_peaks = {}
            state.ers_lap_min_pct = 100.0
        if status is not None:
            pct = status.ers_store_energy / ERS_STORE_MAX_J * 100
            state.ers_lap_min_pct = min(state.ers_lap_min_pct, pct)
        if not in_pit and getattr(telemetry, 'speed', 0) >= 60:
            self._track_brakes(lap, telemetry, corners)
            self._check_engine(now, telemetry.engine_temperature, gap_ahead_s)
        if damage is not None:
            self._check_damage(damage)

    # ── Lap end: fuel, ERS, brakes ──

    def _lap_end(self, lap_num: int, status: Any, *, race: bool, total_laps: int, in_pit: bool) -> None:
        if race and status is not None and not in_pit and total_laps > 0:
            to_go = total_laps - lap_num + 1
            if to_go >= 1 and lap_num >= 2:
                self._check_fuel(lap_num, status, to_go)
            if to_go >= 2:
                self._check_ers(lap_num, status)
        self._check_brakes(lap_num)

    def fuel_margin(self, remaining: float, laps_to_go: int) -> float | None:
        """The fuel margin at the flag in laps (+ spare, - short), or None until known."""
        state = self.state
        if state.fuel_reading is None and laps_to_go >= 5:
            state.fuel_reading = 'absolute' if remaining >= 0.6 * laps_to_go else 'delta'
        if state.fuel_reading is None:
            return None
        return remaining - laps_to_go if state.fuel_reading == 'absolute' else remaining

    def _check_fuel(self, lap_num: int, status: Any, to_go: int) -> None:
        state, cfg = self.state, self.config
        margin = self.fuel_margin(float(status.fuel_remaining_laps), to_go)
        if margin is None:
            return
        band = fuel_band(margin, cfg)
        state.fuel_margin, state.fuel_band = round(margin, 2), band
        repeat = band in ('save', 'lift') and lap_num - state.fuel_called_lap >= cfg.fuel_repeat_laps
        if band == state.fuel_called_band and not repeat:
            return
        lean = getattr(status, 'fuel_mix', 1) == LEAN_MIX
        if band == 'save':
            mix = 'Stay lean' if lean else 'Lean mix'
            message = f'Fuel is tight, we are {laps_words(margin)} short. {mix}, lift and coast.'
            self.speak(message, Priority.STRATEGY, key='fuel_save', ttl=12, cooldown=0)
        elif band == 'lift':
            if margin < 0:
                message = f'Fuel is marginal, {laps_words(margin)} short. Lift slightly into the big stops.'
            else:
                message = 'Fuel is marginal. Lift slightly into the big stops.'
            self.speak(message, Priority.STRATEGY, key='fuel_save', ttl=12, cooldown=0)
        elif band == 'good':
            if to_go < 3 or state.fuel_good_said or not self.chat_ok():
                state.fuel_called_band = band
                return
            state.fuel_good_said = True
            extra = ' You can come off lean mix.' if lean else ' You can push.'
            message = f'Fuel is good, {laps_words(margin)} spare.{extra}'
            self.speak(message, Priority.LOW, key='fuel_status', ttl=12, cooldown=0)
        elif state.fuel_called_band in ('save', 'lift') and self.chat_ok():
            self.speak(
                'Fuel is back on target, normal running.', Priority.LOW, key='fuel_status', ttl=12, cooldown=0
            )
        else:
            state.fuel_called_band = band
            return
        state.fuel_called_band = band
        state.fuel_called_lap = lap_num

    def _check_ers(self, lap_num: int, status: Any) -> None:
        state, cfg = self.state, self.config
        pct = status.ers_store_energy / ERS_STORE_MAX_J * 100

        def due(topic: str) -> bool:
            return lap_num - state.ers_calls.get(topic, -99) >= cfg.ers_repeat_laps and self.chat_ok()

        if pct < cfg.ers_low_pct and due('low'):
            words = say.number_words(int(round(pct)))
            if status.ers_deploy_mode in ERS_HOT_MODES:
                message = f'Battery is low, {words} percent. Off overtake, harvest this lap.'
            else:
                message = f'Battery is low, {words} percent. Harvest this lap.'
            self.speak(message, Priority.NORMAL, key='ers', ttl=10, cooldown=0)
            state.ers_calls['low'] = lap_num
        elif pct >= cfg.ers_full_pct and state.ers_lap_min_pct >= cfg.ers_full_pct - 10 and due('full'):
            self.speak(
                'Battery is full and we are wasting harvest. Use it, overtake on the straights.',
                Priority.NORMAL,
                key='ers',
                ttl=10,
                cooldown=0,
            )
            state.ers_calls['full'] = lap_num

    def _track_brakes(self, lap: Any, telemetry: Any, corners: Sequence[dict[str, Any]]) -> None:
        corner = corner_at(corners, float(getattr(lap, 'lap_distance', -1.0)))
        if corner is None:
            return
        peaks = self.state.brake_peaks
        for wheel, temp in enumerate(telemetry.brakes_temperature):
            key = (corner, wheel)
            if temp > peaks.get(key, 0):
                peaks[key] = int(temp)

    def _check_brakes(self, lap_num: int) -> None:
        state, cfg = self.state, self.config
        hot = {key for key, peak in state.brake_peaks.items() if peak >= cfg.brake_hot_c}
        # A streak continues only through laps where that corner was driven and hot again.
        state.brake_streaks = {key: state.brake_streaks.get(key, 0) + 1 for key in hot}
        state.hot_brakes = sorted(
            f'{WHEELS[wheel]}, {corner}' for (corner, wheel), n in state.brake_streaks.items() if n >= 1
        )
        candidates = [
            (state.brake_peaks[key], key)
            for key, streak in state.brake_streaks.items()
            if streak >= cfg.brake_repeat_laps
            and lap_num - state.brake_called.get(key[0], -99) >= cfg.brake_cooldown_laps
        ]
        if not candidates or not self.chat_ok():
            return
        _, (corner, wheel) = max(candidates)
        direction = 'rearward' if wheel >= 2 else 'forward'
        name = WHEELS[wheel]
        self.speak(
            f'{name[0].upper() + name[1:]} brake is spiking into {corner} every lap. '
            f'A click of bias {direction} if it carries on.',
            Priority.NORMAL,
            key='brake_temp',
            ttl=10,
            cooldown=0,
        )
        state.brake_called[corner] = lap_num

    # ── Engine temperature ──

    def _check_engine(self, now: float, temp: float, gap_ahead_s: float | None) -> None:
        state, cfg = self.state, self.config
        state.engine_hot_since = (state.engine_hot_since or now) if temp >= cfg.engine_hot_c else None
        state.engine_critical_since = (
            (state.engine_critical_since or now) if temp >= cfg.engine_critical_c else None
        )
        words = say.number_words(int(temp))
        if (
            state.engine_critical_since is not None
            and now - state.engine_critical_since >= cfg.engine_sustain_s
            and now - state.engine_critical_called_at >= cfg.engine_cooldown_s
        ):
            self.speak(
                f'Engine temperature critical, {words} degrees. Get out of the tow and lift and coast.',
                Priority.CRITICAL,
                key='engine_temp_critical',
                ttl=8,
                cooldown=0,
            )
            state.engine_critical_called_at = state.engine_called_at = now
        elif (
            state.engine_hot_since is not None
            and now - state.engine_hot_since >= cfg.engine_sustain_s
            and now - state.engine_called_at >= cfg.engine_cooldown_s
            and self.chat_ok()
        ):
            if gap_ahead_s is not None and 0 < gap_ahead_s < 1.0:
                message = (
                    f'Engine temps are up, {words} degrees in the dirty air. Pull out of the tow if you can.'
                )
            else:
                message = f'Engine temps are up, {words} degrees. Keep an eye on it.'
            self.speak(message, Priority.NORMAL, key='engine_temp', ttl=10, cooldown=0)
            state.engine_called_at = now

    # ── Damage ──

    def _check_damage(self, damage: Any) -> None:
        state, cfg = self.state, self.config
        tyres = tuple(int(d) for d in damage.tyres_damage)
        previous = state.last_tyre_damage
        state.last_tyre_damage = tyres
        if previous is not None:
            if sum(tyres) + 20 < sum(previous):
                state.punctures.clear()  # new tyres fitted
            new = [
                i
                for i in range(4)
                if i not in state.punctures
                and (tyres[i] - previous[i] >= cfg.puncture_jump_pct or tyres[i] >= 90 > previous[i])
            ]
            if new:
                state.punctures.update(new)
                wheels = ' and '.join(WHEELS[i] for i in sorted(new))
                self.speak(
                    f'Puncture, puncture, {wheels}. Box this lap, box box.',
                    Priority.CRITICAL,
                    key='puncture',
                    ttl=8,
                    cooldown=0,
                )

        for part, attr in (('engine', 'engine'), ('gearbox', 'gearbox')):
            value = int(getattr(damage, attr, 0) or 0)
            before = state.last_parts.get(part)
            state.last_parts[part] = value
            if before is None or value <= before:
                continue
            self._call_part_damage(part, before, value)

        fault = bool(getattr(damage, 'drs_fault', 0))
        if fault and not state.drs_fault:
            self.speak('DRS fault, we have no DRS.', Priority.RACE, key='drs_fault', ttl=8, cooldown=0)
        state.drs_fault = fault

    def _call_part_damage(self, part: str, before: int, value: int) -> None:
        cfg = self.config
        words = say.number_words(value)
        advice = 'Short shift and stay off the kerbs.' if part == 'gearbox' else 'Lean mix and short shift.'
        key = f'{part}_damage'
        if before < cfg.damage_critical_pct <= value:
            self.speak(
                f'{part.capitalize()} damage is critical, {words} percent. {advice} Bring it home.',
                Priority.CRITICAL,
                key=key,
                ttl=10,
                cooldown=0,
            )
        elif value - before >= cfg.damage_step_pct:
            self.speak(
                f'We are seeing {part} damage, now {words} percent. {advice}',
                Priority.STRATEGY,
                key=key,
                ttl=10,
                cooldown=0,
            )
        elif before < cfg.damage_warn_pct <= value:
            self.speak(
                f'{part.capitalize()} damage at {words} percent. {advice}',
                Priority.STRATEGY,
                key=key,
                ttl=10,
                cooldown=0,
            )

    # ── For the HUD and the AI prompt ──

    def snapshot(self, status: Any = None, damage: Any = None) -> dict[str, Any]:
        state = self.state
        out: dict[str, Any] = {
            'fuel_margin_laps': state.fuel_margin,
            'fuel_status': state.fuel_band,
            'hot_brakes': list(state.hot_brakes),
            'punctures': [WHEELS[i] for i in sorted(state.punctures)],
            'drs_fault': state.drs_fault,
        }
        if status is not None:
            out['fuel_mix'] = FUEL_MIX_NAMES.get(getattr(status, 'fuel_mix', 1), 'standard')
        if damage is not None:
            out['engine_damage_pct'] = int(getattr(damage, 'engine', 0) or 0)
            out['gearbox_damage_pct'] = int(getattr(damage, 'gearbox', 0) or 0)
        return out
