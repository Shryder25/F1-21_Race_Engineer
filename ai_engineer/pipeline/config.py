"""Tuneable settings, loaded from ``config.toml`` at startup.

Every value has a built-in default equal to what the engineer used before the config file
existed, so a missing file (or a missing key) changes nothing. ``config.toml`` at the repo
root documents each key. Command-line flags override the file (see ``main.py``).

The file is plain TOML read with the standard library (``tomllib``), so it needs no extra
package. Unknown sections or keys are an error rather than silently ignored: a typo in a
threshold should not quietly fall back to the default.
"""

from __future__ import annotations

import dataclasses
import tomllib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'config.toml'

# Callouts that break on-demand radio silence: critical events, and strategy calls that
# cannot wait for the driver to ask (covering or taking an undercut is a this-lap decision).
DEFAULT_PROACTIVE_KEYS = (
    'tyre_temperature', 'safety_car', 'safety_car_in', 'weather_rain', 'critical_wear',
    'wing_damage', 'penalty', 'flag_yellow', 'flag_red', 'rival_pit', 'undercut',
    'rival_preempt', 'plan_b',
)


class ConfigError(ValueError):
    """The config file is unreadable, has an unknown key, or a value of the wrong type."""


@dataclass
class TyreConfig:
    high_c: int = 104                     # surface temp above this counts as running hot
    low_c: int = 80                       # surface temp below this counts as running cold
    critical_c: int = 112                 # critical overheat: the only mid-lap tyre call
    rise_rate_c_per_s: float = 1.5        # a critical temp must also be climbing this fast
    critical_sustain_s: float = 0.5       # ... for at least this long
    alert_cooldown_s: float = 40.0        # seconds between two critical overheat calls
    summary_min_s: float = 4.0            # seconds hot/cold in a lap before the summary says so
    summary_repeat_laps: int = 3          # an unchanged summary is repeated at most this often


@dataclass
class RadioConfig:
    on_demand: bool = True                # speak only when asked, except proactive_keys
    proactive_keys: tuple[str, ...] = DEFAULT_PROACTIVE_KEYS
    busy_brake: float = 0.15              # driver is busy above this brake pressure (0-1)...
    busy_steer: float = 0.20              # ... or this steering lock (0-1)...
    busy_lateral_g: float = 1.2           # ... or this lateral G: non-urgent calls wait
    braking_warn_distance_m: float = 220.0  # braking-point call this far before the zone
    delta_report_interval_s: float = 14.0   # seconds between live delta calls


@dataclass
class LapConfig:
    delta_sample_interval_m: float = 10.0   # distance between speed-trace / delta samples
    profile_refresh_laps: int = 3           # rebuild the driver profile every N clean laps


@dataclass
class StrategyConfig:
    rival_react_gap_s: float = 6.0        # only react to a rival's stop within this gap
    rival_forecast_laps: int = 2          # say who is about to box when due within N laps
    rival_preempt_laps: int = 1           # pre-empt an undercut due this lap or the next N
    rival_preempt_prob: float = 0.5       # ... when the stop model gives it at least this chance
    rival_summary_positions: int = 3      # cars this many places either side go in the summary


@dataclass
class SimulationConfig:
    enabled: bool = True                  # Monte Carlo race simulator (win chances, Plan B)
    runs: int = 300                       # races simulated per update
    plan_b_margin_s: float = 3.0          # Plan B must beat Plan A by this much race time...
    plan_b_places: float = 0.5            # ... or by this many finishing places on average


@dataclass
class VoiceConfig:
    tts: str = 'kokoro'                   # 'kokoro' (neural, offline) or 'system' (pyttsx3)
    kokoro_voice: str = 'bm_george'
    kokoro_speed: float = 1.1
    radio_fx: bool = True                 # team-radio filter on the Kokoro voice
    rate_wpm: int = 175                   # system voice speed
    cooldown_s: float = 30.0              # default gap before the same callout repeats


@dataclass
class AIConfig:
    base_url: str = ''                    # '' = $F1_AI_BASE_URL or local Ollama
    model: str = ''                       # '' = $F1_AI_MODEL or deepseek-r1:8b
    timeout_s: float = 30.0
    autostart: bool = True                # start Ollama and pull/preload the model


@dataclass
class NetworkConfig:
    udp_port: int = 20777
    hud_port: int = 5000


@dataclass
class LoggingConfig:
    level: str = 'INFO'                   # DEBUG, INFO, WARNING, ERROR
    format: str = 'text'                  # 'text' for the terminal, 'json' for machines
    file: str = ''                        # also write logs (JSON lines) to this file


@dataclass
class EngineConfig:
    tyres: TyreConfig = field(default_factory=TyreConfig)
    radio: RadioConfig = field(default_factory=RadioConfig)
    laps: LapConfig = field(default_factory=LapConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _coerce(value: Any, hint: Any, where: str) -> Any:
    """Check one TOML value against its field type; ints are accepted for floats."""
    if hint is bool:
        if isinstance(value, bool):
            return value
    elif hint is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif hint is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif hint is str:
        if isinstance(value, str):
            return value
    elif typing.get_origin(hint) is tuple:
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return tuple(value)
    name = getattr(hint, '__name__', str(hint))
    raise ConfigError(f'{where}: expected {name}, got {value!r}')


def _section(cls: type[Any], data: Any, where: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f'[{where}] must be a table')
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        raise ConfigError(f'[{where}] unknown key(s): {", ".join(unknown)}')
    values = {key: _coerce(value, hints[key], f'{where}.{key}') for key, value in data.items()}
    return cls(**values)


def from_dict(data: dict[str, Any]) -> EngineConfig:
    """Build a config from parsed TOML; missing sections and keys keep their defaults."""
    hints = typing.get_type_hints(EngineConfig)
    unknown = sorted(set(data) - set(hints))
    if unknown:
        raise ConfigError(f'unknown section(s): {", ".join(unknown)}')
    sections = {name: _section(hints[name], value, name) for name, value in data.items()}
    return EngineConfig(**sections)


def load_config(path: Path | None = None, *, required: bool = False) -> EngineConfig:
    """Read ``path`` (default: config.toml at the repo root).

    A missing default file gives the built-in defaults; a missing file that was asked for
    explicitly (``required=True``) is an error.
    """
    target = path if path is not None else DEFAULT_CONFIG_PATH
    try:
        with open(target, 'rb') as handle:
            data = tomllib.load(handle)
    except FileNotFoundError:
        if required:
            raise ConfigError(f'config file not found: {target}') from None
        return EngineConfig()
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f'{target}: {error}') from error
    try:
        return from_dict(data)
    except ConfigError as error:
        raise ConfigError(f'{target}: {error}') from error


def apply_overrides(config: EngineConfig, overrides: dict[str, Any]) -> EngineConfig:
    """A copy of ``config`` with ``{'section.key': value}`` overrides; None means "not given".

    main.py passes its command-line flags through here, so a flag always beats the file.
    """
    sections = {f.name: getattr(config, f.name) for f in dataclasses.fields(config)}
    for dotted, value in overrides.items():
        if value is None:
            continue
        section, _, key = dotted.partition('.')
        if section not in sections or key not in {f.name for f in dataclasses.fields(sections[section])}:
            raise ConfigError(f'unknown setting {dotted!r}')
        hint = typing.get_type_hints(type(sections[section]))[key]
        sections[section] = dataclasses.replace(sections[section],
                                                **{key: _coerce(value, hint, dotted)})
    return EngineConfig(**sections)
