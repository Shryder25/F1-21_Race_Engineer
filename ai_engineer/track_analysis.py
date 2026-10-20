"""
Per-track analysis engine for the HUD's Track Database page
===========================================================
Turns a circuit's real-race profile (track_profiles.py) into an engineer's view of the
weekend, separately for dry and wet running:

* tyre degradation per compound (seconds lost per lap of tyre age, fuel-corrected) and a
  Low / Medium / High / Very High label computed from those rates, so it differs by track
  and by conditions instead of being a stored string;
* how long each compound lasts and the fastest pit strategy for the chosen race distance;
* a setup that actually changes for the wet (more wing, softer car, higher ride, more open
  diff, gentler brakes), plus driver handling-preference tweaks;
* a DeepSeek pass (ai_client.chat_json) that reasons over all of the above and returns its
  own degradation verdict, strategy, setup numbers and briefing. Its numbers are validated
  and clamped to the game's ranges, and the model output is the fallback when DeepSeek is
  offline, so the page always has an answer.

The degradation method follows IAmTomShaw/f1-race-replay's Bayesian tyre model: a
compound base rate scaled by a track abrasion factor, with measured real-race rates (from
tools/build_real_race_calibration.py) blended over the prior when available.
"""

from __future__ import annotations

import copy
import threading
import time
from itertools import combinations_with_replacement
from typing import Any, Dict, List, Optional, Tuple

from track_data import TRACK_DATABASE, resolve_track_id
from track_profiles import get_profile

SLICKS = ('SOFT', 'MEDIUM', 'HARD')
COMPOUND_SHORT = {'SOFT': 'S', 'MEDIUM': 'M', 'HARD': 'H', 'INTERMEDIATE': 'I', 'WET': 'W'}

# Fuel-corrected degradation (s/lap of tyre age) on a neutral circuit, and the prior weight
# kept when blending with measured real-race rates (f1-race-replay uses 0.3 for slicks).
BASE_DEG = {'SOFT': 0.100, 'MEDIUM': 0.055, 'HARD': 0.032, 'INTERMEDIATE': 0.055, 'WET': 0.035}
PRIOR_WEIGHT = {'SOFT': 0.3, 'MEDIUM': 0.3, 'HARD': 0.3, 'INTERMEDIATE': 0.4, 'WET': 0.5}
# Fresh-tyre pace gap to the soft (s/lap) and laps before the cliff on a neutral ~5.3 km lap.
PACE_OFFSET = {'SOFT': 0.0, 'MEDIUM': 0.35, 'HARD': 0.7, 'INTERMEDIATE': 0.0, 'WET': 1.5}
BASE_LIFE = {'SOFT': 20, 'MEDIUM': 30, 'HARD': 42, 'INTERMEDIATE': 32, 'WET': 38}
REFERENCE_LAP_M = 5300.0

DRY_LEVELS = ((0.045, 'Low'), (0.058, 'Medium'), (0.068, 'High'))   # on medium-compound rate
WET_LEVELS = ((0.050, 'Low'), (0.062, 'Medium'))                     # on intermediate rate
DEG_LEVELS = ('Low', 'Medium', 'High', 'Very High')

# F1 2021 setup ranges: (min, max, decimals)
SETUP_RANGES: Dict[str, Dict[str, Tuple[float, float, int]]] = {
    'aerodynamics': {'front_wing': (1, 11, 0), 'rear_wing': (1, 11, 0)},
    'transmission': {'diff_on_throttle': (50, 100, 0), 'diff_off_throttle': (50, 100, 0)},
    'suspension_geometry': {'front_camber': (-3.50, -2.50, 2), 'rear_camber': (-2.00, -1.00, 2),
                            'front_toe': (0.05, 0.15, 2), 'rear_toe': (0.20, 0.50, 2)},
    'suspension': {'front_suspension': (1, 11, 0), 'rear_suspension': (1, 11, 0),
                   'front_anti_roll': (1, 11, 0), 'rear_anti_roll': (1, 11, 0),
                   'front_ride_height': (1, 11, 0), 'rear_ride_height': (1, 11, 0)},
    'brakes': {'brake_pressure': (80, 100, 0), 'brake_bias': (50, 70, 0)},
    'tyres': {'front_right_pressure': (21.0, 25.0, 1), 'front_left_pressure': (21.0, 25.0, 1),
              'rear_right_pressure': (19.5, 23.0, 1), 'rear_left_pressure': (19.5, 23.0, 1)},
}
SETUP_CATEGORIES = tuple(SETUP_RANGES)


def is_wet(weather: str) -> bool:
    text = (weather or '').lower()
    return any(word in text for word in ('wet', 'rain', 'storm', 'inter'))


# ──────────────────────────────────────────────
# Tyre model
# ──────────────────────────────────────────────

def demand_factor(profile: Dict[str, Any]) -> float:
    """Track abrasion/energy multiplier on tyre degradation (about 0.5 smooth to 1.6 brutal).

    Weighted from the surface roughness, cornering load, traction and overall tyre stress,
    then nudged for track temperature (hotter asphalt = more thermal degradation).
    """
    w = (0.40 * profile['abrasion'] + 0.30 * profile['lateral']
         + 0.15 * profile['traction'] + 0.15 * profile['tyre_stress'])
    factor = 0.5 + (w - 1.0) * 0.275
    thermal = 1.0 + 0.006 * (profile.get('track_temp_c', 35) - 35)
    measured = (profile.get('measured') or {}).get('abrasion_factor')
    if measured:
        factor = 0.5 * factor + 0.5 * float(measured)      # trust real stints as much as ratings
    return round(max(0.4, min(1.8, factor * thermal)), 3)


def degradation_rates(profile: Dict[str, Any]) -> Dict[str, float]:
    """Seconds lost per lap of tyre age for every compound at this circuit."""
    factor = demand_factor(profile)
    wet_factor = 0.6 + 0.4 * factor                         # standing water cools and lubricates
    measured = (profile.get('measured') or {}).get('deg_s_per_lap') or {}
    rates = {}
    for compound, base in BASE_DEG.items():
        prior = base * (wet_factor if compound in ('INTERMEDIATE', 'WET') else factor)
        observed = measured.get(compound)
        if observed is not None and observed >= 0:
            w = PRIOR_WEIGHT[compound]
            prior = w * prior + (1 - w) * float(observed)
        rates[compound] = round(prior, 4)
    return rates


def degradation_level(rates: Dict[str, float], wet: bool) -> str:
    value, levels = (rates['INTERMEDIATE'], WET_LEVELS) if wet else (rates['MEDIUM'], DRY_LEVELS)
    for limit, label in levels:
        if value < limit:
            return label
    return 'High' if wet else 'Very High'


def compound_life(profile: Dict[str, Any], length_m: float) -> Dict[str, int]:
    """Laps each compound lasts before falling off the cliff."""
    factor = demand_factor(profile)
    wet_factor = 0.6 + 0.4 * factor
    scale = REFERENCE_LAP_M / max(2500.0, float(length_m or REFERENCE_LAP_M))
    measured = (profile.get('measured') or {}).get('max_stint_laps') or {}
    life = {}
    for compound, base in BASE_LIFE.items():
        f = wet_factor if compound in ('INTERMEDIATE', 'WET') else factor
        laps = base * scale / f
        if measured.get(compound):
            laps = 0.4 * laps + 0.6 * float(measured[compound])
        life[compound] = max(5, int(round(laps)))
    return life


def _stint_cost(compound: str, laps: int, rates: Dict[str, float]) -> float:
    return PACE_OFFSET[compound] * laps + rates[compound] * laps * (laps - 1) / 2.0


def _optimise_stints(sequence: Tuple[str, ...], total_laps: int, rates: Dict[str, float],
                     life: Dict[str, int]) -> Optional[Tuple[float, List[int]]]:
    """Cheapest split of the race into len(sequence) stints (dynamic programming)."""
    best: Dict[int, Tuple[float, List[int]]] = {0: (0.0, [])}
    for index, compound in enumerate(sequence):
        remaining_stints = len(sequence) - index - 1
        nxt: Dict[int, Tuple[float, List[int]]] = {}
        for done, (cost, stints) in best.items():
            for laps in range(1, life[compound] + 1):
                covered = done + laps
                if covered > total_laps or (remaining_stints == 0 and covered != total_laps):
                    continue
                if remaining_stints and covered >= total_laps:
                    continue
                total = cost + _stint_cost(compound, laps, rates)
                if covered not in nxt or total < nxt[covered][0]:
                    nxt[covered] = (total, stints + [laps])
        best = nxt
        if not best:
            return None
    return best.get(total_laps)


def plan_strategy(profile: Dict[str, Any], length_m: float, rates: Dict[str, float],
                  life: Dict[str, int], wet: bool, race_fraction: float = 1.0) -> Dict[str, Any]:
    """Fastest stop strategy for the race distance, plus the best plan with a different stop count."""
    total_laps = max(3, int(round(profile['race_laps'] * race_fraction)))
    pit_loss = float(profile['pit_loss_s'])
    # Each stop also risks track position; worth more where passing is hard (Monaco, Hungary).
    position_cost = max(0, profile['overtaking'] - 2) * 3.0
    if wet:
        candidates = [('INTERMEDIATE',) * n for n in range(1, 4)]
    else:
        candidates = [seq for n in range(2, 5) for seq in combinations_with_replacement(SLICKS, n)
                      if len(set(seq)) >= 2]                   # two dry compounds are mandatory
    plans = []
    for seq in candidates:
        result = _optimise_stints(seq, total_laps, rates, life)
        if result is None:
            continue
        cost, stints = result
        stops = len(seq) - 1
        plans.append({
            'stops': stops,
            'stints': [{'compound': c, 'laps': n} for c, n in zip(seq, stints)],
            'label': '-'.join(COMPOUND_SHORT[c] for c in seq),
            'time_loss_s': round(cost + stops * (pit_loss + position_cost), 1),
        })
    plans.sort(key=lambda p: p['time_loss_s'])
    best = plans[0] if plans else None
    alternative = next((p for p in plans if best and p['stops'] != best['stops']), None)
    notes = []
    if best and alternative:
        gap = alternative['time_loss_s'] - best['time_loss_s']
        notes.append(f"{alternative['stops']}-stop is {gap:.1f}s slower on paper")
    if profile['sc_probability'] >= 0.6:
        notes.append(f"Safety Car in {int(profile['sc_probability'] * 100)}% of real races here: a stop under SC "
                     f"costs about {pit_loss * 0.5:.0f}s instead of {pit_loss:.0f}s, so keep the window flexible")
    if profile['overtaking'] >= 4:
        notes.append('Overtaking is very hard here, so track position beats fresher tyres')
    elif profile['overtaking'] <= 1:
        notes.append('Easy to overtake, so an aggressive undercut or extra stop is viable')
    if wet:
        notes.append('Watch the crossover: switch to slicks when inters are about 1s off a dry line lap time')
    return {'race_laps': total_laps, 'pit_loss_s': pit_loss, 'best': best,
            'alternative': alternative, 'notes': notes}


# ──────────────────────────────────────────────
# Setup model
# ──────────────────────────────────────────────

def clamp_setup(setup: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for category, fields in SETUP_RANGES.items():
        values = setup.get(category) or {}
        out[category] = {}
        for name, (low, high, decimals) in fields.items():
            if name not in values:
                continue
            value = max(low, min(high, float(values[name])))
            out[category][name] = int(round(value)) if decimals == 0 else round(value, decimals)
    if 'notes' in setup:
        out['notes'] = setup['notes']
    return out


def _wing_step(value: int) -> int:
    return 3 if value <= 4 else 2 if value <= 8 else 1


def wet_setup(dry: Dict[str, Any], wet_adjustments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Convert a dry setup into a wet one, numerically.

    More wing (most where the dry car is trimmed out), a more open on-throttle diff for
    traction, less camber for a flatter contact patch, a softer car with more ride height
    for bumps and standing water, lower brake pressure and a rearward bias against front
    lock-ups, and a little more tyre pressure to get temperature into the tyres.
    """
    s = copy.deepcopy(dry)
    aero, trans = s['aerodynamics'], s['transmission']
    aero['front_wing'] += _wing_step(aero['front_wing'])
    aero['rear_wing'] += _wing_step(aero['rear_wing'])
    trans['diff_on_throttle'] -= 10
    trans['diff_off_throttle'] -= 2
    geo = s['suspension_geometry']
    geo['front_camber'] += 0.30
    geo['rear_camber'] += 0.20
    sus = s['suspension']
    for key in ('front_suspension', 'rear_suspension', 'front_anti_roll', 'rear_anti_roll'):
        sus[key] -= 1
    sus['front_ride_height'] += 2
    sus['rear_ride_height'] += 2
    brakes = s['brakes']
    brakes['brake_pressure'] -= 6
    bias = str((wet_adjustments or {}).get('brake_bias', '')).rstrip('%').strip()
    try:
        brakes['brake_bias'] = float(bias)
    except ValueError:
        brakes['brake_bias'] -= 2
    for key in s['tyres']:
        s['tyres'][key] += 0.4
    s['notes'] = (wet_adjustments or {}).get('notes') or 'Wet baseline: more wing, softer car, open diff.'
    return clamp_setup(s)


HANDLING_TWEAKS = {
    'oversteer_fix': ({'aerodynamics.rear_wing': 1, 'transmission.diff_on_throttle': -5,
                       'suspension.rear_anti_roll': -1, 'brakes.brake_bias': 1},
                      'Loose rear: +1 rear wing, more open on-throttle diff, softer rear bar, bias forward.'),
    'understeer_fix': ({'aerodynamics.front_wing': 1, 'suspension.front_anti_roll': -1,
                        'suspension_geometry.front_camber': -0.10, 'brakes.brake_bias': -1},
                       'Lazy front: +1 front wing, softer front bar, more front camber, bias rearward.'),
    'top_speed': ({'aerodynamics.front_wing': -1, 'aerodynamics.rear_wing': -2,
                   'suspension.rear_ride_height': -1},
                  'Trimmed for straight-line speed: wings down and rake reduced.'),
}


def apply_handling(setup: Dict[str, Any], handling: str) -> Tuple[Dict[str, Any], Optional[str]]:
    tweak = HANDLING_TWEAKS.get(handling)
    if not tweak:
        return setup, None
    s = copy.deepcopy(setup)
    for path, delta in tweak[0].items():
        category, name = path.split('.')
        s[category][name] += delta
    return clamp_setup(s), tweak[1]


# ──────────────────────────────────────────────
# Full model analysis
# ──────────────────────────────────────────────

def _deg_reasoning(profile, rates, level, wet, track_name) -> str:
    demands = []
    if profile['abrasion'] >= 4:
        demands.append('a very abrasive surface')
    elif profile['abrasion'] <= 1:
        demands.append('a smooth, low-abrasion surface')
    if profile['lateral'] >= 4:
        demands.append('high-energy fast corners')
    if profile['traction'] >= 4:
        demands.append('heavy traction out of slow corners')
    if profile.get('track_temp_c', 35) >= 40 and not wet:
        demands.append(f"hot asphalt (~{profile['track_temp_c']}°C)")
    what = ', '.join(demands) or 'balanced demands'
    axle = 'front' if profile['limiting'] == 'front' else 'rear'
    if wet:
        return (f"{level} degradation in the wet at {track_name}: water keeps the tyres cool, inters lose "
                f"{rates['INTERMEDIATE']:.3f}s per lap. Expect them to overheat fast if the line dries.")
    return (f"{level} degradation at {track_name} from {what}. Mediums lose {rates['MEDIUM']:.3f}s per lap of age, "
            f"softs {rates['SOFT']:.3f}s. The {axle} axle is the limiting tyre.")


def model_analysis(track_id: int, weather: str = 'dry', handling: str = 'balanced',
                   race_fraction: float = 1.0) -> Dict[str, Any]:
    """Instant, data-driven analysis of a circuit for the given conditions (no AI needed)."""
    db_id = resolve_track_id(int(track_id))
    if db_id not in TRACK_DATABASE:
        db_id = 7
    circuit = TRACK_DATABASE[db_id]
    profile = get_profile(db_id)
    wet = is_wet(weather)
    rates = degradation_rates(profile)
    life = compound_life(profile, circuit['length_m'])
    level = degradation_level(rates, wet)
    strategy = plan_strategy(profile, circuit['length_m'], rates, life, wet, race_fraction)

    setup = clamp_setup(circuit['setup'])
    setup['notes'] = circuit['setup'].get('notes', '')
    if wet:
        setup = wet_setup(setup, circuit.get('wet_adjustments'))
    setup, handling_note = apply_handling(setup, handling)

    compounds = ('INTERMEDIATE', 'WET') if wet else SLICKS
    best = strategy['best']
    plan_text = ''
    if best:
        plan_text = ', '.join(f"{s['compound'].title()} {s['laps']} laps" for s in best['stints'])
    briefing = [
        f"{circuit['name']} ({'WET' if wet else 'DRY'}): {_deg_reasoning(profile, rates, level, wet, circuit['short_name'])}",
        f"Strategy for {strategy['race_laps']} laps: {best['stops'] if best else '?'}-stop, {plan_text}." if best else '',
        setup.get('notes', ''),
        handling_note or '',
    ]
    return {
        'track_id': db_id,
        'track_name': circuit['name'],
        'short_name': circuit['short_name'],
        'country': circuit['country'],
        'characteristics': circuit['characteristics'],
        'length_m': circuit['length_m'],
        'turns': circuit['turns'],
        'downforce_level': circuit['downforce'],
        'weather': 'wet' if wet else 'dry',
        'handling': handling,
        'race_fraction': race_fraction,
        'tyre_degradation': level,
        'degradation': {
            'rates_s_per_lap': {c: rates[c] for c in compounds},
            'life_laps': {c: life[c] for c in compounds},
            'demand_factor': demand_factor(profile),
            'limiting_axle': profile['limiting'],
            'reasoning': _deg_reasoning(profile, rates, level, wet, circuit['short_name']),
        },
        'strategy': strategy,
        'demands': {k: profile[k] for k in ('abrasion', 'grip', 'traction', 'braking', 'lateral',
                                             'tyre_stress', 'downforce', 'evolution')},
        'race_facts': {k: profile[k] for k in ('pit_loss_s', 'sc_probability', 'rain_probability',
                                                'overtaking', 'track_temp_c', 'race_laps')},
        'setup': setup,
        'setup_notes': {},
        'key_corners': [name for _d, name, _m in circuit.get('braking_points', [])][:4],
        'wet_adjustments': circuit.get('wet_adjustments') if wet else None,
        'engineer_advice': ' '.join(part for part in briefing if part),
        'analysis_source': 'model',
        'data_source': profile['source'],
        'calibration': (profile.get('measured') or {}).get('races'),
    }


# ──────────────────────────────────────────────
# DeepSeek analysis
# ──────────────────────────────────────────────

ANALYSIS_SYSTEM_PROMPT = """You are the chief race engineer of a Formula 1 team preparing a driver for a race in the F1 2021 game.
You are given real-world data about the circuit (Pirelli severity ratings and real race history) and a quantitative tyre and strategy model built from it.
Analyse THIS circuit in THESE conditions. Your numbers must be specific to the track and must differ between dry and wet.
Reply with ONE JSON object only, with exactly these keys:
{
 "tyre_degradation": "Low" | "Medium" | "High" | "Very High",
 "degradation_reasoning": "1-2 sentences on why, naming the corners or surface that cause it",
 "limiting_tyre": "e.g. front-left",
 "strategy": {"stops": int, "plan": "e.g. Medium 22 laps, Hard 30 laps", "alternative": "...", "reasoning": "1-2 sentences"},
 "setup": {"aerodynamics": {"front_wing": int, "rear_wing": int},
           "transmission": {"diff_on_throttle": int, "diff_off_throttle": int},
           "suspension_geometry": {"front_camber": float, "rear_camber": float, "front_toe": float, "rear_toe": float},
           "suspension": {"front_suspension": int, "rear_suspension": int, "front_anti_roll": int, "rear_anti_roll": int, "front_ride_height": int, "rear_ride_height": int},
           "brakes": {"brake_pressure": int, "brake_bias": int},
           "tyres": {"front_right_pressure": float, "front_left_pressure": float, "rear_right_pressure": float, "rear_left_pressure": float}},
 "setup_notes": {"aerodynamics": "...", "transmission": "...", "suspension_geometry": "...", "suspension": "...", "brakes": "...", "tyres": "..."},
 "key_corners": ["up to 4 corners that decide the lap and why, short"],
 "briefing": "3-4 sentence pre-race radio briefing to the driver"
}
F1 2021 ranges: wings 1-11, diffs 50-100%, front camber -3.50 to -2.50, rear camber -2.00 to -1.00, front toe 0.05-0.15, rear toe 0.20-0.50, suspension/anti-roll/ride height 1-11, brake pressure 80-100%, brake bias 50-70%, front tyres 21.0-25.0 psi, rear tyres 19.5-23.0 psi.
Start from the baseline setup and change values only where the track, conditions or driver preference justify it."""


def build_analysis_prompt(model: Dict[str, Any]) -> Tuple[str, str]:
    profile_lines = ', '.join(f'{k} {v}/5' for k, v in model['demands'].items())
    facts = model['race_facts']
    deg = model['degradation']
    strat = model['strategy']
    best = strat['best'] or {}
    alt = strat['alternative'] or {}
    fmt = lambda plan: ', '.join(f"{s['compound']} {s['laps']}" for s in plan.get('stints', [])) or 'n/a'
    calibration = (f"Calibrated from real races: {', '.join(model['calibration'])}"
                   if model.get('calibration') else 'Ratings from Pirelli circuit data and real race history')
    handling = {'balanced': 'balanced', 'oversteer_fix': 'car is oversteering (loose rear)',
                'understeer_fix': 'car is understeering', 'top_speed': 'wants maximum straight-line speed'}
    user = (
        f"Circuit: {model['track_name']} ({model['country']}), {model['length_m']} m, {model['turns']} turns\n"
        f"Profile: {model['characteristics']}\n"
        f"Conditions: {model['weather'].upper()}\n"
        f"Race distance: {strat['race_laps']} laps\n"
        f"Driver feedback: {handling.get(model['handling'], model['handling'])}\n\n"
        f"Real-world circuit data ({calibration}):\n"
        f"  Severity: {profile_lines}\n"
        f"  Pit-lane loss {facts['pit_loss_s']}s, Safety Car in {int(facts['sc_probability'] * 100)}% of races, "
        f"rain in {int(facts['rain_probability'] * 100)}% of races, typical track temp {facts['track_temp_c']}C, "
        f"overtaking difficulty {facts['overtaking']}/5\n\n"
        f"Tyre model for these conditions:\n"
        f"  Degradation (s lost per lap of tyre age): {deg['rates_s_per_lap']}\n"
        f"  Laps before the cliff: {deg['life_laps']}\n"
        f"  Model verdict: {model['tyre_degradation']} degradation, limiting axle {deg['limiting_axle']}\n"
        f"  Fastest strategy: {best.get('stops', '?')}-stop {fmt(best)} (loss {best.get('time_loss_s', '?')}s); "
        f"alternative {alt.get('stops', '?')}-stop {fmt(alt)} (loss {alt.get('time_loss_s', '?')}s)\n"
        f"  Notes: {'; '.join(strat['notes']) or 'none'}\n\n"
        f"Baseline setup for these conditions: {_setup_compact(model['setup'])}\n"
        f"Key braking zones: {', '.join(model['key_corners'])}\n"
    )
    return ANALYSIS_SYSTEM_PROMPT, user


def _setup_compact(setup: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in setup.items() if k in SETUP_CATEGORIES}


def merge_ai_analysis(model: Dict[str, Any], reply: Dict[str, Any]) -> Dict[str, Any]:
    """Validate DeepSeek's JSON and lay it over the model analysis (model fills any gaps)."""
    if not isinstance(reply, dict):
        raise ValueError('reply is not an object')
    result = copy.deepcopy(model)
    level = str(reply.get('tyre_degradation', '')).strip().title()
    if level in DEG_LEVELS:
        result['tyre_degradation'] = level
    if reply.get('degradation_reasoning'):
        result['degradation']['reasoning'] = str(reply['degradation_reasoning'])[:600]
    if reply.get('limiting_tyre'):
        result['degradation']['limiting_tyre'] = str(reply['limiting_tyre'])[:40]

    ai_setup = reply.get('setup')
    if isinstance(ai_setup, dict):
        merged = copy.deepcopy(_setup_compact(model['setup']))
        for category in SETUP_CATEGORIES:
            values = ai_setup.get(category)
            if not isinstance(values, dict):
                continue
            for name, raw in values.items():
                if name in SETUP_RANGES[category]:
                    try:
                        merged[category][name] = float(raw)
                    except (TypeError, ValueError):
                        pass
        notes = model['setup'].get('notes')
        result['setup'] = clamp_setup(merged)
        result['setup']['notes'] = notes
    notes = reply.get('setup_notes')
    if isinstance(notes, dict):
        result['setup_notes'] = {k: str(v)[:300] for k, v in notes.items() if k in SETUP_CATEGORIES and v}

    strategy = reply.get('strategy')
    if isinstance(strategy, dict):
        result['strategy']['ai'] = {k: str(strategy[k])[:400] for k in ('plan', 'alternative', 'reasoning')
                                    if strategy.get(k)}
        try:
            result['strategy']['ai']['stops'] = int(strategy.get('stops'))
        except (TypeError, ValueError):
            pass
    corners = reply.get('key_corners')
    if isinstance(corners, list) and corners:
        result['key_corners'] = [str(c)[:160] for c in corners[:4]]
    if reply.get('briefing'):
        result['engineer_advice'] = str(reply['briefing'])[:1200]
    result['analysis_source'] = 'deepseek'
    return result


class TrackAnalyst:
    """Runs and caches DeepSeek track analyses for the HUD (thread-safe)."""

    FAILURE_RETRY_S = 60.0

    def __init__(self, ai_client=None, timeout: float = 90.0):
        self.ai_client = None
        self.timeout = timeout
        self._cache: Dict[tuple, Dict[str, Any]] = {}
        self._failures: Dict[tuple, Tuple[float, str]] = {}
        self._inflight: Dict[tuple, threading.Event] = {}
        self._lock = threading.Lock()
        self.set_client(ai_client)

    def set_client(self, client) -> None:
        """Use the engineer's DeepSeek client with a longer timeout (a full analysis is long)."""
        if client is not None and hasattr(client, 'timeout'):
            client = copy.copy(client)
            client.timeout = max(float(client.timeout or 0), self.timeout)
        with self._lock:
            self.ai_client = client
            self._cache.clear()
            self._failures.clear()

    @staticmethod
    def _key(track_id, weather, handling, race_fraction) -> tuple:
        return (resolve_track_id(int(track_id)), 'wet' if is_wet(weather) else 'dry', handling,
                round(float(race_fraction), 2))

    def cached(self, track_id, weather='dry', handling='balanced', race_fraction=1.0) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._cache.get(self._key(track_id, weather, handling, race_fraction))

    def analyze(self, track_id: int, weather: str = 'dry', handling: str = 'balanced',
                race_fraction: float = 1.0, force: bool = False) -> Dict[str, Any]:
        """DeepSeek analysis for these settings, falling back to the model analysis."""
        key = self._key(track_id, weather, handling, race_fraction)
        model = model_analysis(track_id, weather, handling, race_fraction)
        if self.ai_client is None:
            model['ai_error'] = 'DeepSeek is not configured (started with --no-ai)'
            return model
        while True:
            with self._lock:
                if not force and key in self._cache:
                    return self._cache[key]
                failure = self._failures.get(key)
                if not force and failure and time.monotonic() - failure[0] < self.FAILURE_RETRY_S:
                    model['ai_error'] = failure[1]
                    return model
                event = self._inflight.get(key)
                if event is None:
                    event = self._inflight[key] = threading.Event()
                    break
            event.wait(self.timeout * 2)               # someone else is already asking DeepSeek
            force = False
        try:
            system, user = build_analysis_prompt(model)
            reply = self.ai_client.chat_json(system, user, temperature=0.4, max_tokens=2000)
            result = merge_ai_analysis(model, reply)
            with self._lock:
                self._cache[key] = result
                self._failures.pop(key, None)
            return result
        except Exception as error:              # AIUnavailable, bad JSON shape, anything else
            message = f'DeepSeek unavailable: {error}'
            with self._lock:
                self._failures[key] = (time.monotonic(), message)
            model['ai_error'] = message
            return model
        finally:
            with self._lock:
                self._inflight.pop(key, None)
            event.set()
