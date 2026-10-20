"""The engineer's brain: turns a question plus live data into a short radio message.

Flow for every voice command:
  1. ``build_context`` (AnalysisEngine) -> telemetry stats + ML predictions
  2. ``build_prompt`` -> system prompt + data block for DeepSeek
  3. ``AIClient.chat`` -> radio-style answer
  4. If the model is slow, offline or returns junk, ``rule_based_answer`` still
     answers from the same data, so a command never goes unanswered.
"""

from __future__ import annotations

import re
from typing import Callable

from ai_client import AIClient, AIUnavailable
import session_modes

COMMAND_QUESTIONS = {
    'weather': 'Give me a weather report.',
    'tyre_temps': 'What are my tyre temperatures and how are the tyres holding up?',
    'pit_window': 'When is my pit window?',
    'gap_ahead': 'What is the gap to the car ahead?',
    'gap_behind': 'What is the gap to the car behind?',
    'pace': 'Give me a pace update.',
}

SYSTEM_PROMPT = """You are the driver's Formula 1 race engineer speaking over team radio.
Rules:
- Reply in at most two short sentences, under 25 words total, like a real F1 radio call.
- Plain spoken English only: no markdown, lists, emojis or labels. Say numbers naturally (for example "two point four").
- Use ONLY the data provided. Never invent values. If a value is missing or null, say you do not have it yet.
- Be direct and calm. Lead with the answer, then one useful instruction if it adds value.
- The data states the session mode. Time trial: no rivals, no pit stops. Qualifying: no pit stops and rival gaps are not tracked. Free practice: gaps are not tracked. If asked about something the mode does not have, say so in a few words instead of inventing numbers.
- Tyre temperatures are surface temperatures in Celsius; wear is a percentage. Gaps are in seconds.
Example: "Gap ahead is two point four, tyres are good, keep pushing."
"""


def _fmt_lap(ms):
    if not ms:
        return None
    seconds = ms / 1000
    return f'{int(seconds // 60)}:{seconds % 60:06.3f}'


def format_context(ctx: dict) -> str:
    """Compact, labelled data block for the prompt (None values are left out)."""
    lines = []

    def add(label, value):
        if value is not None and value != {} and value != []:
            lines.append(f'{label}: {value}')

    add('Track', ctx.get('track'))
    add('Session', ctx.get('session_type'))
    add('Session mode', ctx.get('session_mode') if ctx.get('session_mode') != 'unknown' else None)
    add('Lap', f"{ctx['lap']} of {ctx['total_laps']}" if ctx.get('lap') and ctx.get('total_laps')
        else ctx.get('lap'))
    add('Position', ctx.get('position'))
    add('Weather', ctx.get('weather'))
    add('Track temp C', ctx.get('track_temp_c'))
    add('Air temp C', ctx.get('air_temp_c'))
    add('Safety car out', ctx.get('safety_car') or None)
    add('Gap to car ahead (s)', None if ctx.get('gap_ahead_s') is None else round(ctx['gap_ahead_s'], 1))
    add('Gap to car behind (s)', None if ctx.get('gap_behind_s') is None else round(ctx['gap_behind_s'], 1))
    add('Last lap', _fmt_lap(ctx.get('last_lap_ms')))
    add('Best lap', _fmt_lap(ctx.get('best_lap_ms')))
    add('Race pace (avg of last 3)', _fmt_lap(ctx.get('race_pace_ms')))
    add('Delta to best lap (s)', None if ctx.get('delta_to_best_ms') is None
        else round(ctx['delta_to_best_ms'] / 1000, 2))
    add('Tyre fall-off, fuel-corrected (s per lap, + means slower)', ctx.get('pace_trend_s_per_lap'))
    add('Tyre compound', ctx.get('compound'))
    add('Tyre age (laps)', ctx.get('tyre_age_laps'))
    add('Tyre surface temps C', ctx.get('tyre_surface_c'))
    add('Tyre wear %', ctx.get('tyre_wear_pct'))
    add('Game pit window (ideal, latest lap)', ctx.get('game_pit_window'))
    prediction = ctx.get('pit_prediction')
    if prediction:
        add('ML pit prediction', (
            f"worst-tyre wear {prediction['wear_now_pct']}%, +{prediction['wear_per_lap_pct']}% per lap, "
            f"ideal stop lap {prediction['ideal_lap']}, latest lap {prediction['latest_lap']}, "
            f"can finish without stopping: {prediction['finish_without_stop']}, "
            f"based on {prediction['samples']} laps"))
    else:
        add('ML pit prediction', 'not enough lap data yet')
    fuel = ctx.get('fuel')
    if fuel:
        add('Fuel', f"{fuel['fuel_kg']} kg, burn {fuel['per_lap_kg']} kg per lap, "
                    f"{fuel['laps_of_fuel']} laps of fuel, margin {fuel['margin_laps']} laps")
    add('Last lap sector times ms', ctx.get('last_lap_sectors_ms'))
    add('Last lap sector deltas to personal best ms', ctx.get('last_lap_sector_deltas_ms'))
    return '\n'.join(lines) if lines else 'No telemetry received yet.'


def build_prompt(command: str, ctx: dict) -> tuple[str, str]:
    """Return (system prompt, user message) for DeepSeek."""
    question = COMMAND_QUESTIONS.get(command, command)
    user = (f'Driver asks: "{question}"\n\nCurrent data:\n{format_context(ctx)}\n\n'
            'Answer as the race engineer.')
    return SYSTEM_PROMPT, user


_MARKDOWN = re.compile(r'[*_`#>]+')


def clean_radio_text(text: str, max_words: int = 45) -> str:
    """Strip formatting and keep the answer radio-short (whole sentences where possible)."""
    text = _MARKDOWN.sub('', text)
    text = re.sub(r'\s+', ' ', text).strip().strip('"')
    if not text:
        return ''
    sentences = re.split(r'(?<=[.!?])\s+', text)
    kept, words = [], 0
    for sentence in sentences:
        count = len(sentence.split())
        if kept and words + count > max_words:
            break
        kept.append(sentence)
        words += count
    result = ' '.join(kept)
    parts = result.split()
    if len(parts) > max_words:
        result = ' '.join(parts[:max_words]).rstrip(',;:') + '.'
    return result


# ──────────────────────────────────────────────
# Rule-based fallback (no AI needed)
# ──────────────────────────────────────────────

def _say_seconds(value: float) -> str:
    return f'{value:.1f}'


def _mode_blocks(command: str, ctx: dict) -> str | None:
    """A short reply when the current session has no such thing as the request, else None."""
    mode = ctx.get('session_mode')
    if not mode or mode == session_modes.UNKNOWN:
        return None
    label = session_modes.MODE_LABELS[mode]
    if command in ('gap_ahead', 'gap_behind') and not session_modes.reports_gaps(mode):
        return 'No other cars in time trial.' if mode == session_modes.TIME_TRIAL \
            else f'Gaps are only tracked in the race, not in {label.lower()}.'
    if command == 'pit_window' and not session_modes.has_pit_strategy(mode):
        return f'No pit stops in {label.lower()}.'
    return None


def rule_based_answer(command: str, ctx: dict) -> str:
    blocked = _mode_blocks(command, ctx)
    if blocked:
        return blocked
    if command == 'weather':
        if not ctx.get('weather'):
            return 'No weather data yet.'
        return (f"{ctx['weather'].capitalize()}. Track {ctx.get('track_temp_c')} degrees, "
                f"air {ctx.get('air_temp_c')}.")
    if command in ('gap_ahead', 'gap_behind'):
        key, name = ('gap_ahead_s', 'ahead') if command == 'gap_ahead' else ('gap_behind_s', 'behind')
        gap = ctx.get(key)
        if gap is None:
            return f'No gap {name} reading right now.'
        return f'Gap {name} is {_say_seconds(gap)}.'
    if command == 'tyre_temps':
        temps = ctx.get('tyre_surface_c')
        if not temps:
            return 'No tyre data yet.'
        text = (f"Fronts {temps['FL']} and {temps['FR']}, rears {temps['RL']} and {temps['RR']}.")
        wear = ctx.get('tyre_wear_pct')
        if wear:
            text += f" Worst wear {max(wear.values()):.0f} percent."
        return text
    if command == 'pit_window':
        prediction = ctx.get('pit_prediction')
        if prediction and prediction.get('finish_without_stop'):
            return 'Tyres should reach the flag. No stop needed at the moment.'
        if prediction and prediction.get('ideal_lap'):
            return (f"Box around lap {prediction['ideal_lap']}, "
                    f"latest lap {prediction['latest_lap'] or 'unknown'}.")
        window = ctx.get('game_pit_window')
        if window:
            return f'Game window is lap {window[0]} to {window[1]}.'
        return 'Not enough laps on these tyres to predict a window yet.'
    if command == 'pace':
        parts = []
        if ctx.get('last_lap_ms'):
            parts.append(f"Last lap {_fmt_lap(ctx['last_lap_ms'])}.")
        delta = ctx.get('delta_to_best_ms')
        if delta is not None:
            parts.append(f"{abs(delta) / 1000:.1f} {'slower' if delta > 0 else 'faster'} than your best.")
        trend = ctx.get('pace_trend_s_per_lap')
        if trend is not None:
            parts.append(f"Tyres costing {max(trend, 0):.2f} a lap.")
        return ' '.join(parts) or 'No pace data yet.'
    return 'Say again.'


class EngineerBrain:
    def __init__(self, context_provider: Callable[[], dict], client: AIClient | None = None):
        self.context_provider = context_provider
        self.client = client
        self.last_source = None   # 'ai' or 'fallback' - useful for the HUD and tests

    def answer(self, command: str) -> str:
        ctx = self.context_provider()
        blocked = _mode_blocks(command, ctx)
        if blocked:                       # e.g. "gap ahead" in time trial: answer instantly, no AI needed
            self.last_source = 'rules'
            return blocked
        if self.client is not None:
            try:
                system, user = build_prompt(command, ctx)
                text = clean_radio_text(self.client.chat(system, user))
                if text:
                    self.last_source = 'ai'
                    return text
            except AIUnavailable as error:
                print(f'[Engineer] AI unavailable ({error}); answering from rules.')
        self.last_source = 'fallback'
        return rule_based_answer(command, ctx)
