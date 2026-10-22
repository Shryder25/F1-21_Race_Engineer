"""The engineer's brain: turns a question plus live data into a short radio message.

Flow for every voice command:
  1. ``build_context`` (AnalysisEngine) -> telemetry stats + ML predictions
  2. ``build_prompt`` -> system prompt + data block for DeepSeek
  3. ``AIClient.chat_json`` -> {"voice": "...", "advice": "..."}, forcing the model's JSON
     mode instead of trusting it to follow a length/format instruction in free text (the
     approach used by Box-Box's analyzer.py). ``voice`` is the short radio line that gets
     spoken; ``advice`` is an optional longer note for the on-screen radio log. Falls back to
     plain-text ``chat`` + prose cleanup for a server that cannot do JSON mode at all.
  4. If the model is slow, offline or returns junk, ``rule_based_answer`` still
     answers from the same data, so a command never goes unanswered.
"""

from __future__ import annotations

import re
from collections.abc import Callable

import session_modes
from ai_client import AIClient, AITimeout, AIUnavailable
from log_setup import get_logger

log = get_logger('engineer')

# Radio lines are under 25 words; with thinking off this leaves plenty of room for the JSON.
RADIO_MAX_TOKENS = 220

COMMAND_QUESTIONS = {
    'weather': 'Give me a weather report.',
    'tyre_temps': 'What are my tyre temperatures and how are the tyres holding up?',
    'pit_window': 'When is my pit window?',
    'gap_ahead': 'What is the gap to the car ahead?',
    'gap_behind': 'What is the gap to the car behind?',
    'pace': 'Give me a pace update.',
    'threats': 'Any threats nearby? Who is attacking or defending?',
    'strategy': 'What is the strategy? How many stops, which tyres, and is the undercut or overcut on?',
    'rivals': 'When are the cars around me going to box, and onto which tyres?',
}

UNKNOWN_QUESTION_REPLY = (
    "Copy, can't answer that one without the AI. "
    "Ask me about gaps, tyres, pace, the pit window, weather or strategy."
)

SYSTEM_PROMPT = """You are the driver's Formula 1 race engineer speaking over team radio.
Rules:
- Reply in at most two short sentences, under 25 words total, like a real F1 radio call.
- Plain spoken English only: no markdown, lists, emojis or labels. Say numbers naturally (for example "two point four").
- Use ONLY the data provided. Never invent values. If a value is missing or null, say you do not have it yet.
- Be direct and calm. Lead with the answer, then one useful instruction if it adds value.
- The data states the session mode. Time trial: no rivals, no pit stops. Qualifying: no pit stops and rival gaps are not tracked. Free practice: gaps are not tracked. If asked about something the mode does not have, say so in a few words instead of inventing numbers.
- Tyre temperatures are surface temperatures in Celsius; wear is a percentage. Gaps are in seconds.
- Tyre wear, degradation, pit-window and fuel predictions come as ranges [low, expected, high] (an 80% range). Quote the expected value and give the range when it matters ("box lap eighteen, window seventeen to twenty"). A wide range or a "built-in estimate" basis means the model is unsure: say so briefly.
- The race strategy line is a model of stop count and compounds; its "slower by" numbers are total race time. Undercut gain is what boxing this lap gains on the car ahead by the time they have stopped too: it works only if the gain beats the gap.
- Rival stop predictions are the model's guess of when nearby cars will box next and which tyre they will fit, from their tyre age, wear and the track's usual strategy. Say "likely" or "expect": they are forecasts, not facts.
- The race simulation line is a Monte Carlo of the whole field (Safety Cars, VSCs, pit-stop spread, traffic, overtaking): quote its odds as odds ("two-stop comes out best 60% of the time"), never as certainties.
- Opponent threats, when present, are nearby cars already ranked by how much pressure they put on the driver; mention one only if it changes what the driver should do right now.
- "Driver's own history at this track" summarises this driver's past laps here. Use it to make advice personal when it fits (e.g. "you usually lose time in sector two here"), never as a substitute for live data.
- When a rival's driver name is given, call them by that name like a real engineer would (for example "Hamilton is one point two ahead"). Never make up a name that is not in the data.
- Sound like a real F1 engineer: calm, clipped, encouraging. Say lap times the way engineers do ("one thirty-two four"), talk about small time differences in tenths ("two tenths up", "a tenth down"), and say "box, box" only for an actual pit call.
- The driver may ask anything in their own words. Answer what they actually asked from the data. If the data cannot answer it, say so in a few words and offer the closest thing you do know.
- Never repeat the driver's question back, never say "as an AI", and avoid filler like "great question" or "currently".
Examples: "Gap ahead two point four, you're catching. Tyres are good, keep pushing." / "Fronts are a bit hot, ninety-eight. Be gentle on turn-in."

Return JSON only, no other text: {"voice": "<the radio line as described above>", "advice": "<one extra sentence of detail for the pit wall screen, or empty string if voice already says it all>"}.
"""


def _fmt_lap(ms):
    if not ms:
        return None
    seconds = ms / 1000
    return f'{int(seconds // 60)}:{seconds % 60:06.3f}'


_BASIS = {'track': 'fitted on this track', 'pooled': 'fitted on other tracks', 'prior': 'built-in estimate'}


def _fmt_range(r) -> str:
    if not r:
        return 'unknown'
    return '[' + ', '.join('?' if v is None else f'{v:g}' for v in r) + ']'


def format_context(ctx: dict, topics: set | None = None) -> str:
    """Compact, labelled data block for the prompt (None values are left out).

    ``topics`` limits the block to the core data plus those topics (see question_topics); None
    sends everything. A shorter prompt is read faster by the model.
    """
    lines = []

    def add(label, value, topic='core'):
        if topics is not None and topic != 'core' and topic not in topics:
            return
        if value is not None and value != {} and value != []:
            lines.append(f'{label}: {value}')

    add('Track', ctx.get('track'))
    add('Session', ctx.get('session_type'))
    add('Session mode', ctx.get('session_mode') if ctx.get('session_mode') != 'unknown' else None)
    add(
        'Lap',
        f"{ctx['lap']} of {ctx['total_laps']}"
        if ctx.get('lap') and ctx.get('total_laps')
        else ctx.get('lap'),
    )
    add('Position', ctx.get('position'))
    add('Weather', ctx.get('weather'), topic='weather')
    conditions = ctx.get('track_conditions')
    if conditions:
        add(
            'Track wetness (0 dry, 1 standing water; from weather and lap times by tyre type)',
            (
                f"{conditions['wetness']} +/- {conditions['sd']}, {conditions['trend']}, quickest tyre "
                f"{conditions['best_tyre']}"
                + (
                    f", switch to {conditions['switch_to']} in {conditions['switch_in_laps']} laps"
                    if conditions.get('switch_to')
                    else ''
                )
                + f"; basis: {conditions['basis']}"
                + (
                    ''.join(
                        f"; forecast +{f['in_min']} min rain {f['rain_pct']}%"
                        for f in conditions['forecast_rain'][:4]
                    )
                )
            ),
            topic='weather',
        )
    add('Track temp C', ctx.get('track_temp_c'), topic='weather')
    add('Air temp C', ctx.get('air_temp_c'), topic='weather')
    add('Safety car out', ctx.get('safety_car') or None)
    add('Gap to car ahead (s)', None if ctx.get('gap_ahead_s') is None else round(ctx['gap_ahead_s'], 1))
    add('Car ahead driver', ctx.get('car_ahead_name'))
    add('Gap to car behind (s)', None if ctx.get('gap_behind_s') is None else round(ctx['gap_behind_s'], 1))
    add('Car behind driver', ctx.get('car_behind_name'))
    add('Last lap', _fmt_lap(ctx.get('last_lap_ms')))
    add('Best lap', _fmt_lap(ctx.get('best_lap_ms')))
    add('Race pace (avg of last 3)', _fmt_lap(ctx.get('race_pace_ms')))
    add(
        'Delta to best lap (s)',
        None if ctx.get('delta_to_best_ms') is None else round(ctx['delta_to_best_ms'] / 1000, 2),
        topic='pace',
    )
    add(
        'Tyre fall-off, fuel-corrected (s per lap, + means slower)',
        ctx.get('pace_trend_s_per_lap'),
        topic='tyres',
    )
    add('Tyre compound', ctx.get('compound'))
    add('Tyre age (laps)', ctx.get('tyre_age_laps'))
    add('Tyre surface temps C', ctx.get('tyre_surface_c'), topic='tyres')
    add('Tyre wear %', ctx.get('tyre_wear_pct'), topic='tyres')
    add('Game pit window (ideal, latest lap)', ctx.get('game_pit_window'), topic='strategy')
    prediction = ctx.get('pit_prediction')
    if prediction:
        add(
            'ML pit prediction',
            (
                f"worst-tyre wear {prediction['wear_now_pct']}%, +{prediction['wear_per_lap_pct']}% per lap "
                f"(range {_fmt_range(prediction.get('wear_per_lap_range'))}), "
                f"ideal stop lap {prediction['ideal_lap']} (range {_fmt_range(prediction.get('ideal_lap_range'))}), "
                f"latest lap {prediction['latest_lap']} (range {_fmt_range(prediction.get('latest_lap_range'))}), "
                f"can finish without stopping: {prediction['finish_without_stop']}"
                + (
                    f" (chance {round(prediction['finish_without_stop_prob'] * 100)}%)"
                    if prediction.get('finish_without_stop_prob') is not None
                    else ''
                )
                + f", based on {prediction['samples']} laps ({_BASIS.get(prediction.get('basis'), 'fit')})"
            ),
            topic='strategy',
        )
    else:
        add('ML pit prediction', 'not enough lap data yet', topic='strategy')
    model = ctx.get('tyre_fuel_model')
    if model:
        add(
            'Tyre model ranges [low, expected, high]',
            (
                f"wear per lap {_fmt_range(model['wear_per_lap_pct'])}%, wear in 5 laps "
                f"{_fmt_range(model['wear_in_5_laps_pct'])}%, fall-off {_fmt_range(model['deg_s_per_lap'])} s/lap; "
                f"{_BASIS.get(model['basis'], 'fit')} from {model['fit_laps']} laps"
                + (' (simulator data only)' if model.get('simulated_fit') else '')
                + f" plus {model['stint_laps']} laps this stint"
            ),
            topic='tyres',
        )
    fuel = ctx.get('fuel')
    if fuel:
        add(
            'Fuel',
            f"{fuel['fuel_kg']} kg, burn {fuel['per_lap_kg']} kg per lap "
            f"(range {_fmt_range(fuel.get('per_lap_kg_range'))}), "
            f"{fuel['laps_of_fuel']} laps of fuel, margin {fuel['margin_laps']} laps "
            f"(range {_fmt_range(fuel.get('margin_laps_range'))})",
            topic='fuel',
        )
    add("Driver's own history at this track", ctx.get('driver_profile'), topic='pace')
    add('Last lap sector times ms', ctx.get('last_lap_sectors_ms'), topic='pace')
    add('Last lap sector deltas to personal best ms', ctx.get('last_lap_sector_deltas_ms'), topic='pace')
    tire = ctx.get('tire_health')
    if tire:
        add(
            'Tyre physics model',
            (
                f"grip {tire['grip_factor']:.2f}, health {tire['health_score']:.0f}%, "
                f"~{tire['remaining_laps']:.0f} competitive laps left (limited by {tire['limiting_factor']}), "
                f"overheat risk {tire['overheat_risk']}"
            ),
            topic='tyres',
        )
    threats = ctx.get('opponent_threats')
    if threats:
        add(
            'Nearby cars ranked by threat',
            '; '.join(
                f"{t.get('driver_name') or 'Unknown driver'} {t['threat_level']} "
                f"({t['gap_s']:+.1f}s{', ' + ', '.join(t['factors']) if t['factors'] else ''})"
                for t in threats
            ),
            topic='field',
        )
    strategy = ctx.get('strategy')
    if strategy:
        best = strategy['best']
        stints = ', '.join(
            f"{c.lower()} {n} laps" for c, n in zip(best['compounds'], best['stint_laps'], strict=False)
        )
        add(
            'Race strategy model',
            (
                f"best is {best['stops']} stop(s): {stints}; box at end of lap(s) {best['pit_laps'] or 'none'}; "
                f"pit loss {strategy['pit_loss_s']}s"
            )
            + ''.join(
                f"; {o['stops']} stop(s) is {o['slower_by_s']}s slower" for o in strategy['alternatives']
            ),
            topic='strategy',
        )
        undercut = strategy.get('undercut_on_car_ahead')
        if undercut:
            add(
                'Undercut on car ahead',
                (
                    f"gap {undercut['gap_s']}s, boxing this lap gains {undercut['gain_s']}s "
                    f"(rival on {undercut['rival_tyre_age']}-lap {undercut['rival_compound']}), "
                    f"works: {undercut['works']}"
                ),
                topic='strategy',
            )
        if strategy.get('last_call'):
            add('Last strategy call', strategy['last_call']['message'], topic='strategy')
        add('Game rejoin position if boxing now', strategy.get('game_rejoin_position'), topic='strategy')
        rivals = strategy.get('rivals')
        if rivals and rivals.get('nearby'):
            add('Rival stop predictions', ' '.join(r['summary'] for r in rivals['nearby']), topic='field')
        sim = strategy.get('simulation')
        if sim and sim.get('options') and not (sim.get('inputs') or {}).get('settled'):
            add(
                f"Race simulation ({sim['runs']} Monte Carlo races from lap {sim['lap']})",
                (
                    '; '.join(
                        f"{o['label']}: best {round(o['p_best'] * 100)}% of runs, expected P{o['position_range'][1]} "
                        f"(P{o['position_range'][0]}-P{o['position_range'][2]}), podium {round(o['p_podium'] * 100)}%"
                        for o in sim['options']
                    )
                    + f"; chance of a Safety Car {round(sim['p_safety_car'] * 100)}%, VSC {round(sim['p_vsc'] * 100)}%, "
                    f"rain {round(sim['p_rain'] * 100)}%; pit loss {sim['pit_loss_s'][1]}s "
                    f"({sim['pit_loss_s'][0]}-{sim['pit_loss_s'][2]}s); overtaking difficulty {sim['overtaking']}/5"
                ),
                topic='strategy',
            )
        replan = strategy.get('replan')
        if replan and replan.get('plan_a'):
            a = replan['plan_a']
            text = f"{a['label']}, box lap(s) {a['pit_laps'] or 'none'}; status {replan['status']}"
            ch = replan.get('challenger')
            if ch:
                text += (
                    f"; best alternative {ch['label']} gains {ch['time_gain_s']}s / "
                    f"{ch['position_gain']} places, ahead in {round(ch['p_better'] * 100)}% of runs"
                )
            if replan.get('call_active') and replan.get('last_call'):
                text += (
                    f"; Plan B called on lap {replan['last_call']['lap']}: {replan['last_call']['message']}"
                )
            add('Current plan (Plan A, the plan we are on)', text, topic='strategy')
        spread = strategy.get('uncertainty')
        if spread and spread.get('worse') and spread.get('better'):
            add(
                'Strategy across the tyre-model range',
                (
                    f"if tyres go off faster: {spread['worse']['stops']} stop(s) at lap(s) "
                    f"{spread['worse']['pit_laps'] or 'none'}; if they last longer: "
                    f"{spread['better']['stops']} stop(s) at lap(s) {spread['better']['pit_laps'] or 'none'}"
                ),
                topic='strategy',
            )
    current = ctx.get('current_lap')
    if current:
        add(
            'Current lap',
            f"sector {current['sector']}, running time {_fmt_lap(current.get('time_ms')) or 'n/a'}"
            + (', lap INVALID (track limits)' if current.get('invalid') else ''),
        )
    add('DRS open now', ctx.get('drs_open'), 'car')
    ers = ctx.get('ers')
    if ers:
        add(
            'ERS battery',
            f"{ers['battery_pct']}% charge, deploy mode {ers['deploy_mode']}, "
            f"{ers['used_this_lap_pct']}% of a full battery used this lap",
            'car',
        )
    add('Tyre inner (carcass) temps C', ctx.get('tyre_inner_c'), 'tyres')
    add('Tyre pressures psi', ctx.get('tyre_pressures_psi'), 'tyres')
    add('Brake temps C', ctx.get('brake_temps_c'), 'car')
    add('Engine temp C', ctx.get('engine_temp_c'), 'car')
    damage = ctx.get('damage_pct')
    if damage is not None:
        hurt = {k: v for k, v in damage.items() if v}
        add('Car damage/wear %', ', '.join(f'{k} {v}' for k, v in hurt.items()) or 'no damage', 'car')
    penalties = ctx.get('penalties')
    if penalties:
        add(
            'Penalties and warnings',
            (
                f"{penalties['time_penalty_s']}s time penalty, {penalties['warnings']} track-limit/corner-cutting "
                f"warnings, unserved drive-through {penalties['unserved_drive_through']}, unserved stop-go "
                f"{penalties['unserved_stop_go']}; pit stops done {penalties['pit_stops_done']}; started "
                f"P{penalties['grid_position']}"
            ),
            'rules',
        )
    add('Recent race control', '; '.join(ctx.get('recent_race_events') or []) or None, 'rules')
    standings = ctx.get('standings')
    if standings:
        add(
            'Running order (P, driver, gap to leader s, interval s, last lap, best lap, tyre and age, stops)',
            ' | '.join(_standing(row) for row in standings),
            'field',
        )
    return '\n'.join(lines) if lines else 'No telemetry received yet.'


def _standing(row: dict) -> str:
    parts = [
        f"P{row.get('position')} {row.get('name') or 'Car'}" + (' (YOU)' if row.get('is_player') else '')
    ]
    if row.get('status') and row['status'] != 'RUN':
        parts.append(row['status'])
    if row.get('laps_down'):
        parts.append(f"+{row['laps_down']} lap(s)")
    elif row.get('gap_s') is not None:
        parts.append(f"+{row['gap_s']:.1f}")
    if row.get('interval_s') is not None:
        parts.append(f"int {row['interval_s']:.1f}")
    if row.get('last_lap_ms'):
        parts.append(f"last {_fmt_lap(row['last_lap_ms'])}")
    if row.get('best_lap_ms'):
        parts.append(f"best {_fmt_lap(row['best_lap_ms'])}" + (' fastest' if row.get('fastest_lap') else ''))
    if row.get('compound'):
        parts.append(f"{row['compound']}{row.get('tyre_age') if row.get('tyre_age') is not None else ''}")
    if row.get('stops'):
        parts.append(f"{row['stops']} stop(s)")
    if row.get('pit'):
        parts.append('in pits')
    if row.get('penalties_s'):
        parts.append(f"+{row['penalties_s']}s pen")
    return ' '.join(parts)


# Words that show what a free-form question is about, so only that data goes to the model.
_TOPIC_WORDS = {
    'car': r'battery|ers|deploy|deployment|energy|hybrid|harvest|overtake (?:mode|button)|damage|damaged|wing|floor|'
    r'gearbox|engine|brakes?|drs|car',
    'tyres': r'tyres?|tires?|wear|worn|grip|deg|degradation|compound|softs?|mediums?|hards?|temps?|temperatures?|'
    r'pressures?|psi|graining|blister\w*|overheat\w*',
    'fuel': r'fuel|lift and coast|lift|coast|save|saving|mix',
    'strategy': r'pit|box|boxing|stop|stops|strategy|plan|undercut|overcut|window|chances?|odds|win|podium|points|'
    r'safety car|vsc|finish',
    'field': r'who|leader|leading|lead|p\d+|positions?|standings|order|ahead|behind|front|rivals?|others|field|'
    r'fastest|team ?mate|threats?|attack\w*|defend\w*|catch\w*|overtak\w*|pass\w*|gap|gaps|driver|everyone',
    'pace': r'pace|lap ?times?|sectors?|faster|slower|delta|improve|corners?|turns?|lose|losing|gain|quicker|'
    r'time|braking|push|best',
    'weather': r'weather|rain\w*|wet|dry|forecast|track temp\w*|air temp\w*|inters?|intermediates?|cloud\w*|sun\w*',
    'rules': r'penalt\w*|warnings?|track limits|limits|drive ?through|stop ?go|invalid|flags?|yellow|blue|red flag|'
    r'race control|investigation|steward\w*|incident',
}
_TOPIC_RE = {topic: re.compile(rf'\b(?:{words})\b') for topic, words in _TOPIC_WORDS.items()}
COMMAND_TOPICS = {
    'weather': {'weather'},
    'tyre_temps': {'tyres'},
    'pit_window': {'strategy', 'tyres'},
    'gap_ahead': {'field'},
    'gap_behind': {'field'},
    'pace': {'pace', 'tyres'},
    'threats': {'field'},
    'strategy': {'strategy', 'tyres', 'field'},
    'rivals': {'field', 'strategy'},
}


def question_topics(question: str) -> set | None:
    """The data topics a question is about, or None (send everything) when unsure."""
    if question in COMMAND_TOPICS:
        return set(COMMAND_TOPICS[question])
    text = (question or '').lower().replace('tire', 'tyre')
    topics = {topic for topic, pattern in _TOPIC_RE.items() if pattern.search(text)}
    return topics or None


def _names_a_driver(question: str, ctx: dict) -> bool:
    words = set(re.findall(r'[a-z]+', (question or '').lower()))
    names = [row.get('name') for row in ctx.get('standings') or []]
    names += [ctx.get('car_ahead_name'), ctx.get('car_behind_name')]
    return any(
        name and any(part in words for part in re.findall(r'[a-z]+', name.lower()) if len(part) > 2)
        for name in names
    )


def build_prompt(command: str, ctx: dict, *, focused: bool = True) -> tuple[str, str]:
    """Return (system prompt, user message) for DeepSeek.

    ``focused`` sends only the data the question is about (plus the core: lap, position,
    gaps, lap times, tyre set), which makes the prompt shorter and the answer quicker.
    """
    question = COMMAND_QUESTIONS.get(command, command)
    topics = question_topics(command) if focused else None
    if topics is not None and 'field' not in topics and _names_a_driver(question, ctx):
        topics.add('field')
    user = (
        f'Driver asks: "{question}"\n\nCurrent data:\n{format_context(ctx, topics)}\n\n'
        'Answer as the race engineer.'
    )
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
    if command in ('gap_ahead', 'gap_behind', 'threats') and not session_modes.reports_gaps(mode):
        return (
            'No other cars in time trial.'
            if mode == session_modes.TIME_TRIAL
            else f'Gaps are only tracked in the race, not in {label.lower()}.'
        )
    if command == 'pit_window' and not session_modes.has_pit_strategy(mode):
        return f'No pit stops in {label.lower()}.'
    if command in ('strategy', 'rivals') and mode != session_modes.RACE:
        return f'No race strategy in {label.lower()}.'
    return None


def rule_based_answer(command: str, ctx: dict) -> str:
    blocked = _mode_blocks(command, ctx)
    if blocked:
        return blocked
    if command == 'weather':
        if not ctx.get('weather'):
            return 'No weather data yet.'
        text = (
            f"{ctx['weather'].capitalize()}. Track {ctx.get('track_temp_c')} degrees, "
            f"air {ctx.get('air_temp_c')}."
        )
        conditions = ctx.get('track_conditions')
        if conditions and conditions.get('radio'):
            text += ' ' + conditions['radio']
        return text
    if command in ('gap_ahead', 'gap_behind'):
        side = 'ahead' if command == 'gap_ahead' else 'behind'
        gap = ctx.get(f'gap_{side}_s')
        if gap is None:
            return f'No gap {side} reading right now.'
        driver = ctx.get(f'car_{side}_name')
        if driver:
            return f'Gap to {driver} {side} is {_say_seconds(gap)}.'
        return f'Gap {side} is {_say_seconds(gap)}.'
    if command == 'tyre_temps':
        temps = ctx.get('tyre_surface_c')
        if not temps:
            return 'No tyre data yet.'
        text = f"Fronts {temps['FL']} and {temps['FR']}, rears {temps['RL']} and {temps['RR']}."
        wear = ctx.get('tyre_wear_pct')
        if wear:
            text += f" Worst wear {max(wear.values()):.0f} percent."
        return text
    if command == 'pit_window':
        prediction = ctx.get('pit_prediction')
        if prediction and prediction.get('finish_without_stop'):
            chance = prediction.get('finish_without_stop_prob')
            if chance is not None and chance < 0.8:
                return (
                    f"Tyres should reach the flag, but it's close: "
                    f"{round(chance * 100)} percent. We'll keep watching."
                )
            return 'Tyres should reach the flag. No stop needed at the moment.'
        if prediction and prediction.get('ideal_lap'):
            soonest, _mid, latest = prediction.get('ideal_lap_range') or (None, None, None)
            window = ''
            if soonest and latest and soonest != latest:
                window = f", window lap {soonest} to {latest}"
            elif soonest and not latest:
                window = f", from lap {soonest}"
            return (
                f"Box around lap {prediction['ideal_lap']}{window}. "
                f"Latest lap {prediction['latest_lap'] or 'unknown'}."
            )
        window = ctx.get('game_pit_window')
        if window:
            return f'Game window is lap {window[0]} to {window[1]}.'
        return 'Not enough laps on these tyres to predict a window yet.'
    if command == 'threats':
        threats = ctx.get('opponent_threats')
        if not threats:
            return 'No immediate threats.'
        top = threats[0]
        side = 'behind' if top['gap_s'] < 0 else 'ahead'
        note = f", {top['factors'][0]}" if top.get('factors') else ''
        driver = top.get('driver_name')
        who = f'from {driver} {side}' if driver else side
        return f"{top['threat_level'].capitalize()} threat {who}, {abs(top['gap_s']):.1f}{note}."
    if command == 'strategy':
        strategy = ctx.get('strategy')
        if not strategy:
            return 'No strategy data yet.'
        text = strategy['radio']
        replan = strategy.get('replan')
        if replan and replan.get('call_active') and replan.get('last_call'):
            text = replan['last_call']['message'] + ' ' + text
        undercut = strategy.get('undercut_on_car_ahead')
        if undercut:
            text += (
                ' Undercut on the car ahead is on.'
                if undercut['works']
                else ' Undercut on the car ahead will not work yet.'
            )
        return text
    if command == 'rivals':
        strategy = ctx.get('strategy') or {}
        rivals = strategy.get('rivals')
        if not rivals or not rivals.get('nearby'):
            return 'No read on the other cars yet.'
        picks = [rivals[side] for side in ('ahead', 'behind') if rivals.get(side)] or rivals['nearby'][:2]
        return ' '.join(r['summary'] for r in picks)
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


# A question this short that matches a fixed command ("gap ahead?", "tyre temps") gets the
# instant built-in answer when rules_first is on; longer questions go to the AI.
INSTANT_MAX_WORDS = 5


class EngineerBrain:
    def __init__(
        self,
        context_provider: Callable[[], dict],
        client: AIClient | None = None,
        *,
        rules_first: bool = False,
        focused: bool = True,
    ):
        """``rules_first``: the fixed commands are answered at once from the live data (no AI
        wait, no load on the graphics card); the AI handles everything else.
        ``focused``: send the AI only the data the question is about (shorter, faster)."""
        self.context_provider = context_provider
        self.client = client
        self.rules_first = rules_first
        self.focused = focused
        self.last_source = None  # 'ai' or 'fallback' or 'rules' - useful for the HUD and tests
        self.last_advice = ''  # longer note from the last AI answer, for an on-screen log

    def answer(self, command: str) -> str:
        ctx = self.context_provider()
        blocked = _mode_blocks(command, ctx)
        if blocked:  # e.g. "gap ahead" in time trial: answer instantly, no AI needed
            self.last_source = 'rules'
            self.last_advice = ''
            return blocked
        if self.rules_first:
            self.last_source = 'rules'
            self.last_advice = ''
            return rule_based_answer(command, ctx)
        text = self._ask_ai(command, ctx)
        if text:
            return text
        self.last_source = 'fallback'
        self.last_advice = ''
        return rule_based_answer(command, ctx)

    def answer_question(self, question: str, command: str | None = None) -> str:
        """Answer anything the driver asked over push-to-talk, in their own words.

        ``command`` is the fixed command the question also matches, if any. It is used for the
        instant session-mode answer and for the rule-based fallback; the AI always gets the
        driver's real words so it can answer exactly what was asked.
        """
        ctx = self.context_provider()
        if command:
            blocked = _mode_blocks(command, ctx)
            if blocked:
                self.last_source = 'rules'
                self.last_advice = ''
                return blocked
        question = re.sub(r'\s+', ' ', question.replace('"', "'")).strip()
        if command and self.rules_first and len(question.split()) <= INSTANT_MAX_WORDS:
            self.last_source = 'rules'
            self.last_advice = ''
            return rule_based_answer(command, ctx)
        text = self._ask_ai(question, ctx) if question else None
        if text:
            return text
        self.last_source = 'fallback'
        self.last_advice = ''
        if command:
            return rule_based_answer(command, ctx)
        return UNKNOWN_QUESTION_REPLY

    def _ask_ai(self, question: str, ctx: dict) -> str | None:
        """The AI's radio line for a command id or a free-form question, or None."""
        if self.client is None:
            return None
        system, user = build_prompt(question, ctx, focused=self.focused)
        # Radio answers are short: no long reasoning, small token budget (much lower latency).
        fast = (
            {'fast': True, 'max_tokens': RADIO_MAX_TOKENS}
            if getattr(self.client, 'supports_fast', False)
            else {}
        )
        if fast and hasattr(self.client, 'radio_voice'):
            # Local Ollama: stream the reply and speak as soon as the radio line is complete.
            try:
                voice = self.client.radio_voice(system, user, max_tokens=RADIO_MAX_TOKENS)
            except AITimeout as error:
                log.warning(f'AI too slow ({error}); answering from rules.')
                return None
            except AIUnavailable:
                voice = None
            voice = clean_radio_text(voice or '')
            if voice:
                self.last_source = 'ai'
                self.last_advice = ''
                return voice
        if hasattr(self.client, 'chat_json'):
            try:
                result = self.client.chat_json(system, user, **fast)
                voice = clean_radio_text(str(result.get('voice') or ''))
                if voice:
                    self.last_source = 'ai'
                    self.last_advice = str(result.get('advice') or '').strip()
                    return voice
            except AITimeout as error:
                log.warning(f'AI too slow ({error}); answering from rules.')
                return None
            except AIUnavailable as error:
                log.info(f'AI JSON reply unavailable ({error}); trying plain text.')
        try:
            text = clean_radio_text(self.client.chat(system, user, **fast))
            if text:
                self.last_source = 'ai'
                self.last_advice = ''
                return text
        except AIUnavailable as error:
            log.warning(f'AI unavailable ({error}); answering from rules.')
        return None
