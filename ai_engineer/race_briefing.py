"""Race information for chatty mode: one short call per lap, rotating through the topics.

    threats      "Hamilton behind is the threat, eight tenths back, three tenths a lap quicker,
                  on you in three laps."
    rival tyres  "Verstappen ahead on mediums, twelve laps old, likely to box lap twenty for hards.
                  Hamilton behind on hards, five laps old."
    safety car   "Safety Car chance for the rest of the race about forty percent."

Each line is built only from what the engineer already tracks (threat scores, the rival stop
model, the race simulation); a topic with nothing to say is skipped that lap.
"""

from __future__ import annotations

from typing import Any

import radio_phrases as say

TOPICS = ('threats', 'rival_tyres', 'safety_car')
MAX_PACE_DELTA_S = 2.0  # a bigger lap-time gap than this is a pit stop or a mistake

_PLURALS = {'soft': 'softs', 'medium': 'mediums', 'hard': 'hards', 'intermediate': 'inters', 'wet': 'wets'}


def compound_plural(compound: str | None) -> str | None:
    if not compound:
        return None
    return _PLURALS.get(str(compound).lower(), f'{str(compound).lower()}s')


def _who(name: str | None, position: int | None) -> str:
    if name:
        return name
    return f'the car in P{position}' if position else 'the car'


def threat_line(threats) -> str | None:
    """The biggest threat around the driver, or None when nobody is one."""
    serious = [t for t in threats or () if t.threat_level in ('HIGH', 'MEDIUM')]
    if not serious:
        return None
    threat = serious[0]
    who = _who(threat.driver_name, threat.position)
    who = who[0].upper() + who[1:]
    gap = say.gap_value(abs(threat.gap_s))
    pace = threat.pace_delta_s  # negative = they are quicker per lap
    if pace is not None and abs(pace) > MAX_PACE_DELTA_S:
        pace = None  # an in-lap, out-lap or a spin, not real pace
    if threat.gap_s < 0:
        line = f'{who} behind is the threat, {gap} back'
        if pace is not None and pace <= -0.1:
            line += f', {say.time_amount(-pace)} a lap quicker'
        if threat.laps_to_catch is not None and threat.laps_to_catch >= 1:
            line += f', on you in {say.number_words(round(threat.laps_to_catch))} laps'
        return line + '.'
    line = f'{who} ahead, {gap} up the road'
    if pace is not None and pace >= 0.1:
        line += f', you are {say.time_amount(pace)} a lap quicker'
        laps = abs(threat.gap_s) / pace
        if 1 <= laps <= 15:
            line += f', catching in {say.number_words(round(laps))} laps'
    return line + '.'


def _rival_part(prediction, where: str, lap_num: int) -> str | None:
    tyres = compound_plural(prediction.compound)
    if not tyres:
        return None
    who = _who(prediction.name, prediction.position)
    part = f'{who} {where} on {tyres}, {say.number_words(prediction.tyre_age)} laps old'
    if prediction.will_stop and prediction.stop_lap and prediction.stop_lap >= lap_num:
        part += f', likely to box lap {say.number_words(prediction.stop_lap)}'
        onto = compound_plural(prediction.next_compound)
        if onto:
            part += f' for {onto}'
    return part + '.'


def rival_tyres_line(predictions, player_position: int | None, lap_num: int) -> str | None:
    """Tyres, tyre age and expected next stop of the cars directly ahead and behind."""
    if not player_position or not predictions:
        return None
    by_position = {p.position: p for p in predictions if p.position}
    parts = []
    for offset, where in ((-1, 'ahead'), (1, 'behind')):
        prediction = by_position.get(player_position + offset)
        part = _rival_part(prediction, where, lap_num) if prediction else None
        if part:
            parts.append(part[0].upper() + part[1:])
    return ' '.join(parts) or None


def safety_car_line(simulation: dict | None) -> str | None:
    """Chance of a Safety Car for the rest of the race, from the race simulation."""
    if not simulation or simulation.get('p_safety_car') is None:
        return None
    percent = int(round(float(simulation['p_safety_car']) * 100 / 5.0) * 5)
    line = f'Safety Car chance for the rest of the race about {say.number_words(percent)} percent.'
    if percent >= 50:
        line += ' Keep the pit stop flexible.'
    return line


class RaceBriefing:
    """Rotates the topics so each lap gets one race-info call, never the same one twice running."""

    def __init__(self) -> None:
        self.next_topic = 0
        self.last_lap: int | None = None

    def reset(self) -> None:
        self.next_topic = 0
        self.last_lap = None

    def line(self, lap_num: int, builders: dict[str, Any]) -> str | None:
        """The call for this lap (once per lap); ``builders`` maps each topic to a function."""
        if lap_num == self.last_lap:
            return None
        self.last_lap = lap_num
        for step in range(len(TOPICS)):
            topic = TOPICS[(self.next_topic + step) % len(TOPICS)]
            builder = builders.get(topic)
            text = builder() if builder else None
            if text:
                self.next_topic = (self.next_topic + step + 1) % len(TOPICS)
                return text
        return None
