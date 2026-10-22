"""Track status for the HUD: red flag, safety car, VSC, yellow flags, blue flag or clear.

F1 2021 sends this in two places:
- the Session packet: safetyCarStatus (0 none, 1 full SC, 2 VSC, 3 formation lap) and
  a flag for every marshal zone (-1 invalid, 0 none, 1 green, 2 blue, 3 yellow, 4 red);
- the Car Status packet: vehicleFiaFlags, the flag the player is being shown right now.
"""

FLAG_BLUE = 2
FLAG_YELLOW = 3
FLAG_RED = 4

LABELS = {
    'RED': 'RED FLAG',
    'SC': 'SAFETY CAR DEPLOYED',
    'VSC': 'VIRTUAL SAFETY CAR',
    'FORMATION': 'FORMATION LAP',
    'YELLOW': 'YELLOW FLAG',
    'BLUE': 'BLUE FLAG',
    'CLEAR': 'TRACK CLEAR',
}

CLEAR = {'state': 'CLEAR', 'label': LABELS['CLEAR'], 'yellow_zones': 0, 'yellow_sectors': []}


def zone_sector(zone_start: float, track_length: int, sector_starts: dict) -> int | None:
    """Which sector (1-3) a marshal zone starts in, once the sector lines are known."""
    s2, s3 = sector_starts.get(2), sector_starts.get(3)
    if not track_length or s2 is None or s3 is None:
        return None
    metres = zone_start * track_length
    return 1 if metres < s2 else 2 if metres < s3 else 3


def describe(session, player_flag: int = 0, sector_starts: dict | None = None) -> dict:
    """The HUD's track status, most serious first."""
    if session is None:
        return dict(CLEAR)
    zones = getattr(session, 'marshal_zones', ()) or ()
    yellow = [start for start, flag in zones if flag == FLAG_YELLOW]
    sectors = sorted(
        {s for s in (zone_sector(start, session.track_length, sector_starts or {}) for start in yellow) if s}
    )
    if any(flag == FLAG_RED for _, flag in zones) or player_flag == FLAG_RED:
        state = 'RED'
    elif session.safety_car_status == 1:
        state = 'SC'
    elif session.safety_car_status == 2:
        state = 'VSC'
    elif session.safety_car_status == 3:
        state = 'FORMATION'
    elif yellow or player_flag == FLAG_YELLOW:
        state = 'YELLOW'
    elif player_flag == FLAG_BLUE:
        state = 'BLUE'
    else:
        state = 'CLEAR'
    label = LABELS[state]
    if state == 'YELLOW' and sectors:
        label += ' · ' + ' '.join(f'S{s}' for s in sectors)
    return {'state': state, 'label': label, 'yellow_zones': len(yellow), 'yellow_sectors': sectors}
