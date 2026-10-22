"""Every car's laps and tyre stints, from the Session History and Final Classification packets.

The game sends one Session History packet (id 11) per car in turn, each carrying that car's
whole session so far: every lap time with its sectors and validity, and every set of tyres
with the lap it came off on. That is the only place F1 2021 tells us *when* a rival stopped
and what they put on, so it is the ground truth for rival stints.

`SessionLog` keeps the latest history per car (each packet replaces the last one, so a
flashback that rewinds the session simply shows fewer laps), the Final Classification
(id 8) and the Car Setups (id 5). It turns them into flat rows:

  lap rows    car, lap, lap time, sectors, valid, compound, stint, laps on the set, in/out lap
  stint rows  car, stint, compound, start lap, end lap, laps on the set, still on the car

which the history database stores (history.py) and the model-fitting and backtest tools
read back from recordings (fitted_models.py, tools/backtest.py).

Tyre age is counted from the stint's first lap, as if every set were new. A car that starts
the race on used qualifying tyres is a few laps older than that; the fall-off fit only uses
how lap time changes within a stint, which an age offset does not change.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from packets import (
    ACTUAL_COMPOUNDS,
    CAR_SETUPS_PACKET_SIZE,
    FINAL_CLASSIFICATION_PACKET_SIZE,
    HEADER_SIZE,
    PARTICIPANTS_PACKET_SIZE,
    SECTOR_VALID,
    SESSION_HISTORY_PACKET_SIZE,
    VISUAL_COMPOUNDS,
    CarSetupData,
    FinalClassificationData,
    PacketHeader,
    SessionHistoryData,
    parse_car_setups,
    parse_final_classification,
    parse_participants,
    parse_session_history,
)

RESULT_STATUS = {0: 'invalid', 1: 'inactive', 2: 'active', 3: 'finished', 4: 'disqualified',
                 5: 'not classified', 6: 'retired'}


def valid_history(history: SessionHistoryData) -> bool:
    """Reject packets that cannot be right (car index, lap/stint counts, stint order)."""
    if not 0 <= history.car_index < 22:
        return False
    return all(stint.end_lap >= stint.start_lap for stint in history.stints)


def stint_for_lap(history: SessionHistoryData, lap_num: int):
    for stint in history.stints:
        if stint.start_lap <= lap_num <= stint.end_lap:
            return stint
    return None


def lap_rows(history: SessionHistoryData, completed_only: bool = True) -> list[dict]:
    """One row per lap of this car (completed laps only by default)."""
    rows = []
    for lap in history.laps:
        if completed_only and lap.lap_time_ms <= 0:
            continue
        stint = stint_for_lap(history, lap.lap_num)
        rows.append({
            'car_index': history.car_index, 'lap_num': lap.lap_num, 'lap_time_ms': lap.lap_time_ms,
            's1_ms': lap.sector_ms[0] or None, 's2_ms': lap.sector_ms[1] or None,
            's3_ms': lap.sector_ms[2] or None, 'valid': lap.valid,
            'sectors_valid': tuple(bool(lap.valid_flags & SECTOR_VALID[s]) for s in (1, 2, 3)),
            'valid_flags': lap.valid_flags,
            'stint_num': stint.stint_num if stint else None,
            'actual_compound': stint.actual_compound if stint else None,
            'visual_compound': stint.visual_compound if stint else None,
            'tyre_age_laps': lap.lap_num - stint.start_lap if stint else None,
            'in_lap': bool(stint and not stint.is_open and lap.lap_num == stint.end_lap),
            'out_lap': bool(stint and stint.stint_num > 0 and lap.lap_num == stint.start_lap),
        })
    return rows


def stint_rows(history: SessionHistoryData) -> list[dict]:
    return [{'car_index': history.car_index, 'stint_num': s.stint_num,
             'actual_compound': s.actual_compound, 'visual_compound': s.visual_compound,
             'compound': s.compound_name, 'start_lap': s.start_lap, 'end_lap': s.end_lap,
             'laps': s.laps, 'is_open': s.is_open}
            for s in history.stints]


def classification_row(c: FinalClassificationData) -> dict:
    return {'car_index': c.car_index, 'position': c.position, 'num_laps': c.num_laps,
            'grid_position': c.grid_position, 'points': c.points, 'num_pit_stops': c.num_pit_stops,
            'result_status': c.result_status, 'result': RESULT_STATUS.get(c.result_status, 'unknown'),
            'best_lap_time_ms': c.best_lap_time_ms, 'total_race_time_s': c.total_race_time_s,
            'penalties_time_s': c.penalties_time_s, 'num_penalties': c.num_penalties,
            'tyre_stints_actual': list(c.tyre_stints_actual),
            'tyre_stints_visual': list(c.tyre_stints_visual),
            'tyre_stints': [VISUAL_COMPOUNDS.get(v, ACTUAL_COMPOUNDS.get(a, 'Unknown'))
                            for a, v in zip(c.tyre_stints_actual, c.tyre_stints_visual, strict=False)]}


@dataclass
class SessionLog:
    """Latest Session History per car, Final Classification and setups for one session."""
    session_uid: int | None = None
    player_index: int | None = None
    histories: dict = field(default_factory=dict)        # car index -> SessionHistoryData
    classification: list = field(default_factory=list)   # FinalClassificationData
    setups: dict = field(default_factory=dict)           # car index -> CarSetupData (visible ones)
    names: dict = field(default_factory=dict)            # car index -> driver name

    # ── live updates; each returns what changed so callers write only that ──

    def update_history(self, history: SessionHistoryData) -> tuple[list[dict], list[dict], int] | None:
        """Store one car's history. Returns (changed lap rows, stint rows, last completed lap)
        or None when nothing changed. A rewound history (flashback) returns every remaining
        row, and the last completed lap lets the caller drop laps past it."""
        previous = self.histories.get(history.car_index)
        self.histories[history.car_index] = history
        if previous is not None and previous.laps == history.laps and previous.stints == history.stints:
            return None
        rows = lap_rows(history)
        if previous is not None and history.num_laps >= previous.num_laps:
            # Only rows that are new or changed (a pit stop turns an earlier lap into an in-lap).
            old = lap_rows(previous)
            rows = [r for r in rows if r not in old]
        return rows, stint_rows(history), max((lap.lap_num for lap in history.completed_laps()), default=0)

    def update_classification(self, classification: list[FinalClassificationData]) -> bool:
        if classification == self.classification:
            return False
        self.classification = list(classification)
        return True

    def update_setups(self, setups: list[CarSetupData]) -> dict:
        """Store visible setups; returns {car index: setup} for those that changed."""
        changed = {}
        for index, setup in enumerate(setups):
            if setup.is_hidden or self.setups.get(index) == setup:
                continue
            self.setups[index] = setup
            changed[index] = setup
        return changed

    # ── reading back ──

    def all_lap_rows(self) -> list[dict]:
        return [row for car in sorted(self.histories) for row in lap_rows(self.histories[car])]

    def all_stint_rows(self) -> list[dict]:
        rows = []
        for car in sorted(self.histories):
            for row in stint_rows(self.histories[car]):
                rows.append(dict(row, driver=self.names.get(car)))
        return rows

    def classification_rows(self) -> list[dict]:
        return [dict(classification_row(c), driver=self.names.get(c.car_index))
                for c in self.classification]

    def pit_laps(self, car_index: int) -> list[int]:
        """Laps the car came in on (the last lap of every set but the one on the car)."""
        history = self.histories.get(car_index)
        if history is None:
            return []
        return [s.end_lap for s in history.stints if not s.is_open and s.stint_num + 1 < len(history.stints)]

    def to_dict(self) -> dict:
        return {'session_uid': self.session_uid, 'player_index': self.player_index,
                'stints': self.all_stint_rows(), 'laps': self.all_lap_rows(),
                'classification': self.classification_rows(),
                'setups': {i: s.as_dict() for i, s in sorted(self.setups.items())}}


# ──────────────────────────────────────────────
# Reading recordings (.f1rec, docs/udp-recording-format.md)
# ──────────────────────────────────────────────

LOGGED_PACKET_SIZES = {4: PARTICIPANTS_PACKET_SIZE, 5: CAR_SETUPS_PACKET_SIZE,
                       8: FINAL_CLASSIFICATION_PACKET_SIZE, 11: SESSION_HISTORY_PACKET_SIZE}


def apply_datagram(logs: dict, data: bytes) -> SessionLog | None:
    """Feed one raw datagram into per-session logs ({session uid: SessionLog}).

    Only packets 4, 5, 8 and 11 are read; anything else, malformed or from another game is
    ignored. Returns the log that changed, if any.
    """
    if len(data) < HEADER_SIZE:
        return None
    try:
        header = PacketHeader.from_bytes(data)
    except (struct.error, ValueError):
        return None
    pid = header.packet_id
    if (header.packet_format != 2021 or LOGGED_PACKET_SIZES.get(pid) != len(data)
            or not 0 <= header.player_car_index < 22):
        return None
    log = logs.get(header.session_uid)
    if log is None:
        log = logs[header.session_uid] = SessionLog(header.session_uid, header.player_car_index)
    log.player_index = header.player_car_index
    try:
        if pid == 11:
            history = parse_session_history(data)
            if not valid_history(history):
                return None
            log.update_history(history)
        elif pid == 8:
            log.update_classification(parse_final_classification(data))
        elif pid == 5:
            log.update_setups(parse_car_setups(data))
        else:
            for index, p in enumerate(parse_participants(data)):
                if p.name:
                    log.names[index] = p.radio_name or p.name
    except struct.error:
        return None
    return log


def session_logs_from_recording(path) -> list[SessionLog]:
    """Every session's car histories, classification and setups in a recording."""
    from udp_recording import RecordingReader

    logs: dict = {}
    with RecordingReader(path) as reader:
        for record in reader:
            apply_datagram(logs, record.data)
    return [log for log in logs.values() if log.histories or log.classification or log.setups]
