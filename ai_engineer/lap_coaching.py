"""End-of-lap coaching: where the lap was lost, said like an engineer would.

After every clean lap the engineer compares each sector with your best sector this session,
finds the one where you lost most, then the corner inside it where you lost most against your
best run through that corner, and says why if the telemetry shows it:

    "You lost three tenths in sector three. Turn 12 cost you two tenths, you braked
     twenty metres early. Fix that next lap."

Nothing is said when the lap was within a tenth of your best sectors everywhere.

That is qualifying and practice coaching. In a race the engineer doesn't coach every lap:
tyres wear and fuel burns, so every corner drifts off its best. ``CornerLossStreaks`` only
speaks when one corner keeps costing time *beyond* that drift three laps running, and then
at most every few laps:

    "Turn 12 again, three laps running. You lost two tenths there last lap, you braked
     twenty metres early."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

import radio_phrases as say

MIN_SECTOR_LOSS_S = 0.1  # smaller than this and the sector isn't worth a call
MIN_CORNER_LOSS_S = 0.05
EARLY_BRAKE_M = 15  # braked at least this much earlier than in the best run
SLOW_APEX_KPH = 5  # carried at least this much less speed at the apex

# Race coaching: a corner must lose this much beyond the lap's general drift off its bests...
REPEAT_MIN_LOSS_S = 0.08
REPEAT_LAPS = 3  # ...on this many consecutive clean laps,
REPEAT_COOLDOWN_LAPS = 3  # and the engineer says it at most once every this many laps.


def corner_seconds(data: dict | None) -> float | None:
    """Time through one corner in seconds, or None when it wasn't timed end to end."""
    if not data:
        return None
    start, end = data.get('start_time_ms'), data.get('end_time_ms')
    if start is None or end is None or end <= start:
        return None
    return (end - start) / 1000.0


def worst_sector(sectors: dict, best_sectors: dict) -> tuple[int, float] | None:
    """(sector, seconds lost) for the sector furthest off its best, if any is worth saying."""
    losses = [
        (int(sector), (ms - best_sectors[sector]) / 1000.0)
        for sector, ms in sectors.items()
        if ms and best_sectors.get(sector)
    ]
    if not losses:
        return None
    sector, loss = max(losses, key=lambda item: item[1])
    return (sector, loss) if loss >= MIN_SECTOR_LOSS_S else None


def worst_corner(corners: dict, best_corners: dict, sector: int | None) -> tuple[str, float] | None:
    """(corner, seconds lost) for the corner furthest off its best run, inside ``sector``."""
    worst = None
    for name, data in corners.items():
        if sector is not None and data.get('sector') not in (None, sector):
            continue
        now, best = corner_seconds(data), corner_seconds(best_corners.get(name))
        if now is None or best is None:
            continue
        loss = now - best
        if loss >= MIN_CORNER_LOSS_S and (worst is None or loss > worst[1]):
            worst = (name, loss)
    return worst


def corner_reason(data: dict, best: dict) -> str | None:
    """Why the corner was slow, from the braking point and apex speed, or None if unclear."""
    brake, best_brake = data.get('brake_point'), best.get('brake_point')
    if brake is not None and best_brake is not None and best_brake - brake >= EARLY_BRAKE_M:
        metres = int(round((best_brake - brake) / 5.0) * 5)
        return f'you braked {say.number_words(metres)} metres early'
    apex, best_apex = data.get('min_speed'), best.get('min_speed')
    if apex is not None and best_apex is not None and best_apex - apex >= SLOW_APEX_KPH:
        return f'{say.number_words(round(best_apex - apex))} k slower at the apex, carry more speed in'
    return None


def coaching_line(sectors: dict, best_sectors: dict, corners: dict, best_corners: dict) -> str | None:
    """The coaching call for a finished lap, or None when there's nothing worth saying."""
    sector_loss = worst_sector(sectors, best_sectors)
    if sector_loss is None:
        return None
    sector, loss = sector_loss
    line = f'You lost {say.time_amount(loss)} in {say.sector_name(sector)}.'
    corner = worst_corner(corners, best_corners, sector)
    if corner:
        name, corner_loss = corner
        line += f' {name} cost you {say.time_amount(corner_loss)}'
        reason = corner_reason(corners[name], best_corners.get(name) or {})
        line += f', {reason}.' if reason else '.'
    return line + ' Fix that next lap.'


def update_best_corners(corners: dict, best_corners: dict) -> None:
    """Keep each corner's quickest run this session as the reference for the next laps."""
    for name, data in corners.items():
        now = corner_seconds(data)
        if now is None:
            continue
        best = corner_seconds(best_corners.get(name))
        if best is None or now < best:
            best_corners[name] = dict(data)


def corner_excess_losses(corners: dict, best_corners: dict) -> dict[str, float]:
    """Seconds each corner lost beyond the lap's typical drift off its bests.

    Tyre wear and fuel make every corner a little slower than its best in a race. The typical
    (median) fractional loss across the lap's corners is that drift; what a corner loses on top
    of it is down to the driver.
    """
    timed = {}
    for name, data in corners.items():
        now, best = corner_seconds(data), corner_seconds(best_corners.get(name))
        if now is not None and best is not None and best > 0:
            timed[name] = (now, best)
    if not timed:
        return {}
    drift = max(0.0, median((now - best) / best for now, best in timed.values()))
    return {name: now - best * (1 + drift) for name, (now, best) in timed.items()}


@dataclass
class CornerLossStreaks:
    """Race coaching: only a corner that keeps costing time, lap after lap, gets a call."""

    streaks: dict[str, int] = field(default_factory=dict)  # corner -> consecutive laps lost there
    losses: dict[str, float] = field(default_factory=dict)  # corner -> excess loss last lap (s)
    last_lap: int | None = None
    last_call_lap: int | None = None

    def reset(self) -> None:
        self.streaks, self.losses = {}, {}
        self.last_lap = self.last_call_lap = None

    def observe(self, lap_num: int, corners: dict, best_corners: dict) -> None:
        """Count a clean lap. A skipped lap (pit, Safety Car, invalid) breaks every streak."""
        if self.last_lap is None or lap_num != self.last_lap + 1:
            self.streaks = {}
        self.last_lap = lap_num
        lost = {
            name: loss
            for name, loss in corner_excess_losses(corners, best_corners).items()
            if loss >= REPEAT_MIN_LOSS_S
        }
        self.streaks = {name: self.streaks.get(name, 0) + 1 for name in lost}
        self.losses = lost

    def line(self, lap_num: int, corners: dict, best_corners: dict) -> str | None:
        """The coaching call for a corner lost on REPEAT_LAPS laps running, or None."""
        due = [name for name, laps in self.streaks.items() if laps >= REPEAT_LAPS]
        if not due:
            return None
        if self.last_call_lap is not None and lap_num - self.last_call_lap < REPEAT_COOLDOWN_LAPS:
            return None
        name = max(due, key=lambda n: self.losses[n])
        self.last_call_lap = lap_num
        laps = say.number_words(self.streaks[name])
        text = (
            f'{name} again, {laps} laps running. You lost {say.time_amount(self.losses[name])} there last lap'
        )
        reason = corner_reason(corners.get(name) or {}, best_corners.get(name) or {})
        return text + (f', {reason}.' if reason else '.')
