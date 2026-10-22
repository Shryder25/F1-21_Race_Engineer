"""Pit wall timing: the whole field's order, gaps, gap history and tyre stints.

F1 timing measures gaps at timing loops round the lap: a car's gap to the leader is the
time between the first car crossing a loop and this car crossing the same loop. That works
for every car, at any gap, without knowing anyone's pace, so it is what `PitWall` does.

The loops sit every ~100 m of *race distance* (``(lap - 1) * track length + lap distance``),
so the gaps refresh several times a second and a lapped car's gap still means something
(the loop it crossed is the one the leader crossed a lap earlier).

What it gives the HUD:

  timing tower   every car in position order: gap to leader (or laps down), interval to the
                 car ahead, last and best lap, tyre and its age, pit stops, in-pit flag
  gap history    each car's gap to the leader at the line, lap by lap (the race chart)
  stints         each car's tyre sets with the laps they ran (from the Session History
                 packet when the game sends it, otherwise from the Car Status compound)

Outside a race (practice, qualifying) there are no running gaps: the order and the gaps
come from best laps instead.

Gap history and stints only change once a lap per car, so each has a revision number and
the caller sends them to the HUD only when it moved.
"""

from __future__ import annotations

import math

LOOP_SPACING_M = 100.0       # timing loops every ~100 m of race distance
MAX_INTERPOLATE_LOOPS = 40   # a bigger jump (teleport, data gap) is not interpolated
MAX_INTERPOLATE_S = 3.0

RESULT_LABELS = {3: 'FIN', 4: 'DSQ', 5: 'NC', 6: 'DNF', 7: 'DNF'}
COMPOUND_LETTERS = {'soft': 'S', 'medium': 'M', 'hard': 'H', 'inter': 'I', 'intermediate': 'I',
                    'wet': 'W', 'super soft': 'S', 'ultra soft': 'S', 'hyper soft': 'S'}


def compound_letter(name) -> str | None:
    """'Soft' -> 'S', 'Intermediate' -> 'I'. Unknown or missing -> None."""
    if not name:
        return None
    lower = str(name).lower()
    if lower in COMPOUND_LETTERS:
        return COMPOUND_LETTERS[lower]
    for key, letter in COMPOUND_LETTERS.items():
        if key in lower:
            return letter
    return None


class PitWall:
    def __init__(self) -> None:
        self.reset()

    def reset(self, track_length: float = 0.0, race: bool = True) -> None:
        self.track_length = float(track_length or 0.0)
        self.race = race
        self.loops_per_lap = max(1, round(self.track_length / LOOP_SPACING_M)) if self.track_length > 0 else 0
        self.loop_m = self.track_length / self.loops_per_lap if self.loops_per_lap else 0.0
        self.crossings: dict[int, dict[int, float]] = {}   # car -> loop -> session time
        self.first_crossing: dict[int, float] = {}          # loop -> earliest session time
        self.last_point: dict[int, tuple[float, float]] = {}  # car -> (race distance, time)
        self.last_loop: dict[int, int] = {}
        self.oldest_loop = 0
        self.gap_history: dict[int, dict[int, float]] = {}   # car -> completed lap -> gap s
        self.best_lap_ms: dict[int, int] = {}
        self.status_stints: dict[int, list] = {}            # fallback stints from Car Status
        self.last_session_time = None
        self.rows: list[dict] = []
        self.history_revision = 0
        self.stint_revision = 0
        self._stint_key = None
        self.histories: dict = {}

    # ── feeding it ──

    def update(self, laps, session_time: float, player_index: int, statuses=None,
               names: dict | None = None, teams: dict | None = None, histories: dict | None = None):
        """One Lap Data frame for every car (plus the latest Car Status and Session History)."""
        if not laps:
            return
        if self.last_session_time is not None and session_time < self.last_session_time - 1.0:
            self._rewind(laps)
        self.last_session_time = session_time
        if self.race and self.loops_per_lap:
            for index, lap in enumerate(laps):
                if lap.result_status != 2 or lap.car_position == 0:
                    continue
                self._advance(index, lap, session_time)
            self._prune()
        for index, lap in enumerate(laps):
            if lap.last_lap_time_ms and lap.result_status in (2, 3):
                best = self.best_lap_ms.get(index)
                if best is None or lap.last_lap_time_ms < best:
                    self.best_lap_ms[index] = lap.last_lap_time_ms
        self._track_status_stints(laps, statuses)
        self.rows = self._build_rows(laps, player_index, statuses, names or {}, teams or {})
        self.histories = histories or {}
        key = self._stints_key()
        if key != self._stint_key:
            self._stint_key = key
            self.stint_revision += 1

    def _race_distance(self, lap) -> float | None:
        dist = lap.lap_distance
        if not math.isfinite(dist) or lap.current_lap_num <= 0:
            return None
        if dist < 0:
            return None                   # before the line on the grid / formation lap
        return (lap.current_lap_num - 1) * self.track_length + dist

    def _advance(self, index: int, lap, now: float):
        dist = self._race_distance(lap)
        if dist is None:
            return
        previous = self.last_point.get(index)
        self.last_point[index] = (dist, now)
        loop = int(dist // self.loop_m)
        if previous is None:
            self.last_loop.setdefault(index, loop)
            return
        prev_dist, prev_time = previous
        prev_loop = self.last_loop.get(index, int(prev_dist // self.loop_m))
        if loop <= prev_loop or dist <= prev_dist:
            if loop < prev_loop:
                self.last_loop[index] = loop      # went backwards (flashback on one car)
            return
        if loop - prev_loop > MAX_INTERPOLATE_LOOPS or now - prev_time > MAX_INTERPOLATE_S:
            self.last_loop[index] = loop          # can't vouch for the loops in between
            return
        mine = self.crossings.setdefault(index, {})
        for k in range(prev_loop + 1, loop + 1):
            at = k * self.loop_m
            t = prev_time + (now - prev_time) * (at - prev_dist) / (dist - prev_dist)
            mine[k] = t
            first = self.first_crossing.get(k)
            if first is None or t < first:
                self.first_crossing[k] = t
            if k % self.loops_per_lap == 0:
                self._record_line(index, k // self.loops_per_lap, t)
        self.last_loop[index] = loop

    def _record_line(self, index: int, lap_done: int, t: float):
        first = self.first_crossing.get(lap_done * self.loops_per_lap)
        if first is None or lap_done < 1:
            return
        self.gap_history.setdefault(index, {})[lap_done] = round(max(0.0, t - first), 2)
        self.history_revision += 1

    def _prune(self):
        """Forget loops every running car has passed (keeps memory flat over a race)."""
        if not self.last_loop:
            return
        keep_from = min(self.last_loop.values()) - 1
        if keep_from <= self.oldest_loop:
            return
        for k in range(self.oldest_loop, keep_from):
            self.first_crossing.pop(k, None)
            for mine in self.crossings.values():
                mine.pop(k, None)
        self.oldest_loop = keep_from

    def _rewind(self, laps):
        """Flashback: drop timing past the restored point; lap history after it goes too."""
        self.crossings.clear()
        self.first_crossing.clear()
        self.last_point.clear()
        self.last_loop.clear()
        self.oldest_loop = 0
        for index, history in self.gap_history.items():
            current = laps[index].current_lap_num if index < len(laps) else 0
            for lap_done in [n for n in history if n >= current]:
                del history[lap_done]
        self.history_revision += 1

    def drop_car_history(self, index: int):
        self.gap_history.pop(index, None)
        self.history_revision += 1

    # ── what the HUD shows ──

    def gap_to_leader(self, index: int) -> tuple[float | None, int]:
        """(seconds behind the leader at the car's last loop, laps down)."""
        loop = self.last_loop.get(index)
        mine = self.crossings.get(index, {})
        if loop is None or loop not in mine:
            return None, 0
        lead_loop = max(self.last_loop.values())
        laps_down = (lead_loop - loop) // self.loops_per_lap if self.loops_per_lap else 0
        first = self.first_crossing.get(loop)
        if first is None:
            return None, laps_down
        return max(0.0, mine[loop] - first), laps_down

    def interval(self, index: int, ahead: int) -> float | None:
        """Seconds between the car ahead and this car at this car's last loop."""
        loop = self.last_loop.get(index)
        mine = self.crossings.get(index, {})
        theirs = self.crossings.get(ahead, {})
        if loop is None or loop not in mine or loop not in theirs:
            return None
        return max(0.0, mine[loop] - theirs[loop])

    def _build_rows(self, laps, player_index, statuses, names, teams) -> list[dict]:
        cars = [(i, lap) for i, lap in enumerate(laps) if lap.car_position > 0 and lap.result_status >= 2]
        cars.sort(key=lambda c: (c[1].result_status not in (2, 3), c[1].car_position))
        fastest = min(self.best_lap_ms.values(), default=None)
        rows, ahead = [], None
        leader_best = None
        for slot, (index, lap) in enumerate(cars):
            running = lap.result_status in (2, 3)
            status = statuses[index] if statuses and index < len(statuses) else None
            best = self.best_lap_ms.get(index)
            row = {
                'index': index, 'position': lap.car_position, 'name': names.get(index),
                'team_id': teams.get(index), 'is_player': index == player_index,
                'gap_s': None, 'laps_down': 0, 'interval_s': None,
                'last_lap_ms': lap.last_lap_time_ms or None, 'best_lap_ms': best,
                'fastest_lap': bool(best and best == fastest),
                'compound': compound_letter(status.compound_name) if status else None,
                'tyre_age': status.tyres_age_laps if status else None,
                'stops': lap.num_pit_stops, 'pit': lap.pit_status,
                'penalties_s': lap.penalties,
                'status': 'RUN' if lap.result_status == 2 else RESULT_LABELS.get(lap.result_status, 'OUT'),
            }
            if self.race:
                if running:
                    gap, down = self.gap_to_leader(index)
                    row['gap_s'] = None if gap is None else round(gap, 3)
                    row['laps_down'] = down
                    if ahead is not None:
                        interval = self.interval(index, ahead)
                        row['interval_s'] = None if interval is None else round(interval, 3)
            elif best:
                if leader_best is None:
                    leader_best = best
                row['gap_s'] = round((best - leader_best) / 1000.0, 3)
                if ahead is not None and self.best_lap_ms.get(ahead):
                    row['interval_s'] = round((best - self.best_lap_ms[ahead]) / 1000.0, 3)
            if slot == 0:
                row['gap_s'] = 0.0 if (self.race or best) else None
            rows.append(row)
            if running:
                ahead = index
        return rows

    # ── stints ──

    def _track_status_stints(self, laps, statuses):
        if not statuses:
            return
        for index, lap in enumerate(laps):
            if index >= len(statuses) or statuses[index] is None or lap.car_position == 0:
                continue
            letter = compound_letter(statuses[index].compound_name)
            if letter is None:
                continue
            lap_num = max(1, lap.current_lap_num)
            stints = self.status_stints.setdefault(index, [])
            if not stints:
                stints.append([letter, lap_num, lap_num, lap.num_pit_stops])
            elif stints[-1][0] != letter or lap.num_pit_stops != stints[-1][3]:
                # Like the Session History packet: the old set ends on the in-lap and the
                # new one starts on the next lap.
                if stints[-1][1] > lap_num:
                    stints.pop()                  # changed again before its first lap
                if stints:
                    stints[-1][2] = lap_num
                stints.append([letter, lap_num + 1, lap_num + 1, lap.num_pit_stops])
            else:
                stints[-1][2] = max(stints[-1][1], lap_num)

    def _stints_key(self):
        hist = tuple((i, tuple((s.start_lap, s.end_lap, s.visual_compound) for s in h.stints))
                     for i, h in sorted(self.histories.items()))
        status = tuple((i, tuple(s[:3] for s in st)) for i, st in sorted(self.status_stints.items()))
        order = tuple(row['index'] for row in self.rows)
        return hist, status, order

    def stints(self) -> list[dict]:
        """Each car's tyre sets, in running order: [{index, name, team_id, is_player,
        stints: [{compound, start_lap, end_lap, open}]}]."""
        out = []
        for row in self.rows:
            index = row['index']
            history = self.histories.get(index)
            sets = []
            if history is not None and history.stints:
                for s in history.stints:
                    letter = compound_letter(s.compound_name)
                    if letter is None:
                        continue
                    sets.append({'compound': letter, 'start_lap': s.start_lap,
                                 'end_lap': max(s.start_lap, s.end_lap), 'open': bool(s.is_open)})
            else:
                fallback = self.status_stints.get(index, [])
                for n, (letter, start, end, _stops) in enumerate(fallback):
                    sets.append({'compound': letter, 'start_lap': start, 'end_lap': max(start, end),
                                 'open': n == len(fallback) - 1})
            out.append({'index': index, 'name': row['name'], 'team_id': row['team_id'],
                        'is_player': row['is_player'], 'position': row['position'], 'stints': sets})
        return out

    def gap_chart(self, names: dict | None = None, teams: dict | None = None,
                  player_index: int | None = None) -> list[dict]:
        """Each car's gap to the leader at the line, lap by lap: [{index, name, team_id,
        is_player, points: [[lap, gap s], ...]}]."""
        names, teams = names or {}, teams or {}
        order = {row['index']: row['position'] for row in self.rows}
        out = []
        for index in sorted(self.gap_history, key=lambda i: order.get(i, 99)):
            history = self.gap_history[index]
            if not history:
                continue
            out.append({'index': index, 'name': names.get(index), 'team_id': teams.get(index),
                        'is_player': index == player_index,
                        'points': [[lap, history[lap]] for lap in sorted(history)]})
        return out


def strategy_odds(result, history: list | None = None) -> dict | None:
    """The Monte Carlo race simulation (race_sim.SimResult) as the HUD's probability panel."""
    if result is None or not result.outcomes:
        return None
    top = result.outcomes[0]
    return {
        'lap': result.lap, 'laps_to_go': result.laps_to_go, 'runs': result.runs,
        'recommended': top.label,
        'p_win': top.p_win, 'p_podium': top.p_podium, 'p_points': top.p_points,
        'mean_position': top.mean_position, 'position_range': top.position_range,
        'position_probs': top.position_probs,
        'plans': [{'label': o.label, 'stops': o.stops, 'compounds': o.compounds, 'pit_laps': o.pit_laps,
                   'p_best': o.p_best, 'p_win': o.p_win, 'p_podium': o.p_podium, 'p_points': o.p_points,
                   'mean_position': o.mean_position, 'position_range': o.position_range,
                   'time_vs_best_s': o.time_vs_best_s}
                  for o in result.outcomes],
        'field': [dict(f) for f in result.field[:10]],
        'p_safety_car': result.p_safety_car, 'p_vsc': result.p_vsc, 'p_rain': result.p_rain,
        'pit_loss_s': result.pit_loss_s,
        'history': list(history or []),
    }
