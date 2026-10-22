"""
Stored track outlines and pit lanes for the HUD's live track map.

F1 2021 world coordinates are fixed per circuit, so a circuit only has to be mapped once:
the first clean, complete lap on a track becomes its outline, and the first full pass
through the pit lane becomes its pit lane. Both are saved (one JSON file for all tracks)
and loaded straight away the next time the game reports that track, so the HUD shows the
whole circuit from the first frame and only moves the car around it.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
from pathlib import Path

from log_setup import get_logger

log = get_logger('track_maps')

OUTLINE_BIN_M = 8.0  # one outline point per 8 m of lap distance
MAX_LINE_GAP_M = 40.0  # a lap with a bigger hole in it (rewind, glitch) is not used
START_TOLERANCE_M = 60.0  # the lap must start this close to the line...
END_TOLERANCE_M = 80.0  # ...and reach this close to the end of the lap
LENGTH_TOLERANCE_M = 50.0  # stored map must match the session's track length
MIN_PIT_POINTS = 15
PIT_POINT_SPACING_M = 4.0


def build_outline(
    dists: list[float], points: list[list[float]], track_length: float, bin_m: float = OUTLINE_BIN_M
) -> list[list[float]] | None:
    """Resample one lap's (distance, [x, z]) trail into an outline, or None if it is not a
    full lap: it must start at the line, reach the end, and have no hole in it."""
    if track_length <= 0 or len(dists) != len(points) or len(points) < 10:
        return None
    pairs = sorted((d, p) for d, p in zip(dists, points, strict=False) if 0 <= d <= track_length + 5)
    if not pairs:
        return None
    if pairs[0][0] > START_TOLERANCE_M or pairs[-1][0] < track_length - END_TOLERANCE_M:
        return None
    if any(b[0] - a[0] > MAX_LINE_GAP_M for a, b in zip(pairs, pairs[1:], strict=False)):
        return None
    outline, last_bin = [], None
    for dist, (x, z) in pairs:
        index = int(dist // bin_m)
        if index != last_bin:
            outline.append([round(x, 1), round(z, 1)])
            last_bin = index
    return outline


def append_pit_point(
    points: list[list[float]], x: float, z: float, spacing: float = PIT_POINT_SPACING_M
) -> bool:
    """Add a pit lane point when the car has moved far enough from the last one."""
    if points and math.hypot(x - points[-1][0], z - points[-1][1]) < spacing:
        return False
    points.append([round(x, 1), round(z, 1)])
    return True


class TrackMapStore:
    """Outline and pit lane per track id. With no path it only keeps them in memory."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path is not None else None
        self.maps: dict[str, dict] = {}
        if self.path is not None and self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding='utf-8'))
                if isinstance(data, dict):
                    self.maps = {str(k): v for k, v in data.items() if isinstance(v, dict)}
            except (OSError, ValueError) as error:
                log.warning(f'Could not read {self.path}: {error}')

    def get(self, track_id, track_length: float) -> dict:
        """{'outline': [...], 'pit_lane': [...]} for this track ([] where not mapped yet)."""
        entry = self.maps.get(str(track_id))
        if (
            not entry
            or track_length <= 0
            or abs(entry.get('track_length', 0) - track_length) > LENGTH_TOLERANCE_M
        ):
            return {'outline': [], 'pit_lane': []}
        return {'outline': list(entry.get('outline') or []), 'pit_lane': list(entry.get('pit_lane') or [])}

    def save(self, track_id, track_length: float, outline=None, pit_lane=None):
        """Store a new outline and/or pit lane for a track (None leaves that part as is)."""
        key = str(track_id)
        entry = dict(self.maps.get(key) or {})
        if abs(entry.get('track_length', track_length) - track_length) > LENGTH_TOLERANCE_M:
            entry = {}  # different layout under the same id: start over
        entry['track_length'] = track_length
        if outline is not None:
            entry['outline'] = outline
        if pit_lane is not None:
            entry['pit_lane'] = pit_lane
        entry['updated'] = int(time.time())
        self.maps[key] = entry
        self._write()

    def forget(self, track_id: int) -> None:
        if self.maps.pop(str(track_id), None) is not None:
            self._write()

    def _write(self):
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix='.tmp')
            with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                json.dump(self.maps, handle, separators=(',', ':'))
            os.replace(tmp, self.path)
        except OSError as error:
            log.warning(f'Could not save {self.path}: {error}')


FIELD_COVERAGE_NEEDED = 0.97  # share of 8 m bins that must be seen before the outline is used
FIELD_MAX_MISSING_BINS = 5  # and no hole longer than this (40 m)
FIELD_SAMPLES_PER_BIN = 8


class FieldMapper:
    """Builds the circuit outline from every car on track, not only the player.

    Each car's (lap distance, x, z) lands in an 8 m bin and the bin keeps the mean
    position, so with a full field the whole circuit is known within a minute or two of
    running, long before the player finishes a lap."""

    def __init__(self, track_length: float, bin_m: float = OUTLINE_BIN_M):
        self.track_length = track_length
        self.bin_m = bin_m
        self.total_bins = max(1, int(track_length // bin_m) + 1) if track_length > 0 else 0
        self.bins: dict[int, list[float]] = {}

    def add(self, dist: float, x: float, z: float) -> bool:
        """Add one sample; True when it filled a new bin (the shape changed visibly)."""
        if not self.total_bins or not (0 <= dist < self.track_length):
            return False
        if not (math.isfinite(x) and math.isfinite(z)) or (x == 0 and z == 0):
            return False
        index = int(dist // self.bin_m)
        entry = self.bins.get(index)
        if entry is None:
            self.bins[index] = [x, z, 1]
            return True
        if entry[2] < FIELD_SAMPLES_PER_BIN:
            entry[0] += x
            entry[1] += z
            entry[2] += 1
        return False

    @property
    def coverage(self) -> float:
        return len(self.bins) / self.total_bins if self.total_bins else 0.0

    def _points(self, index):
        x, z, n = self.bins[index]
        return [round(x / n, 1), round(z / n, 1)]

    def outline(self) -> list[list[float]] | None:
        """The full outline once nearly every bin is seen and no hole is too long."""
        if self.coverage < FIELD_COVERAGE_NEEDED:
            return None
        indexes = sorted(self.bins)
        wrap_gap = indexes[0] + self.total_bins - indexes[-1]
        gaps = [b - a for a, b in zip(indexes, indexes[1:], strict=False)] + [wrap_gap]
        if max(gaps) > FIELD_MAX_MISSING_BINS + 1:
            return None
        return [self._points(i) for i in indexes]

    def partial(self) -> list:
        """What is known so far, in lap order; None marks a hole so no chord is drawn."""
        points, previous = [], None
        for index in sorted(self.bins):
            if previous is not None and index - previous > FIELD_MAX_MISSING_BINS + 1:
                points.append(None)
            points.append(self._points(index))
            previous = index
        return points
