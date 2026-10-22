"""Track map stage: the circuit outline, pit lane, live trail and every car on the map.

The outline comes from the store (track_maps.py) when this circuit was mapped before,
otherwise from the whole field's positions or the player's first clean lap. Flashbacks
cut the trail back to where the restored timeline says the car is.
"""

from __future__ import annotations

import math

import track_maps
import web_hud
from packets import LapData, SessionData

from pipeline.state import JSON
from pipeline.strategy import StrategyStage


class TrackMapStage(StrategyStage):

    def _record_trail_point(self, x: float, z: float) -> None:
        """One Motion packet's player position: extend the live trail, pit lane and zones."""
        if not (self.latest_lap and self.latest_telemetry and not self.map_rewind_pending):
            return
        lap = self.latest_lap
        if lap.pit_status:
            # Pit lane is its own path, never part of the circuit outline.
            self.map_lap_has_pit = True
            track_maps.append_pit_point(self.current_pit_points, x, z)
            return
        if self.current_pit_points:
            self._finish_pit_lane()
        if self.delta.current_lap_num and lap.current_lap_num != self.delta.current_lap_num:
            return                        # lap change not processed yet: keep last lap intact
        dist = lap.lap_distance
        if dist < self.last_map_dist - 30.0:
            self._trim_trail(dist)        # car went backwards along the lap: a rewind
        if abs(dist - self.last_map_dist) >= 6.0:
            self.last_map_dist = dist
            self.current_lap_map_points.append([x, z])
            self.current_lap_map_dists.append(dist)
            self._map_revision += 1
            if self.latest_telemetry.brake > 0.08:
                self.current_lap_braking_zones.append([x, z, round(self.latest_telemetry.brake, 2)])
                self.current_lap_braking_dists.append(dist)
            status = self.latest_status
            if status and (status.ers_deploy_mode > 0 or status.ers_deployed_this_lap > 0):
                self.current_lap_ers_zones.append([x, z])
                self.current_lap_ers_dists.append(dist)

    def _update_field_map(self, positions: list[tuple[float, float]]) -> None:
        """Every car on the map, and every car's laps feeding the outline and pit lane."""
        laps = self.field_laps
        if not laps or not self.session:
            self.field_cars = []
            return
        cars: list[list[float]] = []
        grew = False
        for index, (x, z) in enumerate(positions[:len(laps)]):
            lap = laps[index]
            if (lap.result_status not in (2, 3) or (x == 0 and z == 0)
                    or not (math.isfinite(x) and math.isfinite(z))):
                continue
            cars.append([round(x, 1), round(z, 1), self.car_teams.get(index, -1), lap.car_position,
                         int(index == self.player_index), int(bool(lap.pit_status))])
            if self.track_outline and self.pit_lane_points:
                continue                                   # nothing left to map
            if lap.pit_status:
                track_maps.append_pit_point(self.field_pit_points.setdefault(index, []), x, z)
                continue
            pit_run = self.field_pit_points.pop(index, None)
            if pit_run and not self.pit_lane_points and len(pit_run) >= track_maps.MIN_PIT_POINTS:
                self.pit_lane_points = pit_run
                self.track_store.save(self.track_id, self.session.track_length, pit_lane=pit_run)
                self._push_track_map()
            if lap.driver_status != 0 and self.field_mapper and not self.track_outline:
                grew |= self.field_mapper.add(lap.lap_distance, x, z)
        self.field_cars = cars
        if grew:
            assert self.field_mapper is not None
            outline = self.field_mapper.outline()
            if outline:
                self.track_outline = outline
                self.track_store.save(self.track_id, self.session.track_length, outline=outline)
                self._push_track_map()
            self._map_revision += 1

    def _load_track_map(self, session: SessionData) -> None:
        stored = self.track_store.get(session.track_id, session.track_length)
        self.track_outline = stored['outline']
        self.pit_lane_points = stored['pit_lane']
        self._map_revision += 1
        self._push_track_map()

    def _push_track_map(self) -> None:
        web_hud.update_web_state({'track_outline': self.track_outline,
                                  'pit_lane': self.pit_lane_points,
                                  'track_map_source': 'stored' if self.track_outline else 'recording'})

    def _clear_trail_distances(self) -> None:
        self.current_lap_map_dists.clear()
        self.current_lap_braking_dists.clear()
        self.current_lap_ers_dists.clear()

    def _clear_lap_trail(self) -> None:
        self.current_lap_map_points.clear()
        self.current_lap_braking_zones.clear()
        self.current_lap_ers_zones.clear()
        self._clear_trail_distances()

    def _trim_trail(self, dist: float) -> None:
        """Drop every trail point recorded beyond `dist` on this lap (the rewound future)."""
        for points, dists in ((self.current_lap_map_points, self.current_lap_map_dists),
                              (self.current_lap_braking_zones, self.current_lap_braking_dists),
                              (self.current_lap_ers_zones, self.current_lap_ers_dists)):
            keep = next((i for i, d in enumerate(dists) if d > dist), len(dists))
            del points[keep:], dists[keep:]
        self.last_map_dist = self.current_lap_map_dists[-1] if self.current_lap_map_dists else -999.0
        self._map_revision += 1

    def _finish_pit_lane(self) -> None:
        """Car left the pit lane: keep the path as the pit lane if this track has none yet."""
        points, self.current_pit_points = self.current_pit_points, []
        if self.pit_lane_points or len(points) < track_maps.MIN_PIT_POINTS or not self.session:
            return
        self.pit_lane_points = points
        self.track_store.save(self.track_id, self.session.track_length, pit_lane=points)
        self._push_track_map()

    def _capture_outline(self, lap_num: int, new_lap_num: int) -> None:
        """After a full lap, store it as the circuit outline if this track has none yet."""
        if (self.track_outline or self.map_lap_has_pit or not self.session
                or new_lap_num != lap_num + 1):
            return
        outline = track_maps.build_outline(self.current_lap_map_dists, self.current_lap_map_points,
                                           self.session.track_length)
        if outline:
            self.track_outline = outline
            self.track_store.save(self.track_id, self.session.track_length, outline=outline)
            self._push_track_map()

    def on_flashback(self) -> None:
        """The game rewound (F1 2021 flashback). Stop recording until a lap packet from the
        restored timeline says where the car is, then cut the trail back to there."""
        self.map_rewind_pending = True
        self.current_pit_points.clear()
        self.flashback_count += 1
        web_hud.update_web_state({'flashback_count': self.flashback_count})

    def _resolve_map_rewind(self, lap: LapData) -> None:
        self.map_rewind_pending = False
        if lap.current_lap_num == self.delta.current_lap_num:
            self._trim_trail(lap.lap_distance)
        # A rewind into an earlier lap is handled by the lap change in update_delta.

    def reset_map(self) -> None:
        """Manually wipe the recorded track map/braking/ERS trail (driver pressed 'Reset Map').

        Unlike a new-lap or new-session reset, this does not touch lap timing, tyre or
        AI state — it only clears the coordinate trail so a dirty out-lap or a spin that
        left stray points doesn't junk up the map for the rest of the session.
        """
        self._clear_lap_trail()
        self.track_map_points.clear()
        self.braking_zones_best.clear()
        self.ers_zones_best.clear()
        self.current_pit_points.clear()
        self.last_map_dist = -999.0
        # Also forget the stored outline/pit lane, so a bad one can be mapped again.
        self.track_outline, self.pit_lane_points = [], []
        self.field_pit_points.clear()
        if self.session:
            self.track_store.forget(self.track_id)
            self.field_mapper = track_maps.FieldMapper(self.session.track_length)
        self._map_revision += 1
        self._sent_map_revision = self._map_revision
        web_hud.update_web_state({'track_map': [], 'braking_zones': [], 'ers_zones': []})
        self._push_track_map()

    def map_update(self) -> JSON:
        """The live trail for the HUD, only when it changed, and only until the outline is
        known: the whole lap (not a 400-point tail) so nothing gets erased."""
        if self._map_revision == self._sent_map_revision:
            return {}
        self._sent_map_revision = self._map_revision
        trail = [] if self.track_outline else (self.track_map_points or self.current_lap_map_points)
        if (not self.track_outline and self.field_mapper
                and len(self.field_mapper.bins) * 1.3 > len(trail)):
            trail = self.field_mapper.partial()      # the field has seen more of the track
        return {
            'track_map': trail,
            'current_lap_trail': self.current_lap_map_points,
            'braking_zones': self.braking_zones_best or self.current_lap_braking_zones,
            'ers_zones': self.ers_zones_best or self.current_lap_ers_zones,
        }
