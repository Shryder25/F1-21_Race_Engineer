"""Race control: turn F1 2021 race events into proactive engineer radio calls.

Two sources feed it:
  * Event packets (penalties, fastest laps, retirements, DRS, chequered flag, race winner),
    decoded by packets.parse_race_event().
  * The whole field's lap data and car status, from which it works out what the game
    doesn't send as an event: overtakes, rival pit stops, the player's own stop time,
    flags waved at the player, and the end-of-lap-one and finishing positions.

Every call also lands in a short "race control" feed for the web HUD and the AI prompt,
including calls that on-demand mode keeps off the radio.
"""

import math
from collections import deque

from voice_engine import Priority

RACE_SESSIONS = (10, 11, 12)
ACTIVE, FINISHED = 2, 3
OUT_OF_RACE = (4, 5, 7)            # did not finish, disqualified, retired

# A new position has to hold this long before it is called (lap data jitters across the line).
POSITION_SETTLE_S = 1.5
# Rival calls (pit stops, penalties) only for cars this many places either side of the player.
NEARBY_POSITIONS = 3

# vehicleFiaFlags from Car Status: -1 unknown, 0 none, 1 green, 2 blue, 3 yellow, 4 red.
GREEN, BLUE, YELLOW, RED = 1, 2, 3, 4

# Penalty types (packets.PENALTY_TYPES) grouped by how the engineer reacts.
DRIVE_THROUGH, STOP_GO, GRID, REMINDER, TIME_PENALTY, WARNING, DSQ = 0, 1, 2, 3, 4, 5, 6
LAP_INVALIDATED = range(10, 16)
TYRE_REGS, PARKED_TOO_LONG, BLACK_FLAG_TIMER = 9, 8, 17
SERIOUS_PENALTIES = (DRIVE_THROUGH, STOP_GO, TIME_PENALTY, DSQ)

COMPOUND_PLURALS = {'Soft': 'softs', 'Medium': 'mediums', 'Hard': 'hards',
                    'Intermediate': 'inters', 'Wet': 'wets'}


def lap_time_text(seconds: float) -> str:
    minutes, rest = divmod(seconds, 60)
    return f'{int(minutes)}:{rest:06.3f}' if minutes else f'{rest:.3f}'


def capitalise(text: str) -> str:
    return text[:1].upper() + text[1:]


class RaceControl:
    def __init__(self, speak):
        self.speak = speak
        self.participants: list = []
        self.feed: deque = deque(maxlen=30)
        self.reset(None)

    # ──────────────────────────────────────────
    # State
    # ──────────────────────────────────────────

    def reset(self, session):
        """New session: forget everything except who is driving which car."""
        self.session = session
        self.feed.clear()
        self.retired: set[int] = set()
        self.lap_one_reported = False
        self.finish_reported = False
        self.last_flag = None
        self.reset_tracking()

    def reset_tracking(self):
        """Drop frame-to-frame tracking (also after a flashback), keeping the feed."""
        self.laps = None
        self.player_index = None
        self.statuses = None
        self.session_time = 0.0
        self.player_lap_num = None
        self.confirmed_position = None
        self.candidate_position = None
        self.candidate_since = None
        self.baseline_positions: dict[int, int] = {}
        self.pitted_since_baseline: set[int] = set()
        self.pit_entries: dict[int, dict] = {}       # car index -> state captured at pit entry
        self.result_status: dict[int, int] = {}
        self.leader = self.leader_candidate = None
        self.leader_since = None

    @property
    def is_race(self) -> bool:
        return bool(self.session and self.session.session_type in RACE_SESSIONS)

    def update_participants(self, participants):
        self.participants = list(participants)

    def position_of(self, index):
        if self.laps is None or not 0 <= index < len(self.laps):
            return None
        return self.laps[index].car_position or None

    def name(self, index) -> str:
        participant = self.participants[index] if 0 <= index < len(self.participants) else None
        if participant is not None:
            if participant.short_name:
                return participant.short_name
            if participant.race_number:
                return f'car {participant.race_number}'
        position = self.position_of(index)
        return f'the car in P{position}' if position else 'a rival'

    def is_nearby(self, index, reach=NEARBY_POSITIONS) -> bool:
        mine, theirs = self.position_of(self.player_index), self.position_of(index)
        return bool(mine and theirs and abs(mine - theirs) <= reach)

    def compound(self, index):
        if self.statuses is None or not 0 <= index < len(self.statuses):
            return None
        return COMPOUND_PLURALS.get(self.statuses[index].compound_name)

    def call(self, message, priority, key, kind, ttl=12, cooldown=0, spoken=True):
        position = self.position_of(self.player_index) if self.player_index is not None else None
        lap = self.laps[self.player_index].current_lap_num if position else None
        self.feed.appendleft({'kind': kind, 'text': message, 'lap': lap,
                              'session_time': round(self.session_time, 1)})
        if spoken:
            self.speak(message, priority, key=key, ttl=ttl, cooldown=cooldown)

    def recent(self, limit=8) -> list[dict]:
        return list(self.feed)[:limit]

    # ──────────────────────────────────────────
    # Event packets
    # ──────────────────────────────────────────

    def handle_event(self, event, player_index, session_time=None):
        self.player_index = player_index
        if session_time is not None and math.isfinite(session_time):
            self.session_time = session_time
        handler = getattr(self, f'_event_{event.code.lower()}', None)
        if handler is not None:
            handler(event)

    def _valid_car(self, index):
        return index is not None and 0 <= index < 22

    def _event_pena(self, event):
        if not self._valid_car(event.vehicle_index):
            return
        if event.vehicle_index == self.player_index:
            self._own_penalty(event)
        elif (self.is_race and event.penalty_type in SERIOUS_PENALTIES
              and (event.other_vehicle_index == self.player_index or self.is_nearby(event.vehicle_index)
                   or self.position_of(event.vehicle_index) == 1)):
            who = capitalise(self.name(event.vehicle_index))
            if event.penalty_type == TIME_PENALTY:
                what = f'a {event.time_s} second time penalty'
            elif event.penalty_type == DRIVE_THROUGH:
                what = 'a drive through'
            elif event.penalty_type == STOP_GO:
                what = f'a {event.time_s} second stop go' if event.time_s else 'a stop go'
            else:
                self.call(f'{who} has been disqualified.', Priority.RACE,
                          f'rival_penalty_{event.vehicle_index}', 'penalty')
                return
            reason = ('for the contact with you' if event.other_vehicle_index == self.player_index
                      else f'for {event.infringement_name}')
            self.call(f'{who} has {what}, {reason}.', Priority.RACE,
                      f'rival_penalty_{event.vehicle_index}', 'penalty')

    def _own_penalty(self, event):
        kind, reason = event.penalty_type, event.infringement_name
        if kind == TIME_PENALTY:
            self.call(f'{event.time_s} second time penalty, for {reason}.', Priority.CRITICAL,
                      'penalty', 'penalty')
        elif kind == DRIVE_THROUGH:
            self.call(f'Drive through penalty, for {reason}. We need to serve it, box when told.',
                      Priority.CRITICAL, 'penalty', 'penalty')
        elif kind == STOP_GO:
            seconds = f'{event.time_s} second ' if event.time_s else ''
            self.call(f'{capitalise(seconds)}stop go penalty, for {reason}.', Priority.CRITICAL,
                      'penalty', 'penalty')
        elif kind == WARNING:
            self.call(f'That is a warning, for {reason}. Keep it clean.', Priority.STRATEGY,
                      'penalty', 'warning', cooldown=5)
        elif kind == REMINDER:
            self.call('Reminder, we still have a penalty to serve.', Priority.STRATEGY,
                      'penalty', 'penalty', cooldown=20)
        elif kind == DSQ:
            self.call("We've been disqualified. Sorry mate, box the car.", Priority.CRITICAL,
                      'penalty', 'penalty')
        elif kind == GRID:
            self.call(f'We have a grid penalty, for {reason}.', Priority.NORMAL,
                      'grid_penalty', 'penalty')
        elif kind == TYRE_REGS:
            self.call('Penalty for the tyre regulations.', Priority.STRATEGY, 'penalty', 'penalty')
        elif kind in (PARKED_TOO_LONG, BLACK_FLAG_TIMER):
            self.call('Get the car moving, the black flag timer is running.', Priority.CRITICAL,
                      'penalty', 'penalty', cooldown=10)
        elif kind in LAP_INVALIDATED:
            # Deleted laps only cost anything outside the race; there the driver needs to know.
            self.call(f'Lap deleted, {reason}.', Priority.TIMING, 'lap_deleted', 'lap_deleted',
                      ttl=6, cooldown=5, spoken=not self.is_race)

    def _penalty_served(self, event):
        if event.vehicle_index == self.player_index:
            self.call('Penalty served. Back to racing.', Priority.RACE, 'penalty_served', 'penalty')

    _event_dtsv = _event_sgsv = _penalty_served

    def _event_ftlp(self, event):
        if not self.is_race or not self._valid_car(event.vehicle_index):
            return
        if not (event.lap_time_s and math.isfinite(event.lap_time_s) and event.lap_time_s > 0):
            return
        time_text = lap_time_text(event.lap_time_s)
        if event.vehicle_index == self.player_index:
            self.call(f'Fastest lap, {time_text}. Purple, mega.', Priority.TIMING,
                      'fastest_lap', 'fastest_lap')
        else:
            self.call(f'{capitalise(self.name(event.vehicle_index))} sets the fastest lap, {time_text}.',
                      Priority.NORMAL, 'fastest_lap', 'fastest_lap')

    def _event_rtmt(self, event):
        if self._valid_car(event.vehicle_index):
            self._retirement(event.vehicle_index)

    def _retirement(self, index):
        if index in self.retired or index == self.player_index:
            self.retired.add(index)
            return
        self.retired.add(index)
        if not self.is_race:
            return
        mine, theirs = self.position_of(self.player_index), self.position_of(index)
        message = f'{capitalise(self.name(index))} is out of the race.'
        if mine and theirs and theirs < mine:
            message += ' That is a place for us.'
        self.call(message, Priority.RACE, f'retirement_{index}', 'retirement')

    def _event_tmpt(self, event):
        if self.is_race:
            self.call('Your team mate is in the pits.', Priority.NORMAL, 'teammate_pit', 'pit')

    def _event_drse(self, event):
        if self.is_race:
            self.call('DRS enabled.', Priority.RACE, 'drs', 'drs', ttl=6)

    def _event_drsd(self, event):
        if self.is_race:
            self.call('DRS disabled.', Priority.RACE, 'drs', 'drs', ttl=6)

    def _event_chqf(self, event):
        if self.is_race:
            self.call('Chequered flag is out. Bring it home.', Priority.RACE, 'chequered', 'finish')

    def _event_rcwn(self, event):
        if not self._valid_car(event.vehicle_index):
            return
        if event.vehicle_index == self.player_index:
            self.finish_reported = True
            self.call("Get in there! You've won the race! P1!", Priority.CRITICAL, 'race_winner', 'finish')
        else:
            self.call(f'{capitalise(self.name(event.vehicle_index))} wins the race.', Priority.NORMAL,
                      'race_winner', 'finish')

    def _event_lgot(self, event):
        if self.is_race:
            self.call('Lights out.', Priority.LOW, 'lights_out', 'start', spoken=False)

    # ──────────────────────────────────────────
    # Flags waved at the player (Car Status)
    # ──────────────────────────────────────────

    def update_status(self, statuses, player_index):
        self.statuses = statuses
        self.player_index = player_index
        flag = statuses[player_index].fia_flags
        previous, self.last_flag = self.last_flag, flag
        if previous is None or flag == previous or flag < 0:
            return
        if flag == YELLOW:
            self.call('Yellow flag, yellow flag. No overtaking.', Priority.CRITICAL,
                      'flag_yellow', 'flag', ttl=5, cooldown=8)
        elif flag == RED:
            self.call('Red flag, red flag. Slow down and bring it back to the pit lane.',
                      Priority.CRITICAL, 'flag_red', 'flag', ttl=10)
        elif flag == BLUE and self.is_race:
            self.call('Blue flags. Let the leaders through.', Priority.RACE,
                      'flag_blue', 'flag', ttl=6, cooldown=30)
        elif previous in (YELLOW, RED) and flag in (0, GREEN):
            self.call('Green flag, track is clear.', Priority.RACE, 'flag_green', 'flag',
                      ttl=5, cooldown=8)

    # ──────────────────────────────────────────
    # Field tracking (Lap Data for all cars)
    # ──────────────────────────────────────────

    def update_field(self, laps, player_index, session_time):
        if not math.isfinite(session_time):
            return
        self.laps, self.player_index, self.session_time = laps, player_index, session_time
        for index, lap in enumerate(laps):
            previous = self.result_status.get(index)
            self.result_status[index] = lap.result_status
            if previous == ACTIVE and lap.result_status in OUT_OF_RACE:
                self._retirement(index)
        if not self.is_race:
            return
        player = laps[player_index]
        self._track_pit_stops(laps, player_index)
        self._track_milestones(player)
        if player.result_status == ACTIVE and player.car_position:
            self._track_position(laps, player_index, player, session_time)
        self._track_leader(laps, player_index, session_time)

    def _track_pit_stops(self, laps, player_index):
        for index, lap in enumerate(laps):
            if lap.result_status != ACTIVE:
                self.pit_entries.pop(index, None)
                continue
            entry = self.pit_entries.get(index)
            if lap.pit_status:
                self.pitted_since_baseline.add(index)
                if entry is None:
                    self.pit_entries[index] = {'stops': lap.num_pit_stops, 'position': lap.car_position,
                                               'stationary_ms': lap.pit_stop_timer_ms,
                                               'nearby': self.is_nearby(index) or lap.car_position == 1}
                else:
                    entry['stationary_ms'] = max(entry['stationary_ms'], lap.pit_stop_timer_ms)
                continue
            if entry is None:
                continue
            del self.pit_entries[index]
            if lap.num_pit_stops <= entry['stops'] and not entry['stationary_ms']:
                continue                          # drive through or a pit lane pass, not a stop
            seconds = entry['stationary_ms'] / 1000
            stop = f'{seconds:.1f} second stop' if 0 < seconds < 60 else None
            compound = self.compound(index)
            if index == player_index:
                verdict = 'Good' if seconds < 3.5 else 'Slow'
                parts = [f'{verdict} stop, {seconds:.1f} seconds' if stop else 'Out of the pits']
                if compound:
                    parts.append(f'on the {compound}')
                message = ', '.join(parts) + f'. Rejoining P{lap.car_position}.'
                self.call(message, Priority.RACE, 'own_pit_stop', 'pit')
            elif entry['nearby']:
                parts = [f"{capitalise(self.name(index))} has pitted from P{entry['position']}"]
                if stop:
                    parts.append(stop)
                if compound:
                    parts.append(f'onto the {compound}')
                message = ', '.join(parts) + f'. Rejoins P{lap.car_position}.'
                self.call(message, Priority.RACE, f'rival_pit_{index}', 'pit', ttl=15)

    def _track_milestones(self, player):
        previous_lap, self.player_lap_num = self.player_lap_num, player.current_lap_num
        if (not self.lap_one_reported and player.result_status == ACTIVE and previous_lap == 1
                and player.current_lap_num == 2 and player.car_position):
            self.lap_one_reported = True
            message = f'End of lap one, P{player.car_position}.'
            change = player.grid_position - player.car_position if player.grid_position else 0
            if change:
                places = 'place' if abs(change) == 1 else 'places'
                message += f" {abs(change)} {places} {'gained' if change > 0 else 'lost'} from the grid."
            self.call(message, Priority.RACE, 'lap_one', 'position')
        if player.result_status == FINISHED and not self.finish_reported:
            self.finish_reported = True
            message = f"That's the flag. We finish P{player.car_position}."
            change = player.grid_position - player.car_position if player.grid_position else 0
            if change > 0:
                message += f" Up {change} from the grid, good job."
            self.call(message, Priority.RACE, 'race_finish', 'finish', ttl=20)

    def _track_position(self, laps, player_index, player, session_time):
        position = player.car_position
        if self.confirmed_position is None:
            self._confirm_position(laps, position)
            return
        if position == self.confirmed_position:
            self.candidate_position = None
            return
        if position != self.candidate_position:
            self.candidate_position, self.candidate_since = position, session_time
            return
        if session_time - self.candidate_since < POSITION_SETTLE_S:
            return
        old = self.confirmed_position
        baseline, pitted = self.baseline_positions, self.pitted_since_baseline
        self._confirm_position(laps, position)
        # Lap one is summed up at its end rather than called move by move.
        if player.current_lap_num < 2 or player_index in pitted:
            return
        if position < old:
            swapped = [i for i, before in baseline.items() if i != player_index and before < old
                       and laps[i].car_position > position]
        else:
            swapped = [i for i, before in baseline.items() if i != player_index and before > old
                       and 0 < laps[i].car_position < position]
        on_track = [i for i in swapped if i not in pitted and laps[i].result_status == ACTIVE]
        if not swapped or len(on_track) != len(swapped):
            # Places changed through pit stops or retirements, not wheel to wheel.
            self.call(f'P{position}.', Priority.TIMING, 'position', 'position', ttl=6, cooldown=5)
        elif position < old:
            who = self.name(on_track[0]) if len(on_track) == 1 else f'{len(on_track)} cars'
            self.call(f'Nice move on {who}. P{position}.', Priority.RACE, 'position', 'overtake',
                      ttl=6, cooldown=3)
        else:
            who = capitalise(self.name(on_track[0])) if len(on_track) == 1 else f'{len(on_track)} cars'
            verb = 'is' if len(on_track) == 1 else 'are'
            self.call(f'{who} {verb} through. P{position}.', Priority.RACE, 'position', 'overtake',
                      ttl=6, cooldown=3)

    def _confirm_position(self, laps, position):
        self.confirmed_position = position
        self.candidate_position = None
        self.baseline_positions = {i: lap.car_position for i, lap in enumerate(laps)
                                   if lap.result_status == ACTIVE and lap.car_position}
        self.pitted_since_baseline = {i for i, lap in enumerate(laps) if lap.pit_status}

    def _track_leader(self, laps, player_index, session_time):
        leader = next((i for i, lap in enumerate(laps)
                       if lap.car_position == 1 and lap.result_status == ACTIVE), None)
        if leader is None:
            return
        if self.leader is None:
            self.leader = leader
            return
        if leader == self.leader:
            self.leader_candidate = None
            return
        if leader != self.leader_candidate:
            self.leader_candidate, self.leader_since = leader, session_time
            return
        if session_time - self.leader_since < POSITION_SETTLE_S:
            return
        old, self.leader, self.leader_candidate = self.leader, leader, None
        if player_index in (old, leader) or laps[leader].current_lap_num < 2:
            return                                # the player's own moves are called elsewhere
        if laps[old].pit_status or laps[old].result_status != ACTIVE or old in self.pit_entries:
            return
        self.call(f'{capitalise(self.name(leader))} takes the lead from {self.name(old)}.',
                  Priority.NORMAL, 'lead_change', 'overtake', ttl=8)
