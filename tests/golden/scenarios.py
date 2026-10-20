"""The synthetic golden sessions: one per kind of session the engineer has to handle.

Each scenario is written so the engineer has specific things to react to. The comments
list what each one is meant to exercise; expected.json records what the engineer
actually did, and the regression test fails if that changes.
"""

from synth import Car, Scenario

FIELD = ["Player", "HAMILTON", "VERSTAPPEN", "BOTTAS", "PEREZ", "NORRIS", "LECLERC", "SAINZ",
         "RICCIARDO", "GASLY", "ALONSO", "OCON", "VETTEL", "STROLL", "TSUNODA", "RAIKKONEN",
         "GIOVINAZZI", "RUSSELL", "LATIFI", "SCHUMACHER"]
NUMBERS = [99, 44, 33, 77, 11, 4, 16, 55, 3, 10, 14, 31, 5, 18, 22, 7, 99, 63, 6, 47]


def _backmarkers(first_index, first_grid, pits=None, compound='medium'):
    """The rest of the 20-car field: steadily slower cars further down the grid."""
    cars = []
    for index in range(first_index, len(FIELD)):
        grid = first_grid + index - first_index
        cars.append(Car(FIELD[index], pace=0.8 + 0.4 * (index - first_index), compound=compound,
                        grid=grid, pits=dict(pits or {}), race_number=NUMBERS[index]))
    return cars


def dry_race() -> Scenario:
    """Red Bull Ring, 7 laps, dry.

    Exercises: lights out and lap-one summary, DRS enabled, the car ahead (Perez, on worn
    softs) pitting early -> rival-pit reaction and undercut maths, Norris closing from behind
    -> DRS-range call and being passed, a rival time penalty, fastest-lap events, the game's
    pit window (laps 3-5) calls, the player's own stop, race-pace summaries, chequered flag,
    race winner and finishing position.
    """
    cars = [
        Car('Player', pace=0.30, compound='medium', grid=5, pits={4: 'hard'}, race_number=NUMBERS[0]),
        Car('HAMILTON', pace=0.00, compound='medium', grid=1, pits={4: 'hard'}, race_number=44),
        Car('VERSTAPPEN', pace=-0.05, compound='soft', grid=2, pits={3: 'medium'}, race_number=33),
        Car('BOTTAS', pace=0.35, compound='medium', grid=3, pits={5: 'hard'}, race_number=77),
        Car('PEREZ', pace=0.75, compound='soft', grid=4, pits={3: 'medium'}, race_number=11),
        Car('NORRIS', pace=0.05, compound='medium', grid=6, pits={5: 'hard'}, race_number=4),
    ] + _backmarkers(6, 7, pits={4: 'hard'})
    return Scenario(
        name='race_dry_red_bull_ring', track_id=17, track_length=4318, session_type=10,
        total_laps=7, cars=cars, weather=0, track_temp=38, air_temp=26, pit_window=(3, 5),
        fuel_kg=16.0,
        script=[
            (2.0, 'drs', True),
            (2.3, 'penalty', 4, 27, 4, 255, 5),       # Perez: 5 s for gaining time off track
        ],
        description='Dry race at the Red Bull Ring: undercut, DRS fights, pit window, finish.')


def qualifying() -> Scenario:
    """Silverstone short qualifying: out lap, three flying laps, in lap.

    Exercises: session welcome, the hand-measured Silverstone corner zones and braking
    callouts, sector times against personal bests, a personal-best lap then a slower one,
    a lap invalidated for corner cutting, a sustained hot front-tyre warning, speed traps,
    and that race-only features (gaps, pit window, strategy) stay quiet.
    """
    cars = [Car('Player', pace=8.0, compound='soft', grid=1, race_number=NUMBERS[0])]
    cars += [Car(FIELD[i], pace=0.2 + 0.1 * i, compound='soft', grid=i + 1, race_number=NUMBERS[i])
             for i in range(1, len(FIELD))]
    return Scenario(
        name='qualifying_silverstone', track_id=7, track_length=5891, session_type=8,
        total_laps=0, cars=cars, weather=1, track_temp=31, air_temp=21, fuel_kg=8.0,
        end_after_laps=4.5,
        script=[
            (0.0, "pace", 0, 5.0),       # out lap
            (1.0, 'pace', 0, 0.45),      # first flying lap
            (2.0, 'pace', 0, 0.10),      # personal best
            (2.30, 'hot_tyres', 107, 96),
            (2.42, 'cool_tyres'),
            (3.0, 'pace', 0, 0.20),      # third lap, slower, then invalidated
            (3.40, 'invalidate'),
            (4.0, 'pace', 0, 6.0),       # in lap
        ],
        description='Short qualifying at Silverstone: PB, slower lap, invalid lap, hot fronts.')


def wet_race() -> Scenario:
    """Hungaroring, 6 laps: dry start, light rain on lap 2, heavy rain from lap 4.

    Exercises: the rain-arriving call, how a change of weather group mid-session is handled,
    crossover pit stops onto intermediates (player and rivals), a rival staying out on
    slicks and falling back, wet tyre temperatures and pace changes.
    """
    cars = [
        Car('Player', pace=0.40, compound='medium', grid=4, pits={3: 'inter'}, race_number=NUMBERS[0]),
        Car('HAMILTON', pace=0.00, compound='medium', grid=1, pits={2: 'inter'}, race_number=44),
        Car('VERSTAPPEN', pace=0.05, compound='medium', grid=2, pits={2: 'inter'}, race_number=33),
        Car('BOTTAS', pace=0.30, compound='soft', grid=3, pits={}, race_number=77),   # gambles on slicks
        Car('PEREZ', pace=0.55, compound='medium', grid=5, pits={3: 'inter'}, race_number=11),
        Car('NORRIS', pace=0.45, compound='medium', grid=6, pits={2: 'inter'}, race_number=4),
    ] + _backmarkers(6, 7, pits={3: 'inter'})
    return Scenario(
        name='race_wet_hungaroring', track_id=9, track_length=4381, session_type=10,
        total_laps=6, cars=cars, weather=2, track_temp=24, air_temp=19, fuel_kg=14.0,
        script=[
            (1.50, 'weather', 3),
            (3.60, 'weather', 4),
        ],
        description='Race at the Hungaroring that turns wet: rain call, crossover to inters.')


def safety_car_race() -> Scenario:
    """Zandvoort, 7 laps: crash, yellow flag, Safety Car, cheap stops, restart, later a VSC.

    Exercises: rival retirement, yellow then green flag, Safety Car deployed and in, DRS
    disabled/enabled, front-wing damage and the box call, the player's stop under the
    Safety Car, the pack bunching (gaps collapse), a VSC period, a warning to the player,
    the pit window and the finish.
    """
    cars = [
        Car('Player', pace=0.20, compound='medium', grid=6, pits={3: 'hard'}, race_number=NUMBERS[0]),
        Car('HAMILTON', pace=0.00, compound='medium', grid=1, pits={3: 'hard'}, race_number=44),
        Car('VERSTAPPEN', pace=-0.10, compound='medium', grid=2, pits={5: 'hard'}, race_number=33),
        Car('BOTTAS', pace=0.30, compound='soft', grid=3, pits={3: 'medium'}, race_number=77),
        Car('PEREZ', pace=0.40, compound='medium', grid=4, pits={4: 'hard'}, race_number=11),
        Car('NORRIS', pace=0.25, compound='medium', grid=5, pits={3: 'hard'}, race_number=4),
        Car('LECLERC', pace=0.35, compound='medium', grid=7, pits={4: 'hard'}, race_number=16),
    ] + _backmarkers(7, 8, pits={3: 'hard'})
    return Scenario(
        name='race_safety_car_zandvoort', track_id=26, track_length=4259, session_type=10,
        total_laps=7, cars=cars, weather=0, track_temp=30, air_temp=22, pit_window=(4, 6),
        fuel_kg=16.0,
        script=[
            (1.85, 'retire', 9),         # Gasly crashes
            (1.85, 'flag', 3),           # yellow for the player
            (1.92, 'wing_damage', 55),   # debris
            (2.05, 'safety_car', 1),     # player and most of the field box under it at the end of lap 3
            (3.60, 'safety_car', 0),
            (3.65, 'flag', 1),
            (4.50, 'penalty', 5, 7, 0),  # warning for corner cutting
            (5.20, 'safety_car', 2),     # VSC
            (5.45, 'safety_car', 0),
        ],
        description='Race at Zandvoort with a crash, Safety Car, pit stop under SC and a VSC.')


SCENARIOS = {s.name: s for s in (dry_race(), qualifying(), wet_race(), safety_car_race())}
