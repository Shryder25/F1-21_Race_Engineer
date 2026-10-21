# Channel data store and MoTeC i2 export

The channel store keeps every telemetry sample of the player's car as a row in a Parquet
file, with the lap number and lap distance on each row so laps can be overlaid and compared
point by point. The same file exports to MoTeC i2.

It sits next to the other stores and replaces none of them:

| Store | What it keeps | Used for |
|---|---|---|
| `.f1rec.gz` recording (`udp_recording.py`) | every raw datagram with its arrival time | exact replay, backtests, golden tests; the source of truth |
| `telemetry.sqlite3` (`history.py`) | one row per lap | the learning engine and ML models |
| `fitted_models.json` (`fitted_models.py`) | fitted tyre/fuel curves | predictions with ranges |
| `channels/*.parquet` (`channel_store.py`) | every sample, ~80 channels | lap overlays, driving analysis, MoTeC |

Code: `ai_engineer/channel_store.py` (logger, reader, lap alignment) and
`ai_engineer/motec.py` (MoTeC export). CLI: `tools/channels.py`. Needs `pyarrow`.

## Logging

```bash
python main.py --log-channels channels/                    # live, while you drive
python main.py --replay race.f1rec.gz --replay-speed max --log-channels channels/
python tools/channels.py build recordings/*.f1rec.gz --out channels/   # offline, no engineer
```

The logger reads raw datagrams itself, separately from the engineer's `TelemetryStream`, so
the two cannot break each other. Without pyarrow, `--log-channels` prints a warning and the
engineer runs as normal.

Each session (a new session UID) gets its own file, named
`<date>-<time>_<track>_<session>.parquet`. Laps are written as one Parquet row group each,
once two newer laps have started; the footer is written when the session ends or the
engineer shuts down (Ctrl+C is fine). A hard crash leaves the open file unreadable; rebuild
it from the recording with `tools/channels.py build` if you were recording.

### When a row is written

One row per game frame that has both a Lap Data (2) and a Car Telemetry (6) packet, so the
row rate is the game's UDP send rate (10 to 60 Hz). Motion (0), Car Status (7), Car Damage
(10) and Session (1) values are the latest received; Participants (4) gives the driver name
and team for the file's metadata.

Rows are skipped while the game is paused or you are spectating, and when the session time
does not move forward (duplicates, out-of-order packets).

**Flashbacks.** The rows the flashback undid are dropped from memory, and datagrams still in
flight from the undone timeline (more than a second ahead of the new one) are ignored until
the new timeline catches up. If a flashback reaches back further than the two laps held in
memory, the file ends up with overlapping times; `read_session` drops the overwritten rows
when reading (a row is kept only if every later row is later in session time). The number
of flashbacks is in the file metadata.

## File format

Parquet, zstd-compressed, one row per sample. Wheel channels end in `_rl`, `_rr`, `_fl`,
`_fr` (the game's order). Values are in the game's own units; each field's Parquet metadata
holds its `unit` and `kind` (`linear`, `step` for discrete values like gear, or `angle`).

| Group | Columns |
|---|---|
| Alignment / timing | `session_time` (s), `frame`, `lap`, `lap_distance` (m), `total_distance` (m), `lap_time` (s, current lap), `last_lap_time` (s), `sector` (0-2), `position`, `pit_status` (0 none, 1 pitting, 2 in pit area), `num_pit_stops`, `lap_invalid`, `driver_status` |
| Telemetry | `speed` (km/h), `throttle` (0-1), `brake` (0-1), `steer` (-1..1), `clutch` (%), `gear`, `rpm`, `drs`, `engine_temp` (C), `brake_temp_*` (C), `tyre_surface_temp_*` (C), `tyre_inner_temp_*` (C), `tyre_pressure_*` (psi), `surface_*` |
| Motion | `pos_x/y/z` (m), `g_lat/g_long/g_vert` (G), `yaw/pitch/roll` (rad), `yaw_rate` (rad/s), `front_wheels_angle` (rad), `susp_pos_*` (mm), `susp_vel_*` (mm/s), `wheel_speed_*` (m/s), `wheel_slip_*` |
| Car status | `fuel` (kg), `fuel_laps`, `ers_store` (J), `ers_mode`, `ers_deployed_lap` (J), `tyre_compound` (visual id: 16 soft, 17 medium, 18 hard, 7 inter, 8 wet), `tyre_age` (laps) |
| Damage | `tyre_wear_*` (%), `front_left_wing_damage`, `front_right_wing_damage`, `rear_wing_damage` (%) |
| Session | `track_temp`, `air_temp` (C), `weather` (0 clear ... 5 storm) |

Channels the game had not sent yet (for example damage before the first Car Damage packet)
are null.

File metadata (`f1_session` key, JSON): `track`, `track_id`, `track_length_m`, `session`,
`session_type`, `total_laps`, `driver`, `team`, `session_uid`, `game_version`,
`start_unix`, `rows`, `laps`, `flashbacks`, `channels` (name to unit), `store_version`.

### Reading it

```python
import channel_store
cols, meta = channel_store.read_session('channels/x.parquet')     # numpy arrays, NaN for nulls
laps = channel_store.lap_summary(cols)       # per lap: lap_time, samples, complete/invalid/pit/clean
grid, by_lap = channel_store.align_laps(cols, [4, 7], ['speed', 'throttle'], step_m=1.0)
```

Any Parquet reader works too, e.g. `pandas.read_parquet('channels/x.parquet')`; group by
`lap` and use `lap_distance` as the x axis. A lap is `clean` when it covers the whole lap,
has no track-limits flag and no pit-lane time. The lap time comes from `last_lap_time` on
the first sample of the next lap, so the game's own timing is used.

`tools/channels.py compare FILE 4 7 --channels speed brake --csv out.csv` writes the overlay
for a spreadsheet.

## MoTeC i2 export

```bash
python main.py --export-motec channels/x.parquet               # x.ld + x.ldx, then exit
python tools/channels.py motec channels/x.parquet --rate 60    # same, choose the rate
python tools/channels.py motec channels/x.parquet --format csv # MoTeC-style CSV instead
```

Open the `.ld` in MoTeC i2 (the free Standard edition is enough); keep the `.ldx` next to it,
because that is where the lap markers are. The export:

- resamples every channel onto a fixed rate (default: the nearest of the game's send rates
  10/15/20/30/60 Hz to what was received), because `.ld` channels are fixed-rate. Continuous
  channels are interpolated, discrete ones (gear, lap, sector, flags) hold their last value,
  yaw is interpolated the short way round;
- names channels the way MoTeC users expect (`Ground Speed`, `Throttle Pos`, `Brake Pos`,
  `Steered Angle`, `Engine RPM`, `Gear`, `Lap Distance`, `Tyre Temp Surface FL` ...) and
  converts throttle, brake and steering to %, angles to degrees, wheel speed to km/h and
  ERS energy to MJ;
- writes one lap beacon in the `.ldx` each time the lap number goes up, plus the total laps
  and the fastest valid lap;
- fills the header with the date and time, driver, team ("vehicle"), track ("venue") and
  session.

### Format notes and limitations

- MoTeC has not published the `.ld` format. The writer follows the community's
  reverse-engineered layout (the one ACC and other sim exporters use): a 1762-byte header,
  event / venue / vehicle blocks, 124-byte channel descriptors in a linked list, then the
  samples (float32, or int16 for discrete channels, unscaled). Our files were checked against
  an independent third-party `.ld` parser, which read back every channel exactly; they have
  not been opened in MoTeC i2 itself from this repo's CI (it is Windows-only software). If a
  future i2 version rejects a file, the CSV export is the fallback.
- The CSV uses the layout i2 itself exports (header block, channel names, units, data).
  It is handy for spreadsheets and other tools; how far i2 imports CSV depends on the i2
  version and licence.
- Without the `.ldx` file, i2 opens the data as a single lap.
- Lap distance in i2: the export includes the game's `Lap Distance` channel; i2's own
  distance maths can also be built from `Ground Speed`.
