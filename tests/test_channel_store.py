"""Channel store (Parquet, keyed by lap distance) and the MoTeC i2 export."""

import datetime
import pathlib
import struct
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ai_engineer'))
sys.path.insert(0, str(ROOT / 'tools'))

import motec  # noqa: E402
from packets import CAR_MOTION_SIZE, HEADER_SIZE, MOTION_EX_FORMAT, parse_player_motion_extra  # noqa: E402
from simulate_telemetry import (  # noqa: E402
    build_car_damage_packet,
    build_car_status_packet,
    build_header,
    build_lap_packet,
    build_participants_packet,
    build_session_packet,
    build_telemetry_packet,
)

try:
    import pyarrow  # noqa: F401

    HAVE_PYARROW = True
except ImportError:  # CI installs it; a local checkout may not have it yet
    HAVE_PYARROW = False

if HAVE_PYARROW:
    import channel_store
    from channel_store import ChannelLogger, align_laps, lap_summary, read_session

GOLDEN = ROOT / 'tests' / 'golden' / 'sessions' / 'race_dry_red_bull_ring' / 'session.f1rec.gz'
needs_pyarrow = unittest.skipUnless(HAVE_PYARROW, 'pyarrow not installed (pip install pyarrow)')


def with_uid(packet, uid):
    return packet[:6] + struct.pack('<Q', uid) + packet[14:]


def motion_packet(t, frame, susp=(1.0, 2.0, 3.0, 4.0), wheel_speed=(50.0, 50.0, 51.0, 51.0)):
    header = build_header(0, t, frame)
    player = struct.pack(
        '<ffffffhhhhhhffffff', 10.0, 1.0, -20.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.5, -0.5, 1.0, 0.25, 0.01, 0.02
    )
    cars = player + bytes(CAR_MOTION_SIZE * 21)
    extra = struct.pack(
        MOTION_EX_FORMAT,
        *susp,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        *wheel_speed,
        0.01,
        0.01,
        0.02,
        0.02,
        0.3,
        0,
        0,
        0,
        0.5,
        0,
        0,
        0,
        0,
        0.1,
    )
    return header + cars + extra


def flashback_packet(t, frame, restored_frame, restored_time):
    body = build_header(3, t, frame) + b'FLBK' + struct.pack('<If', restored_frame, restored_time)
    return body + bytes(36 - len(body))


def sample(t, frame, dist, lap=1, speed=200, lap_ms=None):
    lap_ms = int(t * 1000) if lap_ms is None else lap_ms
    return [
        build_lap_packet(t, frame, dist, lap_ms, lap_num=lap),
        build_telemetry_packet(t, frame, speed, 0.8, 0.0, 6, 95),
    ]


def drive(logger, packets):
    return sum(bool(logger.handle(p, wall_time=1_790_000_000.0)) for p in packets)


class MotionExtraTest(unittest.TestCase):
    def test_player_only_block_is_parsed_from_the_end_of_the_motion_packet(self):
        data = motion_packet(1.0, 1)
        self.assertEqual(len(data), 1464)
        self.assertEqual(HEADER_SIZE + 22 * CAR_MOTION_SIZE + struct.calcsize(MOTION_EX_FORMAT), 1464)
        ex = parse_player_motion_extra(data)
        self.assertEqual(ex.suspension_position, (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(ex.wheel_speed, (50.0, 50.0, 51.0, 51.0))
        self.assertAlmostEqual(ex.angular_velocity[1], 0.5)
        self.assertAlmostEqual(ex.local_velocity[0], 0.3, places=5)
        self.assertAlmostEqual(ex.front_wheels_angle, 0.1, places=5)


@needs_pyarrow
class ChannelLoggerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def logger(self):
        return ChannelLogger(self.dir, clock=lambda: 1_790_000_000.0)

    def test_paired_lap_and_telemetry_frames_become_rows_with_every_source(self):
        log = self.logger()
        packets = [
            build_session_packet(0.0, 1, track_id=7, track_length=5891),
            build_participants_packet(0.0, 1),
            build_car_status_packet(0.0, 1, fuel=50.0),
            build_car_damage_packet(0.0, 1, wear=(1.0, 2.0, 3.0, 4.0)),
            motion_packet(0.0, 1),
        ]
        for i in range(10):
            packets += sample(0.05 * (i + 1), i + 2, 100.0 + 10 * i)
        self.assertEqual(drive(log, packets), 10)
        log.close()
        self.assertEqual(len(log.files), 1)
        self.assertIn('Silverstone_Race', log.files[0].name)
        cols, meta = read_session(log.files[0])
        np.testing.assert_allclose(cols['lap_distance'], 100.0 + 10 * np.arange(10))
        self.assertTrue(np.all(cols['speed'] == 200))
        self.assertAlmostEqual(float(cols['throttle'][0]), 0.8, places=5)
        self.assertTrue(np.all(cols['fuel'] == 50.0))
        self.assertEqual(cols['tyre_wear_rl'][0], 1.0)
        self.assertEqual(cols['tyre_wear_fr'][0], 4.0)
        self.assertEqual(cols['susp_pos_fl'][0], 3.0)
        self.assertAlmostEqual(float(cols['g_lat'][0]), 1.5)
        self.assertAlmostEqual(float(cols['yaw_rate'][0]), 0.5)
        self.assertEqual(cols['track_temp'][0], 35)
        self.assertEqual(meta['track'], 'Silverstone')
        self.assertEqual(meta['session'], 'Race')
        self.assertEqual(meta['driver'], 'Player')
        self.assertEqual(meta['rows'], 10)
        self.assertEqual(meta['start_unix'], 1_790_000_000.0)

    def test_telemetry_without_a_lap_packet_for_the_same_frame_is_not_a_row(self):
        log = self.logger()
        packets = [
            build_session_packet(0.0, 1),
            build_lap_packet(0.1, 2, 100.0, 100),
            build_telemetry_packet(0.1, 3, 200, 1, 0, 6, 90),
        ]
        self.assertEqual(drive(log, packets), 0)
        log.close()
        self.assertEqual(log.files, [])  # nothing logged, no empty file left behind

    def test_nothing_is_logged_before_a_session_packet_or_while_paused(self):
        log = self.logger()
        self.assertEqual(drive(log, sample(0.1, 2, 100.0)), 0)
        paused = bytearray(build_session_packet(0.0, 1))
        paused[HEADER_SIZE + 14] = 1  # gamePaused
        self.assertEqual(drive(log, [bytes(paused)] + sample(0.2, 3, 110.0)), 0)
        self.assertEqual(drive(log, [build_session_packet(0.3, 4)] + sample(0.4, 5, 120.0)), 1)
        log.close()

    def test_repeated_session_time_is_dropped(self):
        log = self.logger()
        self.assertEqual(
            drive(
                log,
                [build_session_packet(0.0, 1)]
                + sample(1.0, 2, 100.0)
                + sample(1.0, 3, 100.0)
                + sample(0.9, 4, 99.0),
            ),
            1,
        )
        log.close()

    def test_garbage_datagrams_are_ignored_without_raising(self):
        log = self.logger()
        junk = [b'', b'x' * 10, b'\x00' * 24, bytes(1347), build_session_packet(0.0, 1)[:100]]
        self.assertEqual(drive(log, junk), 0)
        log.close()

    def test_a_new_session_uid_closes_the_file_and_starts_another(self):
        log = self.logger()
        first = [build_session_packet(0.0, 1)] + sample(0.1, 2, 10.0) + sample(0.2, 3, 20.0)
        second = [with_uid(p, 99) for p in [build_session_packet(0.0, 1, track_id=11)] + sample(0.1, 2, 30.0)]
        drive(log, first + second)
        self.assertEqual(len(log.files), 2)
        log.close()
        rows = [len(read_session(f)[0]['session_time']) for f in log.files]
        self.assertEqual(rows, [2, 1])
        self.assertIn('Monza', log.files[1].name)

    def test_laps_are_written_as_row_groups_once_two_newer_laps_exist(self):
        import pyarrow.parquet as pq

        log = self.logger()
        packets = [build_session_packet(0.0, 1)]
        frame, t = 2, 0.0
        for lap in (1, 2, 3, 4):
            for d in (10.0, 2000.0, 5700.0):
                t += 1.0
                packets += sample(t, frame, d, lap=lap)
                frame += 1
        drive(log, packets)
        self.assertEqual(log.rows_written, 6)  # laps 1 and 2 on disk, 3 and 4 still in memory
        log.close()
        self.assertEqual(pq.ParquetFile(log.files[0]).metadata.num_row_groups, 4)

    def test_flashback_drops_undone_samples_and_stale_old_timeline_datagrams(self):
        log = self.logger()
        packets = [build_session_packet(0.0, 1)]
        for i in range(10):  # t = 1..10 s, 100 m per second
            packets += sample(1.0 + i, 10 + i, 100.0 * (i + 1))
        packets.append(flashback_packet(10.1, 30, 15, 5.5))  # rewind to t=5.5
        packets += sample(9.5, 19, 950.0)  # in flight from the old timeline
        packets += sample(6.0, 31, 600.0)  # new timeline
        packets += sample(10.5, 20, 1050.0)  # another stale one
        packets += sample(6.5, 32, 650.0)
        drive(log, packets)
        log.close()
        cols, meta = read_session(log.files[0])
        np.testing.assert_allclose(cols['session_time'], [1, 2, 3, 4, 5, 6.0, 6.5])
        np.testing.assert_allclose(cols['lap_distance'], [100, 200, 300, 400, 500, 600, 650])
        self.assertEqual(meta['flashbacks'], 1)

    def test_rows_overwritten_after_they_reached_the_file_are_dropped_on_read(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.dir / 'overlap.parquet'
        schema = channel_store.arrow_schema()
        rows = [
            {'session_time': t, 'lap': 1, 'lap_distance': d}
            for t, d in [(1, 10), (2, 20), (3, 30), (2.5, 25), (4, 40)]
        ]
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)
        cols, _ = read_session(path)
        np.testing.assert_allclose(cols['session_time'], [1, 2, 2.5, 4])
        self.assertTrue(np.isnan(cols['speed']).all())  # never-sent channels are NaN

    def test_out_of_range_values_never_break_the_writer(self):
        row = channel_store._clean({'gear': 300, 'speed': float('nan'), 'rpm': 99999, 'lap': None})
        self.assertEqual(row, {'gear': 127, 'speed': None, 'rpm': 32767, 'lap': None})

    def test_every_channel_has_a_known_arrow_type_and_unique_name(self):
        names = [c.name for c in channel_store.CHANNELS]
        self.assertEqual(len(names), len(set(names)))
        schema = channel_store.arrow_schema()
        self.assertEqual(schema.field('speed').metadata[b'unit'], b'km/h')
        motec_names = [c.motec[0] for c in channel_store.CHANNELS if c.motec]
        self.assertEqual(len(motec_names), len(set(motec_names)))
        self.assertTrue(all(len(n) <= 32 for n in motec_names))
        self.assertTrue(all(len(c.motec[1]) <= 12 for c in channel_store.CHANNELS if c.motec))


@needs_pyarrow
class FindSessionFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, name, mtime):
        import os

        path = self.dir / name
        log = ChannelLogger(self.dir, clock=lambda: 1_790_000_000.0)
        drive(log, [build_session_packet(0.0, 1)] + sample(0.1, 2, 10.0) + sample(0.2, 3, 20.0))
        log.close()
        log.files[0].rename(path)
        os.utime(path, (mtime, mtime))
        return path

    def test_a_folder_means_its_newest_file(self):
        self.make('old.parquet', 1000)
        new = self.make('new.parquet', 2000)
        self.assertEqual(channel_store.find_session_file(self.dir), new)
        cols, _ = read_session(self.dir)
        self.assertEqual(len(cols['session_time']), 2)

    def test_the_parquet_extension_is_optional(self):
        path = self.make('race.parquet', 1000)
        self.assertEqual(channel_store.find_session_file(self.dir / 'race'), path)

    def test_a_wrong_name_lists_the_files_that_are_there(self):
        self.make('race.parquet', 1000)
        with self.assertRaises(ValueError) as caught:
            channel_store.find_session_file(self.dir / 'THE-FILE-NAME.parquet')
        self.assertIn('race.parquet', str(caught.exception))

    def test_an_empty_or_missing_folder_says_so(self):
        with self.assertRaisesRegex(ValueError, 'No .parquet files'):
            channel_store.find_session_file(self.dir)
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            channel_store.find_session_file(self.dir / 'nope' / 'x.parquet')
        with self.assertRaisesRegex(ValueError, 'no folder called'):
            channel_store.find_session_file(self.dir / 'channels')

    def test_an_unfinished_file_gets_a_plain_explanation(self):
        path = self.make('race.parquet', 1000)
        path.write_bytes(path.read_bytes()[:-8])
        with self.assertRaisesRegex(ValueError, 'never finished'):
            read_session(path)
        with self.assertRaisesRegex(ValueError, 'never finished'):
            motec.export_session(path)


@needs_pyarrow
class LapAlignmentTest(unittest.TestCase):
    def cols(self):
        # Two laps of a 1000 m track; lap 2 is 10% faster everywhere.
        t, lap, dist, speed, gear, last = [], [], [], [], [], []
        clock = 0.0
        for n, pace in ((1, 1.0), (2, 1.1)):
            for d in np.arange(0, 1000, 7.0):
                t.append(clock)
                lap.append(n)
                dist.append(d)
                speed.append(100 * pace + d / 100)
                gear.append(3 if d < 500 else 6)
                last.append(np.nan if n == 1 else 50.0)
                clock += 0.1
        t.append(clock)
        lap.append(3)
        dist.append(1.0)
        speed.append(0)
        gear.append(1)
        last.append(45.5)
        f = lambda v: np.array(v, dtype=np.float32)  # noqa: E731
        return {
            'session_time': np.array(t),
            'lap': f(lap),
            'lap_distance': f(dist),
            'speed': f(speed),
            'gear': f(gear),
            'last_lap_time': f(last),
            'pit_status': f([0] * len(t)),
            'lap_invalid': f([0] * len(t)),
        }

    def test_laps_line_up_on_a_common_distance_grid(self):
        grid, laps = align_laps(self.cols(), [1, 2], ['speed', 'gear'], step_m=10.0)
        self.assertEqual(grid[0], 0.0)
        self.assertEqual(grid[1] - grid[0], 10.0)
        i = int(np.flatnonzero(grid == 350.0)[0])
        self.assertAlmostEqual(laps[1]['speed'][i], 103.5, places=3)
        self.assertAlmostEqual(laps[2]['speed'][i], 113.5, places=3)
        self.assertEqual(laps[1]['gear'][np.flatnonzero(grid == 500.0)[0]], 3)  # held, not blended
        self.assertEqual(laps[1]['gear'][np.flatnonzero(grid == 510.0)[0]], 6)

    def test_lap_summary_takes_lap_time_from_the_next_lap_and_flags_partial_laps(self):
        summary = {s['lap']: s for s in lap_summary(self.cols())}
        self.assertEqual(summary[1]['lap_time'], 50.0)
        self.assertEqual(summary[2]['lap_time'], 45.5)
        self.assertTrue(summary[1]['clean'] and summary[2]['clean'])
        self.assertFalse(summary[3]['complete'])

    def test_angles_interpolate_the_short_way_round(self):
        x = np.array([0.0, 1.0])
        y = np.array([3.1, -3.1])  # just either side of +/-pi
        mid = channel_store.resample(x, y, np.array([0.5]), 'angle')[0]
        self.assertGreater(abs(mid), 3.1)


class LdFormatTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / 'test.ld'

    def tearDown(self):
        self.tmp.cleanup()

    def test_block_sizes_match_the_known_layout(self):
        self.assertEqual(motec.HEADER_SIZE, 0x6E2)
        self.assertEqual(motec.EVENT_SIZE, 1154)
        self.assertEqual(motec.VENUE_SIZE, 1100)
        self.assertEqual(motec.VEHICLE_SIZE, 260)
        self.assertEqual(motec.CHANNEL_SIZE, 124)

    def test_write_then_read_round_trips_channels_and_header(self):
        speed = np.linspace(0, 300, 50)
        gear = np.array([1, 2, 3, 4, 5] * 10, dtype=float)
        start = datetime.datetime(2026, 10, 21, 14, 5, 9)
        motec.write_ld(
            self.path,
            [
                motec.LdChannel('Ground Speed', 'Speed', 'km/h', 20, speed, 'float32'),
                motec.LdChannel('Gear', 'Gear', '', 20, gear, 'int16'),
            ],
            driver='Anvit',
            vehicle='Mercedes F1 2021',
            venue='Silverstone',
            event='F1 2021 Race',
            session='Race',
            comment='hello',
            short_comment='abc',
            start=start,
            vehicle_type='F1 2021',
        )
        head, channels = motec.read_ld(self.path)
        self.assertEqual(head['marker'], 0x40)
        self.assertEqual((head['date'], head['time']), ('21/10/2026', '14:05:09'))
        self.assertEqual(
            (head['driver'], head['vehicle'], head['venue']), ('Anvit', 'Mercedes F1 2021', 'Silverstone')
        )
        self.assertEqual((head['event'], head['session'], head['comment']), ('F1 2021 Race', 'Race', 'hello'))
        self.assertEqual(head['venue_block'], 'Silverstone')
        self.assertEqual(head['vehicle_type'], 'F1 2021')
        self.assertEqual([c.name for c in channels], ['Ground Speed', 'Gear'])
        self.assertEqual([c.freq for c in channels], [20, 20])
        np.testing.assert_allclose(channels[0].data, speed.astype(np.float32))
        np.testing.assert_array_equal(channels[1].data, gear)
        self.assertEqual(
            self.path.stat().st_size,
            motec.HEADER_SIZE
            + motec.EVENT_SIZE
            + motec.VENUE_SIZE
            + motec.VEHICLE_SIZE
            + 2 * motec.CHANNEL_SIZE
            + 50 * 4
            + 50 * 2,
        )

    def test_channel_metadata_is_a_doubly_linked_list_with_contiguous_data(self):
        chans = [motec.LdChannel(f'Ch {i}', f'C{i}', '', 10, np.arange(5.0), 'float32') for i in range(3)]
        motec.write_ld(self.path, chans)
        blob = self.path.read_bytes()
        meta_ptr, data_ptr = struct.unpack_from('<II', blob, 8)
        entries = [struct.unpack_from('<IIII', blob, meta_ptr + i * motec.CHANNEL_SIZE) for i in range(3)]
        self.assertEqual(entries[0][0], 0)
        self.assertEqual(entries[0][1], meta_ptr + motec.CHANNEL_SIZE)
        self.assertEqual(entries[1][0], meta_ptr)
        self.assertEqual(entries[2][1], 0)
        self.assertEqual([e[2] for e in entries], [data_ptr, data_ptr + 20, data_ptr + 40])
        self.assertEqual(struct.unpack_from('<I', blob, 86)[0], 3)  # channel count

    def test_integer_channels_are_rounded_and_clipped(self):
        motec.write_ld(
            self.path, [motec.LdChannel('X', 'X', '', 1, np.array([1.6, 1e9, -1e9, np.nan]), 'int16')]
        )
        _, (ch,) = motec.read_ld(self.path)
        np.testing.assert_array_equal(ch.data, [2, 32767, -32768, 0])

    def test_long_text_is_truncated_to_its_field(self):
        motec.write_ld(
            self.path,
            [motec.LdChannel('N' * 40, 'S' * 20, 'U' * 20, 1, np.zeros(2), 'float32')],
            driver='D' * 100,
        )
        head, (ch,) = motec.read_ld(self.path)
        self.assertEqual((len(ch.name), len(ch.short_name), len(ch.unit)), (32, 8, 12))
        self.assertEqual(len(head['driver']), 64)

    def test_short_names_are_unique_and_fit_eight_characters(self):
        taken = set()
        for name in ['Tyre Temp Surface FL', 'Tyre Temp Surface FR', 'Tyre Temp Inner FL', 'Gear', 'Gear']:
            sn = motec.short_name(name, taken)
            self.assertLessEqual(len(sn), 8)
            self.assertNotIn(sn, taken)
            taken.add(sn)

    def test_ldx_has_one_beacon_per_lap_line_in_microseconds(self):
        ldx = self.path.with_suffix('.ldx')
        motec.write_ldx(ldx, [65.25, 130.5], {'Total Laps': 3, 'Fastest Time': '1:05.250'})
        root = ET.parse(ldx).getroot()
        markers = root.findall('.//MarkerGroup[@Name="Beacons"]/Marker')
        self.assertEqual([float(m.get('Time')) for m in markers], [65.25e6, 130.5e6])
        details = {s.get('Id'): s.get('Value') for s in root.findall('.//Details/String')}
        self.assertEqual(details, {'Total Laps': '3', 'Fastest Time': '1:05.250'})

    def test_lap_time_format(self):
        self.assertEqual(motec.format_lap_time(65.25), '1:05.250')
        self.assertEqual(motec.format_lap_time(59.9994), '0:59.999')

    def test_rate_snaps_to_the_games_send_rates(self):
        self.assertEqual(motec.pick_rate(np.arange(0, 10, 1 / 60)), 60)
        self.assertEqual(motec.pick_rate(np.arange(0, 10, 1 / 19.0)), 20)
        self.assertEqual(motec.pick_rate(np.arange(0, 10, 0.2)), 10)


@needs_pyarrow
class RecordedReplayExportTest(unittest.TestCase):
    """A golden recording through the whole path: UDP datagrams -> Parquet -> MoTeC."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = pathlib.Path(cls.tmp.name)
        cls.files = channel_store.build_from_recording(GOLDEN, cls.dir)
        cls.cols, cls.meta = read_session(cls.files[0])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_session_file_with_every_lap(self):
        self.assertEqual(len(self.files), 1)
        self.assertEqual(self.meta['track'], 'Red Bull Ring')
        laps = lap_summary(self.cols)
        self.assertEqual([lap['lap'] for lap in laps], list(range(1, 9)))
        self.assertTrue(all(np.diff(self.cols['session_time']) > 0))
        self.assertTrue(any(lap['pit'] for lap in laps))
        clean = [lap for lap in laps if lap['clean']]
        self.assertGreaterEqual(len(clean), 4)
        for lap in clean:
            self.assertTrue(60 < lap['lap_time'] < 80, lap)

    def test_ld_export_matches_the_store(self):
        out = self.dir / 'race.ld'
        result = motec.export_session(self.files[0], out, rate_hz=20)
        head, channels = motec.read_ld(out)
        by_name = {c.name: c for c in channels}
        self.assertEqual(result['channels'], len(channels))
        self.assertEqual(head['venue'], 'Red Bull Ring')
        self.assertTrue(all(c.freq == 20 for c in channels))
        self.assertEqual(len({len(c.data) for c in channels}), 1)
        duration = self.cols['session_time'][-1] - self.cols['session_time'][0]
        self.assertAlmostEqual(len(channels[0].data), duration * 20 + 1, delta=1)
        # A raw sample lands exactly on the grid every 0.2 s: values must agree there.
        t0 = self.cols['session_time'][0]
        for raw_i in (50, 700, 1500):
            grid_i = int(round((self.cols['session_time'][raw_i] - t0) * 20))
            self.assertAlmostEqual(by_name['Ground Speed'].data[grid_i], self.cols['speed'][raw_i], places=2)
            self.assertAlmostEqual(
                by_name['Throttle Pos'].data[grid_i], 100 * self.cols['throttle'][raw_i], places=2
            )
            self.assertEqual(by_name['Lap Number'].data[grid_i], self.cols['lap'][raw_i])
        self.assertTrue(set(np.unique(by_name['Gear'].data)) <= set(range(-1, 9)))
        root = ET.parse(result['ldx']).getroot()
        self.assertEqual(len(root.findall('.//Marker')), 7)  # 8 laps -> 7 line crossings

    def test_csv_export_has_the_motec_header_block(self):
        out = self.dir / 'race.csv'
        motec.export_session(self.files[0], out, fmt='csv')
        lines = out.read_text().splitlines()
        self.assertEqual(lines[0], '"Format","MoTeC CSV File"')
        self.assertTrue(lines[1].startswith('"Venue","Red Bull Ring"'))
        names = lines[14].split(',')
        units = lines[15].split(',')
        self.assertEqual(names[0], '"Time"')
        self.assertIn('"Ground Speed"', names)
        self.assertEqual(len(names), len(units))
        first = lines[18].split(',')
        self.assertEqual(len(first), len(names))
        self.assertEqual(first[0], '"0"')

    def test_tools_cli(self):
        import channels as cli

        self.assertEqual(cli.main(['laps', str(self.files[0])]), 0)
        out = self.dir / 'cmp.csv'
        self.assertEqual(
            cli.main(['compare', str(self.files[0]), '2', '3', '--channels', 'speed', '--csv', str(out)]), 0
        )
        self.assertEqual(out.read_text().splitlines()[0], 'lap_distance,speed_lap2,speed_lap3')
        self.assertEqual(cli.main(['motec', str(self.files[0]), '--out', str(self.dir / 'cli.ld')]), 0)
        self.assertTrue((self.dir / 'cli.ldx').exists())
        self.assertEqual(cli.main(['compare', str(self.files[0]), '2', '--channels', 'nope']), 2)


if __name__ == '__main__':
    unittest.main()
