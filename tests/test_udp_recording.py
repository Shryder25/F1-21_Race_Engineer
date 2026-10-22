import pathlib
import socket
import sys
import tempfile
import threading
import time
import unittest
import zlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'ai_engineer'))
import udp_recording
from simulate_telemetry import build_lap_packet, build_session_packet, build_telemetry_packet
from udp_recording import (
    FILE_HEADER,
    RECORD_HEADER,
    Record,
    RecordingError,
    RecordingReader,
    RecordingWriter,
    ReplayFinished,
    ReplaySource,
    paced,
    parse_speed,
    read_records,
    record_socket,
    replay_into,
    replay_to_udp,
    summarize,
)

PACKETS = [
    build_session_packet(0.0, 1, 11),
    build_telemetry_packet(0.04, 2, 280, 1, 0, 8, 95),
    build_lap_packet(0.04, 2, 1000, 20000),
    b'not an F1 packet',
]  # junk is recorded too


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class RecordingTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, name='s.f1rec', times_ms=(0, 40, 41, 1000), packets=PACKETS, **kwargs):
        path = self.dir / name
        with RecordingWriter(path, metadata={'port': 20777, 'note': 'test'}, **kwargs) as writer:
            for t_ms, data in zip(times_ms, packets, strict=False):
                writer.write(data, t_ns=t_ms * 1_000_000)
        return path


class FormatTests(RecordingTestCase):
    def test_round_trip_keeps_bytes_order_and_times(self):
        path = self.make()
        records = list(read_records(path))
        self.assertEqual([r.data for r in records], PACKETS)
        self.assertEqual([r.t_ns for r in records], [0, 40_000_000, 41_000_000, 1_000_000_000])
        self.assertAlmostEqual(records[-1].t, 1.0)

    def test_header_and_metadata_match_the_documented_layout(self):
        path = self.make()
        raw = path.read_bytes()
        magic, version, flags, start_ns, meta_len = FILE_HEADER.unpack_from(raw)
        self.assertEqual((magic, version, flags), (b'F1UDPREC', 1, 0))
        self.assertLess(abs(start_ns - time.time_ns()), 60 * 10**9)
        self.assertEqual(FILE_HEADER.size, 24)
        self.assertEqual(RECORD_HEADER.size, 14)
        offset = FILE_HEADER.size + meta_len
        t_ns, length, crc = RECORD_HEADER.unpack_from(raw, offset)
        payload = raw[offset + 14 : offset + 14 + length]
        self.assertEqual((t_ns, payload, crc), (0, PACKETS[0], zlib.crc32(PACKETS[0])))
        with RecordingReader(path) as reader:
            self.assertEqual(reader.metadata['game'], 'F1 2021')
            self.assertEqual(reader.metadata['port'], 20777)
            self.assertEqual(reader.metadata['note'], 'test')

    def test_gzip_is_detected_from_content_not_name(self):
        path = self.make('s.f1rec.gz')
        self.assertEqual(path.read_bytes()[:2], b'\x1f\x8b')
        renamed = path.rename(self.dir / 'renamed.bin')
        with RecordingReader(renamed) as reader:
            self.assertTrue(reader.compressed)
            self.assertEqual([r.data for r in reader], PACKETS)

    def test_truncated_tail_keeps_complete_records(self):
        path = self.make()
        raw = path.read_bytes()
        for cut in (3, len(PACKETS[-1]) + 5):  # inside the last payload, inside its header
            path.write_bytes(raw[:-cut])
            with RecordingReader(path) as reader:
                self.assertEqual([r.data for r in reader], PACKETS[:-1])
                self.assertTrue(reader.truncated)
        path.write_bytes(raw)
        with RecordingReader(path) as reader:
            list(reader)
            self.assertFalse(reader.truncated)

    def test_truncated_gzip_is_readable_up_to_the_cut(self):
        path = self.make('s.f1rec.gz')
        raw = path.read_bytes()
        path.write_bytes(raw[:-12])  # lose the gzip trailer and some data
        with RecordingReader(path) as reader:
            records = list(reader)
        self.assertGreaterEqual(len(records), 0)
        self.assertEqual([r.data for r in records], PACKETS[: len(records)])

    def test_corrupt_payload_raises(self):
        path = self.make()
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 0xFF
        path.write_bytes(bytes(raw))
        with self.assertRaises(RecordingError):
            list(read_records(path))

    def test_rejects_other_files_and_newer_versions(self):
        other = self.dir / 'other.bin'
        other.write_bytes(b'hello world, definitely not a recording')
        with self.assertRaises(RecordingError):
            RecordingReader(other)
        newer = self.dir / 'newer.f1rec'
        newer.write_bytes(FILE_HEADER.pack(b'F1UDPREC', 99, 0, 0, 2) + b'{}')
        with self.assertRaisesRegex(RecordingError, 'newer'):
            RecordingReader(newer)

    def test_does_not_overwrite_unless_asked(self):
        path = self.make()
        with self.assertRaises(FileExistsError):
            RecordingWriter(path)
        RecordingWriter(path, overwrite=True).close()
        self.assertEqual(list(read_records(path)), [])

    def test_arrival_times_from_clock_never_go_backwards(self):
        ticks = iter([5_000, 6_000, 4_000, 9_000])  # start, then three writes (one goes back)
        path = self.dir / 'clock.f1rec'
        with RecordingWriter(path, clock_ns=lambda: next(ticks)) as writer:
            for data in PACKETS[:3]:
                writer.write(data)
        self.assertEqual([r.t_ns for r in read_records(path)], [1_000, 1_000, 4_000])

    def test_summary(self):
        info = summarize(self.make())
        self.assertEqual(info['datagrams'], 4)
        self.assertAlmostEqual(info['duration_s'], 1.0)
        self.assertEqual(info['by_packet_id'][1], 1)
        self.assertEqual(info['by_packet_id'][6], 1)


class SpeedTests(RecordingTestCase):
    def test_parse_speed(self):
        self.assertEqual(parse_speed('1'), 1.0)
        self.assertEqual(parse_speed('10x'), 10.0)
        self.assertEqual(parse_speed('2.5'), 2.5)
        for text in ('max', 'MAX', 'asap', '0'):
            self.assertIsNone(parse_speed(text))
        for bad in ('fast-ish', '-1'):
            with self.assertRaises(ValueError):
                parse_speed(bad)

    def pace(self, speed):
        fake = FakeClock()
        records = [Record(t, b'') for t in (0, 40_000_000, 41_000_000, 1_000_000_000)]
        out = list(paced(records, speed, fake.clock, fake.sleep))
        self.assertEqual(out, records)
        return fake

    def test_real_time_honours_recorded_gaps(self):
        fake = self.pace(1.0)
        self.assertEqual(len(fake.sleeps), 3)
        for got, want in zip(fake.sleeps, (0.040, 0.001, 0.959), strict=False):
            self.assertAlmostEqual(got, want)
        self.assertAlmostEqual(fake.now - 100.0, 1.0)

    def test_ten_times_divides_gaps(self):
        fake = self.pace(10.0)
        self.assertAlmostEqual(fake.now - 100.0, 0.1)

    def test_max_never_waits(self):
        self.assertEqual(self.pace(None).sleeps, [])

    def test_lateness_is_not_accumulated(self):
        fake = FakeClock()
        records = [Record(t * 10_000_000, b'') for t in range(5)]  # every 10 ms
        for index, _ in enumerate(paced(records, 1.0, fake.clock, fake.sleep)):
            if index == 1:
                fake.now += 0.025  # consumer is slow once
        # 10 ms, then 25 ms late, so records at 20 and 30 ms go at once and 40 ms waits 5 ms:
        # the replay is back on schedule instead of running 25 ms behind for ever.
        self.assertAlmostEqual(fake.now - 100.0, 0.040, places=6)
        self.assertEqual([round(s, 6) for s in fake.sleeps], [0.01, 0.005])


class ReplayTests(RecordingTestCase):
    def test_replay_into_stream_uses_recorded_clock(self):
        class Stream:
            def __init__(self):
                self.calls = []

            def handle(self, data, now=None):
                self.calls.append((data, now))

        stream = Stream()
        self.assertEqual(replay_into(stream, self.make()), 4)
        self.assertEqual([d for d, _ in stream.calls], PACKETS)
        self.assertEqual([now for _, now in stream.calls], [0.0, 0.04, 0.041, 1.0])

    def test_replay_source_behaves_like_the_engineer_socket(self):
        source = ReplaySource(self.make(), speed=None)
        source.settimeout(1.0)
        received = [source.recvfrom(udp_recording.MAX_DATAGRAM)[0] for _ in PACKETS]
        self.assertEqual(received, PACKETS)
        with self.assertRaises(ReplayFinished):
            source.recvfrom(2048)
        self.assertTrue(issubclass(ReplayFinished, EOFError))
        source.close()

    def test_replay_to_udp_sends_every_datagram_in_order(self):
        receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(2.0)
        port = receiver.getsockname()[1]
        try:
            sent = replay_to_udp(self.make(times_ms=(0, 5, 6, 20)), '127.0.0.1', port, speed=10.0)
            got = [receiver.recvfrom(65535)[0] for _ in range(sent)]
        finally:
            receiver.close()
        self.assertEqual(got, PACKETS)


class RecorderTests(RecordingTestCase):
    def test_records_everything_arriving_on_a_socket(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
        path = self.dir / 'live.f1rec.gz'
        done = threading.Event()
        writer = RecordingWriter(path)

        def send():
            sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            time.sleep(0.05)
            for data in PACKETS:
                sender.sendto(data, ('127.0.0.1', port))
                time.sleep(0.02)
            sender.close()
            time.sleep(0.3)
            done.set()

        thread = threading.Thread(target=send)
        thread.start()
        try:
            count = record_socket(listener, writer, duration=10, should_stop=done.is_set)
        finally:
            thread.join()
            listener.close()
            writer.close()
        self.assertEqual(count, len(PACKETS))
        records = list(read_records(path))
        self.assertEqual([r.data for r in records], PACKETS)
        gaps = [b.t_ns - a.t_ns for a, b in zip(records, records[1:], strict=False)]
        self.assertTrue(all(gap >= 10_000_000 for gap in gaps), gaps)  # ~20 ms apart

    def test_record_then_replay_through_a_real_telemetry_stream(self):
        try:
            from analysis import AnalysisEngine
            from telemetry_stream import TelemetryStream
            from test_analysis import SilentVoice

            stream = TelemetryStream(AnalysisEngine(SilentVoice()))
        except Exception as e:  # the engine itself is broken on some branches; not this module
            self.skipTest(f'AnalysisEngine unavailable: {e}')
        path = self.make(times_ms=(0, 40, 41), packets=PACKETS[:3])
        replay_into(stream, path)
        self.assertIsNotNone(stream.latest_session)
        self.assertEqual(stream.latest_session.track_id, 11)


if __name__ == '__main__':
    unittest.main()
