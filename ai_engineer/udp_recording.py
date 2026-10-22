"""Record raw F1 2021 UDP datagrams to a file and replay them later.

A recording keeps every datagram exactly as it arrived (valid or not) together with its
arrival time, so a session can be fed back through the engineer at real speed, faster, or
as fast as possible. It is the raw input for backtesting predictions.

The file format is specified in docs/udp-recording-format.md. In short (little-endian):

    file header   '<8sHHqI'  magic b'F1UDPREC', format version, flags, start time (unix ns),
                             metadata length
    metadata      UTF-8 JSON object, metadata-length bytes
    records       '<QHI'     arrival ns since start, payload length, CRC-32 of payload
                  payload    the datagram bytes
    ... repeated until end of file

The whole byte stream may be gzip-compressed (file name ending .gz when written here);
readers detect gzip by its magic bytes, not by the file name.

Library use:
    with RecordingWriter('race.f1rec', metadata={'port': 20777}) as rec:
        rec.write(datagram)                      # arrival time taken now
    for record in read_records('race.f1rec'):    # Record(t_ns, data); record.t is seconds
        ...
    replay_into(stream, 'race.f1rec')            # feed TelemetryStream.handle(data, now=t)
"""

import gzip
import json
import socket
import struct
import time
import zlib
from collections import namedtuple
from pathlib import Path

MAGIC = b'F1UDPREC'
FORMAT_VERSION = 1
FILE_HEADER = struct.Struct('<8sHHqI')
RECORD_HEADER = struct.Struct('<QHI')
MAX_DATAGRAM = 65535
GZIP_MAGIC = b'\x1f\x8b'
FLUSH_INTERVAL_S = 1.0


class RecordingError(Exception):
    """The file is not a recording, is from a newer format, or is corrupt."""


class ReplayFinished(EOFError):
    """Raised by ReplaySource.recvfrom once every recorded datagram has been delivered."""


class Record(namedtuple('Record', 't_ns data')):
    """One datagram: arrival time in nanoseconds since recording start, and its raw bytes."""
    __slots__ = ()

    @property
    def t(self) -> float:
        return self.t_ns / 1e9


# ──────────────────────────────────────────────
# Writing
# ──────────────────────────────────────────────

class RecordingWriter:
    """Append datagrams to a new recording file. Use as a context manager or call close().

    The file is flushed about once a second, so a crash or power cut loses at most the last
    second; the reader ignores a half-written final record.
    """

    def __init__(self, path, metadata=None, compress=None, overwrite=False, clock_ns=time.monotonic_ns):
        self.path = Path(path)
        compress = self.path.suffix == '.gz' if compress is None else compress
        mode = 'wb' if overwrite else 'xb'
        self._file = gzip.open(self.path, mode) if compress else open(self.path, mode)
        self._clock_ns = clock_ns
        self._start_ns = clock_ns()
        self._last_t_ns = 0
        self._last_flush = time.monotonic()
        self.count = 0
        self.bytes = 0
        meta = {'game': 'F1 2021', 'udp_format': 2021, 'recorder': 'ai_engineer/udp_recording.py'}
        meta.update(metadata or {})
        blob = json.dumps(meta, sort_keys=True).encode('utf-8')
        self.metadata = meta
        self._file.write(FILE_HEADER.pack(MAGIC, FORMAT_VERSION, 0, time.time_ns(), len(blob)) + blob)
        self._file.flush()

    def write(self, data: bytes, t_ns=None):
        """Record one datagram. t_ns defaults to now (ns since this writer was created)."""
        if len(data) > MAX_DATAGRAM:
            raise ValueError(f'datagram of {len(data)} bytes is larger than UDP allows')
        if t_ns is None:
            t_ns = self._clock_ns() - self._start_ns
        t_ns = max(int(t_ns), self._last_t_ns)  # arrival times never go backwards
        self._last_t_ns = t_ns
        self._file.write(RECORD_HEADER.pack(t_ns, len(data), zlib.crc32(data)) + data)
        self.count += 1
        self.bytes += len(data)
        now = time.monotonic()
        if now - self._last_flush >= FLUSH_INTERVAL_S:
            self._last_flush = now
            self.flush()

    def flush(self):
        self._file.flush()  # gzip does a sync flush, so the compressed file stays readable

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ──────────────────────────────────────────────
# Reading
# ──────────────────────────────────────────────

class RecordingReader:
    """Iterate the records of a recording. Opening reads and checks the header.

    After iterating, `truncated` is True if the file ended part-way through a record (for
    example the recorder was killed); everything before that point is still returned.
    """

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, 'rb') as probe:
            compressed = probe.read(2) == GZIP_MAGIC
        self._file = gzip.open(self.path, 'rb') if compressed else open(self.path, 'rb')
        self.compressed = compressed
        self.truncated = False
        try:
            header = self._read_exact(FILE_HEADER.size)
            if header is None:
                raise RecordingError(f'{self.path}: too short to be a recording')
            magic, version, self.flags, self.start_unix_ns, meta_len = FILE_HEADER.unpack(header)
            if magic != MAGIC:
                raise RecordingError(f'{self.path}: not a UDP recording (bad magic)')
            if version > FORMAT_VERSION:
                raise RecordingError(f'{self.path}: format version {version} is newer than '
                                     f'this reader ({FORMAT_VERSION})')
            self.format_version = version
            blob = self._read_exact(meta_len)
            if blob is None:
                raise RecordingError(f'{self.path}: metadata cut short')
            self.metadata = json.loads(blob.decode('utf-8'))
        except Exception:
            self._file.close()
            raise

    def _read_exact(self, size):
        """Read exactly size bytes; None at a clean or truncated end of file."""
        try:
            data = self._file.read(size)
        except EOFError:  # gzip stream cut off mid-member
            data = b''
        if len(data) != size:
            if data:
                self.truncated = True
            return None
        return data

    def __iter__(self):
        index = 0
        while True:
            head = self._read_exact(RECORD_HEADER.size)
            if head is None:
                return
            t_ns, length, crc = RECORD_HEADER.unpack(head)
            data = self._read_exact(length)
            if data is None:
                self.truncated = True
                return
            if zlib.crc32(data) != crc:
                raise RecordingError(f'{self.path}: record {index} is corrupt (CRC mismatch)')
            yield Record(t_ns, data)
            index += 1

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_records(path):
    """Yield every Record in a recording, oldest first."""
    with RecordingReader(path) as reader:
        yield from reader


# ──────────────────────────────────────────────
# Replay
# ──────────────────────────────────────────────

def parse_speed(text):
    """'1', '1x', '10x', '2.5' -> float multiplier; 'max' / 'asap' / '0' -> None (no waiting)."""
    value = str(text).strip().lower()
    if value in ('max', 'asap', 'fast', '0', '0x'):
        return None
    value = value.removesuffix('x')
    try:
        speed = float(value)
    except ValueError:
        raise ValueError(f'speed must be a number like 1 or 10, or "max"; got {text!r}') from None
    if not speed > 0:
        raise ValueError('speed must be greater than zero')
    return speed


def paced(records, speed, clock=time.monotonic, sleep=time.sleep):
    """Yield records on the recorded timeline divided by speed (None = no waiting at all).

    Each record is scheduled against one start anchor, so sleep jitter never accumulates.
    """
    start = first_t = None
    for record in records:
        if speed is not None:
            if start is None:
                start, first_t = clock(), record.t_ns
            delay = start + (record.t_ns - first_t) / 1e9 / speed - clock()
            if delay > 0:
                sleep(delay)
        yield record


def replay_into(stream, path, speed=None, clock=time.monotonic, sleep=time.sleep):
    """Feed a recording straight into a TelemetryStream (no sockets), for backtesting.

    The stream's clock (`now`) is the recorded arrival time, so timeouts and gaps behave as
    they did live whatever the replay speed. Returns the number of datagrams delivered.
    """
    count = 0
    for record in paced(read_records(path), speed, clock, sleep):
        stream.handle(record.data, now=record.t)
        count += 1
    return count


def replay_to_udp(path, host='127.0.0.1', port=20777, speed=1.0, sock=None, on_progress=None):
    """Send a recording to host:port as UDP datagrams, as the game would. Returns the count.

    At speed None (max) a receiver on the same machine can drop datagrams if it cannot keep
    up; use replay_into or `main.py --replay` for lossless as-fast-as-possible replay.
    """
    own_socket = sock is None
    sock = sock or socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    count = 0
    try:
        for record in paced(read_records(path), speed):
            sock.sendto(record.data, (host, port))
            count += 1
            if on_progress:
                on_progress(count, record)
    finally:
        if own_socket:
            sock.close()
    return count


class ReplaySource:
    """A stand-in for the engineer's UDP socket that serves datagrams from a recording.

    Supports the calls main.py makes on its socket (recvfrom, settimeout, close). recvfrom
    waits according to the replay speed and raises ReplayFinished at the end of the file.
    """

    def __init__(self, path, speed=1.0):
        self.path = Path(path)
        self.speed = speed
        self._reader = RecordingReader(path)
        self._records = paced(iter(self._reader), speed)
        self.count = 0

    def recvfrom(self, bufsize):
        record = next(self._records, None)
        if record is None:
            raise ReplayFinished(f'replay of {self.path} finished after {self.count} datagrams')
        self.count += 1
        return record.data[:bufsize], ('replay', 0)

    def settimeout(self, timeout):
        pass

    def close(self):
        self._reader.close()


# ──────────────────────────────────────────────
# Recording from a live socket
# ──────────────────────────────────────────────

def record_socket(sock, writer, duration=None, should_stop=None, on_progress=None,
                  progress_interval=1.0):
    """Record every datagram arriving on a bound UDP socket until duration, should_stop() or
    Ctrl+C. Returns the number recorded."""
    sock.settimeout(0.2)
    started = last_progress = time.monotonic()
    try:
        while True:
            now = time.monotonic()
            if duration is not None and now - started >= duration:
                break
            if should_stop and should_stop():
                break
            if on_progress and now - last_progress >= progress_interval:
                last_progress = now
                on_progress(writer)
            try:
                data, _ = sock.recvfrom(MAX_DATAGRAM)
            except TimeoutError:
                continue
            except ConnectionResetError:  # Windows reports ICMP errors on UDP sockets
                continue
            writer.write(data)
    except KeyboardInterrupt:
        pass
    writer.flush()
    return writer.count


def summarize(path):
    """Counts and duration of a recording, for `replay_udp.py --info`."""
    with RecordingReader(path) as reader:
        count = size = 0
        first = last = None
        by_packet_id = {}
        for record in reader:
            count += 1
            size += len(record.data)
            first = record.t_ns if first is None else first
            last = record.t_ns
            packet_id = record.data[5] if len(record.data) > 5 else None
            by_packet_id[packet_id] = by_packet_id.get(packet_id, 0) + 1
        return {
            'path': str(path), 'format_version': reader.format_version,
            'compressed': reader.compressed, 'metadata': reader.metadata,
            'start_unix_ns': reader.start_unix_ns, 'datagrams': count, 'payload_bytes': size,
            'duration_s': 0.0 if first is None else (last - first) / 1e9,
            'by_packet_id': by_packet_id, 'truncated': reader.truncated,
        }
