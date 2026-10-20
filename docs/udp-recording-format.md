# UDP recording format (`.f1rec`), version 1

A recording is a byte-for-byte copy of every UDP datagram the F1 2021 game sent during a
session, each with the time it arrived. It is written by `ai_engineer/record_udp.py` and
`main.py --record`, and read by `ai_engineer/replay_udp.py`, `main.py --replay` and the
`ai_engineer/udp_recording.py` library (`RecordingReader`, `read_records`, `replay_into`).

Recordings are the raw input for backtesting, so the format is deliberately simple, fixed
and self-describing. Any language can read it with a few lines of code.

## Layout

All integers are **little-endian**. Python `struct` codes are given for each block.

```
+-------------------------------------------------------------+
| File header      24 bytes     '<8sHHqI'                     |
| Metadata         metadata_len bytes, UTF-8 JSON object      |
| Record 0         14-byte record header '<QHI' + payload     |
| Record 1         ...                                        |
| ...              until end of file                          |
+-------------------------------------------------------------+
```

### File header (24 bytes)

| Offset | Size | Type   | Field            | Meaning |
|-------:|-----:|--------|------------------|---------|
| 0      | 8    | bytes  | `magic`          | Always `F1UDPREC` (ASCII). |
| 8      | 2    | uint16 | `format_version` | `1`. Readers must refuse a version newer than they know. |
| 10     | 2    | uint16 | `flags`          | Reserved, written as `0`, ignored by v1 readers. |
| 12     | 8    | int64  | `start_unix_ns`  | Wall-clock time the recording started, nanoseconds since the Unix epoch (UTC). |
| 20     | 4    | uint32 | `metadata_len`   | Length in bytes of the JSON metadata that follows. |

### Metadata (JSON)

A UTF-8 JSON object. Keys written today:

| Key          | Example                              | Meaning |
|--------------|--------------------------------------|---------|
| `game`       | `"F1 2021"`                          | Game the datagrams came from. |
| `udp_format` | `2021`                               | UDP format setting in the game. |
| `recorder`   | `"ai_engineer/udp_recording.py"`     | Program that wrote the file. |
| `port`       | `20777`                              | UDP port that was listened on. |
| `started_at` | `"2026-10-20T10:00:00+01:00"`        | Local start time, for humans (record_udp.py only). |
| `note`       | `"Silverstone race, 50%"`            | Free text from `--note` (optional). |

Readers must ignore keys they do not know. New keys can be added without a version bump.

### Record (14-byte header + payload)

| Offset | Size     | Type   | Field    | Meaning |
|-------:|---------:|--------|----------|---------|
| 0      | 8        | uint64 | `t_ns`   | Arrival time in nanoseconds since the recording started, from a monotonic clock. Never decreases from one record to the next. |
| 8      | 2        | uint16 | `length` | Payload length in bytes (a UDP datagram is at most 65535). |
| 10     | 4        | uint32 | `crc32`  | CRC-32 (zlib / IEEE 802.3) of the payload. |
| 14     | `length` | bytes  | payload  | The datagram exactly as received. |

Every datagram is recorded, including ones the engineer would reject (wrong format, unknown
packet ids, bad lengths), so the recording does not depend on today's parser. The packet id
of an F1 2021 datagram is payload byte 5; the game's own `sessionTime` and `frameIdentifier`
are inside the payload's packet header.

`t_ns` is the time the recorder received the datagram, not the game's `sessionTime`. Replay
uses it to reproduce the original pacing, including pauses (menus, loading screens).

## Compression

The entire byte stream above may be gzip-compressed. The writer compresses when the file name
ends `.gz` (the recorders default to `.f1rec.gz`; a session is roughly 300 KB/s raw at 60 Hz
and compresses several times smaller). Readers detect compression from the gzip magic bytes
`1f 8b`, never from the file name. The writer does a gzip sync flush about once a second, so a
compressed file is readable up to the last flush even if the recorder is killed.

## Robustness rules for readers

* A file that ends part-way through a record (recorder crashed, disk full, power cut) is
  valid: return every complete record before the cut and report that the file was truncated
  (`RecordingReader.truncated`).
* A record whose payload does not match its `crc32` means corruption: stop with an error
  (`RecordingError`) instead of returning bad data.
* Bad `magic`, or a `format_version` above the reader's, is an error.

## Changing the format

Bump `format_version` for any change that an existing reader would misread (a new record
field, a different header size). Adding metadata keys or defining a `flags` bit that old
readers can safely ignore does not need a bump. Keep readers able to read every older
version, since old recordings are backtest fixtures.

## Reading a recording without this repo

```python
import gzip, json, struct, zlib

def read(path):
    raw = open(path, 'rb')
    f = gzip.open(path, 'rb') if raw.read(2) == b'\x1f\x8b' else open(path, 'rb')
    magic, version, flags, start_ns, meta_len = struct.unpack('<8sHHqI', f.read(24))
    assert magic == b'F1UDPREC' and version == 1
    meta = json.loads(f.read(meta_len))
    while True:
        head = f.read(14)
        if len(head) < 14:
            return
        t_ns, length, crc = struct.unpack('<QHI', head)
        data = f.read(length)
        if len(data) < length:
            return                      # truncated tail
        assert zlib.crc32(data) == crc
        yield t_ns, data
```

## Tools

| Command | What it does |
|---------|--------------|
| `python ai_engineer/record_udp.py [--out FILE] [--port 20777] [--duration S] [--note TEXT]` | Record the game without running the engineer. Default output `recordings/<date-time>.f1rec.gz`. Refuses to overwrite unless `--force`. |
| `python main.py --record FILE ...` | Run the engineer as usual and also record every datagram it receives. |
| `python ai_engineer/replay_udp.py FILE [--speed 1\|10\|max] [--host H] [--port P]` | Send a recording over UDP to a running engineer, as the game would. |
| `python ai_engineer/replay_udp.py FILE --info` | Print duration, datagram counts per packet id and metadata. |
| `python main.py --replay FILE [--replay-speed 1\|10\|max]` | Run the engineer straight from a recording, with no socket and no dropped datagrams. |
| `udp_recording.replay_into(stream, FILE)` | Library: feed a `TelemetryStream` with the recorded clock as `now`, for backtests. |

Speeds: `1` honours the recorded gaps between datagrams in real time, `10` divides every gap by
ten, `max` sends without waiting. Pacing is scheduled from a single start anchor, so timing
error does not build up over a long replay. Over UDP at `max`, a receiver that falls behind
can drop datagrams (the operating system's receive buffer fills); `main.py --replay` and
`replay_into` are lossless at any speed.
