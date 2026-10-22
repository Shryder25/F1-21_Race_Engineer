"""Recording I/O for the golden sessions, on top of ai_engineer/udp_recording.py.

Reading uses the recorder's own reader, so a session captured on a PC with
`record_udp.py` or `main.py --record` replays here unchanged. Writing synthetic sessions
uses a small reproducible writer in the same format (docs/udp-recording-format.md): the
recorder stamps the wall-clock start time and a gzip mtime into each file, which would
make every regeneration a binary diff even when nothing changed.
"""

from __future__ import annotations

import gzip
import io
import json
import sys
import zlib
from pathlib import Path

_AI = Path(__file__).resolve().parents[2] / 'ai_engineer'
if str(_AI) not in sys.path:
    sys.path.insert(0, str(_AI))

from udp_recording import (  # noqa: E402,F401
    FILE_HEADER,
    FORMAT_VERSION,
    MAGIC,
    RECORD_HEADER,
    Record,
    RecordingError,
    RecordingReader,
    read_records,
)


def read_metadata(path) -> dict:
    with RecordingReader(path) as reader:
        return dict(reader.metadata)


class ReproducibleWriter:
    """Write a gzipped recording whose bytes depend only on the datagrams and metadata."""

    def __init__(self, path, metadata: dict | None = None):
        self._raw = open(path, 'wb')
        self._file = gzip.GzipFile(fileobj=self._raw, mode='wb', mtime=0, compresslevel=9)
        blob = json.dumps(metadata or {}, sort_keys=True).encode('utf-8')
        self._file.write(FILE_HEADER.pack(MAGIC, FORMAT_VERSION, 0, 0, len(blob)) + blob)
        self._last_t_ns = 0
        self.count = 0

    def write(self, data: bytes, t_ns: int):
        t_ns = max(int(t_ns), self._last_t_ns)
        self._last_t_ns = t_ns
        self._file.write(RECORD_HEADER.pack(t_ns, len(data), zlib.crc32(data)) + data)
        self.count += 1

    def close(self):
        self._file.close()
        self._raw.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def dumps_for_test(records, metadata=None) -> bytes:
    """An uncompressed in-memory recording, for unit tests."""
    out = io.BytesIO()
    blob = json.dumps(metadata or {}).encode()
    out.write(FILE_HEADER.pack(MAGIC, FORMAT_VERSION, 0, 0, len(blob)) + blob)
    for t_ns, data in records:
        out.write(RECORD_HEADER.pack(t_ns, len(data), zlib.crc32(data)) + data)
    return out.getvalue()
