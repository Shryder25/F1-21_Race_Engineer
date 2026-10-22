"""Logging for the engineer: readable lines in the terminal, JSON lines for machines.

Every module logs through ``get_logger(__name__-ish short name)``; ``configure()`` is called
once at startup from main.py with the ``--log-level`` / ``--log-format`` flags (or the
``[logging]`` section of config.toml). Until then Python's default (warnings and up to
stderr) applies, which keeps the unit tests quiet.

Extra fields passed with ``extra={...}`` show up as keys in JSON output, so a log line like
``log.warning('Could not save lap', extra={'lap': 12})`` stays machine-filterable.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any, TextIO

ROOT = 'f1'
LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
FORMATS = ('text', 'json')

# Attributes every LogRecord has: anything else on a record came from ``extra=``.
_STANDARD = frozenset(vars(logging.LogRecord('', 0, '', 0, '', None, None))) | {'message', 'asctime'}


def get_logger(name: str) -> logging.Logger:
    """A logger under the engineer's root, e.g. get_logger('history') -> 'f1.history'."""
    return logging.getLogger(f'{ROOT}.{name}')


def _extras(record: logging.LogRecord) -> dict[str, Any]:
    return {k: v for k, v in vars(record).items() if k not in _STANDARD and not k.startswith('_')}


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, msg, any extra fields, exc."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            'ts': time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created))
            + f'.{int(record.msecs):03d}Z',
            'level': record.levelname,
            'logger': record.name,
            'msg': record.getMessage(),
        }
        entry.update(_extras(record))
        if record.exc_info:
            entry['exc'] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """``12:34:56 INFO  [history] message  key=value``: short and readable in a terminal."""

    def format(self, record: logging.LogRecord) -> str:
        name = record.name.removeprefix(f'{ROOT}.')
        clock = time.strftime('%H:%M:%S', time.localtime(record.created))
        line = f'{clock} {record.levelname:<5} [{name}] {record.getMessage()}'
        extras = _extras(record)
        if extras:
            line += '  ' + ' '.join(f'{k}={v}' for k, v in extras.items())
        if record.exc_info:
            line += '\n' + self.formatException(record.exc_info)
        return line


def configure(
    level: str = 'INFO', fmt: str = 'text', file: str | None = None, stream: TextIO | None = None
) -> logging.Logger:
    """Set up the engineer's loggers. Safe to call more than once (handlers are replaced).

    Logs go to stderr, so they never tangle with the live console dashboard on stdout.
    ``file`` additionally writes every record as JSON lines.
    """
    name = level.upper()
    if name not in LEVELS:
        raise ValueError(f'unknown log level {level!r} (use one of {", ".join(LEVELS)})')
    if fmt not in FORMATS:
        raise ValueError(f'unknown log format {fmt!r} (use text or json)')
    root = logging.getLogger(ROOT)
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setFormatter(JsonFormatter() if fmt == 'json' else TextFormatter())
    root.addHandler(console)
    if file:
        to_file = logging.FileHandler(file, encoding='utf-8')
        to_file.setFormatter(JsonFormatter())
        root.addHandler(to_file)
    root.setLevel(name)
    root.propagate = False
    return root
