"""Export channel-store sessions to MoTeC i2 (.ld + .ldx), with a MoTeC-style CSV fallback.

MoTeC i2 (the free Standard edition is enough) opens `.ld` log files. The format is not
published; the layout here follows the community's reverse-engineered description of it
(the same one used by the Assetto Corsa Competizione and other sim exporters):

    file header   1762 bytes, pointers to the first channel's metadata, its data and the
                  event block, plus date, time, driver, vehicle, venue and a short comment
    event         name, session, comment, pointer to the venue block
    venue         name, pointer to the vehicle block
    vehicle       id, weight, type, comment
    channel meta  124 bytes each, a doubly linked list: prev/next pointers, data pointer,
                  sample count, data type, sample rate, scaling, name, short name, unit
    channel data  each channel's samples back to back (float32 or int16, little-endian)

Every channel in an `.ld` file has a fixed sample rate, while UDP samples arrive at an
uneven rate, so the export resamples each channel onto a regular time grid (default:
the session's own send rate, e.g. 20 or 60 Hz). Continuous channels are interpolated,
discrete ones (gear, lap, flags) hold their last value.

Laps come from the `.ldx` file written next to the `.ld`: one beacon marker each time the
lap number goes up, which is how i2 splits an outing into laps. Without the `.ldx`, i2
still opens the data but shows the whole session as one lap.

Limitations: there is no official spec, so a future i2 release could reject these files
(the CSV is the fallback); values are stored unscaled (float32 / int16 with scale 1),
which i2 reads fine but which means i2's channel "decimal places" display defaults to 0
for integer channels.
"""

import datetime
import math
import struct
from collections import namedtuple
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np

from channel_store import CHANNELS, CHANNELS_BY_NAME, lap_summary, read_session, resample

HEADER_FMT = '<I4xII20xI24xHHHI8sHHI4x16s16x16s16x64s64s64x64s64x1024xI66x64s126x'
EVENT_FMT = '<64s64s1024sH'
VENUE_FMT = '<64s1034xH'
VEHICLE_FMT = '<64s128xI32s32s'
CHANNEL_FMT = '<IIIIHHHHhhhh32s8s12s40x'
HEADER_SIZE = struct.calcsize(HEADER_FMT)      # 1762
EVENT_SIZE = struct.calcsize(EVENT_FMT)        # 1154
VENUE_SIZE = struct.calcsize(VENUE_FMT)        # 1100
VEHICLE_SIZE = struct.calcsize(VEHICLE_FMT)    # 260
CHANNEL_SIZE = struct.calcsize(CHANNEL_FMT)    # 124
# Type codes as written by MoTeC loggers: (type tag, bytes per sample)
DTYPES = {'float32': (0x07, 4, np.dtype('<f4')), 'int16': (0x03, 2, np.dtype('<i2')),
          'int32': (0x05, 4, np.dtype('<i4'))}
STANDARD_RATES = (10, 15, 20, 30, 60)  # the game's UDP send-rate options

LdChannel = namedtuple('LdChannel', 'name short_name unit freq data dtype')


def _text(value, size):
    return str(value or '').encode('ascii', 'replace')[:size]


def short_name(name, taken=()):
    """An 8-character short name, unique within `taken`."""
    words = name.replace('/', ' ').split()
    base = ''.join(w[0].upper() + w[1:3].lower() if i < len(words) - 1 else w[:4]
                   for i, w in enumerate(words))[:8] or 'Ch'
    candidate, n = base, 2
    while candidate in taken:
        suffix = str(n)
        candidate = base[:8 - len(suffix)] + suffix
        n += 1
    return candidate


def write_ld(path, channels, *, driver='', vehicle='', venue='', event='', session='',
             comment='', short_comment='', start=None, vehicle_type='', vehicle_weight=0):
    """Write a MoTeC .ld file. `channels` is a list of LdChannel (data: 1-D numpy array)."""
    if not channels:
        raise ValueError('a MoTeC log needs at least one channel')
    start = start or datetime.datetime.now()
    event_ptr = HEADER_SIZE
    venue_ptr = event_ptr + EVENT_SIZE
    vehicle_ptr = venue_ptr + VENUE_SIZE
    meta_ptr = vehicle_ptr + VEHICLE_SIZE
    data_ptr = meta_ptr + CHANNEL_SIZE * len(channels)
    if venue_ptr > 0xFFFF or vehicle_ptr > 0xFFFF:
        raise AssertionError('venue/vehicle pointers must fit in 16 bits')

    blocks = []
    pointer = data_ptr
    for ch in channels:
        tag, size, np_type = DTYPES[ch.dtype]
        values = np.asarray(ch.data, dtype=np.float64)
        if ch.dtype != 'float32':
            info = np.iinfo(np_type)
            values = np.clip(np.round(np.nan_to_num(values)), info.min, info.max)
        blocks.append((ch, tag, size, values.astype(np_type).tobytes(), pointer))
        pointer += len(values) * size

    with open(path, 'wb') as f:
        f.write(struct.pack(
            HEADER_FMT, 0x40, meta_ptr, data_ptr, event_ptr,
            1, 0x4240, 0xF,                      # static values seen in every MoTeC file
            0x1F44, b'ADL', 420, 0xADB0,         # device serial, type, version
            len(channels),
            _text(start.strftime('%d/%m/%Y'), 16), _text(start.strftime('%H:%M:%S'), 16),
            _text(driver, 64), _text(vehicle, 64), _text(venue, 64),
            0xC81A4,                             # "pro logging" marker
            _text(short_comment, 64)))
        f.write(struct.pack(EVENT_FMT, _text(event, 64), _text(session, 64),
                            _text(comment, 1024), venue_ptr))
        f.write(struct.pack(VENUE_FMT, _text(venue, 64), vehicle_ptr))
        f.write(struct.pack(VEHICLE_FMT, _text(vehicle, 64), int(vehicle_weight),
                            _text(vehicle_type, 32), b''))
        for i, (ch, tag, size, payload, ptr) in enumerate(blocks):
            prev_ptr = meta_ptr + (i - 1) * CHANNEL_SIZE if i else 0
            next_ptr = meta_ptr + (i + 1) * CHANNEL_SIZE if i < len(blocks) - 1 else 0
            f.write(struct.pack(CHANNEL_FMT, prev_ptr, next_ptr, ptr, len(payload) // size,
                                0x2EE1 + i, tag, size, int(ch.freq),
                                0, 1, 1, 0,      # shift, multiplier, scale, decimal places
                                _text(ch.name, 32), _text(ch.short_name, 8), _text(ch.unit, 12)))
        for *_, payload, _ptr in blocks:
            f.write(payload)


def read_ld(path):
    """Read back a .ld file: (header dict, [LdChannel]). Enough to check our own exports."""
    def s(raw):
        return raw.split(b'\0', 1)[0].decode('ascii', 'replace').strip()

    with open(path, 'rb') as f:
        blob = f.read()
    h = struct.unpack_from(HEADER_FMT, blob, 0)
    marker, meta_ptr, data_ptr, event_ptr, n = h[0], h[1], h[2], h[3], h[11]
    head = {'marker': marker, 'channels': n, 'date': s(h[12]), 'time': s(h[13]),
            'driver': s(h[14]), 'vehicle': s(h[15]), 'venue': s(h[16]), 'short_comment': s(h[18])}
    if event_ptr:
        name, sess, comment, venue_ptr = struct.unpack_from(EVENT_FMT, blob, event_ptr)
        head.update(event=s(name), session=s(sess), comment=s(comment))
        if venue_ptr:
            venue, vehicle_ptr = struct.unpack_from(VENUE_FMT, blob, venue_ptr)
            head['venue_block'] = s(venue)
            if vehicle_ptr:
                vid, weight, vtype, _ = struct.unpack_from(VEHICLE_FMT, blob, vehicle_ptr)
                head.update(vehicle_block=s(vid), vehicle_type=s(vtype), vehicle_weight=weight)
    channels, ptr, seen = [], meta_ptr, set()
    while ptr and ptr not in seen:
        seen.add(ptr)
        (_prev, nxt, dptr, count, _c, tag, size, freq, shift, mul, scale, dec,
         name, short, unit) = struct.unpack_from(CHANNEL_FMT, blob, ptr)
        np_type = {(0x07, 4): '<f4', (0x07, 2): '<f2', (0x03, 2): '<i2', (0x05, 4): '<i4',
                   (0x00, 2): '<i2', (0x03, 4): '<i4'}[(tag, size)]
        raw = np.frombuffer(blob, dtype=np_type, count=count, offset=dptr).astype(np.float64)
        values = (raw / scale * 10.0 ** -dec + shift) * mul
        channels.append(LdChannel(s(name), s(short), s(unit), freq, values,
                                  'float32' if tag == 0x07 else 'int16'))
        ptr = nxt
    return head, channels


def write_ldx(path, beacon_times_s, details=None):
    """Write the .ldx sidecar: lap beacons (seconds from log start) and session details."""
    markers = '\n'.join(
        f'     <Marker Version="100" ClassName="BCN" Name="Manual.{i + 1}" Flags="77" '
        f'Time="{t * 1e6:.2f}"/>' for i, t in enumerate(beacon_times_s))
    detail_rows = '\n'.join(f'   <String Id={quoteattr(str(k))} Value={quoteattr(str(v))}/>'
                            for k, v in (details or {}).items())
    Path(path).write_text(
        '<?xml version="1.0"?>\n'
        '<LDXFile Locale="English_United Kingdom.1252" DefaultLocale="C" Version="1.6">\n'
        ' <Layers>\n'
        '  <Layer>\n'
        '   <MarkerBlock>\n'
        '    <MarkerGroup Name="Beacons" Index="3">\n'
        f'{markers}\n'
        '    </MarkerGroup>\n'
        '   </MarkerBlock>\n'
        '   <RangeBlock/>\n'
        '  </Layer>\n'
        '  <Details>\n'
        f'{detail_rows}\n'
        '  </Details>\n'
        ' </Layers>\n'
        '</LDXFile>\n', encoding='utf-8')


def format_lap_time(seconds):
    minutes, rest = divmod(seconds, 60)
    return f'{int(minutes)}:{rest:06.3f}'


def pick_rate(times):
    """The game's send rate closest to the median sample interval."""
    dt = np.diff(times)
    dt = dt[dt > 0]
    if not len(dt):
        return STANDARD_RATES[2]
    measured = 1.0 / float(np.median(dt))
    return min(STANDARD_RATES, key=lambda r: abs(r - measured))


def resample_session(cols, rate_hz):
    """Every MoTeC-exported column on a regular time grid: (grid seconds, [(Channel, data)])."""
    t = cols['session_time']
    grid = np.arange(t[0], t[-1] + 0.5 / rate_hz, 1.0 / rate_hz)
    out = []
    for channel in CHANNELS:
        if channel.motec is None or channel.name not in cols:
            continue
        y = cols[channel.name].astype(np.float64)
        m = np.isfinite(y)
        if not m.any():
            continue  # never received (e.g. no Car Damage packets): leave it out
        values = resample(t[m], y[m], grid, channel.kind) if m.sum() > 1 else np.full(grid.shape, y[m][0])
        # Before a channel's first sample, hold its first value rather than leave a hole.
        first, last = np.flatnonzero(np.isfinite(values))[[0, -1]] if np.isfinite(values).any() else (0, 0)
        values[:first] = values[first]
        values[last + 1:] = values[last]
        out.append((channel, values * channel.motec[2]))
    return grid, out


def beacons(cols, grid_start):
    """Times (seconds from grid start) at which a new lap begins."""
    lap, t = cols['lap'], cols['session_time']
    ok = np.isfinite(lap)
    lap, t = lap[ok], t[ok]
    starts = np.flatnonzero(np.diff(lap) > 0) + 1
    return [float(t[i] - grid_start) for i in starts]


def export_session(parquet_path, out_path=None, rate_hz=None, fmt='ld'):
    """Export one channel-store file. fmt 'ld' writes .ld + .ldx, 'csv' a MoTeC-style CSV.

    Returns a dict describing what was written.
    """
    parquet_path = Path(parquet_path)
    cols, meta = read_session(parquet_path)
    if len(cols['session_time']) < 2:
        raise ValueError(f'{parquet_path} has fewer than two samples; nothing to export')
    rate_hz = int(rate_hz or pick_rate(cols['session_time']))
    if not 1 <= rate_hz <= 1000:
        raise ValueError('rate must be between 1 and 1000 Hz')
    grid, resampled = resample_session(cols, rate_hz)
    marks = beacons(cols, grid[0])
    laps = lap_summary(cols)
    timed = [l for l in laps if l['lap_time'] and not l['invalid']]
    fastest = min(timed, key=lambda l: l['lap_time']) if timed else None
    start = datetime.datetime.fromtimestamp(meta['start_unix']) if meta.get('start_unix') else None
    labels = {
        'driver': meta.get('driver', ''), 'vehicle': f"{meta.get('team', 'F1')} F1 2021".strip(),
        'venue': meta.get('track', ''), 'event': f"F1 2021 {meta.get('session', '')}".strip(),
        'session': meta.get('session', ''),
        'comment': f"Exported by F1-21 Race Engineer from {parquet_path.name}",
        'short_comment': meta.get('session_uid', ''), 'start': start, 'vehicle_type': 'F1 2021'}
    fmt = fmt.lower()
    out_path = Path(out_path) if out_path else parquet_path.with_suffix('.' + fmt)
    result = {'path': out_path, 'rate_hz': rate_hz, 'samples': len(grid),
              'channels': len(resampled), 'laps': len(marks) + 1,
              'fastest_lap': fastest and fastest['lap'], 'fastest_time': fastest and fastest['lap_time']}
    if fmt == 'ld':
        taken = set()
        channels = []
        for channel, values in resampled:
            name, unit, _ = channel.motec
            sn = short_name(name, taken)
            taken.add(sn)
            dtype = 'float32' if channel.type.startswith('float') or channel.motec[2] != 1.0 \
                else 'int16'
            channels.append(LdChannel(name, sn, unit, rate_hz, values, dtype))
        write_ld(out_path, channels, **labels)
        details = {'Total Laps': len(marks) + 1}
        if fastest:
            details['Fastest Time'] = format_lap_time(fastest['lap_time'])
            details['Fastest Lap'] = fastest['lap']
        ldx = out_path.with_suffix('.ldx')
        write_ldx(ldx, marks, details)
        result['ldx'] = ldx
    elif fmt == 'csv':
        write_motec_csv(out_path, grid - grid[0], resampled, labels, rate_hz, marks)
    else:
        raise ValueError("fmt must be 'ld' or 'csv'")
    return result


def write_motec_csv(path, times, resampled, labels, rate_hz, marks):
    """The CSV layout MoTeC i2 itself exports: a header block, channel names, units, data."""
    def q(v):
        return '"' + str(v).replace('"', "'") + '"'

    start = labels.get('start') or datetime.datetime.now()
    duration = float(times[-1]) if len(times) else 0.0
    head = [
        [q('Format'), q('MoTeC CSV File')],
        [q('Venue'), q(labels.get('venue', ''))],
        [q('Vehicle'), q(labels.get('vehicle', ''))],
        [q('Driver'), q(labels.get('driver', ''))],
        [q('Device'), q('ADL')],
        [q('Comment'), q(labels.get('comment', ''))],
        [q('Log Date'), q(start.strftime('%d/%m/%Y'))],
        [q('Log Time'), q(start.strftime('%H:%M:%S'))],
        [q('Sample Rate'), q(f'{rate_hz:.3f}'), q('Hz')],
        [q('Duration'), q(f'{duration:.3f}'), q('s')],
        [q('Range'), q('entire outing')],
        [q('Beacon Markers'), q(' '.join(f'{m:.3f}' for m in marks))],
        [], [],
        [q('Time')] + [q(c.motec[0]) for c, _ in resampled],
        [q('s')] + [q(c.motec[1]) for c, _ in resampled],
        [], [],
    ]
    data = np.column_stack([times] + [v for _, v in resampled])
    with open(path, 'w', encoding='utf-8', newline='\r\n') as f:
        for row in head:
            f.write(','.join(row) + '\n')
        for row in data:
            f.write(','.join(q(f'{v:.6g}') for v in row) + '\n')
