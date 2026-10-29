"""Build a dataset of real F1 engineer radio lines, each paired with the race situation it was said in.

Three steps, each one resumable (files already on disk are skipped):

    python tools/openf1_radio.py fetch      --years 2023 2024 2025 --out data/openf1
    python tools/openf1_radio.py transcribe --out data/openf1 --model small.en
    python tools/openf1_radio.py build      --out data/openf1

fetch       Downloads every race's team radio clips (MP3) and the timing data around them from
            OpenF1 (https://openf1.org, free for 2023 onward): laps, stints, pit stops, gaps,
            positions, race control messages, weather and driver names.
transcribe  Turns each clip into text with faster-whisper (fast on a GPU, so run it in Colab:
            tools/training/openf1_radio_colab.ipynb). Keeps the timed segments.
build       For each clip, rebuilds what the pit wall knew at that moment (lap, position, gaps
            and their trend, tyre and its age, stops, rivals who just pitted, flags, rain), picks
            the engineer's part of the clip, tags the kind of call and writes JSONL:
                data/openf1/examples_train.jsonl, data/openf1/examples_test.jsonl
            The last few races (--holdout) go to the test file only, never to training.

Speaker split is a heuristic: a clip mixes driver and engineer, and Whisper doesn't say who is
talking. Segments that sound like an engineer (second person, "box", "gap", "copy that, ...")
are kept and marked with `speaker_score`; low scores are worth a manual look.

The radio audio belongs to Formula 1. Use the dataset for your own training only and don't
publish the transcripts or a model trained on them. This sandbox can't reach OpenF1, so the
fetch step runs on your PC or in Colab.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

API = 'https://api.openf1.org/v1'
# Free tier: 3 requests/s and 30/min. Stay under both. Audio comes from F1's own file server,
# not the API, so clips only get a short gap.
REQUEST_GAP_S = 2.1
AUDIO_GAP_S = 0.2
# After this many clips in a row are refused (e.g. HTTP 403), stop asking for audio this run.
MAX_REFUSED_IN_A_ROW = 25
ENDPOINTS = (
    'drivers',
    'laps',
    'stints',
    'pit',
    'intervals',
    'position',
    'race_control',
    'weather',
    'team_radio',
)

Row = dict[str, Any]


# ── fetching ────────────────────────────────────────────────────────────


def ascii_url(url: str) -> str:
    """Percent-encode non-ASCII characters (e.g. 'São Paulo' in a clip's path); urllib can't send them."""
    parts = urllib.parse.urlsplit(url)
    safe = "/%:@!$&'()*+,;=-._~"
    return urllib.parse.urlunsplit(
        (
            parts.scheme,
            parts.netloc.encode('idna').decode('ascii') if not parts.netloc.isascii() else parts.netloc,
            urllib.parse.quote(parts.path, safe=safe),
            urllib.parse.quote(parts.query, safe=safe + '?'),
            urllib.parse.quote(parts.fragment, safe=safe),
        )
    )


class Fetcher:
    """Polite OpenF1 client: one request every REQUEST_GAP_S, retries on 429/5xx."""

    def __init__(self, gap_s: float = REQUEST_GAP_S, opener: Callable[..., Any] = urllib.request.urlopen):
        self.gap_s = gap_s
        self.opener = opener
        self._last = 0.0

    def _wait(self, gap_s: float) -> None:
        delay = self._last + gap_s - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        self._last = time.monotonic()

    def get_bytes(self, url: str, tries: int = 5, gap_s: float | None = None) -> bytes:
        url = ascii_url(url)
        for attempt in range(tries):
            self._wait(self.gap_s if gap_s is None else gap_s)
            try:
                with self.opener(url, timeout=60) as response:
                    data: bytes = response.read()
                    return data
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return b''
                if error.code not in (429, 500, 502, 503, 504) or attempt == tries - 1:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == tries - 1:
                    raise
            time.sleep(2**attempt * 5)
        raise RuntimeError(f'unreachable: {url}')

    def get_json(self, endpoint: str, **params: Any) -> list[Row]:
        url = f'{API}/{endpoint}?{urllib.parse.urlencode(params)}'
        rows = json.loads(self.get_bytes(url) or b'[]')
        return rows if isinstance(rows, list) else []


def race_sessions(fetcher: Fetcher, years: Iterable[int]) -> list[Row]:
    sessions: list[Row] = []
    for year in years:
        sessions += fetcher.get_json('sessions', year=year, session_name='Race')
    return sorted(sessions, key=lambda s: s.get('date_start') or '')


def looks_like_audio(data: bytes) -> bool:
    """True for MP3 (ID3 tag or MPEG frame sync), M4A/MP4, Ogg or WAV data.

    A server can answer 200 OK with an HTML or XML error page; that must not be saved as a clip.
    """
    if data[:3] == b'ID3' or data[:4] in (b'OggS', b'RIFF', b'fLaC') or data[4:8] == b'ftyp':
        return True
    return len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0


def _preview(data: bytes) -> str:
    return data[:80].decode('utf-8', 'replace').replace('\n', ' ').strip()


def clip_filename(clip: Row) -> str:
    stamp = re.sub(r'[^0-9]', '', str(clip.get('date', '')))[:17]
    return f'{clip.get("driver_number")}_{stamp}.mp3'


def fetch(out: Path, years: list[int], fetcher: Fetcher | None = None, audio: bool = True) -> int:
    fetcher = fetcher or Fetcher()
    out.mkdir(parents=True, exist_ok=True)
    sessions = race_sessions(fetcher, years)
    (out / 'sessions.json').write_text(json.dumps(sessions, indent=1))
    clips = downloaded = refused = refused_in_a_row = 0
    for session in sessions:
        key = session['session_key']
        folder = out / str(key)
        folder.mkdir(exist_ok=True)
        print(f'{session.get("year")} {session.get("circuit_short_name")} ({key})', flush=True)
        for endpoint in ENDPOINTS:
            path = folder / f'{endpoint}.json'
            if not path.exists():
                path.write_text(json.dumps(fetcher.get_json(endpoint, session_key=key)))
        radio = json.loads((folder / 'team_radio.json').read_text())
        clips += len(radio)
        (folder / 'radio').mkdir(exist_ok=True)
        for clip in radio:
            target = folder / 'radio' / clip_filename(clip)
            url = clip.get('recording_url')
            if target.exists() and not looks_like_audio(target.read_bytes()[:16]):
                target.unlink()  # an error page saved by an older version: fetch it again
            if not url or target.exists() or not audio:
                continue
            try:
                data = fetcher.get_bytes(url, gap_s=min(AUDIO_GAP_S, fetcher.gap_s))
            except (urllib.error.URLError, TimeoutError, ValueError) as error:  # HTTPError is a URLError
                refused += 1
                refused_in_a_row += 1
                print(f'  skipped clip {target.name}: {error}', flush=True)
                if refused_in_a_row >= MAX_REFUSED_IN_A_ROW:
                    print(
                        f'{refused_in_a_row} clips in a row were refused by the audio server. '
                        'Carrying on with timing data only; run fetch again later to pick up '
                        'the missing clips.',
                        flush=True,
                    )
                    audio = False
                continue
            if data and not looks_like_audio(data):
                refused += 1
                refused_in_a_row += 1
                print(f'  skipped clip {target.name}: server sent no audio ({_preview(data)!r})', flush=True)
                if refused_in_a_row >= MAX_REFUSED_IN_A_ROW:
                    print(
                        f'{refused_in_a_row} clips in a row came back without audio. '
                        'Carrying on with timing data only.',
                        flush=True,
                    )
                    audio = False
                continue
            refused_in_a_row = 0
            if data:  # a clip F1 has taken down comes back 404: skip it
                target.write_bytes(data)
                downloaded += 1
    print(f'{len(sessions)} races, {clips} radio clips: {downloaded} downloaded now, {refused} refused')
    return clips


# ── transcription ───────────────────────────────────────────────────────


SAMPLE_RATE = 16000  # what Whisper expects


def decode_audio_pyav(path: Path) -> Any:
    """Same as decode_audio, using the PyAV library that faster-whisper installs (no ffmpeg needed).

    Calls only PyAV's basic API, so it works whichever PyAV version is installed.
    """
    import av
    import numpy as np

    chunks = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format='s16', layout='mono', rate=SAMPLE_RATE)
        for frame in container.decode(audio=0):
            chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(frame)]
        chunks += [out.to_ndarray().reshape(-1) for out in resampler.resample(None)]
    if not chunks:
        raise ValueError('no audio in file')
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def decode_audio(path: Path, ffmpeg: str = 'ffmpeg') -> Any:
    """The clip as 16 kHz mono float32 samples, decoded by ffmpeg.

    faster-whisper's own decoder (PyAV) breaks when Colab has a different PyAV version
    installed ("open() got an unexpected keyword argument 'metadata_errors'"), so the
    audio is decoded here and Whisper gets the samples.
    """
    import numpy as np

    result = subprocess.run(
        [
            ffmpeg,
            '-nostdin',
            '-loglevel',
            'error',
            '-i',
            str(path),
            '-f',
            's16le',
            '-ac',
            '1',
            '-ar',
            str(SAMPLE_RATE),
            '-',
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32) / 32768.0


def load_whisper(whisper_model: Any, model_name: str, device: str) -> Any:
    """Whisper on the GPU when it starts, else on the CPU (slower, same result).

    On Colab the GPU build of CTranslate2 can be newer than the machine's NVIDIA driver
    ("CUDA driver version is insufficient for CUDA runtime version").
    """
    if device == 'cpu':
        return whisper_model(model_name, device='cpu', compute_type='int8')
    try:
        return whisper_model(model_name, device=device, compute_type='auto')
    except (RuntimeError, ValueError) as error:
        if 'cuda' not in str(error).lower():
            raise
        print(f'GPU could not start ({error}). Transcribing on the CPU instead; this is slower.', flush=True)
        return whisper_model(model_name, device='cpu', compute_type='int8')


def transcribe(out: Path, model_name: str = 'small.en', device: str = 'auto') -> int:
    """Writes <session>/transcripts.json: {filename: [{start, end, text}, ...]}."""
    from faster_whisper import WhisperModel  # optional: only this step needs it

    ffmpeg = shutil.which('ffmpeg')
    model = load_whisper(WhisperModel, model_name, device)
    done = failed = 0
    for folder in sorted(p for p in out.iterdir() if (p / 'radio').is_dir()):
        target = folder / 'transcripts.json'
        transcripts: dict[str, list[Row]] = json.loads(target.read_text()) if target.exists() else {}
        for mp3 in sorted((folder / 'radio').glob('*.mp3')):
            if mp3.name in transcripts:
                continue
            head = mp3.read_bytes()[:80]
            if not looks_like_audio(head):
                failed += 1
                print(
                    f'  not audio, skipped {mp3.name} (starts {_preview(head)!r}); run fetch again',
                    flush=True,
                )
                continue
            try:
                audio = decode_audio(mp3, ffmpeg) if ffmpeg else decode_audio_pyav(mp3)
                segments, _ = model.transcribe(audio, language='en', vad_filter=True, beam_size=5)
                transcripts[mp3.name] = [
                    {'start': round(s.start, 2), 'end': round(s.end, 2), 'text': s.text.strip()}
                    for s in segments
                ]
            except (
                subprocess.CalledProcessError,
                ValueError,
                RuntimeError,
                OSError,
            ) as error:  # a damaged clip
                failed += 1
                reason = error
                if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                    reason = error.stderr.decode('utf-8', 'replace').strip().splitlines()[-1]
                print(f'  could not read {mp3.name}: {reason}', flush=True)
                continue
            done += 1
            if done % 50 == 0:
                target.write_text(json.dumps(transcripts))
                print(f'{done} clips transcribed', flush=True)
        target.write_text(json.dumps(transcripts))
    print(f'{done} clips transcribed' + (f', {failed} unreadable clips skipped' if failed else ''))
    return done


# ── situation at a moment ───────────────────────────────────────────────


def parse_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00'))
    except ValueError:
        return None


class Timeline:
    """Rows of one kind sorted by time, with 'last row at or before t' lookups."""

    def __init__(self, rows: Iterable[Row], key: str = 'date'):
        dated = [(t, row) for row in rows if (t := parse_time(row.get(key))) is not None]
        dated.sort(key=lambda pair: pair[0])
        self.times = [t for t, _ in dated]
        self.rows = [row for _, row in dated]

    def at(self, t: datetime) -> Row | None:
        i = bisect.bisect_right(self.times, t)
        return self.rows[i - 1] if i else None

    def between(self, start: datetime, end: datetime) -> list[Row]:
        lo = bisect.bisect_left(self.times, start)
        hi = bisect.bisect_right(self.times, end)
        return self.rows[lo:hi]


def _by_driver(rows: list[Row], key: str = 'date') -> dict[int, Timeline]:
    grouped: dict[int, list[Row]] = {}
    for row in rows:
        if row.get('driver_number') is not None:
            grouped.setdefault(int(row['driver_number']), []).append(row)
    return {number: Timeline(group, key) for number, group in grouped.items()}


COMPOUND_NAMES = {'SOFT': 'soft', 'MEDIUM': 'medium', 'HARD': 'hard', 'INTERMEDIATE': 'inters', 'WET': 'wets'}


@dataclass
class RaceData:
    """Everything OpenF1 has for one race, indexed for lookups by time."""

    session: Row
    names: dict[int, str]
    laps: dict[int, Timeline]
    stints: dict[int, list[Row]]
    pits: Timeline
    intervals: dict[int, Timeline]
    positions: dict[int, Timeline]
    race_control: Timeline
    weather: Timeline
    total_laps: int
    radio: list[Row] = field(default_factory=list)

    @classmethod
    def from_rows(cls, session: Row, rows: dict[str, list[Row]]) -> RaceData:
        names = {
            int(d['driver_number']): str(d.get('name_acronym') or d.get('last_name') or d['driver_number'])
            for d in rows.get('drivers', [])
            if d.get('driver_number') is not None
        }
        stints: dict[int, list[Row]] = {}
        for stint in rows.get('stints', []):
            if stint.get('driver_number') is not None:
                stints.setdefault(int(stint['driver_number']), []).append(stint)
        laps = rows.get('laps', [])
        total = max((int(lap.get('lap_number') or 0) for lap in laps), default=0)
        return cls(
            session=session,
            names=names,
            laps=_by_driver(laps, 'date_start'),
            stints=stints,
            pits=Timeline(rows.get('pit', [])),
            intervals=_by_driver(rows.get('intervals', [])),
            positions=_by_driver(rows.get('position', [])),
            race_control=Timeline(rows.get('race_control', [])),
            weather=Timeline(rows.get('weather', [])),
            total_laps=total,
            radio=rows.get('team_radio', []),
        )

    @classmethod
    def load(cls, folder: Path, session: Row) -> RaceData:
        rows = {}
        for endpoint in ENDPOINTS:
            path = folder / f'{endpoint}.json'
            rows[endpoint] = json.loads(path.read_text()) if path.exists() else []
        return cls.from_rows(session, rows)

    def name(self, number: int) -> str:
        return self.names.get(number, f'car {number}')

    def position_at(self, number: int, t: datetime) -> int | None:
        timeline = self.positions.get(number)
        row = timeline.at(t) if timeline else None
        return int(row['position']) if row and row.get('position') else None

    def car_at_position(self, position: int, t: datetime) -> int | None:
        for number in self.positions:
            if self.position_at(number, t) == position:
                return number
        return None

    def interval_at(self, number: int, t: datetime) -> float | None:
        timeline = self.intervals.get(number)
        row = timeline.at(t) if timeline else None
        value = row.get('interval') if row else None
        return float(value) if isinstance(value, (int, float)) else None

    def lap_at(self, number: int, t: datetime) -> Row | None:
        timeline = self.laps.get(number)
        return timeline.at(t) if timeline else None

    def stint_for_lap(self, number: int, lap: int) -> Row | None:
        for stint in self.stints.get(number, []):
            start = int(stint.get('lap_start') or 0)
            end = stint.get('lap_end')
            if start <= lap and (end is None or lap <= int(end)):
                return stint
        return None


def _trend(now: float | None, before: float | None) -> str:
    if now is None or before is None:
        return ''
    change = now - before
    if abs(change) < 0.15:
        return ', steady'
    return f', {"opening" if change > 0 else "closing"} {abs(change):.1f}s in the last lap'


def _flag_state(race: RaceData, t: datetime) -> str:
    """Safety car / VSC / red flag still in force at t, from race control messages."""
    state = ''
    for row in race.race_control.between(t - timedelta(hours=3), t):
        message = str(row.get('message') or '').upper()
        flag = str(row.get('flag') or '').upper()
        if 'VIRTUAL SAFETY CAR DEPLOYED' in message:
            state = 'VSC deployed'
        elif 'VIRTUAL SAFETY CAR ENDING' in message or 'VSC ENDING' in message:
            state = 'VSC ending'
        elif 'SAFETY CAR DEPLOYED' in message:
            state = 'Safety car deployed'
        elif 'SAFETY CAR IN THIS LAP' in message:
            state = 'Safety car in this lap'
        elif flag == 'RED':
            state = 'Red flag'
        elif flag in ('GREEN', 'CLEAR') and row.get('scope') == 'Track':
            state = ''
    return state


def situation(race: RaceData, number: int, t: datetime) -> str:
    """What the pit wall knew about car `number` at time t, in the engineer prompt's style."""
    parts: list[str] = []
    lap_row = race.lap_at(number, t)
    lap = int(lap_row['lap_number']) if lap_row and lap_row.get('lap_number') else None
    if lap:
        parts.append(f'Lap {lap}/{race.total_laps}.' if race.total_laps else f'Lap {lap}.')
    position = race.position_at(number, t)
    if position:
        parts.append(f'P{position}.')
    if lap:
        stint = race.stint_for_lap(number, lap)
        if stint and stint.get('compound'):
            compound = COMPOUND_NAMES.get(str(stint['compound']).upper(), str(stint['compound']).lower())
            age = int(stint.get('tyre_age_at_start') or 0) + lap - int(stint.get('lap_start') or lap)
            parts.append(f'{compound.capitalize()}, {age} laps old.')
    stops = sum(
        1 for row in race.pits.between(t - timedelta(hours=4), t) if row.get('driver_number') == number
    )
    parts.append(f'{stops} stop{"s" if stops != 1 else ""} so far.')
    lap_before = t - timedelta(seconds=90)
    if position and position > 1:
        ahead = race.car_at_position(position - 1, t)
        gap = race.interval_at(number, t)
        if ahead is not None and gap is not None:
            trend = _trend(gap, race.interval_at(number, lap_before))
            parts.append(f'Ahead {race.name(ahead)} +{gap:.1f}s{trend}.')
    if position:
        behind = race.car_at_position(position + 1, t)
        gap = race.interval_at(behind, t) if behind is not None else None
        if behind is not None and gap is not None:
            trend = _trend(gap, race.interval_at(behind, lap_before))
            parts.append(f'Behind {race.name(behind)} -{gap:.1f}s{trend}.')
    if lap_row and isinstance(lap_row.get('lap_duration'), (int, float)):
        previous = race.lap_at(number, t - timedelta(seconds=float(lap_row['lap_duration']) + 1))
        if previous and isinstance(previous.get('lap_duration'), (int, float)):
            parts.append(f'Last lap {float(previous["lap_duration"]):.3f}s.')
    nearby = set()
    if position:
        nearby = {race.car_at_position(p, t) for p in range(max(1, position - 3), position + 4)} - {
            None,
            number,
        }
    recent = [
        row for row in race.pits.between(t - timedelta(seconds=120), t) if row.get('driver_number') in nearby
    ]
    for row in recent:
        parts.append(f'{race.name(int(row["driver_number"]))} just pitted.')
    flags = _flag_state(race, t)
    if flags:
        parts.append(f'{flags}.')
    weather = race.weather.at(t)
    if weather:
        if weather.get('rainfall'):
            parts.append('Raining.')
        if isinstance(weather.get('track_temperature'), (int, float)):
            parts.append(f'Track {float(weather["track_temperature"]):.0f}C.')
    return ' '.join(parts)


# ── engineer line, tags, filters ────────────────────────────────────────

_ENGINEER_CUES = re.compile(
    r"\b(you|your|you're|we|we're|our|box|pit|gap|push|plan [a-z]|mode|copy|understood|"
    r"tyres?|lap(s)?|behind|ahead|delta|target|keep|stay out|let's|need to)\b",
    re.IGNORECASE,
)
_DRIVER_CUES = re.compile(
    r"\b(i'm|i am|i have|i can't|my tyres|my car|i think|what's the|where am i|should i|"
    r"no grip|how many laps|f\*+|fuck|shit)\b",
    re.IGNORECASE,
)


def speaker_score(text: str) -> float:
    """0..1, how much a segment sounds like the engineer rather than the driver."""
    engineer = len(_ENGINEER_CUES.findall(text))
    driver = len(_DRIVER_CUES.findall(text))
    if engineer + driver == 0:
        return 0.5
    return engineer / (engineer + driver)


TAGS: dict[str, re.Pattern[str]] = {
    'pit_call': re.compile(r'\b(box|pit ?stop|stay out|plan [a-z]|pit this lap|in this lap)\b', re.I),
    'gap': re.compile(r'\b(gap|interval|seconds? (ahead|behind)|tenths?|drs)\b', re.I),
    'tyres': re.compile(
        r'\b(tyres?|tires?|deg|degradation|wear|graining|softs?|mediums?|hards?|inters?)\b', re.I
    ),
    'pace': re.compile(r'\b(pace|push|lap time|purple|fastest|target|delta|quicker|slower)\b', re.I),
    'threat': re.compile(r'\b(under ?cut|over ?cut|defend|attack|coming|closing|behind you|battle)\b', re.I),
    'flags': re.compile(r'\b(safety car|vsc|yellow|red flag|green flag|sc)\b', re.I),
    'weather': re.compile(r'\b(rain|wet|drying|dry line|spits?|drizzle)\b', re.I),
    'car': re.compile(r'\b(mode|strat|engine|brake|battery|ers|fuel|lift and coast|lico|diff)\b', re.I),
}
_ACK_ONLY = re.compile(r"^(ok(ay)?|copy( that)?|yes|yeah|no|understood|thank you|thanks|roger)[.!]*$", re.I)


def tags_for(text: str) -> list[str]:
    return [name for name, pattern in TAGS.items() if pattern.search(text)]


def engineer_text(segments: list[Row], threshold: float = 0.55) -> tuple[str, float]:
    """The engineer-sounding segments of a clip joined, and their average score."""
    kept = [(s['text'], speaker_score(s['text'])) for s in segments if s.get('text')]
    kept = [(text, score) for text, score in kept if score >= threshold]
    if not kept:
        return '', 0.0
    return ' '.join(text for text, _ in kept), sum(score for _, score in kept) / len(kept)


def keep_line(text: str, max_words: int = 35) -> bool:
    words = text.split()
    if len(words) < 2 or len(words) > max_words:
        return False
    return not _ACK_ONLY.match(text.strip())


# ── build ───────────────────────────────────────────────────────────────


def examples_for_race(race: RaceData, transcripts: dict[str, list[Row]]) -> list[Row]:
    out: list[Row] = []
    for clip in race.radio:
        t = parse_time(clip.get('date'))
        segments = transcripts.get(clip_filename(clip))
        if t is None or not segments or clip.get('driver_number') is None:
            continue
        text, score = engineer_text(segments)
        if not keep_line(text):
            continue
        number = int(clip['driver_number'])
        lap_row = race.lap_at(number, t)
        out.append(
            {
                'session_key': race.session.get('session_key'),
                'year': race.session.get('year'),
                'circuit': race.session.get('circuit_short_name'),
                'driver': race.name(number),
                'date': clip.get('date'),
                'lap': lap_row.get('lap_number') if lap_row else None,
                'situation': situation(race, number, t),
                'engineer': text,
                'tags': tags_for(text),
                'speaker_score': round(score, 2),
                'audio_url': clip.get('recording_url'),
            }
        )
    return out


def build(out: Path, holdout: int = 3) -> tuple[int, int]:
    sessions = json.loads((out / 'sessions.json').read_text())
    sessions = [s for s in sessions if (out / str(s['session_key']) / 'transcripts.json').exists()]
    test_keys = {s['session_key'] for s in sessions[-holdout:]} if holdout else set()
    counts = [0, 0]
    with (
        (out / 'examples_train.jsonl').open('w') as train,
        (out / 'examples_test.jsonl').open('w') as test,
    ):
        for session in sessions:
            folder = out / str(session['session_key'])
            race = RaceData.load(folder, session)
            transcripts = json.loads((folder / 'transcripts.json').read_text())
            is_test = session['session_key'] in test_keys
            for example in examples_for_race(race, transcripts):
                (test if is_test else train).write(json.dumps(example) + '\n')
                counts[is_test] += 1
    print(f'{counts[0]} training examples, {counts[1]} test examples ({len(test_keys)} held-out races)')
    return counts[0], counts[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p_fetch = sub.add_parser('fetch', help='download races, timing data and radio clips from OpenF1')
    p_fetch.add_argument('--years', type=int, nargs='+', default=[2023, 2024, 2025])
    p_fetch.add_argument('--no-audio', action='store_true', help='timing data and clip list only')
    p_tr = sub.add_parser('transcribe', help='speech to text for every clip (faster-whisper)')
    p_tr.add_argument('--model', default='small.en')
    p_tr.add_argument('--device', default='auto')
    p_build = sub.add_parser('build', help='pair engineer lines with the race situation')
    p_build.add_argument('--holdout', type=int, default=3, help='latest races kept for testing only')
    for p in (p_fetch, p_tr, p_build):
        p.add_argument('--out', type=Path, default=Path('data/openf1'))
    args = parser.parse_args(argv)
    if args.command == 'fetch':
        fetch(args.out, args.years, audio=not args.no_audio)
    elif args.command == 'transcribe':
        transcribe(args.out, args.model, args.device)
    else:
        build(args.out, args.holdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
