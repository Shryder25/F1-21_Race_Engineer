"""How a real F1 race engineer says things on the radio.

Engineers don't read out "1 minute 32.456 seconds". They say "one thirty-two four",
talk about time in tenths ("two tenths up"), and keep calls short and calm. These
helpers turn raw numbers into that kind of spoken phrase, and ``Phrasebook`` rotates
between a few wordings so the same call doesn't sound like a recording every lap.

Numbers are written out as words so every text-to-speech engine says them the same way.
"""

from __future__ import annotations

_ONES = (
    'zero',
    'one',
    'two',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'eight',
    'nine',
    'ten',
    'eleven',
    'twelve',
    'thirteen',
    'fourteen',
    'fifteen',
    'sixteen',
    'seventeen',
    'eighteen',
    'nineteen',
)
_TENS = ('', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety')
_ORDINAL_SECTOR = {1: 'one', 2: 'two', 3: 'three'}


def number_words(n: int) -> str:
    """0-999 as spoken English ("one hundred and five" style kept short: "one oh five")."""
    n = int(n)
    if n < 0:
        return 'minus ' + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ('' if ones == 0 else '-' + _ONES[ones])
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        if rest == 0:
            return f'{_ONES[hundreds]} hundred'
        return f'{_ONES[hundreds]} {"oh " + _ONES[rest] if rest < 10 else number_words(rest)}'
    return str(n)


def _two_digit_seconds(seconds: int) -> str:
    # "oh five" for 1:05, like engineers say it
    return f'oh {_ONES[seconds]}' if seconds < 10 else number_words(seconds)


def lap_time(ms: int) -> str:
    """92456 ms -> 'one thirty-two four' (minutes, seconds, tenths)."""
    tenths_total = round(ms / 100)
    minutes, rest = divmod(tenths_total, 600)
    seconds, tenths = divmod(rest, 10)
    if minutes == 0:
        return f'{number_words(seconds)} {_ONES[tenths]}'
    return f'{number_words(minutes)} {_two_digit_seconds(seconds)} {_ONES[tenths]}'


def sector_time(ms: int) -> str:
    """31456 ms -> 'thirty-one five'."""
    return lap_time(ms)


def seconds_value(seconds: float) -> str:
    """1.27 -> 'one point three', 0.8 -> 'zero point eight'."""
    tenths = round(abs(seconds) * 10)
    whole, frac = divmod(tenths, 10)
    return f'{number_words(whole)} point {_ONES[frac]}'


def time_amount(seconds: float) -> str:
    """How much time, the way an engineer says it: 'a tenth', 'three tenths', 'one point two seconds'."""
    s = abs(seconds)
    tenths = round(s * 10)
    if tenths == 0:
        return 'less than a tenth'
    if tenths == 1:
        return 'a tenth'
    if tenths == 5:
        return 'half a second'
    if tenths < 10:
        return f'{_ONES[tenths]} tenths'
    if tenths == 10:
        return 'a second'
    return f'{seconds_value(s)} seconds'


def gap_value(seconds: float) -> str:
    """Gap in seconds as spoken: under a second in tenths ('eight tenths'), else 'one point four'."""
    s = abs(seconds)
    if round(s * 10) < 10:
        return time_amount(s)
    return seconds_value(s)


def sector_name(sector: int) -> str:
    return f'sector {_ORDINAL_SECTOR.get(sector, number_words(sector))}'


class Phrasebook:
    """Rotate through wordings per topic so repeated calls don't sound canned.

    Rotation (not random) keeps it predictable for tests and never repeats the
    same wording twice in a row.
    """

    def __init__(self):
        self._next: dict[str, int] = {}

    def pick(self, topic: str, options) -> str:
        options = list(options)
        index = self._next.get(topic, 0) % len(options)
        self._next[topic] = index + 1
        return options[index]

    def reset(self):
        self._next.clear()
