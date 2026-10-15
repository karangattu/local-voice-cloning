"""Spell out numbers, symbols and abbreviations in English scripts.

The voice models read "$5.99" or "3:30 PM" inconsistently. Some skip the
symbol, some read digits one by one. Words such as "five dollars and
ninety-nine cents" are read the same way every time."""

import re

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = ((10**12, "trillion"), (10**9, "billion"), (10**6, "million"), (1000, "thousand"))
_ORDINAL_WORDS = {
    "one": "first",
    "two": "second",
    "three": "third",
    "five": "fifth",
    "eight": "eighth",
    "nine": "ninth",
    "twelve": "twelfth",
}

_ABBREVIATIONS = {
    "Mr.": "Mister",
    "Mrs.": "Missus",
    "Ms.": "Miz",
    "Dr.": "Doctor",
    "Prof.": "Professor",
    "Jr.": "Junior",
    "Sr.": "Senior",
    "vs.": "versus",
    "e.g.": "for example",
    "i.e.": "that is",
    "etc.": "et cetera",
    "approx.": "approximately",
}
_ABBREVIATION_PATTERN = re.compile(
    r"(?<![\w.])(" + "|".join(re.escape(key) for key in _ABBREVIATIONS) + r")(?=\s|$|[,;:!?)])",
    re.IGNORECASE,
)
_CURRENCIES = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "€": ("euro", "euros", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
}
_NUMBER = r"\d{1,3}(?:,\d{3})+|\d+"
_CURRENCY_PATTERN = re.compile(
    r"([$€£])\s?(" + _NUMBER + r")(?:\.(\d{1,2}))?(?:\s?(thousand|million|billion|trillion)\b)?"
)
_TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?:\s?([ap])\.?m\.?\b)?", re.IGNORECASE)
# "9.30am" is a time only with am or pm after it; "9.30" alone is a decimal.
_DOT_TIME_PATTERN = re.compile(r"\b([01]?\d|2[0-3])\.([0-5]\d)(?=\s?[ap]\.?m\b)", re.IGNORECASE)
_PERCENT_PATTERN = re.compile(r"(" + _NUMBER + r")(?:\.(\d+))?\s?%")
_ORDINAL_PATTERN = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_RANGE_PATTERN = re.compile(r"\b(\d+)\s?[-–]\s?(\d+)\b")
_DECIMAL_PATTERN = re.compile(r"(?<![\d.])(" + _NUMBER + r")\.(\d+)\b")
# Digits inside a word such as "mp3" or "B2B" stay as they are.
_INTEGER_PATTERN = re.compile(r"(?<![\w.])(" + _NUMBER + r")(?![\d.]*\d|[^\W\d])")
_HASH_NUMBER_PATTERN = re.compile(r"#\s?(\d+)")


def number_to_words(number: int) -> str:
    """Cardinal words for a non-negative integer, e.g. 1203 -> one thousand two hundred three."""
    if number < 20:
        return _ONES[number]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        return f"{_ONES[hundreds]} hundred" + (f" {number_to_words(rest)}" if rest else "")
    for scale, name in _SCALES:
        if number >= scale:
            head, rest = divmod(number, scale)
            return f"{number_to_words(head)} {name}" + (f" {number_to_words(rest)}" if rest else "")
    raise AssertionError("unreachable")


def ordinal_to_words(number: int) -> str:
    """Ordinal words, e.g. 23 -> twenty-third."""
    words = number_to_words(number)
    head, sep, last = words.rpartition("-" if "-" in words.split(" ")[-1] else " ")
    if last in _ORDINAL_WORDS:
        last = _ORDINAL_WORDS[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return head + sep + last


def year_to_words(year: int) -> str:
    """Words for a year as people say it, e.g. 1999 -> nineteen ninety-nine."""
    if 2000 <= year <= 2009:
        return number_to_words(year)
    high, low = divmod(year, 100)
    if low == 0:
        return f"{number_to_words(high)} hundred"
    low_words = f"oh {_ONES[low]}" if low < 10 else number_to_words(low)
    return f"{number_to_words(high)} {low_words}"


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _cardinal(text: str) -> str:
    value = _int(text)
    # A bare four-digit number in this range is almost always a year.
    if "," not in text and len(text) == 4 and 1100 <= value <= 2099:
        return year_to_words(value)
    return number_to_words(value)


def _digits(text: str) -> str:
    return " ".join(_ONES[int(digit)] for digit in text)


def _currency(match: re.Match) -> str:
    one, many, cent, cents = _CURRENCIES[match.group(1)]
    whole, fraction, scale = _int(match.group(2)), match.group(3), match.group(4)
    if scale:
        amount = number_to_words(whole) + (f" point {_digits(fraction)}" if fraction else "")
        return f"{amount} {scale.lower()} {many}"
    words = f"{number_to_words(whole)} {one if whole == 1 else many}"
    if fraction:
        sub = int(fraction.ljust(2, "0"))
        if sub:
            words += f" and {number_to_words(sub)} {cent if sub == 1 else cents}"
    return words


def _time(match: re.Match) -> str:
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3)
    if minute == 0:
        words = f"{number_to_words(hour)}" + ("" if meridiem else " o'clock")
    elif minute < 10:
        words = f"{number_to_words(hour)} oh {_ONES[minute]}"
    else:
        words = f"{number_to_words(hour)} {number_to_words(minute)}"
    if meridiem:
        words += " AM" if meridiem.lower() == "a" else " PM"
    return words


def _percent(match: re.Match) -> str:
    fraction = f" point {_digits(match.group(2))}" if match.group(2) else ""
    return f"{number_to_words(_int(match.group(1)))}{fraction} percent"


def normalize_english_text(text: str) -> str:
    """Rewrite symbols, numbers and common abbreviations as spoken English words."""
    text = _ABBREVIATION_PATTERN.sub(
        lambda match: _ABBREVIATIONS[
            next(key for key in _ABBREVIATIONS if key.lower() == match.group(1).lower())
        ],
        text,
    )
    text = _CURRENCY_PATTERN.sub(_currency, text)
    text = _DOT_TIME_PATTERN.sub(r"\1:\2", text)
    text = _TIME_PATTERN.sub(_time, text)
    text = _PERCENT_PATTERN.sub(_percent, text)
    text = _ORDINAL_PATTERN.sub(lambda match: ordinal_to_words(int(match.group(1))), text)
    text = _HASH_NUMBER_PATTERN.sub(lambda match: f"number {match.group(1)}", text)
    text = _RANGE_PATTERN.sub(
        lambda match: f"{_cardinal(match.group(1))} to {_cardinal(match.group(2))}", text
    )
    text = _DECIMAL_PATTERN.sub(
        lambda match: f"{number_to_words(_int(match.group(1)))} point {_digits(match.group(2))}",
        text,
    )
    text = _INTEGER_PATTERN.sub(lambda match: _cardinal(match.group(1)), text)
    return re.sub(r"\s*&\s*", " and ", text)


def should_normalize(text: str, language: str, engine: str) -> bool:
    """English rules apply to English scripts only. With auto-detect, a script
    with no non-ASCII letters is treated as English."""
    if engine == "chatterbox" or language.lower() == "english":
        return True
    return language == "auto" and not re.search(r"[^\x00-\x7f]", text)
