"""Numbers recovered from narration text.

The script writer spells numbers out because the TTS needs them that way --
"twenty five states", "eighteen seventy seven" -- but the art director needs
them back as values to choose a shot: a sentence carrying two comparable
quantities wants a measure_compare, one carrying a small count wants an
icon_multiply, one carrying a year wants a date card.

Years are their own problem. "Eighteen seventy seven" is not 18 + 77 and it is
not 1877 by any general rule; it is two two-digit groups read as a date, and
English only does that for years. It is handled as its own case rather than
bent into the cardinal parser.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
        "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000, "billion": 1_000_000_000}
WORDS = set(UNITS) | set(TENS) | set(SCALES) | {"and", "point"}

#: Units worth showing on a card. Bare numbers are common; units are the signal
#: that two numbers in a sentence are actually comparable.
#: Maps a spoken unit to (singular, plural). A card reading "1 ACRES" looks
#: like a bug to a viewer even though the number is right.
UNIT_WORDS = {
    "kilogram": ("KG", "KG"), "kilograms": ("KG", "KG"),
    "kilo": ("KG", "KG"), "kilos": ("KG", "KG"),
    "pound": ("LB", "LB"), "pounds": ("LB", "LB"),
    "ton": ("TON", "TONS"), "tons": ("TON", "TONS"),
    "acre": ("ACRE", "ACRES"), "acres": ("ACRE", "ACRES"),
    "mile": ("MILE", "MILES"), "miles": ("MILE", "MILES"),
    "foot": ("FT", "FT"), "feet": ("FT", "FT"),
    "percent": ("%", "%"), "dollar": ("$", "$"), "dollars": ("$", "$"),
    "year": ("YEAR", "YEARS"), "years": ("YEAR", "YEARS"),
    "state": ("STATE", "STATES"), "states": ("STATE", "STATES"),
    "day": ("DAY", "DAYS"), "days": ("DAY", "DAYS"),
    "hour": ("HOUR", "HOURS"), "hours": ("HOUR", "HOURS"),
}


def unit_for(word: str | None, value: float) -> str | None:
    """The unit as it should read beside `value`, or None if not a unit."""
    pair = UNIT_WORDS.get((word or "").lower())
    if pair is None:
        return None
    return pair[0] if abs(value) == 1 else pair[1]


@dataclass(frozen=True)
class Number:
    value: float
    unit: str | None
    surface: str
    is_year: bool = False

    @property
    def display(self) -> str:
        """How it should read on a card: no trailing .0, thousands separated."""
        if self.is_year or float(self.value).is_integer():
            n = int(self.value)
            text = str(n) if self.is_year or abs(n) < 10000 else f"{n:,}"
            if not self.is_year and abs(n) >= 1000:
                text = f"{n:,}"
            return f"{text} {self.unit}" if self.unit else text
        return f"{self.value:g} {self.unit}" if self.unit else f"{self.value:g}"


def _cardinal(words: list[str]) -> float | None:
    """Fold a run of number words into one value."""
    total, current, seen = 0.0, 0.0, False
    i = 0
    while i < len(words):
        w = words[i]
        if w == "and":
            i += 1
            continue
        if w == "point":
            digits = []
            i += 1
            while i < len(words) and words[i] in UNITS and UNITS[words[i]] < 10:
                digits.append(str(UNITS[words[i]]))
                i += 1
            if digits:
                return (total + current) + float("0." + "".join(digits))
            break
        if w in UNITS:
            current += UNITS[w]
            seen = True
        elif w in TENS:
            current += TENS[w]
            seen = True
        elif w == "hundred":
            current = (current or 1) * 100
            seen = True
        elif w in SCALES:
            total += (current or 1) * SCALES[w]
            current = 0.0
            seen = True
        else:
            break
        i += 1
    return total + current if seen else None


def _as_year(words: list[str]) -> int | None:
    """'eighteen seventy seven' -> 1877. Only for plausible year shapes.

    The whole difficulty is telling a year from an ordinary cardinal that
    begins the same way: "twenty twenty four" is 2024 but "twenty five" is 25.
    English separates them by the SECOND group -- a year's second group is
    itself a two-digit number, so it opens with a tens word (or is "hundred",
    as in "eighteen hundred"). "twenty five" opens its second group with a
    unit, and is one number.
    """
    if len(words) < 2:
        return None
    century = UNITS.get(words[0], TENS.get(words[0]))
    if century is None or not 10 <= century <= 20:
        return None

    if words[1] == "hundred":
        year = century * 100
        return year if 1000 <= year <= 2099 else None
    if words[1] not in TENS:
        return None

    rest = _cardinal(words[1:])
    if rest is None or not 0 <= rest <= 99 or rest != int(rest):
        return None
    year = century * 100 + int(rest)
    return year if 1000 <= year <= 2099 else None


def parse_numbers(text: str) -> list[Number]:
    """Every number in `text`, digits or words, with its unit where it has one."""
    # The decimal part requires a digit after the point: "\\.?\\d*" happily
    # swallows a sentence-ending period, turning "1880." into a five-character
    # token that then fails the four-digit year test.
    tokens = re.findall(r"[A-Za-z]+|\d[\d,]*(?:\.\d+)?", text)
    lowered = [t.lower() for t in tokens]
    found: list[Number] = []
    i = 0
    while i < len(tokens):
        tok = lowered[i]
        if re.fullmatch(r"\d[\d,]*(?:\.\d+)?", tok):
            value = float(tok.replace(",", ""))
            unit = unit_for(lowered[i + 1] if i + 1 < len(tokens) else None, value)
            found.append(Number(value, unit, tokens[i],
                                is_year=1000 <= value <= 2099 and value.is_integer()
                                and len(tok) == 4))
            i += 1
            continue
        if tok in WORDS and tok not in {"and", "point"}:
            j = i
            while j < len(tokens) and lowered[j] in WORDS:
                j += 1
            run = lowered[i:j]
            year = _as_year(run)
            if year is not None:
                found.append(Number(float(year), None, " ".join(tokens[i:j]), is_year=True))
            else:
                value = _cardinal(run)
                if value is not None:
                    unit = unit_for(lowered[j] if j < len(tokens) else None, value)
                    found.append(Number(value, unit, " ".join(tokens[i:j])))
            i = j
            continue
        i += 1
    return found
