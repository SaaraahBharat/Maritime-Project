"""Parse a spoken/written position back into latitude/longitude.

Inverse of `mdi.data.spoken`. Handles digit-by-digit and natural English
numbers, native Hindi/Tamil/Malayalam digit words, DDM, DMS, decimal degrees,
written DDM (09°48.7'N ...) and bearing/range from a known landmark.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from mdi.data import gazetteer as gz
from mdi.data import spoken
from mdi.schema import Language

_DIGIT_WORDS = {w: i for lang in Language for i, w in enumerate(spoken.DIGITS[lang])}
_DIGIT_WORDS["niner"] = 9
_EN_SMALL = {w: i for i, w in enumerate(spoken._EN_ONES)}  # zero..nineteen
_EN_TENS = {w: i * 10 for i, w in enumerate(spoken._EN_TENS) if w}
_MILES = {"mile", "miles", "मील", "மைல்", "മൈൽ"}
_COMPASS = {c: i * 45 for i, c in enumerate(spoken.COMPASS)}
_WRITTEN_DDM = re.compile(r"(\d{1,2})°(\d{1,2}(?:\.\d+)?)'?\s*([NS])\s+(\d{1,3})°(\d{1,2}(?:\.\d+)?)'?\s*([EW])")


@dataclass
class ParsedPosition:
    lat: float
    lon: float
    fmt: str
    uncertainty_nm: float  # inherent to the format (e.g. 8-point compass bearing)


def _kind(tok: str) -> tuple[str, int] | None:
    if tok in _DIGIT_WORDS:
        return "digit", _DIGIT_WORDS[tok]
    if tok.isdigit():
        return ("digit", int(tok)) if len(tok) == 1 else ("literal", int(tok))
    if tok in _EN_SMALL:
        return "compound", _EN_SMALL[tok]
    if tok in _EN_TENS:
        return "compound", _EN_TENS[tok]
    if tok == "hundred":
        return "hundred", 100
    return None


def _flush(run: list[tuple[str, int, str]]) -> str:
    if all(k == "digit" for k, _, _ in run):
        return "".join(str(v) for _, v, _ in run)  # digit-by-digit: keep leading zeros
    total = 0
    for k, v, _ in run:
        total = total * 100 if k == "hundred" else total + v
    return str(total)


def normalise_numbers(text: str) -> list[str]:
    """Tokenise and collapse runs of number words into numeric strings."""
    tokens = [t.strip(",.;:!?").lower() for t in text.split()]
    out: list[str] = []
    run: list[tuple[str, int, str]] = []
    for i, tok in enumerate(tokens):
        k = _kind(tok)
        # "and" joins compounds ("one hundred and twenty")
        if k is None and tok == "and" and run and i + 1 < len(tokens) and _kind(tokens[i + 1]):
            continue
        if k is None:
            if run:
                out.append(_flush(run))
                run = []
            if tok:
                out.append(tok)
        else:
            run.append((k[0], k[1], tok))
    if run:
        out.append(_flush(run))
    return out


def parse_count(text: str) -> int | None:
    """First number in a span such as 'twelve', 'one five' or 'ஐந்து'."""
    for tok in normalise_numbers(text):
        if tok.isdigit():
            return int(tok)
    return None


def _sign(v: float, hemi: str) -> float:
    return -v if hemi in ("south", "west", "s", "w") else v


def parse_position(text: str) -> ParsedPosition | None:
    m = _WRITTEN_DDM.search(text)
    if m:
        lat = _sign(int(m[1]) + float(m[2]) / 60, m[3].lower())
        lon = _sign(int(m[4]) + float(m[5]) / 60, m[6].lower())
        return ParsedPosition(lat, lon, "ddm", 0.1)

    norm = " ".join(normalise_numbers(text))
    m = re.search(r"(\d+) degrees? (\d+) decimal (\d+) (north|south) (\d+) degrees? (\d+) decimal (\d+) (east|west)", norm)
    if m:
        lat = _sign(int(m[1]) + float(f"{m[2]}.{m[3]}") / 60, m[4])
        lon = _sign(int(m[5]) + float(f"{m[6]}.{m[7]}") / 60, m[8])
        return ParsedPosition(lat, lon, "ddm", 0.1)

    # ASR output style: "18 degrees 44.8 north, 72 degrees 26.9 east"
    m = re.search(r"(\d+) degrees? (\d+(?:\.\d+)?) (?:minutes? )?(north|south) (\d+) degrees? (\d+(?:\.\d+)?) (?:minutes? )?(east|west)", norm)
    if m:
        lat = _sign(int(m[1]) + float(m[2]) / 60, m[3])
        lon = _sign(int(m[4]) + float(m[5]) / 60, m[6])
        return ParsedPosition(lat, lon, "ddm", 0.1)

    m = re.search(r"(\d+) degrees? (\d+) minutes? (\d+) seconds? (north|south) (\d+) degrees? (\d+) minutes? (\d+) seconds? (east|west)", norm)
    if m:
        lat = _sign(int(m[1]) + int(m[2]) / 60 + int(m[3]) / 3600, m[4])
        lon = _sign(int(m[5]) + int(m[6]) / 60 + int(m[7]) / 3600, m[8])
        return ParsedPosition(lat, lon, "dms", 0.02)

    m = re.search(r"(\d+) point (\d+) (?:degrees? )?(north|south) (\d+) point (\d+) (?:degrees? )?(east|west)", norm)
    if m:
        lat = _sign(float(f"{m[1]}.{m[2]}"), m[3])
        lon = _sign(float(f"{m[4]}.{m[5]}"), m[6])
        return ParsedPosition(lat, lon, "decimal", 0.5)

    return _parse_relative(text, norm.split())


def _parse_relative(text: str, toks: list[str]) -> ParsedPosition | None:
    low = text.lower()
    landmark = next((lm for lm in gz.LANDMARKS if lm.name.lower() in low), None)
    compass = next((t for t in toks if t in _COMPASS), None)
    mi = next((i for i, t in enumerate(toks) if t in _MILES), None)
    if landmark is None or compass is None or mi is None:
        return None
    j = mi - 1
    while j >= 0 and not toks[j].isdigit():  # skip "nautical", "about", ...
        j -= 1
    if j < 0:
        return None
    dist = float(toks[j])
    lat, lon = spoken.destination(landmark.lat, landmark.lon, _COMPASS[compass], dist)
    # An 8-point compass is +/-22.5 degrees: worst-case cross-track error.
    return ParsedPosition(lat, lon, "relative", dist * math.sin(math.radians(22.5)) + 0.5)
