"""Render numbers and positions the way they are actually spoken on VHF.

SMCP prescribes digit-by-digit numbers, but real callers mix in natural numbers,
native-language numerals and written forms (what an ASR system often emits).
Every renderer takes an `rng` so variety is reproducible from a seed.
"""

from __future__ import annotations

import math
import random

from mdi.schema import Language, Position

DIGITS = {
    Language.EN: ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"],
    Language.HI: ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"],
    Language.TA: ["பூஜ்யம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"],
    Language.ML: ["പൂജ്യം", "ഒന്ന്", "രണ്ട്", "മൂന്ന്", "നാല്", "അഞ്ച്", "ആറ്", "ഏഴ്", "എട്ട്", "ഒൻപത്"],
}

_EN_ONES = DIGITS[Language.EN] + [
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_EN_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

COMPASS = ["north", "northeast", "east", "southeast", "south", "southwest", "west", "northwest"]


def en_words(n: int) -> str:
    """English words for 0 <= n < 1000."""
    if n < 20:
        return _EN_ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _EN_TENS[tens] + (f" {_EN_ONES[ones]}" if ones else "")
    hundreds, rest = divmod(n, 100)
    return f"{_EN_ONES[hundreds]} hundred" + (f" and {en_words(rest)}" if rest else "")


def spoken_digits(digits: str, lang: Language, rng: random.Random, native_prob: float = 0.4) -> str:
    """Digit-by-digit reading. Indic callers often read digits in English."""
    use_native = lang != Language.EN and rng.random() < native_prob
    table = DIGITS[lang] if use_native else DIGITS[Language.EN]
    words = [table[int(d)] for d in digits]
    if not use_native and rng.random() < 0.2:
        words = ["niner" if w == "nine" else w for w in words]  # ITU radiotelephony habit
    return " ".join(words)


def spoken_count(n: int, lang: Language, rng: random.Random) -> str:
    """A count such as persons on board, in one of several spoken forms."""
    r = rng.random()
    if r < 0.3:
        return str(n)  # ASR-normalised written form
    if r < 0.55 and n >= 10:
        return spoken_digits(str(n), lang, rng)
    if lang != Language.EN and n < 10 and r < 0.8:
        return DIGITS[lang][n]
    return en_words(n)


def _ddm(value: float, width: int) -> tuple[str, str]:
    """Split |value| into zero-padded whole degrees and minutes to one decimal."""
    v = abs(value)
    deg = int(v)
    minutes = round((v - deg) * 60, 1)
    if minutes >= 60:
        deg, minutes = deg + 1, 0.0
    return f"{deg:0{width}d}", f"{minutes:04.1f}"


def _hemis(pos: Position) -> tuple[str, str]:
    return ("north" if pos.lat >= 0 else "south"), ("east" if pos.lon >= 0 else "west")


def position_ddm(pos: Position, rng: random.Random) -> str:
    ns, ew = _hemis(pos)
    lat_d, lat_m = _ddm(pos.lat, 2)
    lon_d, lon_m = _ddm(pos.lon, 3)
    if rng.random() < 0.25:  # written form, as ASR or a typed log might give it
        return f"{lat_d}°{lat_m}'{ns[0].upper()} {lon_d}°{lon_m}'{ew[0].upper()}"

    def part(d: str, m: str, h: str) -> str:
        whole, frac = m.split(".")
        return (
            f"{spoken_digits(d, Language.EN, rng)} degrees "
            f"{spoken_digits(whole, Language.EN, rng)} decimal {spoken_digits(frac, Language.EN, rng)} {h}"
        )

    return f"{part(lat_d, lat_m, ns)}, {part(lon_d, lon_m, ew)}"


def position_dms(pos: Position, rng: random.Random) -> str:
    ns, ew = _hemis(pos)

    def part(value: float, h: str) -> str:
        v = abs(value)
        d = int(v)
        m = int((v - d) * 60)
        s = round(((v - d) * 60 - m) * 60)
        if s == 60:
            m, s = m + 1, 0
        if m == 60:
            d, m = d + 1, 0
        return f"{en_words(d)} degrees {en_words(m)} minutes {en_words(s)} seconds {h}"

    return f"{part(pos.lat, ns)}, {part(pos.lon, ew)}"


def position_decimal(pos: Position, rng: random.Random) -> str:
    ns, ew = _hemis(pos)

    # Two styles: "nine point eight one north" and "nine point eight one two three degrees north"
    precision, degrees = (4, " degrees") if rng.random() < 0.5 else (2, "")
    digit_whole = rng.random() < 0.5

    def part(value: float, h: str) -> str:
        whole, frac = f"{abs(value):.{precision}f}".split(".")
        w = spoken_digits(whole, Language.EN, rng) if digit_whole else en_words(int(whole))
        return f"{w} point {spoken_digits(frac, Language.EN, rng)}{degrees} {h}"

    return f"{part(pos.lat, ns)}, {part(pos.lon, ew)}"


def compass_point(bearing_deg: float) -> str:
    return COMPASS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


# --- geodesy helpers (spherical earth is plenty for synthetic data) ---

EARTH_RADIUS_NM = 3440.065


def destination(lat: float, lon: float, bearing_deg: float, dist_nm: float) -> tuple[float, float]:
    phi1, lam1, theta = map(math.radians, (lat, lon, bearing_deg))
    delta = dist_nm / EARTH_RADIUS_NM
    phi2 = math.asin(math.sin(phi1) * math.cos(delta) + math.cos(phi1) * math.sin(delta) * math.cos(theta))
    lam2 = lam1 + math.atan2(
        math.sin(theta) * math.sin(delta) * math.cos(phi1),
        math.cos(delta) - math.sin(phi1) * math.sin(phi2),
    )
    return math.degrees(phi2), (math.degrees(lam2) + 540) % 360 - 180


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlam = phi2 - phi1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_NM * math.asin(math.sqrt(a))
