"""Phase 1 dataset factory: structured incident first, dialogue second.

Labels are sampled programmatically BEFORE any text exists, then the dialogue is
rendered from them with exact character spans. Nothing here calls an LLM, so the
labels can never be "whatever the generator happened to write" (no circularity
when an LLM extractor is evaluated later).

Leakage controls:
  * vessel names are partitioned: train pool vs eval pool (val/test)
  * the last phrasing of every template slot is reserved for eval splits
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from mdi.data import gazetteer as gz
from mdi.data import spoken
from mdi.data.templates import FILLERS, MRCC_ASK, PHRASES, VESSEL_TYPE_WORDS
from mdi.schema import (
    CRITICAL_FIELDS,
    CallType,
    Dialogue,
    EntityLabel,
    Incident,
    IncidentType,
    Language,
    Position,
    PositionFormat,
    Severity,
    Span,
    Turn,
    Vessel,
    VesselType,
)

STYLES = ("textbook", "hurried", "panicked")
SPLITS = ("train", "val", "test")

BASE_SEVERITY = {
    IncidentType.FIRE: 3, IncidentType.FLOODING: 3, IncidentType.ENGINE_FAILURE: 1,
    IncidentType.CAPSIZING: 4, IncidentType.MAN_OVERBOARD: 3, IncidentType.MEDICAL: 2,
    IncidentType.COLLISION: 3, IncidentType.GROUNDING: 2,
}
# Secondary incidents that plausibly follow a primary one.
FOLLOW_ON = {
    IncidentType.COLLISION: [IncidentType.FLOODING, IncidentType.MEDICAL],
    IncidentType.GROUNDING: [IncidentType.FLOODING],
    IncidentType.FLOODING: [IncidentType.CAPSIZING, IncidentType.ENGINE_FAILURE],
    IncidentType.FIRE: [IncidentType.MEDICAL, IncidentType.ENGINE_FAILURE],
    IncidentType.ENGINE_FAILURE: [IncidentType.GROUNDING],
}
POSITION_FORMAT_WEIGHTS = {
    PositionFormat.DDM: 0.4, PositionFormat.RELATIVE: 0.35,
    PositionFormat.DECIMAL: 0.15, PositionFormat.DMS: 0.1,
}
# Probability each critical field is left out of the caller's first transmission.
OMIT_PROB = {"textbook": 0.03, "hurried": 0.25, "panicked": 0.4}


@dataclass
class GeneratorConfig:
    n: int = 1000
    seed: int = 13
    languages: tuple[Language, ...] = (Language.EN, Language.HI, Language.TA, Language.ML)
    split_weights: tuple[float, float, float] = (0.8, 0.1, 0.1)
    answer_prob: float = 0.75  # caller answers an MRCC question on a given attempt
    contradiction_prob: float = 0.08


@dataclass
class _Builder:
    """Accumulates transcript text and entity spans with exact offsets."""

    text: str = ""
    spans: list[Span] = field(default_factory=list)

    def add(self, s: str, label: EntityLabel | None = None) -> None:
        if label is not None and s:
            self.spans.append(Span(start=len(self.text), end=len(self.text) + len(s), label=label, text=s))
        self.text += s


# --- incident sampling -------------------------------------------------------

def _derive_severity(types: list[IncidentType], injured: int) -> Severity:
    sev = max(BASE_SEVERITY[t] for t in types)
    if len(types) > 1:
        sev += 1
    if injured >= 2:
        sev += 1
    return Severity(min(sev, 4))


def sample_incident(rng: random.Random, assign_split) -> tuple[Incident, str]:
    """Sample an incident; `assign_split(severity)` returns (split, vessel_pool, landmark_pool).

    The split is decided after severity is known (so it can be stratified) but
    before the vessel name and landmark are drawn (so those can be held out).
    """
    vtype = rng.choices(list(gz.VESSEL_TYPE_WEIGHTS), weights=list(gz.VESSEL_TYPE_WEIGHTS.values()))[0]
    mmsi = None
    if rng.random() < gz.HAS_MMSI_PROB[vtype]:
        mmsi = gz.INDIA_MID + "".join(str(rng.randrange(10)) for _ in range(6))

    primary = rng.choice(list(IncidentType))
    types = [primary]
    if primary in FOLLOW_ON and rng.random() < 0.25:
        types.append(rng.choice(FOLLOW_ON[primary]))

    injured = 0
    if IncidentType.MEDICAL in types:
        injured = rng.randint(1, 2)
    elif {IncidentType.COLLISION, IncidentType.FIRE} & set(types) and rng.random() < 0.3:
        injured = rng.randint(1, 3)

    lo, hi = gz.POB_RANGE[vtype]
    pob = max(rng.randint(lo, hi), injured, 1)
    severity = _derive_severity(types, injured)
    # Callers misuse PAN-PAN vs MAYDAY ~10% of the time, so call type is a noisy cue.
    call_type = CallType.MAYDAY if severity >= Severity.HIGH else CallType.PANPAN
    if rng.random() < 0.1:
        call_type = CallType.PANPAN if call_type == CallType.MAYDAY else CallType.MAYDAY

    split, vessel_pool, landmark_pool = assign_split(severity)
    vessel = Vessel(name=rng.choice(vessel_pool), vessel_type=vtype, mmsi=mmsi)
    lm = rng.choice(landmark_pool)
    lat, lon = spoken.destination(lm.lat, lm.lon, rng.uniform(*lm.seaward_bearing), rng.uniform(3, 40))
    fmt = rng.choices(list(POSITION_FORMAT_WEIGHTS), weights=list(POSITION_FORMAT_WEIGHTS.values()))[0]

    return Incident(
        call_type=call_type, vessel=vessel, position=Position(lat=round(lat, 4), lon=round(lon, 4)),
        position_format=fmt, incident_types=types, pob=pob, injured=injured,
        severity=severity, nearest_landmark=lm.name,
    ), split


# --- rendering ---------------------------------------------------------------

class _Renderer:
    def __init__(self, inc: Incident, lang: Language, style: str, split: str, rng: random.Random):
        self.inc, self.lang, self.style, self.rng = inc, lang, style, rng
        self.allow_heldout = split != "train"
        self.used_heldout = False
        self.bank = PHRASES[lang]
        # Surface-form variety so models cannot key on our own writing conventions
        # (found via MARTTS dev error analysis: "Mayday" tagged as a vessel name).
        case = rng.choices(["title", "upper", "lower"], weights=[0.5, 0.35, 0.15])[0]
        self.vname = {"title": inc.vessel.name, "upper": inc.vessel.name.upper(), "lower": inc.vessel.name.lower()}[case]
        self.call = rng.choices([inc.call_type.value, inc.call_type.value.title(), inc.call_type.value.lower()],
                                weights=[0.5, 0.35, 0.15])[0]
        big = inc.vessel.vessel_type in (VesselType.CARGO, VesselType.TANKER, VesselType.FERRY, VesselType.TUG)
        self.prefix = rng.choice(["MV ", "M/V ", "motor vessel "]) if big and rng.random() < 0.5 else ""
        self.sentence_case = rng.random() < 0.5

    def pick(self, slot: str | IncidentType) -> str:
        options = self.bank[slot]
        choice = self.rng.randrange(len(options) if self.allow_heldout else len(options) - 1)
        if choice == len(options) - 1:
            self.used_heldout = True
        return options[choice]

    def fill(self, b: _Builder, template: str, values: dict[str, tuple[str, EntityLabel | None]]) -> None:
        """Write `template` into the builder, labelling substituted placeholders."""
        i = 0
        while i < len(template):
            if template[i] == "{":
                j = template.index("}", i)
                value, label = values[template[i + 1 : j]]
                b.add(value, label)
                i = j + 1
            else:
                j = template.find("{", i)
                j = len(template) if j == -1 else j
                b.add(template[i:j])
                i = j

    def position_text(self) -> str:
        inc, rng = self.inc, self.rng
        if inc.position_format == PositionFormat.RELATIVE:
            lm = next(l for l in gz.LANDMARKS if l.name == inc.nearest_landmark)
            dist = spoken.haversine_nm(lm.lat, lm.lon, inc.position.lat, inc.position.lon)
            bearing = _bearing(lm.lat, lm.lon, inc.position.lat, inc.position.lon)
            out = self.pick("relative")
            return (
                out.replace("{L}", lm.name)
                .replace("{D}", spoken.spoken_count(round(dist), self.lang, rng))
                .replace("{C}", spoken.compass_point(bearing))
            )
        render = {
            PositionFormat.DDM: spoken.position_ddm,
            PositionFormat.DMS: spoken.position_dms,
            PositionFormat.DECIMAL: spoken.position_decimal,
        }[inc.position_format]
        return render(inc.position, rng)

    def clause(self, b: _Builder, name: str, pob_override: int | None = None) -> None:
        """Render one critical-field clause into the builder."""
        inc, rng, lang = self.inc, self.rng, self.lang
        if name == "position":
            self.fill(b, self.pick("position"), {"P": (self.position_text(), EntityLabel.POSITION)})
        elif name == "pob":
            n = inc.pob if pob_override is None else pob_override
            self.fill(b, self.pick("pob"), {"N": (spoken.spoken_count(n, lang, rng), EntityLabel.POB)})
        elif name == "incident_types":
            for k, t in enumerate(inc.incident_types):
                if k:
                    b.add(", ")
                b.add(self.pick(t), EntityLabel.INCIDENT)
        elif name == "vessel_name":
            self.fill(b, self.pick("opening").replace("{V}", self.prefix + "{V}"), {
                "CALL": (self.call, None), "V": (self.vname, EntityLabel.VESSEL),
            })
        elif name == "mmsi":
            digits = spoken.spoken_digits(inc.vessel.mmsi, lang, rng) if rng.random() < 0.8 else inc.vessel.mmsi
            self.fill(b, self.pick("mmsi"), {"M": (digits, EntityLabel.MMSI)})
        elif name == "vessel_type":
            word = VESSEL_TYPE_WORDS[lang][inc.vessel.vessel_type]
            self.fill(b, self.pick("vessel_type"), {"T": (word, EntityLabel.VESSEL_TYPE)})

    def filler(self, b: _Builder) -> None:
        if self.style == "panicked" and self.rng.random() < 0.5:
            b.add(self.rng.choice(FILLERS[self.lang]) + ", ")


def _capitalise_sentences(b: _Builder, start: int) -> None:
    """Upper-case the first letter of each sentence in b.text[start:], keeping spans in sync."""
    chars = list(b.text)
    at_start = True
    for i in range(start, len(chars)):
        if at_start and chars[i].isalpha():
            chars[i] = chars[i].upper()
            at_start = False
        elif chars[i] in ".?!":
            at_start = True
    b.text = "".join(chars)
    for sp in b.spans:
        sp.text = b.text[sp.start : sp.end]


def _bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def render_dialogue(
    inc: Incident, lang: Language, style: str, split: str, rng: random.Random, cfg: GeneratorConfig, idx: int,
    unseen_vessel: bool,
) -> Dialogue:
    r = _Renderer(inc, lang, style, split, rng)
    b = _Builder()
    turns: list[Turn] = []
    rescue_centre = next(l.rescue_centre for l in gz.LANDMARKS if l.name == inc.nearest_landmark)

    def caller_turn(write) -> None:
        start = len(b.text)
        b.add("CALLER: ")
        write()
        close = r.pick("close")
        if close:
            b.add(", " + close)
        turns.append(Turn(speaker="CALLER", text=b.text[start + 8 :], lang=lang))
        b.add("\n")

    def mrcc_turn(text: str) -> None:
        b.add("MRCC: " + text + "\n")
        turns.append(Turn(speaker="MRCC", text=text, lang=Language.EN))

    # Decide which critical fields the first transmission leaves out.
    omitted = [f for f in CRITICAL_FIELDS if rng.random() < OMIT_PROB[style]]
    stated: set[str] = set()

    # Optional (non-critical) fields: MMSI and vessel type.
    extras = []
    if inc.vessel.mmsi and rng.random() < (0.8 if style == "textbook" else 0.35):
        extras.append("mmsi")
    if rng.random() < (0.7 if style == "textbook" else 0.4):
        extras.append("vessel_type")

    body = [f for f in ("position", "incident_types", "pob") if f not in omitted]
    if style != "textbook":
        rng.shuffle(body)  # stressed callers do not follow the SMCP order

    def first_call() -> None:
        r.filler(b)
        if "vessel_name" in omitted:
            b.add(r.pick("opening_noname").replace("{CALL}", r.call))
        else:
            r.clause(b, "vessel_name")
            stated.add("vessel_name")
        for f in extras:
            b.add(". ")
            r.clause(b, f)
        for f in body:
            b.add(". ")
            r.filler(b)
            r.clause(b, f)
            stated.add(f)
        b.add(". " + r.pick("assist"))

    caller_turn(first_call)

    station = r.prefix + r.vname if "vessel_name" in stated else "station calling"
    ack = f"{station}, this is {rescue_centre}, received {inc.call_type.value}"

    # MRCC asks for each missing field; the caller may fail to answer.
    pending = [f for f in CRITICAL_FIELDS if f not in stated]
    contradiction = False
    first_question = True
    for f in pending:
        for _attempt in range(2):
            q = rng.choice(MRCC_ASK[f])
            mrcc_turn(f"{ack}. {q}, over" if first_question else f"{q}, over")
            first_question = False
            if rng.random() < cfg.answer_prob:
                caller_turn(lambda f=f: r.clause(b, f))
                stated.add(f)
                break
            caller_turn(lambda: b.add(r.pick("say_again")))

    if first_question:
        mrcc_turn(f"{ack}. Rescue units proceeding to your position, over")

    # Inject a conflicting persons-on-board report the operator would have to resolve.
    if "pob" in stated and rng.random() < cfg.contradiction_prob:
        wrong = inc.pob + rng.choice([-2, -1, 1, 2, 3])
        if wrong >= 1 and wrong != inc.pob:
            mrcc_turn("confirm number of persons on board, over")
            caller_turn(lambda: r.clause(b, "pob", pob_override=wrong))
            contradiction = True

    mrcc_turn("received, keep watch on channel one six, over")

    return Dialogue(
        id=f"{split}-{idx:06d}",
        split=split,
        language=lang,
        style=style,
        unseen_phrasing=r.used_heldout,
        unseen_vessel=unseen_vessel,
        incident=inc,
        turns=turns,
        transcript=b.text,
        spans=b.spans,
        missing_fields=[f for f in CRITICAL_FIELDS if f not in stated],
        contradiction=contradiction,
    )


def _partition(items: list, seed: int, eval_frac: float) -> dict[str, list]:
    """Disjoint train / val / test pools (val and test each get `eval_frac`)."""
    items = sorted(items, key=str)
    random.Random(seed).shuffle(items)
    k = max(1, round(len(items) * eval_frac))
    return {"val": items[:k], "test": items[k : 2 * k], "train": items[2 * k :]}


class StratifiedSplitter:
    """Exact split proportions inside every (language, severity) stratum.

    Each stratum draws from its own shuffled block of 10 labels (8 train,
    1 val, 1 test), so every language x severity cell is split 80/10/10.
    """

    def __init__(self, rng: random.Random, weights: tuple[float, float, float]):
        self.rng = rng
        self.block = [s for s, w in zip(SPLITS, weights) for _ in range(round(w * 10))]
        self.queues: dict[tuple, list[str]] = {}

    def __call__(self, stratum: tuple) -> str:
        q = self.queues.setdefault(stratum, [])
        if not q:
            q.extend(self.block)
            self.rng.shuffle(q)
        return q.pop()


def generate(cfg: GeneratorConfig) -> list[Dialogue]:
    rng = random.Random(cfg.seed)
    names = _partition(gz.VESSEL_NAMES, cfg.seed, 0.15)
    landmarks = _partition(gz.LANDMARKS, cfg.seed + 1, 0.125)  # 16 -> 12 / 2 / 2
    splitter = StratifiedSplitter(random.Random(cfg.seed + 2), cfg.split_weights)

    out: list[Dialogue] = []
    for idx in range(cfg.n):
        lang = rng.choice(cfg.languages)
        style = rng.choice(STYLES)

        def assign(severity, lang=lang):
            split = splitter((lang, int(severity)))
            return split, names[split], landmarks[split]

        inc, split = sample_incident(rng, assign)
        out.append(render_dialogue(inc, lang, style, split, rng, cfg, idx, unseen_vessel=split != "train"))
    return out
