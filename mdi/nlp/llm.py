"""Small open LLM (Qwen2.5-1.5B-Instruct, 4-bit GGUF, Apache 2.0) as a zero-shot extractor.

Design: the LLM only SELECTS evidence. It must copy each field verbatim from the
transcript; decoding is constrained by a JSON-schema grammar (llama.cpp), so the
output always parses. Two guards stop hallucination:

  1. grounding: every quoted span must actually occur in the transcript,
     otherwise the field is rejected (counted in `rejected`);
  2. normalisation: numbers, MMSI digits and positions are converted by the same
     deterministic parsers the CRF pipeline uses, never by the LLM.

`hybrid()` keeps the CRF pipeline's fields and lets the LLM fill only the
critical fields the CRF missed, the cheap way to combine a fast specialised model
with a general one.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from mdi.nlp.position import normalise_numbers, parse_count, parse_position

GGUF_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
GGUF_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

SCHEMA = {
    "type": "object",
    "properties": {
        # maxLength keeps quotes short so the JSON always closes within max_tokens.
        "vessel_name": {"type": ["string", "null"], "maxLength": 40},
        "mmsi_quote": {"type": ["string", "null"], "maxLength": 90},
        "position_quote": {"type": ["string", "null"], "maxLength": 140},
        "persons_on_board_quote": {"type": ["string", "null"], "maxLength": 60},
    },
    "required": ["vessel_name", "mmsi_quote", "position_quote", "persons_on_board_quote"],
}

SYSTEM = (
    "You extract facts from marine VHF radio transcripts (often with speech-recognition errors). "
    "Copy text EXACTLY as it appears in the transcript, do not convert numbers or fix spelling. "
    "Use null when a fact is not stated. Fields:\n"
    "vessel_name: name of the vessel that is calling/in trouble (not the coast station).\n"
    "mmsi_quote: the words or digits that give the 9-digit MMSI number.\n"
    "position_quote: the words giving latitude and longitude (or bearing/distance from a place).\n"
    "persons_on_board_quote: the words giving how many people are on board.\n"
    "Example transcript: CALLER: mayday mayday this is fishing vessel Blue Marlin, MMSI two three five "
    "one two three four five six, position one zero degrees zero five decimal two north zero seven five "
    "degrees five eight decimal one east, engine fire, six persons on board.\n"
    'Example answer: {"vessel_name": "Blue Marlin", "mmsi_quote": "two three five one two three four five six", '
    '"position_quote": "one zero degrees zero five decimal two north zero seven five degrees five eight decimal one east", '
    '"persons_on_board_quote": "six persons on board"}'
)


def _squash(s: str) -> str:
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def grounded(quote: str | None, transcript: str) -> bool:
    """Quote occurs in the transcript, ignoring case and punctuation."""
    return bool(quote) and _squash(quote) in _squash(transcript)


@dataclass
class LlmFields:
    vessel_name: str | None = None
    mmsi: str | None = None
    position: dict | None = None
    pob: int | None = None
    raw: dict = field(default_factory=dict)
    rejected: list[str] = field(default_factory=list)  # fields dropped by the grounding guard


@lru_cache(maxsize=1)
def _llm():
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    return Llama(model_path=hf_hub_download(GGUF_REPO, GGUF_FILE), n_ctx=2048, n_threads=6, verbose=False)


CACHE = Path(__file__).resolve().parents[2] / "data/cache/llm_raw.jsonl"


@lru_cache(maxsize=1)
def _disk_cache() -> dict:
    if not CACHE.exists():
        return {}
    return {r["key"]: r["content"] for r in map(json.loads, open(CACHE, encoding="utf-8"))}


def _complete(transcript: str) -> str:
    """Raw LLM output, cached on disk by transcript hash (temperature 0, so it is deterministic)."""
    key = hashlib.md5((GGUF_FILE + SYSTEM + transcript).encode()).hexdigest()
    cache = _disk_cache()
    if key not in cache:
        out = _llm().create_chat_completion(
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Transcript:\n" + transcript}],
            response_format={"type": "json_object", "schema": SCHEMA}, temperature=0.0, max_tokens=320)
        cache[key] = out["choices"][0]["message"]["content"]
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "content": cache[key]}, ensure_ascii=False) + "\n")
    return cache[key]


@lru_cache(maxsize=4096)
def extract(transcript: str, max_chars: int = 2500) -> LlmFields:
    try:
        raw = json.loads(_complete(transcript[:max_chars]))
    except json.JSONDecodeError:  # truncated output: treat as "nothing extracted"
        return LlmFields(rejected=["unparseable"])
    f = LlmFields(raw=raw)

    if raw.get("vessel_name"):
        if grounded(raw["vessel_name"], transcript):
            f.vessel_name = raw["vessel_name"].strip()
        else:
            f.rejected.append("vessel_name")
    if raw.get("mmsi_quote"):
        q = re.sub(r"(mmsi|number|is|no)", " ", raw["mmsi_quote"].lower())
        if re.search(r"\d", q):  # written digits: read them directly ("235 123 456")
            digits, toks = re.sub(r"\D", "", q), re.findall(r"\d+|[a-z]+", q)
        else:  # spoken digits ("two three five ...")
            toks = normalise_numbers(q)
            digits = "".join(t for t in toks if t.isdigit())
        # An MMSI quote must be (almost) only digits: rejects positions or POB mislabelled as MMSI.
        mostly_digits = bool(toks) and sum(t.isdigit() for t in toks) / len(toks) >= 0.8
        if grounded(raw["mmsi_quote"], transcript) and mostly_digits and 9 <= len(digits) <= 10:
            f.mmsi = digits[:9]
        else:
            f.rejected.append("mmsi")
    if raw.get("position_quote"):
        p = parse_position(raw["position_quote"]) if grounded(raw["position_quote"], transcript) else None
        if p is not None:
            f.position = {"lat": round(p.lat, 4), "lon": round(p.lon, 4), "format": p.fmt,
                          "uncertainty_nm": round(p.uncertainty_nm, 2)}
        else:
            f.rejected.append("position")
    if raw.get("persons_on_board_quote"):
        n = parse_count(raw["persons_on_board_quote"]) if grounded(raw["persons_on_board_quote"], transcript) else None
        if n:
            f.pob = n
        else:
            f.rejected.append("pob")
    return f


CRITICAL = ("vessel_name", "mmsi", "position", "pob")


def apply(report, llm: LlmFields, policy: dict[str, str], confidence: float = 0.6):
    """Merge LLM fields into a pipeline IncidentReport, field by field.

    policy[field]: "crf_first" = keep the CRF value, LLM only fills a gap;
                   "llm_first" = take the (grounded) LLM value, fall back to the CRF;
                   "llm"       = LLM only, drop the CRF value.
    """
    from mdi.data.templates import MRCC_ASK
    from mdi.pipeline import Field

    values = {"vessel_name": llm.vessel_name, "mmsi": llm.mmsi, "position": llm.position, "pob": llm.pob}
    evidence = {"vessel_name": llm.raw.get("vessel_name"), "mmsi": llm.raw.get("mmsi_quote"),
                "position": llm.raw.get("position_quote"), "pob": llm.raw.get("persons_on_board_quote")}
    for k, v in values.items():
        rule = policy.get(k, "crf_first")
        if rule == "crf_first" and k in report.fields:
            continue
        if v is not None:
            report.fields[k] = Field(v, confidence, f"[LLM] {evidence[k]}")
        elif rule == "llm":
            report.fields.pop(k, None)
    report.missing_fields = [m for m in report.missing_fields if m not in report.fields]
    report.suggested_questions = [MRCC_ASK[m][0] for m in report.missing_fields]
    return report


POLICIES = {"llm": dict.fromkeys(CRITICAL, "llm"), "hybrid": dict.fromkeys(CRITICAL, "crf_first"),
            "llm_first": dict.fromkeys(CRITICAL, "llm_first")}
ROUTING_FILE = Path(__file__).resolve().parents[2] / "models/llm_routing.json"


def learned_policy() -> dict[str, str]:
    """Per-field routing chosen on dev data by scripts/evaluate_llm.py (falls back to hybrid)."""
    if ROUTING_FILE.exists():
        return json.loads(ROUTING_FILE.read_text(encoding="utf-8"))["policy"]
    return POLICIES["hybrid"]


class LlmPipeline:
    """Same `.extract()` interface as Pipeline, so existing evaluators can score it unchanged.

    mode: "llm" (LLM replaces the four critical fields), "hybrid" (LLM fills CRF gaps),
    "llm_first" (LLM wins when grounded), "routed" (per-field policy learned on dev data),
    or pass an explicit `policy` dict.
    """

    def __init__(self, pipe, mode: str = "hybrid", policy: dict[str, str] | None = None):
        self.pipe, self.mode = pipe, mode
        self.policy = policy or (learned_policy() if mode == "routed" else POLICIES[mode])
        self.rejected: list[str] = []

    def extract(self, transcript: str):
        rep = self.pipe.extract(transcript)
        if all(r == "crf_first" for r in self.policy.values()) and all(k in rep.fields for k in CRITICAL):
            return rep  # nothing missing: skip the slow LLM call
        f = extract(transcript)
        self.rejected += f.rejected
        return apply(rep, f, self.policy)
