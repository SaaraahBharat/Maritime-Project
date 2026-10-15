"""End-to-end extraction: transcript -> structured, explained incident report.

Every extracted field carries the transcript span it came from (evidence) and a
confidence, so the operator can verify it and nothing is invented.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from mdi.data.templates import MRCC_ASK, VESSEL_TYPE_WORDS
from mdi.nlp.models import INCIDENT_CLASSES, caller_text, severity_features
from mdi.nlp.position import parse_count, parse_position, normalise_numbers
from mdi.schema import CRITICAL_FIELDS

SEVERITY_NAMES = {1: "LOW", 2: "MODERATE", 3: "HIGH", 4: "CRITICAL"}
_VT_LOOKUP = {w.lower(): vt.value for words in VESSEL_TYPE_WORDS.values() for vt, w in words.items()}
LOW_CONFIDENCE = 0.6


@dataclass
class Field:
    value: object
    confidence: float
    evidence: str | None = None


@dataclass
class IncidentReport:
    fields: dict[str, Field] = field(default_factory=dict)
    incident_proba: dict[str, float] = field(default_factory=dict)
    severity: int | None = None
    severity_proba: dict[int, float] = field(default_factory=dict)
    severity_set: list[int] = field(default_factory=list)  # conformal prediction set
    missing_fields: list[str] = field(default_factory=list)
    low_confidence: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    suggested_questions: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    severity_features: np.ndarray | None = None

    def to_json(self) -> dict:
        return {
            "fields": {k: {"value": v.value, "confidence": round(v.confidence, 3), "evidence": v.evidence}
                       for k, v in self.fields.items()},
            "severity": SEVERITY_NAMES.get(self.severity),
            "severity_proba": {SEVERITY_NAMES[k]: round(v, 3) for k, v in self.severity_proba.items()},
            "severity_90pct_set": [SEVERITY_NAMES[k] for k in self.severity_set],
            "missing_fields": self.missing_fields,
            "low_confidence": self.low_confidence,
            "contradictions": self.contradictions,
            "suggested_questions": self.suggested_questions,
        }


def _speaker_at(transcript: str, offset: int) -> str:
    line_start = transcript.rfind("\n", 0, offset) + 1
    return transcript[line_start:offset].split(":", 1)[0]


_COUNT_BLOCKERS = {"mile", "miles", "nautical", "मील", "மைல்", "മൈൽ", "channel", "degrees", "decimal", "point"}


def _fallback_count(transcript: str, ents: list[dict]) -> tuple[int, str] | None:
    """A number in caller speech that is not part of a position, MMSI or distance.

    Used only when NER found no POB; result is low-confidence by design.
    """
    masked = list(transcript)
    for e in ents:
        if e["label"] in ("POSITION", "MMSI", "VESSEL"):
            masked[e["start"]:e["end"]] = " " * (e["end"] - e["start"])
    caller = caller_text("".join(masked))
    toks = normalise_numbers(caller)
    for i, t in enumerate(toks):
        # 1 is skipped: "एक"/"one"/"ஒரு" mostly act as articles ("one side", "a ship")
        if t.isdigit() and 2 <= int(t) <= 500:
            near = set(toks[max(0, i - 1): i + 3])
            if not near & _COUNT_BLOCKERS:
                return int(t), t
    return None


class Pipeline:
    incident_stated_threshold = 0.5

    def __init__(self, ner, incident_clf, severity_model, conformal):
        self.ner, self.incident_clf = ner, incident_clf
        self.severity_model, self.conformal = severity_model, conformal

    @classmethod
    def load(cls, path: str | Path = "models/pipeline.joblib") -> "Pipeline":
        return joblib.load(path)

    def save(self, path: str | Path = "models/pipeline.joblib") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    def extract(self, transcript: str) -> IncidentReport:
        rep = IncidentReport()
        ents = self.ner.predict(transcript)
        rep.entities = ents
        by = {}
        for e in ents:
            by.setdefault(e["label"], []).append(e)

        if "VESSEL" in by:
            # "Mercy, Mercy" (a repeated call sign tagged as one span) -> "Mercy"
            for e in by["VESSEL"]:
                e["name"] = e["text"].split(",")[0].strip()
            # SMCP: the caller identifies itself first ("this is X"), so prefer the first
            # VESSEL span on a CALLER line; other names may be responding ships.
            caller_spans = [e for e in by["VESSEL"] if _speaker_at(transcript, e["start"]) == "CALLER"]
            if caller_spans:
                name, conf = caller_spans[0]["name"], caller_spans[0]["confidence"]
            else:
                name, _ = Counter(e["name"] for e in by["VESSEL"]).most_common(1)[0]
                conf = max(e["confidence"] for e in by["VESSEL"] if e["name"] == name)
            rep.fields["vessel_name"] = Field(name, conf, name)

        for e in by.get("MMSI", []):
            digits = "".join(t for t in normalise_numbers(e["text"]) if t.isdigit())
            if len(digits) >= 9:  # spans can run past the number; MMSI is the first 9 digits
                rep.fields["mmsi"] = Field(digits[:9], e["confidence"], e["text"])
                break
            rep.low_confidence.append("mmsi")  # heard an MMSI but not 9 digits: ask again

        for e in by.get("VESSEL_TYPE", []):
            vt = _VT_LOOKUP.get(e["text"].lower())
            if vt:
                rep.fields["vessel_type"] = Field(vt, e["confidence"], e["text"])
                break

        for e in by.get("POSITION", []):
            p = parse_position(e["text"])
            if p:
                rep.fields["position"] = Field(
                    {"lat": round(p.lat, 4), "lon": round(p.lon, 4), "format": p.fmt, "uncertainty_nm": round(p.uncertainty_nm, 2)},
                    e["confidence"], e["text"],
                )
                break

        pobs = [(parse_count(e["text"]), e) for e in by.get("POB", [])]
        pobs = [(v, e) for v, e in pobs if v]
        if not pobs:
            fb = _fallback_count(transcript, ents)
            if fb:
                pobs = [(fb[0], {"text": fb[1], "confidence": 0.4})]  # rule backoff: operator must verify
        if pobs:
            v, e = pobs[0]
            rep.fields["pob"] = Field(v, e["confidence"], e["text"])
            distinct = sorted({v for v, _ in pobs})
            if len(distinct) > 1:
                rep.contradictions.append(f"persons on board reported as {', '.join(map(str, distinct))}")

        text = caller_text(transcript) or transcript
        proba = self.incident_clf.predict_proba([text])[0]
        rep.incident_proba = {c: float(p) for c, p in zip(INCIDENT_CLASSES, proba)}
        types = self.incident_clf.predict([text])[0]
        evidence = "; ".join(e["text"] for e in by.get("INCIDENT", [])) or None
        rep.fields["incident_types"] = Field(types, float(min(proba[INCIDENT_CLASSES.index(t)] for t in types)), evidence)

        # Critical fields never stated -> missing; stated but uncertain -> verify.
        # Incident counts as stated if NER found a phrase OR the classifier is confident.
        incident_stated = evidence is not None or float(proba.max()) >= self.incident_stated_threshold
        stated = set(rep.fields) - (set() if incident_stated else {"incident_types"})
        rep.missing_fields = [f for f in CRITICAL_FIELDS if f not in stated]
        rep.low_confidence += [k for k, v in rep.fields.items() if v.confidence < LOW_CONFIDENCE and k not in rep.missing_fields]
        rep.suggested_questions = [MRCC_ASK[f][0] for f in rep.missing_fields]
        if rep.contradictions:
            rep.suggested_questions.append("confirm number of persons on board")

        is_mayday = "mayday" in transcript.lower()
        pob = rep.fields["pob"].value if "pob" in rep.fields else None
        vt = rep.fields["vessel_type"].value if "vessel_type" in rep.fields else None
        x = severity_features(proba, pob, is_mayday, vt)[None, :]
        sp = self.severity_model.predict_proba(x)[0]
        rep.severity_proba = {k: float(p) for k, p in zip((1, 2, 3, 4), sp)}
        rep.severity = int(np.argmax(sp)) + 1
        rep.severity_set = [i + 1 for i in self.conformal.predict_sets(sp[None, :])[0]]
        rep.severity_features = x[0]
        return rep

    def explain_severity(self, rep: IncidentReport) -> list[tuple[str, float]]:
        """Per-feature contribution to the log-odds of severity >= HIGH.

        For a linear model, w_i * (x_i - mean_i) is the exact SHAP value
        (under feature independence), so no sampling approximation is needed.
        """
        from mdi.nlp.models import FEATURE_NAMES, OrdinalLogistic

        if isinstance(self.severity_model, OrdinalLogistic):
            w = self.severity_model.models[1].coef_[0]
        else:  # multinomial: HIGH+CRITICAL minus LOW+MODERATE logits
            c = self.severity_model.lr.coef_
            w = c[2:].sum(0) - c[:2].sum(0)
        means = getattr(self, "feature_means", np.zeros_like(w))
        contrib = w * (rep.severity_features - means)
        return sorted(zip(FEATURE_NAMES, map(float, contrib)), key=lambda kv: -abs(kv[1]))
