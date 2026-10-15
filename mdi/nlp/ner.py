"""Multilingual NER with a linear-chain CRF (CPU, trains in minutes).

Features are deliberately script/shape based rather than gazetteer based:
using the vessel-name list as a feature would leak the held-out test names.
A fine-tuned XLM-R (Colab notebook) is the stronger comparison model.
"""

from __future__ import annotations

import re
import unicodedata

import sklearn_crfsuite

from mdi.data import gazetteer as gz
from mdi.nlp.position import _COMPASS, _DIGIT_WORDS, _EN_SMALL, _EN_TENS, _MILES

_NUMBERISH = set(_DIGIT_WORDS) | set(_EN_SMALL) | set(_EN_TENS) | {"hundred", "decimal", "point", "degrees", "minutes", "seconds"}
_TOKEN = re.compile(r"\S+")


def tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _script(tok: str) -> str:
    for ch in tok:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            return name.split()[0]  # LATIN, DEVANAGARI, TAMIL, MALAYALAM
    return "OTHER"


_LANDMARK_TOKENS = {t.lower() for lm in gz.LANDMARKS for t in lm.name.split()}


def _protocol_vocabulary() -> set[str]:
    """Every word our phrase banks / MRCC side use, plus radio procedure words.

    These keep their identity in any casing ("Mayday" == "MAYDAY" == "mayday").
    """
    from mdi.data.templates import MRCC_ASK, PHRASES

    texts = [t for bank in PHRASES.values() for opts in bank.values() for t in opts]
    texts += [q for qs in MRCC_ASK.values() for q in qs]
    words = {re.sub(r"\{\w+\}", "", w).strip(",.?").lower() for t in texts for w in t.split()}
    words |= {"mayday", "pan-pan", "panpan", "securite", "sécurité", "mv", "m/v", "motor", "vessel", "mrcc", "mrsc",
              "this", "is", "over", "out", "received", "roger", "affirmative", "negative", "position", "all", "stations"}
    return {w for w in words if w} | set(_NUMBERISH) | set(_COMPASS)


def _is_title(w: str) -> bool:
    return w[:1].isupper() and w[1:2].islower()


def _is_cap(w: str) -> bool:
    return w[:1].isupper()


def _word_id(w: str) -> str:
    """Word identity, except unknown capitalised/upper-case Latin words become a shape token.

    Otherwise the CRF memorises training vessel names (fails on new ones) and
    learns our casing habits (tags "Mayday" as a vessel on external data).
    """
    lw = w.strip(",.-?").lower()
    if lw in _VOCAB or lw in _LANDMARK_TOKENS:
        return lw
    if _is_cap(w) and _script(w) == "LATIN":
        return "<CAP>"
    return lw


def _feats(tokens: list[str], i: int) -> dict:
    w = tokens[i]
    lw = w.strip(",.").lower()
    wid = _word_id(w)
    f = {
        "bias": 1.0,
        "lower": wid,
        "suf3": lw[-3:] if wid != "<CAP>" else "<CAP>",
        "known_word": lw.strip("?") in _VOCAB,
        "landmark": lw.split("-")[0] in _LANDMARK_TOKENS,
        "is_title": _is_title(w),
        "is_upper": w.isupper(),
        "has_digit": any(c.isdigit() for c in w),
        "has_degree": "°" in w,
        "numberish": lw in _NUMBERISH or lw.isdigit(),
        "compass": lw in _COMPASS,
        "miles": lw in _MILES,
        "script": _script(w),
        "ends_comma": w.endswith(","),
        "speaker": w in ("CALLER:", "MRCC:"),
    }
    for off in (-2, -1, 1, 2):
        j = i + off
        if 0 <= j < len(tokens):
            o = tokens[j].strip(",.").lower()
            f[f"{off}:lower"] = _word_id(tokens[j])
            f[f"{off}:numberish"] = o in _NUMBERISH or o.isdigit()
            f[f"{off}:is_title"] = _is_title(tokens[j])
            f[f"{off}:ends_comma"] = tokens[j].endswith(",")
            f[f"{off}:script"] = _script(tokens[j])
        else:
            f[f"{off}:pad"] = True
    return f


_VOCAB = _protocol_vocabulary()


def sent_features(tokens: list[str]) -> list[dict]:
    return [_feats(tokens, i) for i in range(len(tokens))]


def bio_tags(text: str, spans) -> list[str]:
    tags = []
    for _, s, _e in tokenize(text):
        tag = "O"
        for sp in spans:
            if sp.start <= s < sp.end:
                tag = ("B-" if s == sp.start else "I-") + sp.label.value
                break
        tags.append(tag)
    return tags


class CrfNer:
    def __init__(self, c1: float = 0.05, c2: float = 0.05, max_iterations: int = 150):
        self.crf = sklearn_crfsuite.CRF(
            algorithm="lbfgs", c1=c1, c2=c2, max_iterations=max_iterations, all_possible_transitions=True
        )

    def fit(self, dialogues) -> "CrfNer":
        X, y = [], []
        for d in dialogues:
            toks = [t for t, _, _ in tokenize(d.transcript)]
            X.append(sent_features(toks))
            y.append(bio_tags(d.transcript, d.spans))
        self.crf.fit(X, y)
        return self

    def predict_tags(self, text: str) -> list[str]:
        toks = [t for t, _, _ in tokenize(text)]
        return self.crf.predict_single(sent_features(toks)) if toks else []

    def predict(self, text: str) -> list[dict]:
        """Entity spans with character offsets and a marginal-probability confidence."""
        toks = tokenize(text)
        if not toks:
            return []
        feats = sent_features([t for t, _, _ in toks])
        tags = self.crf.predict_single(feats)
        marg = self.crf.predict_marginals_single(feats)
        ents: list[dict] = []
        for (tok, s, e), tag, m in zip(toks, tags, marg):
            if tag == "O":
                if ents:
                    ents[-1]["_open"] = False
                continue
            label = tag[2:]
            if tag.startswith("I-") and ents and ents[-1]["label"] == label and ents[-1]["_open"]:
                ents[-1]["end"] = e
                ents[-1]["_probs"].append(m[tag])
            else:
                if ents:
                    ents[-1]["_open"] = False
                ents.append({"label": label, "start": s, "end": e, "_probs": [m[tag]], "_open": True})
        out = []
        for ent in ents:
            # Trailing punctuation belongs to the sentence, not the entity.
            end = ent["end"]
            while end > ent["start"] and text[end - 1] in ",.":
                end -= 1
            out.append({
                "label": ent["label"], "start": ent["start"], "end": end, "text": text[ent["start"]:end],
                "confidence": float(min(ent["_probs"])),
            })
        return out
