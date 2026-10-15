"""SeaAlert (Atia et al., IEEE Access 2026, MIT): 1,872 English VHF messages,
4 classes x 468 (Distress / Urgency / Safety / Routine), with Whisper transcripts
at three noise levels and gold fields (vessel, MMSI, position, POB).

Two experiments:

A. Benchmark reproduction (message category). SeaAlert's own task with its own
   train/test sizes (1310/281/281, stratified; their exact split is not
   published, so ours uses seed 42). Compares against their reported numbers and
   tests our noise-augmented training technique (mdi.data.noise_text).

B. Zero-shot field extraction. Our pipeline, trained only on our synthetic data,
   run on SeaAlert test messages (clean text and ASR transcripts). SeaAlert data
   is never used to train the main pipeline.

Usage:  python scripts/evaluate_seaalert.py
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.data.noise_text import corrupt  # noqa: E402
from mdi.data.spoken import haversine_nm  # noqa: E402
from mdi.nlp.models import IncidentClassifier  # noqa: E402
from mdi.nlp.position import parse_count, parse_position  # noqa: E402
from mdi.pipeline import Pipeline  # noqa: E402

CSV = Path("data/external/seaalert/data/processed/03seaalert_with_asr.csv")
SETTINGS = ["text", "asr_low", "asr_med", "asr_high"]
# Published in the SeaAlert repo (results/csv/all_model_results.csv, macro F1).
PUBLISHED = {
    "logistic_regression": {"text": 0.6742, "asr_med": 0.4268, "asr_high": 0.4231},
    "roberta-base": {"text": 0.6787, "asr_med": 0.6198, "asr_high": 0.6081},
}
# Published extraction accuracy at ASR-high (README).
PUBLISHED_EXTRACTION_ASR_HIGH = {
    "regex": {"vessel": 0.465, "mmsi": 0.015, "position": 0.083, "pob": 0.305},
    "gpt-4o-mini": {"vessel": 0.627, "mmsi": 0.239, "position": 0.323, "pob": 0.796},
}
RNG = np.random.default_rng(0)


def split(df: pd.DataFrame):
    train, rest = train_test_split(df, test_size=562, stratify=df["label"], random_state=42)
    val, test = train_test_split(rest, test_size=281, stratify=rest["label"], random_state=42)
    return train, val, test


def macro_f1(y, p) -> float:
    return float(f1_score(y, p, average="macro"))


def bootstrap(y, p, n=1000):
    y, p = np.asarray(y), np.asarray(p)
    vals = [macro_f1(y[i], p[i]) for i in (RNG.integers(0, len(y), len(y)) for _ in range(n))]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def experiment_a(train, test) -> dict:
    """Message-category classification: clean training vs noise-augmented training."""
    def fit(texts, labels):
        clf = IncidentClassifier(threshold=0.0)  # reuse TF-IDF(char+word) + LR; single label via argmax
        clf.mlb.classes = sorted(set(labels))
        clf.model.fit(texts, clf.mlb.fit_transform([[l] for l in labels]))
        return clf

    def predict(clf, texts):
        return [clf.mlb.classes_[i] for i in clf.predict_proba(texts).argmax(1)]

    rng = random.Random(0)
    clean = fit(train["text"].tolist(), train["label"].tolist())
    aug_texts = train["text"].tolist() + [corrupt("CALLER: " + t, rng.choice([0.1, 0.2, 0.3]), rng)[8:]
                                           for t in train["text"]]
    augmented = fit(aug_texts, train["label"].tolist() * 2)

    out = {"published_macro_f1": PUBLISHED, "ours": {}}
    for name, clf in (("tfidf_lr_clean_training", clean), ("tfidf_lr_noise_augmented", augmented)):
        out["ours"][name] = {}
        for s in SETTINGS:
            texts = test[s].fillna("").tolist()
            pred = predict(clf, texts)
            out["ours"][name][s] = {"macro_f1": macro_f1(test["label"], pred), "ci95": bootstrap(test["label"], pred)}
    return out


# --- B. zero-shot extraction ---------------------------------------------------

def _norm_vessel(v: str) -> str:
    v = re.sub(r"[^a-z0-9 ]", " ", str(v).lower())
    v = re.sub(r"\b(the|mv|m v|fishing vessel|sailing vessel|motor vessel|vessel|yacht|tug|sv|fv)\b", " ", v)
    return " ".join(v.split())


def experiment_b(pipe: Pipeline, test) -> dict:
    out = {}
    for s in SETTINGS:
        res = defaultdict(list)
        for _, row in test.iterrows():
            text = str(row[s]) if pd.notna(row[s]) else ""
            r = pipe.extract("CALLER: " + text)
            f = r.fields
            if pd.notna(row["vessel"]) and str(row["vessel"]).strip():
                res["vessel"].append(f.get("vessel_name") is not None
                                     and _norm_vessel(f["vessel_name"].value) == _norm_vessel(row["vessel"]))
            mmsi = re.sub(r"\D", "", str(row["mmsi"]))
            if len(mmsi) == 9:
                res["mmsi"].append(f.get("mmsi") is not None and f["mmsi"].value == mmsi)
            gold_pos = parse_position(str(row["location"])) if pd.notna(row["location"]) else None
            if gold_pos is not None and gold_pos.fmt != "relative":
                p = f.get("position")
                res["position"].append(p is not None and haversine_nm(
                    p.value["lat"], p.value["lon"], gold_pos.lat, gold_pos.lon) <= 1.0)
            pob = parse_count(str(row["pob"])) if pd.notna(row["pob"]) else None
            if pob:
                res["pob"].append(f.get("pob") is not None and f["pob"].value == pob)
            # Severity sanity: distress messages should be rated higher than routine ones.
            res[f"severity_{row['label']}"].append(r.severity)
        out[s] = {k: {"value": float(np.mean(v)), "n": len(v)} for k, v in res.items()}
    out["published_asr_high"] = PUBLISHED_EXTRACTION_ASR_HIGH
    return out


def main() -> None:
    df = pd.read_csv(CSV).rename(columns={})
    train, val, test = split(df)
    report = {"sizes": {"train": len(train), "val": len(val), "test": len(test)},
              "A_message_category": experiment_a(train, test),
              "B_zero_shot_extraction": experiment_b(Pipeline.load(), test)}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/seaalert.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    a = report["A_message_category"]["ours"]
    print("A. message category macro-F1 (ours vs published LR / RoBERTa)")
    for s in SETTINGS:
        pub = {m: PUBLISHED[m].get(s) for m in PUBLISHED}
        print(f"  {s:9s} clean-trained {a['tfidf_lr_clean_training'][s]['macro_f1']:.3f}  "
              f"noise-augmented {a['tfidf_lr_noise_augmented'][s]['macro_f1']:.3f}  published {pub}")
    print("B. zero-shot extraction accuracy")
    for s in SETTINGS:
        b = report["B_zero_shot_extraction"][s]
        print(f"  {s:9s}", {k: round(v["value"], 3) for k, v in b.items() if not k.startswith("severity")},
              {k: round(v["value"], 2) for k, v in b.items() if k.startswith("severity")})


if __name__ == "__main__":
    main()
