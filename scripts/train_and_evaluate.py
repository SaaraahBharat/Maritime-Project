"""Train every model, evaluate on the held-out test split, save pipeline + metrics.

Usage:
    python scripts/train_and_evaluate.py --data data/synthetic

Outputs models/pipeline.joblib and reports/metrics.json (read by the dashboard).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import jiwer
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.data.noise_text import corrupt  # noqa: E402
from mdi.data.spoken import haversine_nm  # noqa: E402
from mdi.nlp.models import (  # noqa: E402
    FEATURE_NAMES, INCIDENT_CLASSES, IncidentClassifier, KeywordIncidentBaseline, OrdinalLogistic,
    SplitConformal, TemperatureScaled, caller_text, expected_calibration_error, severity_features,
)
from mdi.nlp.ner import CrfNer  # noqa: E402
from mdi.pipeline import Pipeline  # noqa: E402
from mdi.schema import CRITICAL_FIELDS, Dialogue  # noqa: E402

RNG = np.random.default_rng(0)


def load(path: Path, split: str, limit: int | None = None) -> list[Dialogue]:
    rows = [Dialogue.model_validate_json(l) for l in open(path / f"{split}.jsonl", encoding="utf-8")]
    return rows[:limit] if limit else rows


def bootstrap_ci(per_item: list, stat, n: int = 1000) -> tuple[float, float]:
    idx = np.arange(len(per_item))
    vals = [stat([per_item[i] for i in RNG.choice(idx, len(idx))]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# --- NER ---------------------------------------------------------------------

def ner_counts(ner: CrfNer, d: Dialogue) -> dict:
    gold = {(s.label.value, s.start, s.end) for s in d.spans}
    pred = {(e["label"], e["start"], e["end"]) for e in ner.predict(d.transcript)}
    c = defaultdict(lambda: [0, 0, 0])
    for lab in {g[0] for g in gold | pred}:
        g = {x for x in gold if x[0] == lab}
        p = {x for x in pred if x[0] == lab}
        c[lab] = [len(g & p), len(p - g), len(g - p)]
    return dict(c)


def micro_f1(counts: list[dict], label: str | None = None) -> float:
    tp = fp = fn = 0
    for c in counts:
        for lab, (a, b, d) in c.items():
            if label is None or lab == label:
                tp, fp, fn = tp + a, fp + b, fn + d
    return 2 * tp / max(1, 2 * tp + fp + fn)


# --- severity feature extraction ------------------------------------------------

def sev_X(ner, incident_proba, dialogues):
    from mdi.pipeline import _VT_LOOKUP
    from mdi.nlp.position import parse_count

    X = []
    for d, p in zip(dialogues, incident_proba):
        ents = ner.predict(d.transcript)
        pob = next((parse_count(e["text"]) for e in ents if e["label"] == "POB"), None)
        vt = next((_VT_LOOKUP.get(e["text"].lower()) for e in ents if e["label"] == "VESSEL_TYPE"), None)
        X.append(severity_features(p, pob, "mayday" in d.transcript.lower(), vt))
    return np.array(X)


# --- end-to-end field evaluation --------------------------------------------

def field_eval(pipe: Pipeline, dialogues: list[Dialogue], transcripts: list[str] | None = None) -> dict:
    transcripts = transcripts or [d.transcript for d in dialogues]
    res = defaultdict(list)
    miss_true, miss_pred = defaultdict(list), defaultdict(list)
    contra_true, contra_pred = [], []
    inc_true, inc_pred = [], []
    for d, t in zip(dialogues, transcripts):
        r = pipe.extract(t)
        f = r.fields
        if "vessel_name" not in d.missing_fields:
            res["vessel_exact"].append(f.get("vessel_name") is not None
                                       and f["vessel_name"].value.lower() == d.incident.vessel.name.lower())
        if "pob" not in d.missing_fields:
            res["pob_exact"].append(f.get("pob") is not None and f["pob"].value == d.incident.pob)
        if d.incident.vessel.mmsi and any(s.label.value == "MMSI" for s in d.spans):
            res["mmsi_exact"].append(f.get("mmsi") is not None and f["mmsi"].value == d.incident.vessel.mmsi)
        if "position" not in d.missing_fields:
            ok = False
            if "position" in f:
                err = haversine_nm(f["position"].value["lat"], f["position"].value["lon"],
                                   d.incident.position.lat, d.incident.position.lon)
                res["position_err_nm"].append(err)
                res[f"position_err_nm_{d.incident.position_format.value}"].append(err)
                # correct = within the format's inherent precision (relative bearings are coarse)
                ok = err <= max(1.0, f["position"].value["uncertainty_nm"])
            res["position_recovered"].append("position" in f)
            res["position_correct"].append(ok)
        for fld in CRITICAL_FIELDS:
            miss_true[fld].append(fld in d.missing_fields)
            miss_pred[fld].append(fld in r.missing_fields)
        contra_true.append(d.contradiction)
        contra_pred.append(bool(r.contradictions))
        inc_true.append([t.value for t in d.incident.incident_types])
        inc_pred.append(f["incident_types"].value)

    out = {k: float(np.mean(v)) for k, v in res.items() if not k.startswith("position_err")}
    for k, v in res.items():
        if k.startswith("position_err"):
            out[k.replace("err_nm", "median_err_nm")] = float(np.median(v))
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=INCIDENT_CLASSES)
    out["incident_macro_f1"] = float(f1_score(mlb.fit_transform(inc_true), mlb.transform(inc_pred), average="macro", zero_division=0))
    out["missing_info"] = {}
    for fld in CRITICAL_FIELDS:
        yt, yp = np.array(miss_true[fld]), np.array(miss_pred[fld])
        tp = int((yt & yp).sum())
        out["missing_info"][fld] = {
            "support": int(yt.sum()),
            "precision": tp / max(1, int(yp.sum())),
            "recall": tp / max(1, int(yt.sum())),
        }
    yt, yp = np.array(contra_true), np.array(contra_pred)
    out["contradiction"] = {"support": int(yt.sum()), "precision": int((yt & yp).sum()) / max(1, int(yp.sum())),
                            "recall": int((yt & yp).sum()) / max(1, int(yt.sum()))}
    return out


def end_to_end(pipe: Pipeline, test: list[Dialogue], metrics: dict, robustness_n: int) -> None:
    # 4. End-to-end field extraction + missing info
    metrics["end_to_end"] = field_eval(pipe, test)

    # 5. Robustness: extraction quality vs simulated ASR word error rate
    sub = test[:robustness_n]
    curve = []
    for rate in (0.0, 0.05, 0.1, 0.2, 0.3):
        rng = random.Random(42)
        noisy = [corrupt(d.transcript, rate, rng) for d in sub]
        wer = jiwer.wer([caller_text(d.transcript) for d in sub], [caller_text(t) for t in noisy])
        fe = field_eval(pipe, sub, noisy)
        curve.append({"corruption_rate": rate, "measured_wer": float(wer),
                      **{k: fe.get(k) for k in ("vessel_exact", "pob_exact", "position_recovered",
                                                 "position_correct", "incident_macro_f1")}})
        print(f"  corruption {rate:.2f}: WER={wer:.3f} pos_ok={fe.get('position_correct'):.2f} pob={fe.get('pob_exact'):.2f}")
    metrics["robustness_curve"] = curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--max-train", type=int, default=8000, help="cap CRF training size for laptop CPUs")
    ap.add_argument("--robustness-n", type=int, default=300)
    ap.add_argument("--eval-only", action="store_true", help="reuse models/pipeline.joblib, refresh end-to-end metrics")
    args = ap.parse_args()
    if args.eval_only:
        metrics = json.loads(Path("reports/metrics.json").read_text(encoding="utf-8"))
        test = load(args.data, "test")
        end_to_end(Pipeline.load(), test, metrics, args.robustness_n)
        Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return

    t0 = time.time()
    train, val, test = load(args.data, "train", args.max_train), load(args.data, "val"), load(args.data, "test")
    print(f"train={len(train)} val={len(val)} test={len(test)}")
    metrics: dict = {"sizes": {"train": len(train), "val": len(val), "test": len(test)}}

    # 1. NER ---------------------------------------------------------------------
    ner = CrfNer().fit(train)
    print(f"CRF trained in {time.time() - t0:.0f}s")
    counts = [ner_counts(ner, d) for d in test]
    labels = sorted({l for c in counts for l in c})
    metrics["ner"] = {
        "micro_f1": micro_f1(counts),
        "micro_f1_ci95": bootstrap_ci(counts, micro_f1, 300),
        "per_label_f1": {l: micro_f1(counts, l) for l in labels},
        "by_language": {}, "by_style": {}, "by_phrasing": {},
    }
    for key, attr in (("by_language", lambda d: d.language.value), ("by_style", lambda d: d.style),
                      ("by_phrasing", lambda d: "unseen" if d.unseen_phrasing else "seen")):
        groups = defaultdict(list)
        for d, c in zip(test, counts):
            groups[attr(d)].append(c)
        metrics["ner"][key] = {g: micro_f1(cs) for g, cs in groups.items()}

    # 2. Incident classification: keyword baseline vs TF-IDF + LR ---------------
    def texts(ds):
        return [caller_text(d.transcript) for d in ds]

    def gold(ds):
        return [[t.value for t in d.incident.incident_types] for d in ds]

    def phrases(d):
        return [s.text for s in d.spans if s.label.value == "INCIDENT"]

    baseline = KeywordIncidentBaseline().fit([(phrases(d), gold([d])[0]) for d in train])
    clf = IncidentClassifier().fit(texts(train), gold(train))
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer(classes=INCIDENT_CLASSES).fit([INCIDENT_CLASSES])

    def mf1(y, p):
        return float(f1_score(mlb.transform(y), mlb.transform(p), average="macro", zero_division=0))

    seen = [i for i, d in enumerate(test) if not d.unseen_phrasing]
    unseen = [i for i, d in enumerate(test) if d.unseen_phrasing]
    yb, yc, yg = baseline.predict(texts(test)), clf.predict(texts(test)), gold(test)
    metrics["incident"] = {
        "keyword_baseline_macro_f1": mf1(yg, yb),
        "tfidf_lr_macro_f1": mf1(yg, yc),
        "tfidf_lr_macro_f1_ci95": bootstrap_ci(list(zip(yg, yc)), lambda xs: mf1([a for a, _ in xs], [b for _, b in xs]), 300),
        "seen_phrasing": {"baseline": mf1([yg[i] for i in seen], [yb[i] for i in seen]),
                          "tfidf_lr": mf1([yg[i] for i in seen], [yc[i] for i in seen])},
        "unseen_phrasing": {"baseline": mf1([yg[i] for i in unseen], [yb[i] for i in unseen]),
                            "tfidf_lr": mf1([yg[i] for i in unseen], [yc[i] for i in unseen])},
        "per_class_f1": dict(zip(INCIDENT_CLASSES, map(float, f1_score(mlb.transform(yg), mlb.transform(yc), average=None, zero_division=0)))),
    }

    # 3. Severity: ordinal vs multinomial, temperature scaling, conformal --------
    oof = cross_val_predict(IncidentClassifier().model, texts(train), mlb.transform(gold(train)), cv=3, method="predict_proba")
    Xtr = sev_X(ner, oof, train)
    Xva = sev_X(ner, clf.predict_proba(texts(val)), val)
    Xte = sev_X(ner, clf.predict_proba(texts(test)), test)
    ytr, yva, yte = (np.array([int(d.incident.severity) for d in ds]) for ds in (train, val, test))

    ordinal = OrdinalLogistic().fit(Xtr, ytr)
    multi = TemperatureScaled().fit(Xtr, ytr)
    p_multi_raw = multi.predict_proba(Xte, calibrated=False)
    T = multi.calibrate(Xva, yva)
    p_multi = multi.predict_proba(Xte)
    p_ord = ordinal.predict_proba(Xte)

    def sev_metrics(p):
        pred = p.argmax(1) + 1
        return {"accuracy": float((pred == yte).mean()), "macro_f1": float(f1_score(yte, pred, average="macro")),
                "mae": float(mean_absolute_error(yte, pred)),
                "quadratic_kappa": float(cohen_kappa_score(yte, pred, weights="quadratic")),
                "ece": expected_calibration_error(p, yte - 1)}

    metrics["severity"] = {
        "majority_baseline_accuracy": float((yte == np.bincount(ytr).argmax()).mean()),
        "multinomial_uncalibrated": sev_metrics(p_multi_raw),
        "multinomial_temperature_scaled": {**sev_metrics(p_multi), "temperature": T},
        "ordinal_frank_hall": sev_metrics(p_ord),
    }
    chosen = ordinal if metrics["severity"]["ordinal_frank_hall"]["mae"] <= metrics["severity"]["multinomial_temperature_scaled"]["mae"] else multi
    conformal = SplitConformal(alpha=0.1).fit(chosen.predict_proba(Xva), yva - 1)
    sets = conformal.predict_sets(chosen.predict_proba(Xte))
    metrics["severity"]["chosen_model"] = type(chosen).__name__
    metrics["severity"]["conformal_90"] = {
        "empirical_coverage": float(np.mean([y - 1 in s for y, s in zip(yte, sets)])),
        "mean_set_size": float(np.mean([len(s) for s in sets])),
    }
    # Linear-model explanation: global feature weights for P(severity >= HIGH).
    w = ordinal.models[1].coef_[0]
    metrics["severity"]["top_features_high_or_worse"] = sorted(
        zip(FEATURE_NAMES, map(float, w)), key=lambda kv: -abs(kv[1]))[:8]

    pipe = Pipeline(ner, clf, chosen, conformal)
    pipe.feature_means = Xtr.mean(0)
    pipe.save("models/pipeline.joblib")

    end_to_end(pipe, test, metrics, args.robustness_n)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("ner", "incident", "severity")}, indent=2)[:4000])
    print(f"done in {time.time() - t0:.0f}s -> models/pipeline.joblib, reports/metrics.json")


if __name__ == "__main__":
    main()
