"""Confusion matrices and reliability diagram data for the saved pipeline (test split).

* Severity: 4x4 confusion matrix (counts and row-normalised = recall per true class).
* Incident types: one 2x2 confusion matrix per class (multi-label).
* Calibration: reliability bins (mean confidence vs accuracy per bin), before and
  after temperature scaling; ECE is the weighted gap between the two.

Usage:  python scripts/severity_diagnostics.py
Output: reports/severity_diagnostics.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdi.nlp.models import INCIDENT_CLASSES, TemperatureScaled, caller_text, expected_calibration_error  # noqa: E402
from mdi.pipeline import SEVERITY_NAMES, Pipeline  # noqa: E402
from train_and_evaluate import load, sev_X  # noqa: E402


def reliability(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> list[dict]:
    conf, correct = proba.max(1), proba.argmax(1) == y_idx
    out = []
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out.append({"bin": [round(lo, 1), round(hi, 1)], "n": int(m.sum()),
                        "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return out


def main() -> None:
    pipe = Pipeline.load()
    test = load(Path("data/synthetic"), "test")
    texts = [caller_text(d.transcript) for d in test]
    inc_proba = pipe.incident_clf.predict_proba(texts)
    X = sev_X(pipe.ner, inc_proba, test)
    y = np.array([int(d.incident.severity) for d in test])
    model = pipe.severity_model
    p = model.predict_proba(X)
    names = [SEVERITY_NAMES[k] for k in (1, 2, 3, 4)]

    cm = confusion_matrix(y, p.argmax(1) + 1, labels=[1, 2, 3, 4])
    report = {
        "n": int(len(y)), "model": type(model).__name__, "labels": names,
        "severity_confusion": cm.tolist(),
        "severity_confusion_row_normalised": (cm / cm.sum(1, keepdims=True)).round(4).tolist(),
        "reliability_calibrated": reliability(p, y - 1),
        "ece_calibrated": expected_calibration_error(p, y - 1),
    }
    # Does the 90% conformal set still contain the true level when the top-1 guess is wrong?
    sets = pipe.conformal.predict_sets(p)
    report["conformal_coverage_by_true_class"] = {
        n: float(np.mean([k - 1 in s for k, s in zip(y, sets) if k == lvl])) for lvl, n in zip((1, 2, 3, 4), names)}
    if isinstance(model, TemperatureScaled):
        raw = model.predict_proba(X, calibrated=False)
        report["reliability_uncalibrated"] = reliability(raw, y - 1)
        report["ece_uncalibrated"] = expected_calibration_error(raw, y - 1)
        report["temperature"] = model.T

    mlb = MultiLabelBinarizer(classes=INCIDENT_CLASSES).fit([INCIDENT_CLASSES])
    gold = mlb.transform([[t.value for t in d.incident.incident_types] for d in test])
    pred = mlb.transform(pipe.incident_clf.predict(texts))
    report["incident_confusion"] = {c: {"tn": int(m[0, 0]), "fp": int(m[0, 1]), "fn": int(m[1, 0]), "tp": int(m[1, 1])}
                                    for c, m in zip(INCIDENT_CLASSES, multilabel_confusion_matrix(gold, pred))}

    Path("reports/severity_diagnostics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("severity confusion (rows = true, cols = predicted):", names)
    for n, row in zip(names, cm):
        print(f"  {n:9s}", row.tolist())
    print("conformal coverage by true class:", {k: round(v, 3) for k, v in report["conformal_coverage_by_true_class"].items()})
    print("ECE", round(report.get("ece_uncalibrated", float("nan")), 3), "->", round(report["ece_calibrated"], 3))
    print("incident FN/FP:", {c: (v["fn"], v["fp"]) for c, v in report["incident_confusion"].items()})


if __name__ == "__main__":
    main()
