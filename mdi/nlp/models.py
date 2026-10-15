"""Phase 3/4 models: incident classification, ordinal severity, calibration, conformal sets."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from mdi.schema import IncidentType

INCIDENT_CLASSES = [t.value for t in IncidentType]


def caller_text(transcript: str) -> str:
    return " ".join(line[len("CALLER: "):] for line in transcript.splitlines() if line.startswith("CALLER: "))


# --- incident type (multi-label) -------------------------------------------

class KeywordIncidentBaseline:
    """Baseline: remember every incident phrase seen in training, match by substring."""

    def fit(self, texts_and_spans: list[tuple[list[str], list[str]]]) -> "KeywordIncidentBaseline":
        self.lexicon: dict[str, str] = {}
        for phrases, labels in texts_and_spans:
            if len(phrases) == len(labels):
                self.lexicon.update({p.lower(): l for p, l in zip(phrases, labels)})
        return self

    def predict(self, texts: list[str]) -> list[list[str]]:
        return [sorted({l for p, l in self.lexicon.items() if p in t.lower()}) for t in texts]


class IncidentClassifier:
    """TF-IDF (char + word n-grams) + one-vs-rest logistic regression."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.mlb = MultiLabelBinarizer(classes=INCIDENT_CLASSES)
        feats = FeatureUnion([
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ])
        self.model = make_pipeline(feats, OneVsRestClassifier(LogisticRegression(C=8.0, max_iter=2000)))

    def fit(self, texts: list[str], labels: list[list[str]]) -> "IncidentClassifier":
        self.model.fit(texts, self.mlb.fit_transform(labels))
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(texts)

    def predict(self, texts: list[str]) -> list[list[str]]:
        out = []
        for p in self.predict_proba(texts):
            idx = np.where(p >= self.threshold)[0]
            if len(idx) == 0:
                idx = [int(np.argmax(p))]  # a distress call always has some nature
            out.append([INCIDENT_CLASSES[i] for i in idx])
        return out


# --- severity -----------------------------------------------------------------

VESSEL_TYPES = ["fishing_boat", "trawler", "cargo", "tanker", "ferry", "yacht", "tug"]


def severity_features(incident_proba: np.ndarray, pob: int | None, is_mayday: bool, vessel_type: str | None) -> np.ndarray:
    """Tabular features built from the *extracted* fields, so errors propagate realistically."""
    vt = np.array([vessel_type == v for v in VESSEL_TYPES], dtype=float)
    pob_f = [np.log1p(pob) if pob else 0.0, float(pob is None)]
    n_types = float((incident_proba >= 0.5).sum())
    return np.concatenate([incident_proba, pob_f, [float(is_mayday), n_types], vt])


FEATURE_NAMES = (
    [f"p_{c}" for c in INCIDENT_CLASSES] + ["log_pob", "pob_missing", "is_mayday", "n_incident_types"]
    + [f"vt_{v}" for v in VESSEL_TYPES]
)


class OrdinalLogistic:
    """Frank & Hall ordinal classifier: K-1 binary models for P(y > k).

    Respects the order LOW < MODERATE < HIGH < CRITICAL, unlike plain
    multinomial regression; the Phase 4 roadmap's CORAL is the neural analogue.
    """

    def __init__(self, classes=(1, 2, 3, 4), C: float = 1.0):
        self.classes = list(classes)
        self.models = [LogisticRegression(C=C, max_iter=2000) for _ in self.classes[:-1]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "OrdinalLogistic":
        for k, m in zip(self.classes[:-1], self.models):
            m.fit(X, (y > k).astype(int))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        gt = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models])
        gt = np.minimum.accumulate(gt, axis=1)  # enforce P(y>1) >= P(y>2) >= ...
        cum = np.column_stack([np.ones(len(X)), gt, np.zeros(len(X))])
        return np.clip(cum[:, :-1] - cum[:, 1:], 1e-9, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.classes)[self.predict_proba(X).argmax(1)]


class TemperatureScaled:
    """Multinomial LR whose logits are divided by a temperature fit on validation data."""

    def __init__(self, C: float = 1.0):
        self.lr = LogisticRegression(C=C, max_iter=2000)
        self.T = 1.0

    def fit(self, X, y):
        self.lr.fit(X, y)
        self.classes = list(self.lr.classes_)
        return self

    def calibrate(self, X_val, y_val) -> float:
        logits = self.lr.decision_function(X_val)
        idx = np.searchsorted(self.classes, y_val)

        def nll(t: float) -> float:
            p = softmax(logits / t, axis=1)
            return -np.mean(np.log(p[np.arange(len(idx)), idx] + 1e-12))

        self.T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        return self.T

    def predict_proba(self, X, calibrated: bool = True):
        return softmax(self.lr.decision_function(X) / (self.T if calibrated else 1.0), axis=1)

    def predict(self, X):
        return np.array(self.classes)[self.predict_proba(X).argmax(1)]


def expected_calibration_error(proba: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(1)
    correct = proba.argmax(1) == y_idx
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


class SplitConformal:
    """Split-conformal prediction sets with marginal coverage >= 1 - alpha."""

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha

    def fit(self, cal_proba: np.ndarray, cal_y_idx: np.ndarray) -> "SplitConformal":
        scores = 1 - cal_proba[np.arange(len(cal_y_idx)), cal_y_idx]
        n = len(scores)
        q = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self.qhat = float(np.quantile(scores, q, method="higher"))
        return self

    def predict_sets(self, proba: np.ndarray) -> list[list[int]]:
        return [list(np.where(1 - p <= self.qhat)[0]) or [int(p.argmax())] for p in proba]
