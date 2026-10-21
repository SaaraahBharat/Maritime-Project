import random

import numpy as np
import pytest

from mdi.data.generator import GeneratorConfig, generate
from mdi.data.noise_text import corrupt
from mdi.data.spoken import haversine_nm
from mdi.nlp.models import ClassConditionalConformal, OrdinalLogistic, SplitConformal, expected_calibration_error
from mdi.nlp.position import normalise_numbers, parse_count, parse_position
from mdi.response import engine


@pytest.mark.parametrize("text,lat,lon", [
    ("zero nine degrees four eight decimal seven north, zero seven six degrees one four decimal five east", 9.8117, 76.2417),
    ("09°48.7'N 076°14.5'E", 9.8117, 76.2417),
    ("nine point eight one north, seventy six point two four east", 9.81, 76.24),
    ("nine degrees forty eight minutes forty two seconds north, seventy six degrees fourteen minutes thirty seconds east", 9.8117, 76.2417),
])
def test_parse_absolute_positions(text, lat, lon):
    p = parse_position(text)
    assert p is not None and haversine_nm(p.lat, p.lon, lat, lon) < 0.5


def test_parse_relative_position():
    p = parse_position("Kochi lighthouse से 12 मील southwest")
    assert p is not None and p.fmt == "relative"
    assert haversine_nm(p.lat, p.lon, 9.97, 76.24) == pytest.approx(12, abs=0.1)


def test_number_normalisation():
    assert normalise_numbers("one five persons") == ["15", "persons"]
    assert parse_count("forty two") == 42
    assert parse_count("ஐந்து பேர்") == 5
    assert parse_count("one hundred and twenty") == 120


def test_parser_recovers_all_generated_positions():
    for d in generate(GeneratorConfig(n=300, seed=11)):
        for s in d.spans:
            if s.label.value == "POSITION":
                p = parse_position(s.text)
                assert p is not None, s.text
                assert haversine_nm(p.lat, p.lon, d.incident.position.lat, d.incident.position.lon) <= max(1.0, p.uncertainty_nm)


def test_corrupt_only_touches_caller_lines():
    t = "CALLER: mayday this is sea hawk position one two\nMRCC: say again, over"
    out = corrupt(t, 1.0, random.Random(0))
    assert out.splitlines()[1] == "MRCC: say again, over"
    assert out.splitlines()[0] != t.splitlines()[0]


def test_ordinal_probabilities_valid():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    y = np.clip(np.round(X[:, 0] * 1.2 + 2.5), 1, 4).astype(int)
    p = OrdinalLogistic().fit(X, y).predict_proba(X)
    assert np.allclose(p.sum(1), 1, atol=1e-6) and (p >= 0).all()


def test_conformal_coverage_on_synthetic_probs():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4000, 4))
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rng.choice(4, p=row) for row in p])  # labels drawn from the probs: calibrated
    cp = SplitConformal(0.1).fit(p[:2000], y[:2000])
    sets = cp.predict_sets(p[2000:])
    cov = np.mean([yy in s for yy, s in zip(y[2000:], sets)])
    assert 0.87 <= cov <= 0.93
    assert expected_calibration_error(p, y) < 0.05


def test_class_conditional_conformal_covers_a_hard_class():
    # Class 3 is "hard": the model puts little mass on it, so marginal sets under-cover it.
    rng = np.random.default_rng(2)
    n = 6000
    y = rng.choice(4, size=n, p=[0.32, 0.32, 0.31, 0.05])
    p = np.full((n, 4), 0.1)
    p[np.arange(n), y] = 0.7
    hard = y == 3
    p[hard] = [0.15, 0.15, 0.5, 0.2]
    p /= p.sum(1, keepdims=True)
    cal, te = slice(0, 3000), slice(3000, None)

    def cov_hard(cp):
        sets = cp.predict_sets(p[te])
        return np.mean([3 in s for s, yy in zip(sets, y[te]) if yy == 3])

    assert cov_hard(SplitConformal(0.1).fit(p[cal], y[cal])) < 0.5
    assert cov_hard(ClassConditionalConformal(0.1).fit(p[cal], y[cal])) >= 0.85


def test_assignment_prefers_critical_incident():
    a = [engine.Asset("near", "boat", 10.0, 76.0, 30), engine.Asset("far", "boat", 11.0, 76.0, 30)]
    inc = [{"id": "low", "lat": 10.05, "lon": 76.0, "severity": 1},
           {"id": "crit", "lat": 10.1, "lon": 76.0, "severity": 4}]
    out = {r["incident"]: r["asset"] for r in engine.assign(a, inc)}
    assert out["crit"] == "near"


def test_drift_cloud_grows_with_time():
    cond = {"wind_kn": 15, "wind_from_deg": 240, "current_kn": 0.5, "current_to_deg": 60}
    r1 = engine.search_area(engine.drift_particles(10, 75.5, 1, cond, "fishing_boat", 0.5, n=200))["r90_nm"]
    r6 = engine.search_area(engine.drift_particles(10, 75.5, 6, cond, "fishing_boat", 0.5, n=200))["r90_nm"]
    assert r6 > r1
