import random

import numpy as np
import pytest

from mdi.audio.channel import VHF_SR, simulate_vhf
from mdi.data import spoken
from mdi.data.generator import GeneratorConfig, generate
from mdi.schema import Language, Position


@pytest.fixture(scope="module")
def dialogues():
    return generate(GeneratorConfig(n=600, seed=7))


def test_spans_match_transcript(dialogues):
    for d in dialogues:
        for s in d.spans:
            assert d.transcript[s.start : s.end] == s.text


def test_generation_is_deterministic():
    a = generate(GeneratorConfig(n=50, seed=3))
    b = generate(GeneratorConfig(n=50, seed=3))
    assert [d.transcript for d in a] == [d.transcript for d in b]


def _pool(dialogues, split, key):
    return {key(d) for d in dialogues if d.split == split}


@pytest.mark.parametrize("key", [lambda d: d.incident.vessel.name, lambda d: d.incident.nearest_landmark])
def test_vessel_and_landmark_pools_disjoint(dialogues, key):
    tr, va, te = (_pool(dialogues, s, key) for s in ("train", "val", "test"))
    assert tr and va and te
    assert not (tr & va) and not (tr & te) and not (va & te)


def test_split_is_stratified():
    ds = generate(GeneratorConfig(n=4000, seed=5))
    strata = {}
    for d in ds:
        strata.setdefault((d.language, int(d.incident.severity)), []).append(d.split)
    for splits in strata.values():
        if len(splits) >= 100:
            assert abs(splits.count("train") / len(splits) - 0.8) < 0.03
            assert abs(splits.count("test") / len(splits) - 0.1) < 0.03


def test_train_never_uses_heldout_phrasing(dialogues):
    assert not any(d.unseen_phrasing for d in dialogues if d.split == "train")
    assert any(d.unseen_phrasing for d in dialogues if d.split != "train")


def test_missing_fields_really_missing(dialogues):
    labels = {"vessel_name": "VESSEL", "position": "POSITION", "incident_types": "INCIDENT", "pob": "POB"}
    for d in dialogues:
        present = {s.label.value for s in d.spans}
        for f in d.missing_fields:
            assert labels[f] not in present, (d.id, f)


def test_all_languages_and_styles_present(dialogues):
    assert {d.language for d in dialogues} == set(Language)
    assert {d.style for d in dialogues} == {"textbook", "hurried", "panicked"}


def test_ddm_rounding_never_gives_60_minutes():
    text = spoken.position_ddm(Position(lat=9.99999, lon=76.0), random.Random(0))
    assert "six zero decimal" not in text and "60.0" not in text


def test_destination_roundtrip():
    lat, lon = spoken.destination(10.0, 76.0, 250, 20)
    assert spoken.haversine_nm(10.0, 76.0, lat, lon) == pytest.approx(20, rel=1e-3)


def test_vhf_channel_shape_and_band():
    rng = np.random.default_rng(0)
    sr = 16000
    t = np.arange(sr) / sr
    tone = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 100 * t)  # 100 Hz should be cut
    y, params = simulate_vhf(tone, sr, rng, snr_db=30)
    assert y.dtype == np.float32 and np.max(np.abs(y)) <= 1.0
    assert len(y) >= VHF_SR
    spec = np.abs(np.fft.rfft(y[:VHF_SR]))
    freqs = np.fft.rfftfreq(VHF_SR, 1 / VHF_SR)
    assert spec[np.argmin(abs(freqs - 1000))] > 10 * spec[np.argmin(abs(freqs - 100))]
    assert params.snr_db == 30
