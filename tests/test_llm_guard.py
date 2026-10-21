"""LLM extractor safety: grounding guard and per-field merge rules (LLM stubbed, no model needed)."""

import json

import pytest

from mdi.nlp import llm

T = ("CALLER: mayday this is Sea Hawk MMSI two three five one two three four five six, position one zero "
     "degrees zero five decimal two north zero seven five degrees five eight decimal one east, six persons on board")


@pytest.fixture
def stub(monkeypatch):
    def make(raw: dict):
        llm.extract.cache_clear()
        monkeypatch.setattr(llm, "_complete", lambda transcript: json.dumps(raw))
    yield make
    llm.extract.cache_clear()


def test_grounded_fields_are_normalised_by_parsers(stub):
    stub({"vessel_name": "Sea Hawk", "mmsi_quote": "two three five one two three four five six",
          "position_quote": "one zero degrees zero five decimal two north zero seven five degrees five eight decimal one east",
          "persons_on_board_quote": "six persons on board"})
    f = llm.extract(T)
    assert (f.vessel_name, f.mmsi, f.pob, f.rejected) == ("Sea Hawk", "235123456", 6, [])
    assert abs(f.position["lat"] - 10.0867) < 1e-3


def test_hallucinated_or_mislabelled_spans_are_rejected(stub):
    stub({"vessel_name": "Ocean Queen",  # not in the transcript
          "mmsi_quote": "one zero degrees zero five decimal two north",  # a position, not an MMSI
          "position_quote": None, "persons_on_board_quote": None})
    f = llm.extract(T)
    assert f.vessel_name is None and f.mmsi is None
    assert set(f.rejected) == {"vessel_name", "mmsi"}


def test_unparseable_output_extracts_nothing(monkeypatch):
    llm.extract.cache_clear()
    monkeypatch.setattr(llm, "_complete", lambda transcript: '{"vessel_name": "Sea Ha')
    assert llm.extract(T).rejected == ["unparseable"]
    llm.extract.cache_clear()


def test_merge_rules():
    from mdi.pipeline import Field, IncidentReport

    rep = IncidentReport(fields={"vessel_name": Field("SEA HAWK", 0.9, "SEA HAWK")}, missing_fields=["pob"])
    f = llm.LlmFields(vessel_name="Sea Hawk", pob=6, raw={"vessel_name": "Sea Hawk", "persons_on_board_quote": "six"})
    out = llm.apply(rep, f, {"vessel_name": "crf_first", "pob": "crf_first"})
    assert out.fields["vessel_name"].value == "SEA HAWK" and out.fields["pob"].value == 6
    assert out.missing_fields == []
    out = llm.apply(rep, f, {"vessel_name": "llm_first"})
    assert out.fields["vessel_name"].value == "Sea Hawk"
