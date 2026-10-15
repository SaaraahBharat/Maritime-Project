"""External evaluation on MARTTS (DLR, CC BY 4.0): 240 English maritime radio
dialogues made by a different team, generator and TTS voice than ours.

The corpus is split by *base scenario* (variants _a/_b/_c share a script and
differ only in vessel names), so no script appears in both parts:
  dev  (~30%): may be read for error analysis and fixes
  test (~70%): touched only for final reporting

Gold labels come from the corpus: the topic gives the incident type; vessel
name, MMSI and position are read from the distressed vessel's first turn, whose
format is fixed by the MARTTS templates.

Usage:
    python scripts/evaluate_martts.py                 # text transcripts, dev + test
    python scripts/evaluate_martts.py --audio --part test --model small
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.data.spoken import haversine_nm  # noqa: E402
from mdi.nlp.position import normalise_numbers  # noqa: E402
from mdi.pipeline import Pipeline  # noqa: E402

ROOT = Path("data/external/martts")
# MARTTS topic -> acceptable incident types in our taxonomy (None = not a distress incident).
TOPIC_MAP = {
    "Collision": {"collision"}, "Fire": {"fire"}, "Flooding": {"flooding"},
    "Sinking": {"flooding", "capsizing"}, "PanPan": None, "SAR": None, "Regular": None, "Safety": None,
}
DISTRESS_TOPICS = {"Collision", "Fire", "Flooding", "Sinking"}


def part_of(dialog_id: str) -> str:
    base = dialog_id.rsplit("_", 1)[0]
    return "dev" if int(hashlib.md5(base.encode()).hexdigest(), 16) % 10 < 3 else "test"


def to_transcript(d: dict) -> str:
    """Speaker 1 opens the call (the vessel in trouble) -> CALLER; everyone else -> MRCC side."""
    caller = d["utterances"][0]["speaker_id"]
    return "\n".join(f"{'CALLER' if u['speaker_id'] == caller else 'MRCC'}: {u['text']}" for u in d["utterances"])


def gold_fields(d: dict) -> dict:
    first = d["utterances"][0]["text"]
    g: dict = {"topic": d["topic"], "incident": TOPIC_MAP[d["topic"]]}
    m = re.search(r"[Tt]his is (?:MV |M/V |SV |FV )?([A-Z0-9][A-Z0-9 \-']+?)(?:,| MMSI|\.)", first)
    g["vessel"] = m[1].strip() if m else None
    m = re.search(r"MMSI(?: number)? ((?:(?:zero|one|two|three|four|five|six|seven|eight|nine|niner)[ ,]*){9})", first, re.I)
    g["mmsi"] = "".join(t for t in normalise_numbers(m[1]) if t.isdigit()) if m else None
    norm = " ".join(normalise_numbers(first))
    m = re.search(r"(\d+) point (\d+) degrees (north|south) (\d+) point (\d+) degrees (east|west)", norm)
    if m:
        lat = float(f"{m[1]}.{m[2]}") * (1 if m[3] == "north" else -1)
        lon = float(f"{m[4]}.{m[5]}") * (1 if m[6] == "east" else -1)
        g["position"] = (lat, lon)
    else:
        g["position"] = None
    return g


def evaluate(pipe: Pipeline, items: list[tuple[dict, str]]) -> dict:
    res = defaultdict(list)
    sev_by_topic = defaultdict(list)
    for d, transcript in items:
        g = gold_fields(d)
        r = pipe.extract(transcript)
        f = r.fields
        pred_types = set(f["incident_types"].value)
        sev_by_topic[g["topic"]].append(r.severity)
        if g["incident"] is not None:
            res["incident_hit"].append(bool(pred_types & g["incident"]))
            res["incident_top1_hit"].append(max(r.incident_proba, key=r.incident_proba.get) in g["incident"])
        if g["vessel"]:
            v = f.get("vessel_name")
            res["vessel_found"].append(v is not None)
            res["vessel_exact"].append(v is not None and v.value.upper().replace("M V - ", "") == g["vessel"].upper())
        if g["mmsi"] and len(g["mmsi"]) == 9:
            res["mmsi_exact"].append(f.get("mmsi") is not None and f["mmsi"].value == g["mmsi"])
        if g["position"]:
            p = f.get("position")
            res["position_recovered"].append(p is not None)
            res["position_within_1nm"].append(
                p is not None and haversine_nm(p.value["lat"], p.value["lon"], *g["position"]) <= 1.0)
        # Our system has no "not distress" output; measure how it treats routine traffic.
        if g["topic"] in ("Regular", "Safety"):
            res["routine_flagged_high_or_critical"].append(r.severity >= 3)
    out = {k: {"rate": float(np.mean(v)), "n": len(v)} for k, v in res.items()}
    out["mean_severity_by_topic"] = {t: float(np.mean(s)) for t, s in sorted(sev_by_topic.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["dev", "test", "both"], default="both")
    ap.add_argument("--audio", action="store_true", help="run Whisper on MARTTS noisy audio first")
    ap.add_argument("--model", default="small")
    ap.add_argument("--per-topic", type=int, default=None, help="audio: cap dialogues per topic (CPU time)")
    args = ap.parse_args()

    pipe = Pipeline.load()
    dialogues = [json.loads(l) for l in open(ROOT / "scenarios.jsonl", encoding="utf-8")]
    parts = ["dev", "test"] if args.part == "both" else [args.part]
    report: dict = {"counts": Counter(part_of(d["dialog_id"]) for d in dialogues)}

    for part in parts:
        subset = [d for d in dialogues if part_of(d["dialog_id"]) == part]
        report[f"{part}_text"] = evaluate(pipe, [(d, to_transcript(d)) for d in subset])
        if args.audio:
            sample = subset
            if args.per_topic:
                seen: Counter = Counter()
                sample = [d for d in subset if (seen.update([d["topic"]]) or seen[d["topic"]] <= args.per_topic)]
            report[f"{part}_asr"] = evaluate_audio(pipe, sample, args.model)
            report[f"{part}_text_same_sample"] = evaluate(pipe, [(d, to_transcript(d)) for d in sample])

    Path("reports").mkdir(exist_ok=True)
    out = Path(f"reports/martts{'_audio' if args.audio else ''}.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def evaluate_audio(pipe: Pipeline, subset: list[dict], model: str) -> dict:
    """Whisper on each MARTTS noisy dialogue WAV, then the same field evaluation."""
    import jiwer

    from mdi.speech import transcribe
    from scripts.evaluate_asr import norm

    audio_dir = ROOT / "dataset_noise"
    wavs = {p.stem: p for p in audio_dir.rglob("*.wav")}
    items, refs, hyps = [], [], []
    for d in subset:
        topic, rest = d["dialog_id"].split("/")
        wav = wavs.get(f"{topic}_{rest.replace('scenario_', '')}")  # "Collision/scenario_10_a" -> Collision_10_a.wav
        if wav is None:
            continue
        text = transcribe(str(wav), None, size=model)["text"]
        # Whole-dialogue audio has no speaker turns; treat it all as the caller side.
        items.append((d, "CALLER: " + text))
        refs.append(norm(" ".join(u["text"] for u in d["utterances"])))
        hyps.append(norm(text) or "<empty>")
        print(f"  {d['dialog_id']} transcribed")
    out = evaluate(pipe, items)
    out["wer"] = float(jiwer.wer(refs, hyps)) if refs else None
    out["n_audio"] = len(items)
    return out


if __name__ == "__main__":
    main()
