"""CRF pipeline vs zero-shot LLM vs hybrid vs learned per-field routing, on external data only.

Routing is chosen on DEV data only (SeaAlert validation split + MARTTS dev part):
for each critical field, keep whichever merge rule (CRF first / LLM first / LLM
only) is most accurate there, then score it once on the test samples.

Same samples, same metrics as evaluate_seaalert.py (experiment B) and
evaluate_martts.py, so numbers are directly comparable with earlier reports.
Small samples keep CPU time low (the LLM runs ~5-10 s per call on a laptop).

Usage:  python scripts/evaluate_llm.py --seaalert 40 --martts-per-topic 5
Output: reports/llm_extraction.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate_martts as em  # noqa: E402
import evaluate_seaalert as es  # noqa: E402
from mdi.nlp.llm import CRITICAL, ROUTING_FILE, LlmPipeline  # noqa: E402
from mdi.pipeline import Pipeline  # noqa: E402


# (SeaAlert key, MARTTS key) per critical field; MARTTS has no POB gold.
FIELD_KEYS = {"vessel_name": ("vessel", "vessel_exact"), "mmsi": ("mmsi", "mmsi_exact"),
              "position": ("position", "position_within_1nm"), "pob": ("pob", None)}
RULES = {"crf_first": "hybrid", "llm_first": "llm_first", "llm": "llm"}


def score(system, sa, mt) -> dict:
    return {"seaalert": es.experiment_b(system, sa), "martts": em.evaluate(system, [(d, em.to_transcript(d)) for d in mt])}


def pooled(res: dict, field: str) -> float:
    """n-weighted accuracy over SeaAlert (clean + ASR-high) and MARTTS."""
    sa_key, mt_key = FIELD_KEYS[field]
    parts = [res["seaalert"][s][sa_key] for s in es.SETTINGS if sa_key in res["seaalert"][s]]
    if mt_key and mt_key in res["martts"]:
        parts.append({"value": res["martts"][mt_key]["rate"], "n": res["martts"][mt_key]["n"]})
    n = sum(p["n"] for p in parts)
    return sum(p["value"] * p["n"] for p in parts) / n if n else float("nan")


def martts_sample(part: str, per_topic: int) -> list[dict]:
    dialogues = [json.loads(l) for l in open(em.ROOT / "scenarios.jsonl", encoding="utf-8")]
    seen: Counter = Counter()
    return [d for d in dialogues if em.part_of(d["dialog_id"]) == part
            and (seen.update([d["topic"]]) or seen[d["topic"]] <= per_topic)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seaalert", type=int, default=30, help="SeaAlert test messages")
    ap.add_argument("--martts-per-topic", type=int, default=3, help="MARTTS test dialogues per topic")
    ap.add_argument("--dev-seaalert", type=int, default=20, help="SeaAlert validation messages for routing")
    ap.add_argument("--dev-martts-per-topic", type=int, default=3)
    args = ap.parse_args()

    es.SETTINGS = ["text", "asr_high"]  # clean text and the noisiest ASR condition only
    _, sa_val, sa_test = es.split(pd.read_csv(es.CSV))
    sa_test, sa_dev = sa_test.head(args.seaalert), sa_val.head(args.dev_seaalert)
    mt, mt_dev = martts_sample("test", args.martts_per_topic), martts_sample("dev", args.dev_martts_per_topic)
    base = Pipeline.load()

    # 1. Learn the per-field routing on dev data.
    dev = {rule: score(LlmPipeline(base, mode), sa_dev, mt_dev) for rule, mode in RULES.items()}
    dev_acc = {f: {rule: round(pooled(dev[rule], f), 4) for rule in RULES} for f in CRITICAL}
    policy = {f: max(RULES, key=lambda r: (dev_acc[f][r], r == "crf_first")) for f in CRITICAL}  # ties -> CRF
    ROUTING_FILE.write_text(json.dumps({"policy": policy, "dev_accuracy": dev_acc,
                                        "dev_sizes": {"seaalert_val": len(sa_dev), "martts_dev": len(mt_dev)}},
                                       indent=2), encoding="utf-8")
    print("routing learned on dev:", policy, dev_acc)

    # 2. Score every system once on test.
    systems = {"crf_pipeline": base, "llm_zero_shot": LlmPipeline(base, "llm"), "hybrid": LlmPipeline(base, "hybrid"),
               "routed": LlmPipeline(base, policy=policy)}
    report: dict = {"model": "Qwen2.5-1.5B-Instruct Q4_K_M (llama.cpp, JSON-schema constrained)",
                    "sizes": {"seaalert_test": len(sa_test), "martts_test": len(mt)},
                    "routing": {"policy": policy, "dev_accuracy": dev_acc}}
    for name, sys_ in systems.items():
        t0 = time.perf_counter()
        r = score(sys_, sa_test, mt)
        report[name] = {"seaalert": r["seaalert"], "martts_test_text": r["martts"]}
        report[name]["seaalert"].pop("published_asr_high", None)
        report[name]["seconds"] = round(time.perf_counter() - t0, 1)
        if isinstance(sys_, LlmPipeline):
            report[name]["grounding_rejections"] = dict(Counter(sys_.rejected))
        print(f"{name} done in {report[name]['seconds']} s")
    report["published_asr_high"] = es.PUBLISHED_EXTRACTION_ASR_HIGH

    Path("reports").mkdir(exist_ok=True)
    Path("reports/llm_extraction.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in systems:
        sa = report[name]["seaalert"]
        mm = report[name]["martts_test_text"]
        print(f"\n{name}")
        for s in es.SETTINGS:
            print(f"  SeaAlert {s:9s}", {k: round(v["value"], 3) for k, v in sa[s].items() if not k.startswith("severity")})
        print("  MARTTS test     ", {k: round(v["rate"], 3) for k, v in mm.items()
                                     if k in ("vessel_exact", "mmsi_exact", "position_within_1nm")})
        print("  rejected by grounding guard:", report[name].get("grounding_rejections"))


if __name__ == "__main__":
    main()
