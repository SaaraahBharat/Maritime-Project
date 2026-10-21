"""Rebuild the Kathbath/IndicVoices ASR report from a (possibly partial) predictions.jsonl.

evaluate_indic_asr.py writes one prediction line per utterance as it goes, but
only aggregates at the end; if a run is interrupted, this recovers the metrics
for every utterance that finished. Same normalisation and bootstrap as the evaluator.

Usage:  python scripts/summarize_indic_asr.py reports/indic_asr_n8
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_asr import norm  # noqa: E402
from scripts.evaluate_indic_asr import bootstrap_ci, paired_wer_delta_ci  # noqa: E402


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "reports/indic_asr_n8")
    rows = [json.loads(l) for l in open(out_dir / "predictions.jsonl", encoding="utf-8")]
    by = defaultdict(lambda: defaultdict(lambda: {"refs": [], "hyps": [], "latency": []}))
    for r in rows:
        for cond, preds in r["predictions"].items():
            for model, p in preds.items():
                v = by[(r["language"], cond)][model]
                v["refs"].append(norm(r["reference"]))
                v["hyps"].append(norm(p["text"]))
                v["latency"].append(p["latency_s"])

    results = []
    for (lang, cond), models in sorted(by.items()):
        names = sorted(models, key=lambda m: not m.startswith("faster-whisper"))  # baseline first
        for m in names:
            v = models[m]
            results.append({"language": lang, "condition": cond, "model": m, "n": len(v["refs"]),
                            "wer": jiwer.wer(v["refs"], v["hyps"]),
                            "wer_ci95": bootstrap_ci(v["refs"], v["hyps"], "wer", 500, 13),
                            "cer": jiwer.cer(v["refs"], v["hyps"]),
                            "cer_ci95": bootstrap_ci(v["refs"], v["hyps"], "cer", 500, 13)})
        if len(names) > 1:
            b, c = models[names[0]], models[names[1]]
            results.append({"language": lang, "condition": cond, "comparison": f"{names[1]} minus {names[0]}",
                            "delta_wer": jiwer.wer(b["refs"], c["hyps"]) - jiwer.wer(b["refs"], b["hyps"]),
                            "delta_wer_ci95": paired_wer_delta_ci(b["refs"], b["hyps"], c["hyps"], 500, 13)})
    report = {"dataset": rows[0].get("dataset", "ai4bharat/Kathbath") if rows else None, "split": "valid",
              "scope_note": "Real Indian speech with simulated VHF channel effects; not real maritime distress calls.",
              "recovered_from_predictions": True, "n_utterances": len(rows), "results": results}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for r in results:
        if "comparison" in r:
            print(f"{r['language']} {r['condition']:5s} delta WER {r['delta_wer']:+.3f} CI {[round(x, 3) for x in r['delta_wer_ci95']]}")
        else:
            print(f"{r['language']} {r['condition']:5s} {r['model']:32s} n={r['n']} WER {r['wer']:.3f} CER {r['cer']:.3f}")


if __name__ == "__main__":
    main()
