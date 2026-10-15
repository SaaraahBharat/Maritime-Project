"""Phase 2: Whisper on the synthetic VHF audio -> WER/CER by language and SNR,
then the full pipeline on ASR transcripts (audio-to-incident evaluation).

Usage:
    python scripts/evaluate_asr.py --split test --model small
"""

from __future__ import annotations

import argparse
import re
import json
import sys
from collections import defaultdict
from pathlib import Path

import jiwer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.nlp.position import normalise_numbers  # noqa: E402
from mdi.schema import Dialogue, Language  # noqa: E402
from mdi.speech import DOMAIN_PROMPT, transcribe  # noqa: E402

SNR_BUCKETS = [(-10, 0), (0, 5), (5, 10), (10, 30)]


def norm(text: str) -> str:
    """Same normalisation for reference and hypothesis: case, punctuation, numbers.

    Whisper writes "18 degrees" where the reference says "one eight degrees";
    without this, correct recognitions are scored as errors.
    """
    text = re.sub(r"[^\w\s°'.]", " ", text.lower()).replace("-", " ")
    toks = normalise_numbers(text)
    # digit strings -> individual digits so "18" == "one eight" == "1 8"
    return " ".join(" ".join(t) if t.isdigit() else t for t in toks)


def bucket(snr: float) -> str:
    return next(f"{lo}..{hi} dB" for lo, hi in SNR_BUCKETS if lo <= snr < hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", type=Path, default=Path("data/audio"))
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="small", help="faster-whisper size: tiny/base/small/medium/large-v3")
    ap.add_argument("--language-hint", action="store_true", help="tell Whisper the caller language")
    ap.add_argument("--no-prompt", action="store_true", help="disable domain-vocabulary prompt (ablation)")
    args = ap.parse_args()

    from scripts.train_and_evaluate import field_eval  # reuse the end-to-end metrics
    from mdi.pipeline import Pipeline

    pipe = Pipeline.load()
    manifest = [json.loads(l) for l in open(args.audio / args.split / "manifest.jsonl", encoding="utf-8")]
    gold = {d.id: d for d in (Dialogue.model_validate_json(l) for l in open(args.data / f"{args.split}.jsonl", encoding="utf-8"))}

    refs, hyps, groups = [], [], defaultdict(lambda: ([], []))
    dialogues, asr_transcripts, rows = [], [], []
    for m in manifest:
        lines = []
        for t in m["turns"]:
            hint = Language(t["lang"]) if args.language_hint else None
            r = transcribe(t["wav"], hint, size=args.model, prompt=None if args.no_prompt else DOMAIN_PROMPT)
            lines.append(f"{t['speaker']}: {r['text']}")
            if t["speaker"] == "CALLER":
                ref, hyp = norm(t["text"]), norm(r["text"]) or "<empty>"
                refs.append(ref)
                hyps.append(hyp)
                for g in (f"lang={t['lang']}", f"snr={bucket(t['snr_db'])}"):
                    groups[g][0].append(ref)
                    groups[g][1].append(hyp)
        dialogues.append(gold[m["id"]])
        asr_transcripts.append("\n".join(lines))
        rows.append({"id": m["id"], "asr": asr_transcripts[-1], "reference": gold[m["id"]].transcript})
        print(f"{m['id']} done")

    report = {
        "model": args.model,
        "language_hint": args.language_hint,
        "domain_prompt": not args.no_prompt,
        "caller_wer": jiwer.wer(refs, hyps),
        "caller_cer": jiwer.cer(refs, hyps),  # CER is fairer for Tamil/Malayalam agglutination
        "by_group": {g: {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h), "n": len(r)} for g, (r, h) in sorted(groups.items())},
        "pipeline_on_asr": field_eval(pipe, dialogues, asr_transcripts),
        "pipeline_on_reference": field_eval(pipe, dialogues),
    }
    Path("reports").mkdir(exist_ok=True)
    Path(f"reports/asr_{args.model}{'_noprompt' if args.no_prompt else ''}.json").write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    with open(f"reports/asr_{args.model}{'_noprompt' if args.no_prompt else ''}_transcripts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    print(json.dumps({k: report[k] for k in ("caller_wer", "caller_cer", "by_group")}, indent=2, default=float))


if __name__ == "__main__":
    main()
