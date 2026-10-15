"""Generate the Phase 1 synthetic dialogue dataset.

Usage:
    python scripts/generate_dataset.py --n 20000 --out data/synthetic

Writes {train,val,test}.jsonl (full Dialogue objects), {split}.conll (BIO tags
over whitespace tokens, for NER training) and stats.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.data.generator import SPLITS, GeneratorConfig, generate  # noqa: E402
from mdi.schema import Dialogue  # noqa: E402


def to_conll(d: Dialogue) -> list[tuple[str, str]]:
    """BIO-tag whitespace tokens; a token takes the label of the span it starts in."""
    rows = []
    for m in re.finditer(r"\S+", d.transcript):
        tag = "O"
        for s in d.spans:
            if s.start <= m.start() < s.end:
                tag = ("B-" if m.start() == s.start else "I-") + s.label.value
                break
        rows.append((m.group(), tag))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = ap.parse_args()

    dialogues = generate(GeneratorConfig(n=args.n, seed=args.seed))
    args.out.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        rows = [d for d in dialogues if d.split == split]
        with open(args.out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for d in rows:
                f.write(d.model_dump_json() + "\n")
        with open(args.out / f"{split}.conll", "w", encoding="utf-8") as f:
            for d in rows:
                f.write(f"# id = {d.id}\n")
                f.writelines(f"{tok}\t{tag}\n" for tok, tag in to_conll(d))
                f.write("\n")

    pools = {sp: {"vessels": sorted({d.incident.vessel.name for d in dialogues if d.split == sp}),
                  "landmarks": sorted({d.incident.nearest_landmark for d in dialogues if d.split == sp})}
             for sp in SPLITS}
    stats = {
        "dataset_version": "v1",
        "seed": args.seed,
        "split_method": "stratified 80/10/10 by (language, severity); disjoint vessel-name and landmark pools per split; "
                        "one phrasing per template slot held out of train",
        "n": len(dialogues),
        "split": Counter(d.split for d in dialogues),
        "language": Counter(d.language.value for d in dialogues),
        "style": Counter(d.style for d in dialogues),
        "severity": Counter(int(d.incident.severity) for d in dialogues),
        "incident_type": Counter(t.value for d in dialogues for t in d.incident.incident_types),
        "position_format": Counter(d.incident.position_format.value for d in dialogues),
        "has_missing_field": sum(bool(d.missing_fields) for d in dialogues),
        "contradiction": sum(d.contradiction for d in dialogues),
        "eval_unseen_phrasing": sum(d.unseen_phrasing for d in dialogues if d.split != "train"),
        "split_by_language": {sp: Counter(d.language.value for d in dialogues if d.split == sp) for sp in SPLITS},
        "split_by_severity": {sp: Counter(int(d.incident.severity) for d in dialogues if d.split == sp) for sp in SPLITS},
        "pool_sizes": {sp: {k: len(v) for k, v in p.items()} for sp, p in pools.items()},
        "heldout_landmarks": {sp: pools[sp]["landmarks"] for sp in ("val", "test")},
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
