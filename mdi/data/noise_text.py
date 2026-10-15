"""Simulate ASR errors on a transcript at a controlled word error rate.

Lets the robustness experiment (extraction quality vs WER) run on text without
needing hours of audio. Errors mimic what Whisper does on VHF audio: deleted
words, digit confusions ("five"/"nine"), split/merged names and substitutions.
Only CALLER lines are corrupted; the MRCC side is clear landline audio.
"""

from __future__ import annotations

import random

_CONFUSABLE = {
    "five": ["nine", "fife"], "nine": ["five", "niner"], "niner": ["nine", "minor"],
    "one": ["won", "nine"], "two": ["to", "too"], "four": ["for", "or"], "eight": ["ate", "a"],
    "three": ["tree", "free"], "six": ["sick", "fix"], "zero": ["hero", "oh"], "seven": ["eleven"],
    "north": ["nought", "forth"], "east": ["yeast", "is"], "miles": ["mild", "smiles"],
    "decimal": ["the small", "decimals"], "degrees": ["decrees", "agrees"],
}


def corrupt(transcript: str, wer: float, rng: random.Random) -> str:
    out_lines = []
    for line in transcript.splitlines():
        if not line.startswith("CALLER: ") or wer <= 0:
            out_lines.append(line)
            continue
        words = line[len("CALLER: "):].split()
        new: list[str] = []
        for w in words:
            if rng.random() >= wer:
                new.append(w)
                continue
            r = rng.random()
            key = w.strip(",.").lower()
            if r < 0.35:
                continue  # deletion (fading / squelch)
            if r < 0.7 and key in _CONFUSABLE:
                new.append(rng.choice(_CONFUSABLE[key]))
            elif r < 0.85 and len(w) > 4:
                cut = rng.randrange(2, len(w) - 1)
                new.extend([w[:cut], w[cut:]])  # word split
            else:
                new.append(w[: max(1, len(w) // 2)])  # truncated word
        out_lines.append("CALLER: " + " ".join(new))
    return "\n".join(out_lines)
