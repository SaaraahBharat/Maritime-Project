"""Phase 1/2: turn synthetic dialogues into noisy VHF audio.

Each turn is spoken by MMS-TTS (caller in the dialogue's language, MRCC in
English), passed through the VHF channel simulator, optionally mixed with real
background noise (ESC-50 engine/sea/wind/rain clips, see scripts/download_noise.py),
and saved per turn plus as one full-call WAV.

Usage:
    python scripts/synthesize_audio.py --split test --n 100 --noise-dir data/noise
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.audio.channel import VHF_SR, resample, simulate_vhf  # noqa: E402
from mdi.schema import Dialogue, Language  # noqa: E402
from mdi.speech import synthesize  # noqa: E402

OUT_SR = 16000  # Whisper's native rate


def tts_ready(text: str) -> str:
    """Written forms the TTS cannot read aloud ("09°48.7'N") -> words."""
    text = text.replace("°", " degrees ").replace("'", " minutes ")
    text = re.sub(r"(\d)\.(\d)", r"\1 decimal \2", text)
    return re.sub(r"\b([NSEW])\b", lambda m: {"N": "north", "S": "south", "E": "east", "W": "west"}[m[1]], text)


def load_noise(noise_dir: Path | None) -> list[np.ndarray]:
    if not noise_dir or not noise_dir.exists():
        return []
    clips = []
    for p in sorted(noise_dir.glob("*.wav")):
        x, sr = sf.read(p, dtype="float32", always_2d=False)
        x = x.mean(1) if x.ndim > 1 else x
        clips.append(resample(x, sr, VHF_SR))
    return clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("data/audio"))
    ap.add_argument("--noise-dir", type=Path, default=Path("data/noise"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-language", type=int, help="take N calls per language instead of the first --n")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    noise = load_noise(args.noise_dir)
    print(f"background noise clips: {len(noise)}")
    rows = [Dialogue.model_validate_json(l) for l in open(args.data / f"{args.split}.jsonl", encoding="utf-8")]
    if args.per_language:
        seen: dict = {}
        rows = [d for d in rows if seen.setdefault(d.language, []).append(d) or len(seen[d.language]) <= args.per_language]
    else:
        rows = rows[: args.n]
    out_dir = args.out / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "manifest.jsonl", "w", encoding="utf-8") as man:
        for k, d in enumerate(rows):
            call_snr = float(rng.uniform(-5, 20))  # one radio link per call; turns vary around it
            pieces, turns = [], []
            for i, t in enumerate(d.turns):
                lang = t.lang if t.speaker == "CALLER" else Language.EN
                wav, sr = synthesize(tts_ready(t.text), lang, seed=int(rng.integers(1e6)),
                                     speaking_rate=float(rng.uniform(1.0, 1.25) if d.style == "panicked" else 1.0))
                bg = noise[int(rng.integers(len(noise)))] if noise and t.speaker == "CALLER" else None
                snr = call_snr + rng.normal(0, 2) if t.speaker == "CALLER" else 25.0  # shore station is clear
                y, params = simulate_vhf(wav, sr, rng, snr_db=snr, background=bg)
                y = resample(y, VHF_SR, OUT_SR).astype(np.float32)
                path = out_dir / f"{d.id}_t{i:02d}.wav"
                sf.write(path, y, OUT_SR)
                turns.append({"speaker": t.speaker, "lang": lang.value, "text": t.text, "wav": str(path), **params.as_dict()})
                pieces += [y, np.zeros(int(0.4 * OUT_SR), np.float32)]
            full = out_dir / f"{d.id}.wav"
            sf.write(full, np.concatenate(pieces), OUT_SR)
            man.write(json.dumps({"id": d.id, "language": d.language.value, "style": d.style,
                                  "call_snr_db": call_snr, "wav": str(full), "turns": turns}, ensure_ascii=False) + "\n")
            print(f"[{k + 1}/{len(rows)}] {d.id} {d.language.value} snr={call_snr:.1f} dB")


if __name__ == "__main__":
    main()
