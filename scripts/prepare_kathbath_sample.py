"""Download a small, speaker-diverse real Indian speech sample.

Kathbath and IndicVoices access is gated by Hugging Face. Accept the dataset
terms and run ``hf auth login`` locally before using this script. It streams
only the validation split and saves one utterance per speaker per language.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

LANGUAGES = {"hi": "hindi", "ta": "tamil", "ml": "malayalam", "mr": "marathi"}
DATASETS = {
    "kathbath": "ai4bharat/Kathbath",
    "indicvoices": "ai4bharat/IndicVoices",
}


def extract_fields(row: dict) -> tuple[dict, str, str]:
    """Resolve the shared audio, text, and speaker fields from a dataset row."""
    audio = next((row[key] for key in ("audio_filepath", "audio") if row.get(key) is not None), None)
    transcript = next((row[key] for key in ("text", "transcript", "sentence") if row.get(key)), None)
    speaker = next((row[key] for key in (
        "speaker_id", "speakerid", "speaker", "client_id", "contributor_id", "user_id",
    ) if row.get(key) is not None), None)
    if not isinstance(audio, dict) or not any(audio.get(key) is not None for key in ("array", "bytes", "path")):
        raise ValueError(f"Could not find audio payload in columns: {sorted(row)}")
    if not transcript:
        raise ValueError(f"Could not find a transcript field in columns: {sorted(row)}")
    if speaker is None:
        raise ValueError(f"Could not find a speaker identifier in columns: {sorted(row)}")
    return audio, str(transcript).strip(), str(speaker)


def decode_audio(audio: dict) -> tuple[np.ndarray, int]:
    """Decode HF audio payloads with soundfile, without requiring torchcodec."""
    if audio.get("array") is not None:
        samples = np.asarray(audio["array"], dtype=np.float32)
        sample_rate = audio.get("sampling_rate")
    elif audio.get("bytes") is not None:
        samples, sample_rate = sf.read(io.BytesIO(bytes(audio["bytes"])), dtype="float32", always_2d=False)
    elif audio.get("path"):
        samples, sample_rate = sf.read(audio["path"], dtype="float32", always_2d=False)
    else:
        raise ValueError("Audio payload contains no array, bytes, or readable path")

    if sample_rate is None:
        raise ValueError("Audio payload is missing its sample rate")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        channel_axis = -1 if samples.shape[-1] <= 8 else 0
        samples = samples.mean(axis=channel_axis)
    return samples, int(sample_rate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="kathbath")
    parser.add_argument("--per-language", type=int, default=200)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.per_language < 1:
        parser.error("--per-language must be at least 1")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            'Install the optional dependency with: .venv\\Scripts\\python.exe -m pip install -e ".[real-speech]"'
        ) from exc

    output_dir = args.out or Path("data/real_speech") / args.dataset
    dataset_id = DATASETS[args.dataset]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    manifest_rows = []

    for language, config in LANGUAGES.items():
        print(f"Streaming {dataset_id} {config}/valid")
        try:
            from datasets import Audio

            rows = load_dataset(dataset_id, config, split="valid", streaming=True, token=True)
            audio_column = next(
                (name for name, feature in rows.features.items() if isinstance(feature, Audio)), None
            )
            if audio_column is None:
                raise ValueError(f"No audio feature found in dataset columns: {list(rows.features)}")
            rows = rows.cast_column(audio_column, Audio(decode=False)).shuffle(
                seed=args.seed, buffer_size=512
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not access {dataset_id}. Accept its Hugging Face dataset terms, then authenticate "
                "locally with `hf auth login`; credentials stay in your local Hugging Face login."
            ) from exc

        lang_dir = output_dir / language
        lang_dir.mkdir(parents=True, exist_ok=True)
        seen_speakers: set[str] = set()
        saved = 0
        for row in rows:
            audio, transcript, speaker = extract_fields(row)
            if speaker in seen_speakers:
                continue
            samples, sample_rate = decode_audio(audio)
            if not transcript or samples.size == 0:
                continue

            filename = f"{language}_{saved + 1:04d}.wav"
            sf.write(lang_dir / filename, samples, sample_rate)
            manifest_rows.append({
                "id": f"{language}_{saved + 1:04d}",
                "language": language,
                "speaker_id": speaker,
                "reference": transcript,
                "audio": f"{language}/{filename}",
                "sample_rate": sample_rate,
                "duration_s": samples.size / sample_rate,
                "dataset": dataset_id,
                "config": config,
                "split": "valid",
            })
            seen_speakers.add(speaker)
            saved += 1
            if saved >= args.per_language:
                break

        print(f"  saved {saved} utterances from {len(seen_speakers)} speakers")
        if saved == 0:
            raise RuntimeError(f"No usable examples found for {dataset_id} {config}/valid")

    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest_rows)} examples to {manifest_path}")


if __name__ == "__main__":
    main()