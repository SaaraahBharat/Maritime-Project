"""Paired real-Indian-speech ASR evaluation: baseline vs Indic Whisper models.

Uses the local manifest produced by prepare_kathbath_sample.py. It compares
clean recordings with deterministic simulated-VHF copies; neither condition is
a maritime-distress benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdi.audio.channel import VHF_SR, resample, simulate_vhf
from mdi.speech import transcribe
from scripts.evaluate_asr import norm

LANGUAGES = {"hi": "hindi", "ta": "tamil", "ml": "malayalam", "mr": "marathi"}
SPECIALIZED_MODELS = {
    "hi": "vasista22/whisper-hindi-small",
    "ta": "vasista22/whisper-tamil-small",
    "ml": "thennal/whisper-medium-ml",
}
CONDITIONS = ("clean", "vhf")


class TransformersWhisper:
    def __init__(self, model_id: str, language: str):
        from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor, pipeline

        processor = WhisperProcessor.from_pretrained(model_id)
        model = WhisperForConditionalGeneration.from_pretrained(model_id)
        base_model = "openai/whisper-medium" if "medium" in model_id else "openai/whisper-small"
        model.generation_config = GenerationConfig.from_pretrained(base_model)
        model.generation_config.max_new_tokens = 440
        self.language = language
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=-1,
        )

    def __call__(self, samples: np.ndarray) -> str:
        result = self.asr(
            {"raw": samples, "sampling_rate": 16_000},
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )
        return result["text"].strip()


def load_manifest(path: Path, limit_per_language: int) -> dict[str, list[dict]]:
    samples: dict[str, list[dict]] = {language: [] for language in LANGUAGES}
    seen_speakers: dict[str, set[str]] = {language: set() for language in LANGUAGES}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        language = row["language"]
        if language not in samples or len(samples[language]) >= limit_per_language:
            continue
        speaker = str(row["speaker_id"])
        if speaker in seen_speakers[language]:
            continue
        row["audio_path"] = (path.parent / row["audio"]).resolve()
        samples[language].append(row)
        seen_speakers[language].add(speaker)

    empty = [language for language, rows in samples.items() if not rows]
    if empty:
        raise ValueError(f"Manifest has no examples for: {', '.join(empty)}")
    return samples


def bootstrap_ci(refs: list[str], hyps: list[str], metric: str, samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        indices = rng.integers(0, len(refs), size=len(refs))
        sampled_refs = [refs[i] for i in indices]
        sampled_hyps = [hyps[i] for i in indices]
        values.append(jiwer.wer(sampled_refs, sampled_hyps) if metric == "wer"
                      else jiwer.cer(sampled_refs, sampled_hyps))
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def paired_wer_delta_ci(
    refs: list[str], baseline: list[str], candidate: list[str], samples: int, seed: int
) -> list[float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(samples):
        indices = rng.integers(0, len(refs), size=len(refs))
        sampled_refs = [refs[i] for i in indices]
        base = [baseline[i] for i in indices]
        proposed = [candidate[i] for i in indices]
        deltas.append(jiwer.wer(sampled_refs, proposed) - jiwer.wer(sampled_refs, base))
    return [float(x) for x in np.percentile(deltas, [2.5, 97.5])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/real_speech/kathbath/manifest.jsonl"))
    parser.add_argument("--limit-per-language", type=int, default=200)
    parser.add_argument("--baseline-size", default="small")
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.limit_per_language < 1 or args.bootstrap_samples < 1:
        parser.error("sample and bootstrap counts must be positive")

    grouped = load_manifest(args.manifest, args.limit_per_language)
    dataset_rows = [row for rows in grouped.values() for row in rows]
    dataset_ids = {row.get("dataset", "unknown") for row in dataset_rows}
    dataset_splits = {row.get("split", "unknown") for row in dataset_rows}
    if len(dataset_ids) != 1 or len(dataset_splits) != 1:
        raise ValueError("A manifest must contain examples from exactly one dataset and split")
    dataset_id = dataset_ids.pop()
    dataset_split = dataset_splits.pop()
    out_dir = args.out
    if out_dir is None:
        out_dir = Path("reports/indic_asr")
        if dataset_id.lower().endswith("indicvoices"):
            out_dir /= "indicvoices"
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = out_dir / "predictions.jsonl"
    metric_rows = []

    with predictions_path.open("w", encoding="utf-8") as output:
        for language, rows in grouped.items():
            model_id = SPECIALIZED_MODELS.get(language)
            print(f"Evaluating {language}; {len(rows)} speakers")
            specialized = TransformersWhisper(model_id, language) if model_id else None
            baseline_name = f"faster-whisper-{args.baseline_size}"
            model_names = [baseline_name] + ([model_id] if model_id else [])
            collected: dict[str, dict[str, dict[str, list]]] = {
                condition: {name: {"refs": [], "hyps": [], "latency": []} for name in model_names}
                for condition in CONDITIONS
            }

            for index, row in enumerate(rows):
                audio, sample_rate = sf.read(row["audio_path"], dtype="float32", always_2d=False)
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)
                clean = resample(audio, sample_rate, 16_000).astype(np.float32)
                noisy_8k, channel_params = simulate_vhf(
                    clean, 16_000, np.random.default_rng(args.seed + index), snr_db=args.snr_db
                )
                noisy = resample(noisy_8k, VHF_SR, 16_000).astype(np.float32)
                audio_conditions = {"clean": clean, "vhf": noisy}
                predictions = {}
                reference = norm(row["reference"])

                for condition, samples_audio in audio_conditions.items():
                    predictions[condition] = {}
                    for name in model_names:
                        started = time.perf_counter()
                        if name == model_names[0]:
                            hypothesis = transcribe(
                                samples_audio, language, size=args.baseline_size, prompt=None
                            )["text"]
                        else:
                            assert specialized is not None
                            hypothesis = specialized(samples_audio)
                        elapsed = time.perf_counter() - started
                        normalized_hypothesis = norm(hypothesis)
                        predictions[condition][name] = {
                            "text": hypothesis,
                            "latency_s": elapsed,
                        }
                        collected[condition][name]["refs"].append(reference)
                        collected[condition][name]["hyps"].append(normalized_hypothesis)
                        collected[condition][name]["latency"].append(elapsed)

                output.write(json.dumps({
                    "id": row["id"], "language": language, "speaker_id": row["speaker_id"],
                    "reference": row["reference"], "snr_db": args.snr_db,
                    "channel": channel_params.as_dict(), "predictions": predictions,
                }, ensure_ascii=False) + "\n")
                output.flush()
                print(f"  [{index + 1}/{len(rows)}] {row['id']}")

            for condition in CONDITIONS:
                for name in model_names:
                    values = collected[condition][name]
                    wer = jiwer.wer(values["refs"], values["hyps"])
                    cer = jiwer.cer(values["refs"], values["hyps"])
                    metric_rows.append({
                        "language": language,
                        "condition": condition,
                        "model": name,
                        "n": len(values["refs"]),
                        "wer": wer,
                        "wer_ci95": bootstrap_ci(values["refs"], values["hyps"], "wer", args.bootstrap_samples, args.seed),
                        "cer": cer,
                        "cer_ci95": bootstrap_ci(values["refs"], values["hyps"], "cer", args.bootstrap_samples, args.seed),
                        "median_latency_s": float(np.median(values["latency"])),
                    })

                if len(model_names) > 1:
                    baseline_values = collected[condition][baseline_name]
                    candidate_values = collected[condition][model_names[1]]
                    metric_rows.append({
                        "language": language,
                        "condition": condition,
                        "comparison": f"{model_names[1]} minus {baseline_name}",
                        "delta_wer": jiwer.wer(baseline_values["refs"], candidate_values["hyps"])
                        - jiwer.wer(baseline_values["refs"], baseline_values["hyps"]),
                        "delta_wer_ci95": paired_wer_delta_ci(
                            baseline_values["refs"], baseline_values["hyps"], candidate_values["hyps"],
                            args.bootstrap_samples, args.seed,
                        ),
                    })

    report = {
        "dataset": dataset_id,
        "split": dataset_split,
        "license": "CC BY 4.0; follow dataset terms and cite the dataset",
        "scope_note": "Real Indian speech with simulated VHF channel effects; not real maritime distress calls.",
        "normalization": "scripts.evaluate_asr.norm applied equally to references and hypotheses",
        "seed": args.seed,
        "snr_db": args.snr_db,
        "bootstrap_samples": args.bootstrap_samples,
        "results": metric_rows,
        "models": {
            "baseline": f"faster-whisper-{args.baseline_size}",
            "specialized": SPECIALIZED_MODELS,
            "specialized_not_available": [language for language in LANGUAGES if language not in SPECIALIZED_MODELS],
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved predictions to {predictions_path}")
    print(f"Saved metrics to {report_path}")


if __name__ == "__main__":
    main()