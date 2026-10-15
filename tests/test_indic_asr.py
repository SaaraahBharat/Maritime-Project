import io
import json
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from scripts.evaluate_indic_asr import bootstrap_ci, load_manifest, paired_wer_delta_ci
from scripts.prepare_kathbath_sample import decode_audio, extract_fields
from mdi.schema import Language
from mdi.speech import WHISPER_LANG, transcribe


def test_bootstrap_ci_is_reproducible():
    refs = ["alpha bravo", "charlie delta", "echo foxtrot"]
    hyps = ["alpha bravo", "charlie wrong", "echo foxtrot"]

    first = bootstrap_ci(refs, hyps, "wer", samples=100, seed=7)
    second = bootstrap_ci(refs, hyps, "wer", samples=100, seed=7)

    assert first == second
    assert first[0] <= first[1]


def test_paired_wer_delta_is_negative_when_candidate_is_better():
    refs = ["alpha bravo", "charlie delta", "echo foxtrot"]
    baseline = ["alpha wrong", "charlie wrong", "echo wrong"]
    candidate = refs.copy()

    interval = paired_wer_delta_ci(refs, baseline, candidate, samples=100, seed=7)

    assert interval[1] < 0


def test_manifest_limit_keeps_one_utterance_per_speaker(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    rows = [
        {"id": "hi1", "language": "hi", "speaker_id": "1", "audio": "hi/a.wav"},
        {"id": "hi2", "language": "hi", "speaker_id": "1", "audio": "hi/b.wav"},
        {"id": "hi3", "language": "hi", "speaker_id": "2", "audio": "hi/c.wav"},
        {"id": "ta1", "language": "ta", "speaker_id": "1", "audio": "ta/a.wav"},
        {"id": "ml1", "language": "ml", "speaker_id": "1", "audio": "ml/a.wav"},
        {"id": "mr1", "language": "mr", "speaker_id": "1", "audio": "mr/a.wav"},
    ]
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    grouped = load_manifest(manifest, limit_per_language=1)

    assert [row["id"] for row in grouped["hi"]] == ["hi1"]
    assert [row["id"] for row in grouped["mr"]] == ["mr1"]
    assert grouped["hi"][0]["audio_path"] == (tmp_path / "hi/a.wav").resolve()


def test_whisper_language_code_accepts_marathi_and_existing_enum():
    assert WHISPER_LANG[Language.HI] == "hi"
    assert "mr" not in WHISPER_LANG


def test_transcribe_passes_marathi_code_to_whisper(monkeypatch):
    captured = {}

    class FakeWhisper:
        def transcribe(self, audio, **kwargs):
            captured.update(kwargs)
            return [], SimpleNamespace(language="mr", language_probability=1.0)

    monkeypatch.setattr("mdi.speech._whisper", lambda size: FakeWhisper())

    result = transcribe(np.zeros(16_000, dtype=np.float32), "mr", prompt=None)

    assert captured["language"] == "mr"
    assert result["language"] == "mr"


def test_indicvoices_fields_resolve_audio_text_and_speaker():
    audio = {"bytes": b"wav-payload", "path": None}

    resolved_audio, transcript, speaker = extract_fields({
        "audio": audio,
        "text": " नमस्ते ",
        "speaker_id": "speaker-7",
    })

    assert resolved_audio is audio
    assert transcript == "नमस्ते"
    assert speaker == "speaker-7"


def test_decode_audio_bytes_without_torchcodec():
    expected = np.array([0.0, 0.25, -0.25], dtype=np.float32)
    encoded = io.BytesIO()
    sf.write(encoded, expected, 16_000, format="WAV")

    samples, sample_rate = decode_audio({"bytes": encoded.getvalue(), "path": None})

    assert sample_rate == 16_000
    np.testing.assert_allclose(samples, expected, atol=1e-4)