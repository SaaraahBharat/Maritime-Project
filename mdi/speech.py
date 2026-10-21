"""Open-source speech + translation models, loaded lazily so the text-only
pipeline and dashboard work without them.

* TTS:  Meta MMS-TTS (facebook/mms-tts-{eng,hin,tam,mal}), VITS, runs on CPU.
        Licence CC-BY-NC 4.0: fine for coursework, not for commercial use.
* ASR:  OpenAI Whisper via faster-whisper (MIT), int8 on CPU.
* MT:   Meta NLLB-200 distilled 600M (CC-BY-NC) by default; AI4Bharat
        IndicTrans2 (MIT) is the stronger Indic option if IndicTransToolkit installs.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from mdi.schema import Language

MMS_TTS = {Language.EN: "facebook/mms-tts-eng", Language.HI: "facebook/mms-tts-hin",
           Language.TA: "facebook/mms-tts-tam", Language.ML: "facebook/mms-tts-mal"}
NLLB_CODES = {Language.EN: "eng_Latn", Language.HI: "hin_Deva", Language.TA: "tam_Taml", Language.ML: "mal_Mlym"}
WHISPER_LANG = {Language.EN: "en", Language.HI: "hi", Language.TA: "ta", Language.ML: "ml"}


# --- TTS ----------------------------------------------------------------------

@lru_cache(maxsize=4)
def _tts(lang: Language):
    from transformers import AutoTokenizer, VitsModel

    return AutoTokenizer.from_pretrained(MMS_TTS[lang]), VitsModel.from_pretrained(MMS_TTS[lang]).eval()


@lru_cache(maxsize=1)
def _uroman():
    import uroman

    return uroman.Uroman()


def synthesize(text: str, lang: Language, seed: int = 0, speaking_rate: float = 1.0) -> tuple[np.ndarray, int]:
    """Text -> (float32 mono audio, sample_rate). MMS checkpoints for Indic
    scripts expect romanised input, which also lets embedded English words through."""
    import torch

    tok, model = _tts(lang)
    if getattr(tok, "is_uroman", False):
        text = _uroman().romanize_string(text)
    inputs = tok(text, return_tensors="pt")
    model.speaking_rate = speaking_rate
    torch.manual_seed(seed)  # VITS samples durations/noise: seed varies the "speaker" delivery
    with torch.no_grad():
        wav = model(**inputs).waveform[0].numpy()
    return wav.astype(np.float32), model.config.sampling_rate


# --- ASR ----------------------------------------------------------------------

@lru_cache(maxsize=2)
def _whisper(size: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device="cpu", compute_type="int8")


# Contextual biasing: radio vocabulary Whisper otherwise mishears ("MRCC" -> "Rick").
DOMAIN_PROMPT = (
    "Marine VHF distress call. MAYDAY MAYDAY MAYDAY, PAN-PAN. This is MRCC Mumbai, MRCC Chennai, MRSC Kochi. "
    "MMSI four one niner. Position zero nine degrees four eight decimal seven north, zero seven six degrees east. "
    "Persons on board. Nature of distress. Channel one six. Over."
)


# Transfer learning: Whisper checkpoints already fine-tuned on Indian-language speech
# (Apache 2.0). Used when the caller language is known; English stays on faster-whisper.
INDIC_WHISPER = {Language.HI: "vasista22/whisper-hindi-small", Language.TA: "vasista22/whisper-tamil-small",
                 Language.ML: "thennal/whisper-medium-ml"}


@lru_cache(maxsize=1)  # one checkpoint in RAM at a time (medium-ml alone is ~3 GB)
def _indic_whisper(model_id: str):
    from transformers import GenerationConfig, WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id).eval()
    # Community checkpoints ship old generation configs; borrow the base model's (language/task tokens).
    model.generation_config = GenerationConfig.from_pretrained(
        "openai/whisper-medium" if "medium" in model_id else "openai/whisper-small")
    return processor, model


def _transcribe_indic(audio: np.ndarray, lang: Language) -> dict:
    import torch

    processor, model = _indic_whisper(INDIC_WHISPER[lang])
    feats = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        out = model.generate(feats, language=WHISPER_LANG[lang], task="transcribe", num_beams=1,
                             max_new_tokens=220, return_dict_in_generate=True, output_scores=True)
    # Mean probability of the chosen tokens = ASR confidence, comparable to faster-whisper's word probs.
    probs = [float(torch.softmax(s[0], -1).max()) for s in out.scores]
    return {"text": processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip(),
            "language": WHISPER_LANG[lang], "language_probability": 1.0,
            "confidence": float(np.mean(probs)) if probs else 0.0}


def transcribe(audio: np.ndarray | str, lang: Language | str | None = None, size: str = "small",
               prompt: str | None = DOMAIN_PROMPT, indic: bool = False) -> dict:
    """Returns text, detected language, and a mean word probability as ASR confidence.

    indic=True routes Hindi/Tamil/Malayalam to the fine-tuned INDIC_WHISPER checkpoint (needs `lang`).
    """
    if isinstance(audio, str):
        # Decode with soundfile instead of faster-whisper's PyAV path (version clashes on Windows).
        import soundfile as sf

        from mdi.audio.channel import resample

        x, sr = sf.read(audio, dtype="float32", always_2d=False)
        audio = resample(x.mean(1) if x.ndim > 1 else x, sr, 16000).astype(np.float32)
    if indic and lang is not None and Language(lang) in INDIC_WHISPER:
        return _transcribe_indic(audio, Language(lang))
    language_code = WHISPER_LANG[lang] if isinstance(lang, Language) else lang
    segs, info = _whisper(size).transcribe(
        audio, language=language_code, vad_filter=True, word_timestamps=True, beam_size=5,
        initial_prompt=prompt, condition_on_previous_text=False,  # limits repetition loops
    )
    segs = list(segs)
    probs = [w.probability for s in segs for w in (s.words or [])]
    return {
        "text": " ".join(s.text.strip() for s in segs),
        "language": info.language,
        "language_probability": float(info.language_probability),
        "confidence": float(np.mean(probs)) if probs else 0.0,
    }


# --- Translation ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    name = "facebook/nllb-200-distilled-600M"
    return AutoTokenizer.from_pretrained(name), AutoModelForSeq2SeqLM.from_pretrained(name).eval()


def translate_to_english(text: str, src: Language) -> str:
    if src == Language.EN or not text.strip():
        return text
    import torch

    tok, model = _nllb()
    tok.src_lang = NLLB_CODES[src]
    batch = tok(text, return_tensors="pt", truncation=True, max_length=400)
    with torch.no_grad():
        out = model.generate(**batch, forced_bos_token_id=tok.convert_tokens_to_ids("eng_Latn"), max_new_tokens=256)
    return tok.batch_decode(out, skip_special_tokens=True)[0]
