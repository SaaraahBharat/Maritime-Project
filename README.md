# Maritime Distress Call Intelligence & Response Support System

Prototype decision-support system that turns noisy multilingual (English, Hindi,
Tamil, Malayalam) VHF distress calls into structured incident reports for coast
guard operators: who, where, what, how many people, how severe, what is still
missing, and which rescue asset can get there first. A human operator verifies
every field; the system never dispatches on its own.

- [docs/DATASETS.md](docs/DATASETS.md): every dataset and open-source model used, and why
- [reports/](reports/): every evaluation result as JSON

## Quick start (laptop, CPU only)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,speech,web,ml]"
.venv/Scripts/python scripts/generate_dataset.py --n 20000          # Phase 1: synthetic dialogues
.venv/Scripts/python scripts/train_and_evaluate.py                  # Phases 3-4: train + evaluate
.venv/Scripts/uvicorn mdi.api:app --port 8000                       # Phase 7: operator console -> http://localhost:8000
```

The operator console (`web/`, served by the FastAPI app in `mdi/api.py`) has a
call queue, a live map with the drift search area and asset routes, the
extracted incident with evidence and confidence, and a model-performance view
read from `reports/*.json`. **▶ Live demo** plays one distress call end to end.

Heavier ML components (each has its own script and report):

```bash
.venv/Scripts/python scripts/train_drift_ml.py --n 1600 --stride 6  # LightGBM drift correction on NOAA buoys
.venv/Scripts/python scripts/evaluate_llm.py                        # CRF vs Qwen2.5-1.5B vs learned routing
.venv/Scripts/python scripts/synthesize_audio.py --per-language 6 --out data/audio_indic --noise-dir data/noise_musan
.venv/Scripts/python scripts/evaluate_asr.py --audio data/audio_indic --indic --tag _indic_ft   # fine-tuned Indic Whisper
```

Optional audio (Phase 1-2):

```bash
.venv/Scripts/python scripts/download_noise.py                      # real ESC-50 background noise (~600 MB download)
.venv/Scripts/python scripts/synthesize_audio.py --split test --n 100
.venv/Scripts/python scripts/evaluate_asr.py --model small          # WER by language/SNR + audio-to-incident metrics
```

## Key results

| Component | Test data | Result | Baseline |
|---|---|---|---|
| Entity extraction (CRF) | synthetic, held-out names and phrasings | micro-F1 0.89 | — |
| Incident type (TF-IDF + LR) | synthetic | macro-F1 0.89 | keyword rules 0.80 |
| Severity (calibrated LR) | synthetic | accuracy 81%, ECE 0.073, conformal coverage 89% | majority 46%, ECE 0.092 |
| Field extraction | MARTTS real radio dialogues (test split) | position 1.00, MMSI 0.82 | zero-shot 0.00 / 0.44 |
| Message category | SeaAlert, heavy ASR noise | macro-F1 0.56 | published LR 0.42 |
| Indic ASR (fine-tuned Whisper) | Kathbath real speech over a VHF channel | Hindi WER 0.28 | Whisper-small 0.73 |
| CRF + LLM routing | SeaAlert, heavy ASR noise | position 0.64, POB 0.69 | CRF alone 0.55 / 0.55 |
| Drift (physics + LightGBM) | 11 real NOAA buoys, buoy-grouped CV | 24 h error 8.9 nm (+2.1%, CI 0.6–3.6%) | physics 9.1 nm |

Known weak spots: 43% of CRITICAL calls are predicted HIGH
(`reports/severity_diagnostics.json`); class-conditional conformal sets compensate,
raising CRITICAL coverage from 68% to 88%. The labelled training data is synthetic.

## Pipeline

```
VHF audio ──► VHF channel sim (train/eval only) ──► Whisper ASR ──► transcript
transcript ─► CRF NER ─► field normalisers (position/number/MMSI parsers) ─┐
           └► TF-IDF + logistic regression (incident type, multi-label) ───┤
                                                                           ▼
     severity model (ordinal vs multinomial, temperature scaling, conformal sets, linear SHAP)
     missing-info + contradiction detection ─► suggested follow-up questions
     Monte Carlo drift ─► 90% search area ─► asset ETA ranking ─► Hungarian assignment
                                                                           ▼
                                         web operator console (FastAPI + MapLibre)
```

## Data science methods used

| Concept | Where |
|---|---|
| Synthetic data with labels sampled before text; leakage control by held-out entities and phrasings | `mdi/data/generator.py` |
| Sequence labelling (linear-chain CRF), feature engineering, entity-level F1 | `mdi/nlp/ner.py` |
| Multi-label text classification, TF-IDF char/word n-grams, baseline comparison | `mdi/nlp/models.py` |
| Ordinal regression (Frank & Hall) vs multinomial | `mdi/nlp/models.py` |
| Probability calibration (temperature scaling, ECE) | `mdi/nlp/models.py` |
| Uncertainty quantification (class-conditional split conformal prediction sets) | `mdi/nlp/models.py` |
| Explainability (exact linear SHAP contributions) | `mdi/pipeline.py` |
| Robustness analysis (metric vs word error rate) | `scripts/train_and_evaluate.py` |
| Bootstrap confidence intervals, stratified evaluation (language, style, seen/unseen) | `scripts/train_and_evaluate.py` |
| Signal processing (band-pass, SNR mixing, clipping) | `mdi/audio/channel.py` |
| Monte Carlo simulation (drift), optimisation (assignment problem), geodesy | `mdi/response/engine.py` |
| Physics-informed ML: LightGBM learns the physics model's residual; buoy-grouped CV, shrinkage, conformal radius | `scripts/train_drift_ml.py` |
| Transfer learning: Whisper checkpoints fine-tuned on Hindi/Tamil/Malayalam speech | `mdi/speech.py` |
| LLM structured extraction (grammar-constrained JSON), grounding guard, per-field routing learned on dev data | `mdi/nlp/llm.py`, `scripts/evaluate_llm.py` |

## Layout

```
mdi/schema.py            Incident / Dialogue data contracts
mdi/data/                gazetteer, spoken-number renderer, phrase banks, generator, ASR-noise simulator
mdi/audio/channel.py     VHF channel simulation
mdi/nlp/                 position/number parsing, CRF NER, classifiers, calibration, conformal
mdi/pipeline.py          transcript -> IncidentReport (with evidence spans and confidence)
mdi/response/engine.py   assets, ETA, drift, search area, assignment, live weather
mdi/speech.py            MMS-TTS, Whisper + Indic Whisper ASR, NLLB translation (lazy-loaded)
mdi/nlp/llm.py           Qwen2.5-1.5B extractor (llama.cpp), grounding guard, CRF/LLM merge rules
mdi/api.py               FastAPI backend for the operator console
web/                     operator console (HTML/CSS/JS, MapLibre map, no build step)
scripts/                 generate / synthesise / train / evaluate / download
tests/                   pytest suite
```
