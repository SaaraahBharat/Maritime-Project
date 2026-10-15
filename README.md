# Maritime Distress Call Intelligence & Response Support System

Prototype decision-support system that turns noisy multilingual (English, Hindi,
Tamil, Malayalam) VHF distress calls into structured incident reports for coast
guard operators: who, where, what, how many people, how severe, what is still
missing, and which rescue asset can get there first. A human operator verifies
every field; the system never dispatches on its own.

- [docs/PLAN.md](docs/PLAN.md): phase-by-phase plan and what each roadmap phase became
- [docs/DATASETS.md](docs/DATASETS.md): every dataset and open-source model used, and why
- [Roadmap image](Maritime%20Distress%20Intelligence%20Roadmap.png): original design

## Quick start (laptop, CPU only)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev,app,speech]"
.venv/Scripts/python scripts/generate_dataset.py --n 20000          # Phase 1: synthetic dialogues
.venv/Scripts/python scripts/train_and_evaluate.py                  # Phases 3-4: train + evaluate
.venv/Scripts/streamlit run app/streamlit_app.py                    # Phase 7: dashboard + simulator
```

Optional audio (Phase 1-2):

```bash
.venv/Scripts/python scripts/download_noise.py                      # real ESC-50 background noise (~600 MB download)
.venv/Scripts/python scripts/synthesize_audio.py --split test --n 100
.venv/Scripts/python scripts/evaluate_asr.py --model small          # WER by language/SNR + audio-to-incident metrics
```

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
                                         Streamlit operator dashboard / training simulator
```

## Data science methods used

| Concept | Where |
|---|---|
| Synthetic data with labels sampled before text; leakage control by held-out entities and phrasings | `mdi/data/generator.py` |
| Sequence labelling (linear-chain CRF), feature engineering, entity-level F1 | `mdi/nlp/ner.py` |
| Multi-label text classification, TF-IDF char/word n-grams, baseline comparison | `mdi/nlp/models.py` |
| Ordinal regression (Frank & Hall) vs multinomial | `mdi/nlp/models.py` |
| Probability calibration (temperature scaling, ECE) | `mdi/nlp/models.py` |
| Uncertainty quantification (split conformal prediction sets) | `mdi/nlp/models.py` |
| Explainability (exact linear SHAP contributions) | `mdi/pipeline.py` |
| Robustness analysis (metric vs word error rate) | `scripts/train_and_evaluate.py` |
| Bootstrap confidence intervals, stratified evaluation (language, style, seen/unseen) | `scripts/train_and_evaluate.py` |
| Signal processing (band-pass, SNR mixing, clipping) | `mdi/audio/channel.py` |
| Monte Carlo simulation (drift), optimisation (assignment problem), geodesy | `mdi/response/engine.py` |

## Layout

```
mdi/schema.py            Incident / Dialogue data contracts
mdi/data/                gazetteer, spoken-number renderer, phrase banks, generator, ASR-noise simulator
mdi/audio/channel.py     VHF channel simulation
mdi/nlp/                 position/number parsing, CRF NER, classifiers, calibration, conformal
mdi/pipeline.py          transcript -> IncidentReport (with evidence spans and confidence)
mdi/response/engine.py   assets, ETA, drift, search area, assignment, live weather
mdi/speech.py            MMS-TTS, Whisper ASR, NLLB translation (lazy-loaded)
scripts/                 generate / synthesise / train / evaluate / download
app/streamlit_app.py     dashboard: live ops, incident board, training simulator, evaluation, dataset
tests/                   pytest suite
```
