# Build Plan: full roadmap, solo, laptop + free Colab

The roadmap image keeps all nine phases. Because this is a solo project on a CPU
laptop, each phase ships as a **thin but real** version: every box on the
roadmap exists and works end to end, and the heavy items are swapped for
equivalents that fit the hardware. Substitutions are listed explicitly so the
report can justify them.

Environment: Windows, Python 3.13, no Docker, no local GPU. GPU work runs on
Colab/Kaggle T4. Everything else runs locally.

## Guiding rules (from the advisory review)

1. **Labels before text.** The incident is sampled first and the dialogue is
   rendered from it, so labels are exact and no LLM decides ground truth.
2. **Leakage control.** Vessel names and one phrasing per template slot are
   held out of train. Always report scores separately for seen vs. unseen.
3. **Small real test set.** Record 30-50 calls yourself (scripts read aloud in
   a noisy room or over a phone or radio) and transcribe some public MAYDAY
   recordings. Report the synthetic-vs-real gap; that gap is itself a result.
4. **Different model families.** If an LLM is used anywhere in generation
   (e.g. paraphrase), use a different family for extraction.
5. **Decision support only.** Every extracted field links to a transcript
   span. The system never auto-dispatches or downgrades a call.

## Phases

| Phase | Roadmap item | Prototype build | Status |
|---|---|---|---|
| 0 | Repo, schema, architecture | Python package, Pydantic schema (`mdi/schema.py`), 21 tests | done |
| 1 | Dataset factory | 20k labelled dialogues in 4 languages (`mdi/data/`), VHF channel sim, MMS-TTS audio + ESC-50 noise (`scripts/synthesize_audio.py`) | done; scale audio to 100-1000 calls |
| 2 | ASR | faster-whisper (CPU), WER/CER by language and SNR, audio-to-incident metrics (`scripts/evaluate_asr.py`) | done (small-scale); LoRA fine-tune on Colab = stretch |
| 3 | NLP extraction | CRF NER, position/number/MMSI normalisers, TF-IDF+LR incident classifier vs keyword baseline, rule backoff for counts | done; XLM-R / Qwen comparison on Colab = next |
| 4 | Decision intelligence | Multinomial vs ordinal severity, temperature scaling + ECE, split conformal sets, linear SHAP, contradiction + missing-info detection | done |
| 5 | Response intelligence | Open-Meteo live wind/current, Monte Carlo drift + 90% search area, asset ETA ranking, Hungarian assignment | done (simulated AIS) |
| 6 | Backend | Pipeline is a library; FastAPI wrapper + local PostgreSQL 16 (already installed) | todo |
| 7 | Operator UI + simulator | Streamlit dashboard: live ops, incident board, trainee simulator, evaluation, dataset explorer | done |
| 8 | MLOps / testing | pytest suite, metrics report with bootstrap CIs; MLflow tracking + drift report | partial |

## Data splitting (dataset v1)

- Stratified 80/10/10 split inside every (language, severity) cell.
- Disjoint vessel-name pools (70 / 15 / 15 names) and landmark pools (12 / 2 / 2) per split.
- One phrasing per template slot held out of train (reported as seen vs unseen).
- 3-fold out-of-fold predictions inside train for the stacked severity model.
- External test: MARTTS split by base scenario into dev / test; fixes developed on dev only,
  test scored once zero-shot and once after fixes (`reports/martts_zeroshot.json`, `reports/martts_final.json`).

## Evaluation the report must contain

- ASR: WER/CER per language and SNR bucket, clean vs simulated VHF vs real recordings.
- NER: entity-level F1 (seqeval) per label and language, seen vs unseen vessel/phrasing.
- Incident classification: macro F1, per-class confusion.
- Missing-info detection: recall at fixed precision (missing a field is the costly error).
- Severity: macro F1 + MAE (ordinal), ECE before/after temperature scaling, conformal coverage.
- Position: exact match and error in nautical miles.
- Bootstrap 95% confidence intervals and 3 seeds for all headline numbers.
- Limitations and ethics section: synthetic-to-real gap, TTS voices vs stressed speakers, non-native Indic templates, no operational claim.

## Known open items

- Indic phrase banks in `mdi/data/templates.py` need native-speaker review.
- Severity labels are rule-derived; a model learning them is partly learning our
  rules. Say this in the report and evaluate severity mainly for calibration.
