# Data science and ML concepts used, and exactly where

Every concept below is implemented in this repository (file:line points to the code)
and, where it produces a number, that number is saved in `reports/`.

Legend: **[model]** a learned model · **[metric]** an evaluation measure ·
**[stats]** statistical method · **[data]** data engineering / experimental design ·
**[sim]** simulation / optimisation / signal processing

---

## 1. Data creation and experimental design

| # | Concept | Where | What it does here |
|---|---|---|---|
| 1 | **Synthetic data generation** [data] | [mdi/data/generator.py:97](../mdi/data/generator.py#L97) `sample_incident`, [:251](../mdi/data/generator.py#L251) `render_dialogue` | 20,000 labelled distress dialogues. The incident (labels) is sampled first; text is rendered from it, so labels are exact by construction |
| 2 | **Span-level annotation / BIO tagging** [data] | [mdi/data/generator.py:80](../mdi/data/generator.py#L80) `_Builder.add`, [mdi/nlp/ner.py:116](../mdi/nlp/ner.py#L116) `bio_tags` | Character spans recorded while writing text, converted to B-/I-/O token tags for the CRF |
| 3 | **Prior distributions / sampling** [stats] | [mdi/data/gazetteer.py](../mdi/data/gazetteer.py) (per-vessel-type priors), [mdi/data/generator.py:88](../mdi/data/generator.py#L88) `_derive_severity` | Vessel type, POB, incident types and severity drawn from realistic priors |
| 4 | **Data augmentation** [data] | [mdi/data/generator.py:142](../mdi/data/generator.py#L142) `_Renderer` (casing, "MV" prefixes, 3 caller styles, fillers), [:228](../mdi/data/generator.py#L228) | Makes models robust to casing and phrasing seen in real radio text |
| 5 | **Stratified train/val/test split (80/10/10)** [data] | [mdi/data/generator.py:363](../mdi/data/generator.py#L363) `StratifiedSplitter` | Stratified by (language, severity) in shuffled blocks of 10 |
| 6 | **Leakage control: held-out entities** [data] | [mdi/data/generator.py:355](../mdi/data/generator.py#L355) `_partition`, [:385-386](../mdi/data/generator.py#L385) | Vessel names (70/15/15) and landmarks (12/2/2) never shared across splits |
| 7 | **Leakage control: held-out phrasings** [data] | [mdi/data/templates.py](../mdi/data/templates.py) (last option of each slot), [mdi/data/generator.py:158](../mdi/data/generator.py#L158) `pick` | One phrasing per template slot appears only in val/test; results reported seen vs unseen |
| 8 | **External test sets / domain shift** [data] | [scripts/evaluate_martts.py](../scripts/evaluate_martts.py), [scripts/evaluate_seaalert.py](../scripts/evaluate_seaalert.py), [scripts/evaluate_indic_asr.py](../scripts/evaluate_indic_asr.py) | Train on synthetic, test on real data (MARTTS, SeaAlert, Kathbath) |
| 9 | **Dev/test protocol (no test peeking)** [data] | [scripts/evaluate_martts.py:45](../scripts/evaluate_martts.py#L45) `part_of` | MARTTS split by base scenario (md5 hash): fixes on dev only, test scored only |
| 10 | **Group-aware splitting** [data] | [scripts/evaluate_martts.py:45](../scripts/evaluate_martts.py#L45) (by scenario), [scripts/validate_drift.py:158](../scripts/validate_drift.py#L158) and [scripts/train_drift_ml.py:143](../scripts/train_drift_ml.py#L143) (by buoy) | Prevents near-duplicates from leaking between train and test |
| 11 | **Stratified sampling of a benchmark** [data] | [scripts/evaluate_seaalert.py:57-58](../scripts/evaluate_seaalert.py#L57) `train_test_split(stratify=…)` | Reproduces SeaAlert's 1310/281/281 split |
| 12 | **Simulated measurement noise (ASR errors)** [sim] | [mdi/data/noise_text.py:22](../mdi/data/noise_text.py#L22) `corrupt` | Injects substitutions/deletions/insertions at a target WER |

## 2. Feature engineering and text processing

| # | Concept | Where | What it does here |
|---|---|---|---|
| 13 | **Tokenisation** | [mdi/nlp/ner.py:22](../mdi/nlp/ner.py#L22) | Script-aware tokens (Latin, Devanagari, Tamil, Malayalam) |
| 14 | **Hand-crafted lexical features** | [mdi/nlp/ner.py:74](../mdi/nlp/ner.py#L74) `_feats` | Word identity, prefixes/suffixes, script, digit/shape flags, ±2 context window |
| 15 | **Shape features / generalisation token** | [mdi/nlp/ner.py:60](../mdi/nlp/ner.py#L60) `_word_id` (`<CAP>`) | Unknown capitalised words become `<CAP>`, so the CRF learns *context* not names (vessel exact 66% → 99.6%) |
| 16 | **TF-IDF (char 2–5 + word 1–2 n-grams)** | [mdi/nlp/models.py:46-47](../mdi/nlp/models.py#L46) | Text vectorisation for incident classification, robust to misspellings |
| 17 | **Sublinear TF, min_df pruning** | [mdi/nlp/models.py:46-47](../mdi/nlp/models.py#L46) | Standard TF-IDF regularisation |
| 18 | **Engineered tabular features** | [mdi/nlp/models.py:73](../mdi/nlp/models.py#L73) `severity_features` | Incident probabilities, POB, MAYDAY flag, vessel type → severity model |
| 19 | **Rule-based normalisation (numbers, positions)** | [mdi/nlp/position.py:58](../mdi/nlp/position.py#L58) `normalise_numbers`, [:93](../mdi/nlp/position.py#L93) `parse_position` | Spoken numbers/coordinates (DDM, DMS, decimal, relative) → values |
| 20 | **Physics-derived features** | [scripts/train_drift_ml.py:64](../scripts/train_drift_ml.py#L64) `build_rows` | Wind/current u-v components, horizon means, physics displacement as LightGBM inputs |
| 21 | **Feature removal to stop leakage** | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) (`FEATURES` comment) | lat/lon/day-of-year dropped: they acted as buoy identifiers |

## 3. Models

| # | Concept | Where | What it does here |
|---|---|---|---|
| 22 | **Linear-chain CRF (sequence labelling)** [model] | [mdi/nlp/ner.py:128](../mdi/nlp/ner.py#L128) `CrfNer` | Named-entity recognition: vessel, type, MMSI, position, POB, incident |
| 23 | **L1 + L2 regularisation (elastic net)** [model] | [mdi/nlp/ner.py:129](../mdi/nlp/ner.py#L129) (`c1`, `c2`) | Regularises the CRF |
| 24 | **Marginal probabilities as confidence** [model] | [mdi/nlp/ner.py:154](../mdi/nlp/ner.py#L154) | Per-entity confidence shown to the operator |
| 25 | **Multi-label classification (one-vs-rest logistic regression)** [model] | [mdi/nlp/models.py:39](../mdi/nlp/models.py#L39) `IncidentClassifier`, [:49](../mdi/nlp/models.py#L49) | Several incident types per call (e.g. fire + flooding) |
| 26 | **Decision threshold** [model] | [mdi/nlp/models.py:42](../mdi/nlp/models.py#L42), [mdi/pipeline.py](../mdi/pipeline.py) (`incident_stated_threshold`) | Probability → label cut-off |
| 27 | **Baseline model (keyword rules)** [model] | [mdi/nlp/models.py:25](../mdi/nlp/models.py#L25) `KeywordIncidentBaseline` | Every model is compared with a simple baseline |
| 28 | **Majority-class baseline** [model] | [scripts/train_and_evaluate.py](../scripts/train_and_evaluate.py) (`majority_baseline_accuracy`) | Severity baseline (46%) |
| 29 | **Multinomial logistic regression** [model] | [mdi/nlp/models.py:113](../mdi/nlp/models.py#L113) `TemperatureScaled` | Severity classifier (chosen model) |
| 30 | **Ordinal regression (Frank & Hall)** [model] | [mdi/nlp/models.py:87](../mdi/nlp/models.py#L87) `OrdinalLogistic` | Uses LOW < MODERATE < HIGH < CRITICAL order; compared with multinomial |
| 31 | **Model selection** [model] | [scripts/train_and_evaluate.py:248](../scripts/train_and_evaluate.py#L248) | Ordinal vs multinomial vs calibrated, picked on validation |
| 32 | **Gradient-boosted trees (LightGBM)** [model] | [scripts/train_drift_ml.py:116](../scripts/train_drift_ml.py#L116) `fit_lgb` | Learns the drift physics model's residual error |
| 33 | **Residual / hybrid (grey-box) modelling** [model] | [scripts/train_drift_ml.py:93](../scripts/train_drift_ml.py#L93) `frame` (target = true − physics) | ML corrects physics instead of replacing it |
| 34 | **Shrinkage of a learned correction** [model] | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) (`LAMBDAS`, per-horizon λ on calibration buoys) | λ = 0 falls back to pure physics |
| 35 | **Grid-search parameter calibration** [model] | [scripts/train_drift_ml.py:107](../scripts/train_drift_ml.py#L107) `fit_scale`, [scripts/validate_drift.py:165](../scripts/validate_drift.py#L165) | Current scale factor fitted on training buoys |
| 36 | **Transfer learning (pre-trained speech models)** [model] | [mdi/speech.py](../mdi/speech.py) `INDIC_WHISPER`, `_transcribe_indic` | Whisper checkpoints fine-tuned on Hindi/Tamil/Malayalam reused, no training |
| 37 | **Contextual biasing (prompting ASR)** [model] | [mdi/speech.py](../mdi/speech.py) `DOMAIN_PROMPT` | Radio vocabulary prompt; ablation with `--no-prompt` |
| 38 | **LLM zero-/few-shot structured extraction** [model] | [mdi/nlp/llm.py](../mdi/nlp/llm.py) `extract`, `SYSTEM` (one example) | Qwen2.5-1.5B, 4-bit quantised |
| 39 | **Constrained decoding (JSON-schema grammar)** [model] | [mdi/nlp/llm.py](../mdi/nlp/llm.py) `SCHEMA`, `_complete` | Output always parses; quote lengths capped |
| 40 | **Hallucination guard / grounding** [model] | [mdi/nlp/llm.py](../mdi/nlp/llm.py) `grounded`, MMSI digit check | Rejects answers not in the transcript (counted in reports) |
| 41 | **Ensembling / learned model routing** [model] | [mdi/nlp/llm.py](../mdi/nlp/llm.py) `apply`, `learned_policy`; [scripts/evaluate_llm.py](../scripts/evaluate_llm.py) | Per-field choice CRF vs LLM, selected on dev data |
| 42 | **Machine translation (NLLB-200)** [model] | [mdi/speech.py](../mdi/speech.py) `translate_to_english` | Operator context only; extraction stays in the original language |
| 43 | **Text-to-speech (MMS-TTS, VITS)** [model] | [mdi/speech.py](../mdi/speech.py) `synthesize` | Generates the synthetic audio test set |

## 4. Evaluation metrics

| # | Metric | Where | Result file |
|---|---|---|---|
| 44 | **Entity-level precision / recall / F1 (micro)** | [scripts/train_and_evaluate.py:52](../scripts/train_and_evaluate.py#L52) `ner_counts`, [:63](../scripts/train_and_evaluate.py#L63) `micro_f1` | `metrics.json → ner` |
| 45 | **Per-class F1** | [scripts/train_and_evaluate.py:231](../scripts/train_and_evaluate.py#L231) | `metrics.json → incident.per_class_f1`, `ner.per_label_f1` |
| 46 | **Macro-F1** | [scripts/train_and_evaluate.py:130](../scripts/train_and_evaluate.py#L130), [:217](../scripts/train_and_evaluate.py#L217); [scripts/evaluate_seaalert.py:63](../scripts/evaluate_seaalert.py#L63) | incident, severity, SeaAlert category |
| 47 | **Precision and recall (detection tasks)** | [scripts/train_and_evaluate.py:137-142](../scripts/train_and_evaluate.py#L137) | missing-field and contradiction detection |
| 48 | **Accuracy** | [scripts/train_and_evaluate.py:250](../scripts/train_and_evaluate.py#L250) | severity |
| 49 | **Mean absolute error (ordinal)** | [scripts/train_and_evaluate.py:251](../scripts/train_and_evaluate.py#L251) | severity (how many levels off) |
| 50 | **Quadratic weighted Cohen's kappa (QWK)** | [scripts/train_and_evaluate.py:252](../scripts/train_and_evaluate.py#L252) | severity agreement, penalises far-off errors |
| 51 | **Expected calibration error (ECE)** | [mdi/nlp/models.py:143](../mdi/nlp/models.py#L143) | severity 0.092 → 0.073 |
| 52 | **Negative log-likelihood** | [mdi/nlp/models.py:129](../mdi/nlp/models.py#L129) `nll` | objective for temperature scaling |
| 53 | **Exact-match field accuracy** | [scripts/train_and_evaluate.py:89](../scripts/train_and_evaluate.py#L89) `field_eval`, [scripts/evaluate_martts.py:74](../scripts/evaluate_martts.py#L74) | vessel/MMSI/POB end-to-end |
| 54 | **Distance-threshold accuracy (≤ 1 nm)** | [scripts/train_and_evaluate.py:108](../scripts/train_and_evaluate.py#L108), [scripts/evaluate_martts.py:96](../scripts/evaluate_martts.py#L96) | position correctness via haversine distance |
| 55 | **Word error rate (WER) and character error rate (CER)** | [scripts/evaluate_asr.py:90-92](../scripts/evaluate_asr.py#L90), [scripts/evaluate_indic_asr.py:196](../scripts/evaluate_indic_asr.py#L196) | ASR quality per language and SNR |
| 56 | **Text normalisation for fair WER** | [scripts/evaluate_asr.py:28](../scripts/evaluate_asr.py#L28) `norm` | "18" = "one eight" |
| 57 | **Mean / median error (regression)** | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) (`mean_error_nm`, `median_error_nm`) | drift position error in nm |
| 58 | **Skill score vs persistence baseline** | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py), [scripts/validate_drift.py:142](../scripts/validate_drift.py#L142) | 1 − error/error(no-drift baseline) |
| 59 | **Empirical coverage (of prediction sets/regions)** | [scripts/train_and_evaluate.py](../scripts/train_and_evaluate.py) (`conformal_90`), [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) (`conformal_90pct_coverage`) | 89.3% severity sets; ~85% drift radius |
| 60 | **Prediction-set size** | [scripts/train_and_evaluate.py](../scripts/train_and_evaluate.py) (`mean_set_size`) | 1.40 severity levels on average |
| 61 | **Latency** | [scripts/evaluate_indic_asr.py](../scripts/evaluate_indic_asr.py) (`median_latency_s`), [scripts/evaluate_llm.py](../scripts/evaluate_llm.py) (`seconds`) | real-time feasibility |
| 61a | **Confusion matrix (4×4 severity, row-normalised)** | [scripts/severity_diagnostics.py](../scripts/severity_diagnostics.py) | `severity_diagnostics.json`: 43% of CRITICAL predicted as HIGH |
| 61b | **Multi-label confusion matrices (TP/FP/FN/TN per incident type)** | [scripts/severity_diagnostics.py](../scripts/severity_diagnostics.py) | `severity_diagnostics.json → incident_confusion` |
| 61c | **Reliability diagram (calibration curve)** | [scripts/severity_diagnostics.py](../scripts/severity_diagnostics.py) `reliability` | before vs after temperature scaling |
| 61d | **Class-conditional coverage check** | [scripts/severity_diagnostics.py](../scripts/severity_diagnostics.py) | conformal covers 99% of HIGH but only 68% of CRITICAL |

## 5. Statistics and uncertainty

| # | Concept | Where | What it does here |
|---|---|---|---|
| 62 | **Bootstrap confidence intervals** [stats] | [scripts/train_and_evaluate.py:44](../scripts/train_and_evaluate.py#L44), [scripts/evaluate_seaalert.py:66](../scripts/evaluate_seaalert.py#L66), [scripts/evaluate_indic_asr.py:83](../scripts/evaluate_indic_asr.py#L83) | 95% CI on F1, WER, etc. |
| 63 | **Paired bootstrap (difference of two systems)** [stats] | [scripts/evaluate_indic_asr.py:95](../scripts/evaluate_indic_asr.py#L95) `paired_wer_delta_ci` | Is fine-tuned Whisper significantly better? |
| 64 | **Cluster (grouped) bootstrap** [stats] | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) ("Paired bootstrap BY BUOY") | Resamples whole buoys: honest CI for correlated samples |
| 65 | **Temperature scaling (probability calibration)** [stats] | [mdi/nlp/models.py:125](../mdi/nlp/models.py#L125) `calibrate`, bounded optimisation [:133](../mdi/nlp/models.py#L133) | Makes "70% HIGH" mean 70% |
| 66 | **Split conformal prediction (classification)** [stats] | [mdi/nlp/models.py:155](../mdi/nlp/models.py#L155) `SplitConformal` | Severity sets with a 90% coverage guarantee |
| 67 | **Split conformal regression (search radius)** [stats] | [scripts/validate_drift.py:173](../scripts/validate_drift.py#L173), [scripts/train_drift_ml.py:171](../scripts/train_drift_ml.py#L171) | 90% search radius from calibration-buoy errors |
| 68 | **Quantiles / empirical distributions** [stats] | [mdi/response/engine.py:149](../mdi/response/engine.py#L149) `search_area` | 50%/90% containment radius of the particle cloud |
| 69 | **Explainability: exact linear SHAP** [stats] | [mdi/pipeline.py:190](../mdi/pipeline.py#L190) `explain_severity` | weight × (value − mean) per feature |
| 70 | **Feature importance (gain)** [stats] | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) (`feature_importance_gain`) | wind east–west component is the top driver |
| 71 | **Ablation study** [stats] | [scripts/evaluate_asr.py:51](../scripts/evaluate_asr.py#L51) `--no-prompt`; noise-augmentation in [scripts/evaluate_seaalert.py:72](../scripts/evaluate_seaalert.py#L72) | Measures the effect of one component |
| 72 | **Robustness curve (metric vs noise level)** [stats] | [scripts/train_and_evaluate.py:146](../scripts/train_and_evaluate.py#L146) `end_to_end` | Extraction quality as WER rises |
| 73 | **Stratified / subgroup evaluation** [stats] | [scripts/train_and_evaluate.py](../scripts/train_and_evaluate.py) (`by_language`, `by_style`, `by_phrasing`), [scripts/evaluate_asr.py](../scripts/evaluate_asr.py) (`by_group`) | Per language, caller style, SNR bucket, seen/unseen |
| 74 | **Descriptive statistics from real data** [stats] | [scripts/ais_stats.py](../scripts/ais_stats.py) | Median/P10/P90 vessel speeds by type from real AIS |

## 6. Cross-validation and tuning

| # | Concept | Where |
|---|---|---|
| 75 | **GroupKFold cross-validation (by buoy)** | [scripts/train_drift_ml.py:143](../scripts/train_drift_ml.py#L143) |
| 76 | **Nested split: fit / calibrate / test inside each fold** | [scripts/train_drift_ml.py:144-171](../scripts/train_drift_ml.py#L144) |
| 77 | **Hyperparameter regularisation (tree depth, leaves, L2, subsampling)** | [scripts/train_drift_ml.py](../scripts/train_drift_ml.py) `LGB_PARAMS` |
| 78 | **Overfitting diagnosis** | first LightGBM run memorised buoys via lat/lon; fixed by feature removal and shrinkage (documented in the script) |

## 7. Simulation, signal processing, optimisation, geospatial

| # | Concept | Where |
|---|---|---|
| 79 | **Band-pass filtering (Butterworth, 300–3400 Hz)** [sim] | [mdi/audio/channel.py:44](../mdi/audio/channel.py#L44) |
| 80 | **SNR-controlled noise mixing** [sim] | [mdi/audio/channel.py:49](../mdi/audio/channel.py#L49) `add_noise` |
| 81 | **Non-linear distortion (soft clipping), dropouts, squelch** [sim] | [mdi/audio/channel.py:58-71](../mdi/audio/channel.py#L58) |
| 82 | **Resampling** [sim] | [mdi/audio/channel.py:37](../mdi/audio/channel.py#L37) |
| 83 | **Monte Carlo simulation (particle drift ensemble)** [sim] | [mdi/response/engine.py:127](../mdi/response/engine.py#L127) `drift_particles` |
| 84 | **Numerical integration (hourly advection)** [sim] | [scripts/validate_drift.py:74](../scripts/validate_drift.py#L74) `integrate` |
| 85 | **Assignment problem (Hungarian algorithm)** [sim] | [mdi/response/engine.py:81](../mdi/response/engine.py#L81) `assign` |
| 86 | **Weighted cost function (severity-weighted ETA)** [sim] | [mdi/response/engine.py:81-91](../mdi/response/engine.py#L81) |
| 87 | **Ranking** [sim] | [mdi/response/engine.py:74](../mdi/response/engine.py#L74) `rank_assets` |
| 88 | **Geodesy (haversine distance, destination point)** [sim] | [mdi/data/spoken.py:147](../mdi/data/spoken.py#L147) |
| 89 | **Geospatial raster lookup (GEBCO bathymetry, land mask)** [data] | [mdi/response/geo.py](../mdi/response/geo.py) `elevation`, `on_land` |
| 90 | **Reanalysis / API data ingestion with caching** [data] | [mdi/response/geo.py:65](../mdi/response/geo.py#L65) `historical_forcing`, [mdi/response/engine.py:99](../mdi/response/engine.py#L99) `fetch_conditions` |

## 8. Product / MLOps

| # | Concept | Where |
|---|---|---|
| 91 | **Model persistence (serialisation)** | [mdi/pipeline.py:98-101](../mdi/pipeline.py#L98) joblib; `models/drift_lgbm.joblib`, `models/llm_routing.json` |
| 92 | **Inference API / model serving** | [mdi/api.py](../mdi/api.py) (FastAPI) |
| 93 | **Lazy model loading and caching** | [mdi/speech.py](../mdi/speech.py) `lru_cache`, [mdi/nlp/llm.py](../mdi/nlp/llm.py) disk cache |
| 94 | **Data contracts / schema validation** | [mdi/schema.py](../mdi/schema.py) (Pydantic) |
| 95 | **Automated tests** | [tests/](../tests/) (34 tests) |
| 96 | **Reproducibility (seeds, versioned dataset stats)** | `GeneratorConfig.seed`, `data/synthetic/stats.json` |
| 97 | **Human-in-the-loop design** | operator verify buttons in [web/app.js](../web/app.js); missing-field questions in [mdi/pipeline.py](../mdi/pipeline.py) |

---

### Not used yet (worth adding)

- **Class-conditional (Mondrian) conformal prediction**: fixes the 68% coverage on CRITICAL
  found by the diagnostics above.
- **ROC / PR curves and AUC** for the incident classifier.
- **k-fold cross-validation for the text models** (they use a fixed validation split).
- **Hyperparameter search** (C for logistic regression, c1/c2 for the CRF) is fixed by hand.
