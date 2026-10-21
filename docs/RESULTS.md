# Results summary (for the presentation)

All numbers come from `reports/*.json`; `presentation/build_deck.js` reads the same files.
Synthetic = our generated data (held-out vessels, landmarks and phrasings).
External = real data never used for training.

## Language understanding (synthetic test, n = 2,001)

| Model | Metric | Result | Baseline |
|---|---|---|---|
| CRF NER | entity micro-F1 | **0.892** (95% CI 0.885–0.898) | — |
| TF-IDF + LR | incident macro-F1 | **0.887** | keyword rules 0.800 |
| Severity (temperature-scaled LR) | accuracy / QWK | **80.8%** / 0.82 | majority class 45.8% |
| Calibration | ECE | 0.092 → **0.073** | uncalibrated |
| Conformal 90% sets | coverage / mean size | **89.3%** / 1.40 | target 90% |

Diagnostics (`severity_diagnostics.json`): CRITICAL is the weak class. 43% of CRITICAL
calls are predicted HIGH, and conformal sets cover only 68% of CRITICAL (99% of HIGH).
Next step: class-conditional conformal prediction.

## External validation

| Test | Metric | Result | Comparison |
|---|---|---|---|
| MARTTS test (n = 177) | position ≤ 1 nm | 0.00 → **1.00** | zero-shot → after dev-split fixes |
| MARTTS test | MMSI exact | 0.44 → **0.82** | |
| MARTTS test | vessel exact | 0.00 → **0.37** | |
| MARTTS test | routine calls rated HIGH/CRITICAL | 62% → **2%** | |
| SeaAlert category | macro-F1 clean / ASR-high | **0.686 / 0.558** | published LR 0.674 / 0.423; RoBERTa 0.679 / 0.608 |

## Speech (transfer learning, no training run)

| Data | Language | Whisper-small → fine-tuned (WER) |
|---|---|---|
| Kathbath real speech, simulated VHF | Hindi (n = 8) | 0.73 → **0.28** (paired CI excludes 0) |
| | Tamil (n = 8) | 0.73 → **0.37** |
| | Malayalam (n = 2) | 1.00 → **0.23** |
| Our 24 synthetic TTS calls | all | 0.94 → 0.78 WER, CER 0.76 → 0.77 (no real gain: synthetic audio is the bottleneck) |

Caveat: Kathbath may overlap the checkpoints' training data.

## LLM extraction (Qwen2.5-1.5B, 4-bit, CPU)

SeaAlert test (n = 30), noisiest ASR, field accuracy; routing learned on dev data
(vessel, MMSI, POB: LLM first; position: CRF first).

| Field | CRF only | CRF + LLM routed | GPT-4o-mini (published) |
|---|---|---|---|
| vessel | 0.03 | **0.27** | 0.63 |
| position | 0.55 | **0.64** | 0.32 |
| persons on board | 0.55 | **0.69** | 0.80 |
| MMSI | 0.00 | **0.09** | 0.24 |

MARTTS (n = 24): MMSI 0.67 → **0.92**, vessel 0.42 → 0.21 (the per-field rule trades MARTTS
for SeaAlert). Grounding guard blocked 123 LLM answers.

## Drift (11 real NOAA buoys, 1,600 start points, cross-validated by buoy)

| Horizon | Persistence | Physics | Physics + LightGBM |
|---|---|---|---|
| 6 h | 2.79 nm | 2.40 | 2.41 |
| 12 h | 5.48 | 4.68 | 4.66 |
| 24 h | 10.70 | 9.08 | **8.89** |

LightGBM gain over physics at 24 h: **+2.1%** (buoy-bootstrap 95% CI 0.6% to 3.6%).
Skill vs persistence at 24 h: **16.9%**. Conformal 90% radius coverage ≈ 0.85 (few calibration buoys).
