---
license: cc-by-4.0
task_categories:
- text-classification
- text-to-speech
- automatic-speech-recognition
tags:
- synthetic
size_categories:
- n<1K
language:
- en
pretty_name: MARTTS
---

# MARTTS: Maritime Radio Text-To-Speech Synthetic Corpus
### Synthetic VHF Maritime Communication Data for Robust ASR Evaluation  
**Dataset accompanying the paper:**  
*A Text-to-Speech Framework for Generating Synthetic Maritime Radio Communications in ASR Evaluation*

---

## Dataset Summary

MARTTS is an open-source synthetic speech corpus designed to evaluate and stress-test **Automatic Speech Recognition (ASR)** systems operating in **maritime VHF radiotelephony** environments.

The dataset contains 240 **realistic multi-speaker distress, urgency, SAR, and routine maritime dialogues**, generated through:

- SMCP-compliant templates  
- LLM-based scenario generation  
- AIS-derived ship names, MMSI identifiers, and positions  
- Synthesis with the **Chatterbox TTS** model  
- A multi-stage **radio post-processing pipeline**

The dataset emulates true operational VHF conditions, including channel artifacts, background noise, environmental ship noise, dropouts, squelch clicks, and band-limiting.
It is intended for stress-testing and validating ASR systems under realistic maritime conditions where authentic data is scarce or sensitive.
This dataset is, to our knowledge, the first publicly available synthetic dataset tailored to maritime distress communication 

### 1. **Scenario Generation (SMCP + LLM)**  
Distress situations include:

- Fire / Explosion  
- Flooding  
- Sinking  
- Collision  
- Grounding  
- PAN PAN & SAR coordination  

LLM-generated dialogues include placeholders for vessel names, MMSI, and positions, which were populated with real AIS-derived values.

---

### 2. **Domain-Specific Text Normalization**

A custom text normalization pipeline ensures correct maritime-style pronunciation. Examples:

- `ETA` → `E T A`  
- `13:45` → `thirteen forty five U T C`  
- `MV Aurora` → `M V Aurora`  
- `52.1234 N` → `five two point one two three four degrees north`

Generic TTS normalizers fail on maritime abbreviations; this dedicated pipeline avoids transcription drift.

---

### 3. **Audio Synthesis & Speaker Profiling**

For each utterance:

- Chatterbox parameters sampled as:  
  - `exag ~ N(0.65, 0.08²)`  
  - `cfg ~ N(0.45, 0.08²)`  
- Speakers optionally cloned from VCTK (5–15s samples)  
- Inter-utterance pauses sampled from a truncated log-normal distribution:  
  - μ = 500 ms, σ = 0.4  
  - range = between 50 ms and 2 s

---

### 4. **VHF Post-Processing Pipeline**

The following transformations were applied:

1. **Speaker variation** (pitch, gain, stretching)  
2. **Band-limiting**
3. **Real maritime background noise**
4. **Environmental ship noise** (ESC-50: waves, wind, engine, rain, thunder)  
5. **Radio hiss**  
6. **Push-to-talk squelch clicks**  
7. **Dropouts / attenuation events**

## Dataset License 

**CC-BY 4.0**

### Components  
- **Chatterbox TTS** — MIT License  
  https://github.com/resemble-ai/chatterbox  
  License: https://opensource.org/licenses/MIT  

- **CSTR VCTK Corpus** — CC-BY 4.0  
  https://datashare.ed.ac.uk/handle/10283/3443  
  License: https://creativecommons.org/licenses/by/4.0/  

- **ESC-50 Environmental Sound Dataset** — CC-BY 4.0  
  https://github.com/karoldvl/ESC-50  
  License: https://creativecommons.org/licenses/by/4.0/  

- **Marine Radio Chatter (Bridge-to-Bridge)** — Kaggle dataset  
  https://www.kaggle.com/datasets/linogova/marine-radio-chatter-bridge-2-bridge-communication  
  License: https://www.kaggle.com/terms  

---