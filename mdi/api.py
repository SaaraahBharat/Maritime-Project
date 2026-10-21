"""HTTP API + web console for the operator frontend (web/).

Run:  uvicorn mdi.api:app --port 8000      then open http://localhost:8000

Every heavy model (Whisper, Indic Whisper, NLLB, the LLM) is loaded lazily on
first use, so the console starts in seconds and works text-only without them.
"""

from __future__ import annotations

import json
import math
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mdi.data.spoken import haversine_nm
from mdi.pipeline import SEVERITY_NAMES, Pipeline
from mdi.response import engine
from mdi.schema import Language

ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIRS = [ROOT / "data/audio/test", ROOT / "data/audio_indic/test"]

app = FastAPI(title="Maritime Distress Intelligence API", version="0.2.0")


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Browsers revalidate the console's HTML/CSS/JS, so an update shows up on reload."""
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# --- cached resources -----------------------------------------------------------

@lru_cache(maxsize=1)
def pipe() -> Pipeline:
    return Pipeline.load(ROOT / "models/pipeline.joblib")


@lru_cache(maxsize=1)
def calls() -> dict[str, dict]:
    rows = {}
    for line in open(ROOT / "data/synthetic/test.jsonl", encoding="utf-8"):
        d = json.loads(line)
        rows[d["id"]] = d
    return rows


@lru_cache(maxsize=1)
def audio_manifest() -> dict[str, dict]:
    out = {}
    for d in AUDIO_DIRS:
        p = d / "manifest.jsonl"
        if p.exists():
            out.update({m["id"]: m for m in map(json.loads, open(p, encoding="utf-8"))})
    return out


@lru_cache(maxsize=1)
def assets() -> list[engine.Asset]:
    return engine.coast_guard_assets() + engine.simulated_ais(np.random.default_rng(7))


@lru_cache(maxsize=256)
def conditions(lat: float, lon: float) -> dict:
    return engine.fetch_conditions(lat, lon)


def _report(name: str) -> dict | None:
    p = ROOT / "reports" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# --- schemas --------------------------------------------------------------------

class ExtractIn(BaseModel):
    transcript: str
    language: Language = Language.EN
    llm: bool = False  # Qwen merged per field with the routing learned on dev data


class ResponseIn(BaseModel):
    lat: float
    lon: float
    uncertainty_nm: float = 0.5
    hours: float = 2.0
    target: str | None = None
    ml_drift: bool = True


class TranslateIn(BaseModel):
    text: str
    language: Language


class AssignIn(BaseModel):
    incidents: list[dict]


# --- endpoints ------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"pipeline": (ROOT / "models/pipeline.joblib").exists(),
            "drift_ml": (ROOT / "models/drift_lgbm.joblib").exists(),
            "calls": len(calls()), "calls_with_audio": len(audio_manifest()),
            "llm_available": _module("llama_cpp"), "asr_available": _module("faster_whisper")}


def _module(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


@app.get("/api/calls")
def list_calls(language: str | None = None, audio_only: bool = False, limit: int = 60) -> list[dict]:
    am = audio_manifest()
    out = []
    for d in calls().values():
        if language and d["language"] != language:
            continue
        if audio_only and d["id"] not in am:
            continue
        inc = d["incident"]
        out.append({"id": d["id"], "language": d["language"], "style": d["style"], "call_type": inc["call_type"],
                    "vessel": inc["vessel"]["name"], "has_audio": d["id"] in am})
        if len(out) >= limit and not audio_only:
            break
    out.sort(key=lambda c: not c["has_audio"])  # calls with audio first
    return out[:limit]


@app.get("/api/calls/{call_id}")
def get_call(call_id: str) -> dict:
    d = calls().get(call_id)
    if d is None:
        raise HTTPException(404, "unknown call")
    m = audio_manifest().get(call_id)
    return {"id": d["id"], "language": d["language"], "style": d["style"], "transcript": d["transcript"],
            "gold": d["incident"], "audio": f"/api/audio/{call_id}" if m else None,
            "snr_db": m["call_snr_db"] if m else None}


@app.get("/api/audio/{call_id}")
def get_audio(call_id: str):
    m = audio_manifest().get(call_id)
    if m is None:
        raise HTTPException(404, "no audio for this call")
    return FileResponse(ROOT / m["wav"], media_type="audio/wav")


@app.post("/api/calls/{call_id}/transcribe")
def transcribe_call(call_id: str, indic: bool = True) -> dict:
    """Whisper on each recorded turn (turn-level speaker labels come from the radio channel)."""
    from mdi.speech import transcribe

    m = audio_manifest().get(call_id)
    if m is None:
        raise HTTPException(404, "no audio for this call")
    t0, lines, confs = time.perf_counter(), [], []
    for t in m["turns"]:
        r = transcribe(str(ROOT / t["wav"]), Language(t["lang"]), indic=indic)
        lines.append(f"{t['speaker']}: {r['text']}")
        confs.append(r["confidence"])
    return {"transcript": "\n".join(lines), "asr_confidence": float(np.mean(confs)),
            "seconds": round(time.perf_counter() - t0, 1), "indic": indic}


@app.post("/api/transcribe")
async def transcribe_upload(file: UploadFile = File(...), language: Language = Form(Language.EN),
                            indic: bool = Form(True)) -> dict:
    from mdi.speech import transcribe

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "a.wav").suffix, delete=False) as tmp:
        tmp.write(await file.read())
    t0 = time.perf_counter()
    r = transcribe(tmp.name, language, indic=indic)
    return {"transcript": "CALLER: " + r["text"], "asr_confidence": r["confidence"],
            "detected_language": r["language"], "seconds": round(time.perf_counter() - t0, 1), "indic": indic}


@app.post("/api/extract")
def extract(body: ExtractIn) -> dict:
    p = pipe()
    t0 = time.perf_counter()
    if body.llm:
        from mdi.nlp.llm import LlmPipeline

        rep = LlmPipeline(p, "routed").extract(body.transcript)
    else:
        rep = p.extract(body.transcript)
    j = rep.to_json()
    j["severity_level"] = rep.severity
    j["incident_proba"] = {k: round(v, 3) for k, v in sorted(rep.incident_proba.items(), key=lambda kv: -kv[1])}
    j["entities"] = rep.entities
    j["explanation"] = [{"feature": f, "contribution": round(c, 4)} for f, c in p.explain_severity(rep)[:8]]
    j["seconds"] = round(time.perf_counter() - t0, 3)
    j["llm"] = body.llm
    return j


@lru_cache(maxsize=1)
def _drift_model() -> dict | None:
    p = ROOT / "models/drift_lgbm.joblib"
    if not p.exists():
        return None
    import joblib

    return joblib.load(p)


def ml_correction_nm(cond: dict, hours: float) -> tuple[float, float] | None:
    """LightGBM residual (east, north nm) learned on real NOAA buoys (scripts/train_drift_ml.py).

    Trained for 6-24 h; below 6 h the 6 h correction is scaled linearly with time.
    """
    m = _drift_model()
    if m is None or hours <= 0:
        return None
    h = float(min(max(hours, 6.0), 24.0))
    to = lambda spd, deg: (spd * math.sin(math.radians(deg)), spd * math.cos(math.radians(deg)))  # noqa: E731
    cu, cv = to(cond.get("current_kn", 0.0), cond.get("current_to_deg", 0.0))
    wu, wv = to(cond["wind_kn"], cond["wind_from_deg"] + 180)
    scale, leeway = m["current_scale"], 0.01
    row = {"h": h, "drogued": 0, "cur_u0": cu, "cur_v0": cv, "cur_u_mean": cu, "cur_v_mean": cv,
           "wind_u_mean": wu, "wind_v_mean": wv, "wind_speed_mean": cond["wind_kn"], "cur_speed_std": 0.0,
           "phys_dx": (scale * cu + leeway * wu) * h, "phys_dy": (scale * cv + leeway * wv) * h}
    import pandas as pd

    X = pd.DataFrame([row])[m["features"]]
    lam = np.interp(h, [6, 12, 24], [m["shrinkage_lambda"][k] for k in ("6", "12", "24")])
    dx, dy = (float(mod.predict(X)[0]) * lam for mod in m["models"])
    frac = min(1.0, hours / 6.0) if hours < 6 else 1.0
    return dx * frac, dy * frac


@app.post("/api/response")
def response(body: ResponseIn) -> dict:
    cond = conditions(round(body.lat, 2), round(body.lon, 2))
    parts = engine.drift_particles(body.lat, body.lon, body.hours, cond, body.target, body.uncertainty_nm, n=400)
    area = engine.search_area(parts)
    physics_centre = {"lat": area["lat"], "lon": area["lon"]}
    corr = ml_correction_nm(cond, body.hours) if body.ml_drift else None
    if corr:
        dlat = corr[1] / 60
        dlon = corr[0] / (60 * math.cos(math.radians(area["lat"])))
        parts = parts + np.array([dlat, dlon])
        area["lat"] += dlat
        area["lon"] += dlon
    cal = engine.calibrated_radius90(body.hours)
    if cal:  # conformal radius from real drifters: Monte Carlo alone was overconfident
        area["r90_nm"] = max(area["r90_nm"], cal + body.uncertainty_nm)
        area["area90_nm2"] = math.pi * area["r90_nm"] ** 2
    ranked = engine.rank_assets(assets(), area["lat"], area["lon"], k=6)
    idx = np.random.default_rng(0).choice(len(parts), size=min(250, len(parts)), replace=False)
    return {"conditions": cond, "area": area, "physics_centre": physics_centre,
            "ml_shift_nm": round(math.hypot(*corr), 2) if corr else None,
            "particles": [[round(float(a), 5), round(float(b), 5)] for a, b in parts[idx]],
            "assets": ranked, "drift_nm": round(haversine_nm(body.lat, body.lon, area["lat"], area["lon"]), 2)}


@app.get("/api/assets")
def all_assets() -> list[dict]:
    return [{"name": a.name, "kind": a.kind, "lat": a.lat, "lon": a.lon, "speed_kn": round(a.speed_kn, 1)} for a in assets()]


@app.post("/api/assign")
def assign(body: AssignIn) -> list[dict]:
    return engine.assign(assets(), body.incidents)


@app.post("/api/translate")
def translate(body: TranslateIn) -> dict:
    from mdi.speech import translate_to_english

    t0 = time.perf_counter()
    lines = []
    for line in body.text.splitlines():
        spk, _, txt = line.partition(": ")
        lines.append(f"{spk}: {translate_to_english(txt, body.language) if spk == 'CALLER' else txt}")
    return {"text": "\n".join(lines), "seconds": round(time.perf_counter() - t0, 1)}


@app.get("/api/metrics")
def metrics() -> list[dict]:
    """Headline numbers for the Model performance view, read from reports/*.json."""
    cards: list[dict] = []

    def card(group, title, value, unit="", baseline=None, note="", ci=None):
        cards.append({"group": group, "title": title, "value": value, "unit": unit,
                      "baseline": baseline, "note": note, "ci": ci})

    m = _report("metrics.json")
    if m:
        card("Language understanding", "NER micro-F1", m["ner"]["micro_f1"], ci=m["ner"].get("micro_f1_ci95"),
             note="Linear-chain CRF · held-out vessels & phrasings")
        card("Language understanding", "Incident type macro-F1", m["incident"]["tfidf_lr_macro_f1"],
             baseline=m["incident"]["keyword_baseline_macro_f1"], ci=m["incident"].get("tfidf_lr_macro_f1_ci95"),
             note="TF-IDF + logistic regression vs keyword rules")
        sev = m["severity"]
        ts = sev.get("multinomial_temperature_scaled")
        if ts:
            card("Severity & uncertainty", "Severity accuracy", ts["accuracy"], baseline=sev["majority_baseline_accuracy"],
                 note=f"Temperature-scaled logistic · QWK {ts['quadratic_kappa']:.2f} · vs majority class")
            card("Severity & uncertainty", "Calibration error (ECE)", ts["ece"],
                 baseline=sev["multinomial_uncalibrated"]["ece"], note="After vs before temperature scaling (lower is better)")
        if "conformal_90" in sev:
            card("Severity & uncertainty", "Conformal 90% coverage", sev["conformal_90"]["empirical_coverage"],
                 note=f"Class-conditional conformal sets · mean size {sev['conformal_90']['mean_set_size']:.2f} · "
                      f"CRITICAL covered {sev['conformal_90'].get('coverage_by_true_class', {}).get('CRITICAL', float('nan')):.0%}")
    sa = _report("seaalert.json")
    if sa:
        ours = sa["A_message_category"]["ours"]["tfidf_lr_clean_training"]
        card("External benchmarks", "SeaAlert category F1 (clean)", ours["text"]["macro_f1"], baseline=0.674,
             note="vs published logistic regression 0.674 / RoBERTa 0.679")
        card("External benchmarks", "SeaAlert category F1 (ASR-high)", ours["asr_high"]["macro_f1"], baseline=0.423,
             note="vs published logistic regression 0.423 / RoBERTa 0.608")
    mt = _report("martts_final.json") or _report("martts.json")
    if mt and "test_text" in mt:
        t = mt["test_text"]
        for k, label in (("position_within_1nm", "MARTTS position ≤1 nm"), ("mmsi_exact", "MARTTS MMSI exact")):
            if k in t:
                card("External benchmarks", label, t[k]["rate"], note=f"Real radio dialogues (DLR), held-out test, n={t[k]['n']}")
    dm = _report("drift_ml.json")
    if dm:
        card("Drift & search", "24 h drift error", dm["mean_error_nm"]["physics+lgbm"]["24"], "nm",
             baseline=dm["mean_error_nm"]["physics"]["24"],
             note=f"Physics + LightGBM vs physics only · {dm['n_buoys']} NOAA buoys, buoy-grouped CV")
        card("Drift & search", "Skill vs persistence (24 h)", dm["skill_vs_persistence"]["physics+lgbm"]["24"],
             baseline=dm["skill_vs_persistence"]["physics"]["24"], note="1 − error / error of 'object stays put'")
        card("Drift & search", "90% search-area coverage (24 h)", dm["conformal_90pct_coverage"]["physics+lgbm"]["24"],
             baseline=0.36, note="Split-conformal radius vs uncalibrated Monte Carlo (≈0.36)")
    base, ft = _report("asr_small_indic_base.json"), _report("asr_small_indic_ft.json")
    if base and ft:
        card("Speech", "Caller WER, all 24 VHF calls", ft["caller_wer"], baseline=base["caller_wer"],
             note="Fine-tuned Indic Whisper vs Whisper-small (both told the language); synthetic VHF audio")
        for lang, name in (("hi", "Hindi"), ("ta", "Tamil"), ("ml", "Malayalam")):
            b, f = base["by_group"].get(f"lang={lang}"), ft["by_group"].get(f"lang={lang}")
            if b and f:
                card("Speech", f"{name} caller CER", f["cer"], baseline=b["cer"],
                     note=f"Character error rate, n={f['n']} turns · transfer learning, no training run")
    kb = _report("indic_asr_n8/report.json") or _report("indic_asr/report.json")
    if kb:
        for row in kb["results"]:
            if row.get("comparison") and row["condition"] == "vhf":
                bl = next((r for r in kb["results"] if r.get("model", "").startswith("faster-whisper")
                           and r["language"] == row["language"] and r["condition"] == "vhf"), None)
                if bl:
                    card("Speech", f"Kathbath {row['language'].upper()} WER, VHF channel", bl["wer"] + row["delta_wer"],
                         baseline=bl["wer"], ci=None,
                         note=f"Real human speech (AI4Bharat), n={bl['n']} · fine-tuned vs Whisper-small")
    llm = _report("llm_extraction.json")
    if llm:
        best = "routed" if "routed" in llm else "hybrid"
        for field, label in (("position", "position"), ("pob", "persons on board"), ("vessel", "vessel name")):
            m, c = llm[best]["seaalert"]["asr_high"].get(field), llm["crf_pipeline"]["seaalert"]["asr_high"].get(field)
            if m and c:
                card("LLM extraction", f"SeaAlert ASR-high {label}", m["value"], baseline=c["value"],
                     note=f"{best.capitalize()} CRF+Qwen2.5-1.5B vs CRF alone · n={m['n']} · grounding guard on")
    return cards


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


app.mount("/", StaticFiles(directory=ROOT / "web"), name="web")
