"""Operator dashboard + simulator (Phase 7 prototype).

Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mdi.data.spoken import haversine_nm  # noqa: E402
from mdi.pipeline import SEVERITY_NAMES, Pipeline  # noqa: E402
from mdi.response import engine  # noqa: E402
from mdi.schema import Dialogue, Language  # noqa: E402

st.set_page_config(page_title="Maritime Distress Intelligence", page_icon="⚓", layout="wide")

# Status palette (reserved for severity) and categorical palette for entities/series.
SEV_COLOR = {"LOW": "#0ca30c", "MODERATE": "#fab219", "HIGH": "#ec835a", "CRITICAL": "#d03b3b"}
SEV_ICON = {"LOW": "🟢", "MODERATE": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}
ENT_COLOR = {"VESSEL": "#2a78d6", "VESSEL_TYPE": "#4a3aa7", "MMSI": "#1baf7a",
             "POSITION": "#eb6834", "POB": "#e87ba4", "INCIDENT": "#e34948"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


# --- cached resources ---------------------------------------------------------

@st.cache_resource
def load_pipeline() -> Pipeline | None:
    p = ROOT / "models/pipeline.joblib"
    return Pipeline.load(p) if p.exists() else None


@st.cache_data
def load_split(split: str) -> list[dict]:
    p = ROOT / f"data/synthetic/{split}.jsonl"
    return [json.loads(l) for l in open(p, encoding="utf-8")] if p.exists() else []


@st.cache_data
def load_audio_manifest() -> dict[str, dict]:
    p = ROOT / "data/audio/test/manifest.jsonl"
    return {m["id"]: m for m in map(json.loads, open(p, encoding="utf-8"))} if p.exists() else {}


@st.cache_resource
def assets() -> list[engine.Asset]:
    return engine.coast_guard_assets() + engine.simulated_ais(np.random.default_rng(7))


@st.cache_data(ttl=1800)
def conditions(lat: float, lon: float) -> dict:
    return engine.fetch_conditions(lat, lon)


pipe = load_pipeline()
if pipe is None:
    st.error("No trained model found. Run `python scripts/generate_dataset.py` then "
             "`python scripts/train_and_evaluate.py` first.")
    st.stop()

st.session_state.setdefault("board", [])
st.session_state.setdefault("call", None)


# --- helpers ------------------------------------------------------------------

def highlight(text: str, ents: list[dict]) -> str:
    out, pos = [], 0
    for e in sorted(ents, key=lambda e: e["start"]):
        out.append(html.escape(text[pos:e["start"]]))
        c = ENT_COLOR[e["label"]]
        out.append(f'<mark style="background:{c}26;border-bottom:2px solid {c};padding:0 2px;border-radius:3px" '
                   f'title="{e["label"]} · conf {e["confidence"]:.2f}">{html.escape(text[e["start"]:e["end"]])}'
                   f'<sub style="font-size:.6em;opacity:.75"> {e["label"]}</sub></mark>')
        pos = e["end"]
    out.append(html.escape(text[pos:]))
    return '<div style="white-space:pre-wrap;line-height:1.9;font-size:.95rem">' + "".join(out) + "</div>"


def legend() -> str:
    return " ".join(f'<span style="border-bottom:2px solid {c};padding:0 4px;margin-right:6px;font-size:.8rem">{l}</span>'
                    for l, c in ENT_COLOR.items())


def severity_badge(name: str) -> str:
    return (f'<span style="background:{SEV_COLOR[name]};color:#0b0b0b;padding:4px 12px;border-radius:6px;'
            f'font-weight:700">{SEV_ICON[name]} {name}</span>')


def response_map(lat: float, lon: float, particles: np.ndarray, area: dict, ranked: list[dict]) -> pdk.Deck:
    layers = [
        pdk.Layer("ScatterplotLayer", pd.DataFrame(particles, columns=["lat", "lon"]),
                  get_position="[lon, lat]", get_radius=250, get_fill_color=[42, 120, 214, 70]),
        pdk.Layer("ScatterplotLayer", pd.DataFrame([area]), get_position="[lon, lat]",
                  get_radius=area["r90_nm"] * 1852, stroked=True, filled=False,
                  get_line_color=[42, 120, 214, 200], line_width_min_pixels=2),
        pdk.Layer("LineLayer", pd.DataFrame([{**r, "tlat": lat, "tlon": lon} for r in ranked[:3]]),
                  get_source_position="[lon, lat]", get_target_position="[tlon, tlat]",
                  get_color=[27, 175, 122, 220], get_width=2),
        pdk.Layer("ScatterplotLayer", pd.DataFrame(ranked), get_position="[lon, lat]", get_radius=1500,
                  get_fill_color=[27, 175, 122, 230], pickable=True),
        pdk.Layer("ScatterplotLayer", pd.DataFrame([{"lat": lat, "lon": lon, "asset": "DISTRESS"}]),
                  get_position="[lon, lat]", get_radius=2200, get_fill_color=[208, 59, 59, 240],
                  stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True),
    ]
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=7.2)
    return pdk.Deck(layers=layers, initial_view_state=view,
                    tooltip={"text": "{asset}\n{kind}\nETA {eta_h} h · {distance_nm} nm"})


def run_report(transcript: str, language: Language, gold: dict | None = None, key: str = "live") -> None:
    rep = pipe.extract(transcript)
    j = rep.to_json()
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("##### Transcript")
        st.markdown(legend(), unsafe_allow_html=True)
        st.markdown(highlight(transcript, rep.entities), unsafe_allow_html=True)
        if language != Language.EN:
            if st.button("Translate caller lines to English (NLLB-200)", key=f"tr_{key}"):
                from mdi.speech import translate_to_english
                with st.spinner("Loading translation model (first time ~2.5 GB download)…"):
                    lines = [translate_to_english(l[8:], language) if l.startswith("CALLER: ") else l[6:]
                             for l in transcript.splitlines()]
                st.info("\n\n".join(lines))
                st.caption("Machine translation for operator context only; fields are extracted from the original.")

    with right:
        sev = j["severity"]
        st.markdown(f"##### Severity &nbsp; {severity_badge(sev)}", unsafe_allow_html=True)
        st.caption(f"90% conformal set: **{' / '.join(j['severity_90pct_set'])}**. The true severity is in this set "
                   "for ~90% of calls; a wider set means the model is less sure.")
        sp = pd.DataFrame({"severity": list(j["severity_proba"]), "probability": list(j["severity_proba"].values())})
        st.altair_chart(
            alt.Chart(sp).mark_bar(cornerRadiusEnd=4, height=16).encode(
                x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1]), title=None),
                y=alt.Y("severity:N", sort=list(SEV_COLOR), title=None),
                color=alt.Color("severity:N", scale=alt.Scale(domain=list(SEV_COLOR), range=list(SEV_COLOR.values())), legend=None),
                tooltip=["severity", alt.Tooltip("probability:Q", format=".2f")],
            ).properties(height=130), width="stretch")

        for f in j["missing_fields"]:
            st.error(f"⚠ Missing: **{f.replace('_', ' ')}**")
        for f in j["low_confidence"]:
            st.warning(f"? Verify (low confidence): **{f.replace('_', ' ')}**")
        for c in j["contradictions"]:
            st.warning(f"⇄ Contradiction: {c}")
        if j["suggested_questions"]:
            st.markdown("**Ask the caller:** " + " · ".join(f"“{q}?”" for q in j["suggested_questions"]))

    st.markdown("##### Extracted incident")
    rows = []
    for name, f in j["fields"].items():
        val = f["value"]
        if name == "position":
            val = f"{val['lat']:.4f}, {val['lon']:.4f} (±{val['uncertainty_nm']} nm, {val['format']})"
        row = {"field": name, "value": ", ".join(sorted(val)) if isinstance(val, list) else str(val),
               "confidence": f["confidence"], "evidence (transcript)": f["evidence"] or "—"}
        if gold is not None:
            row["ground truth"] = gold.get(name, "—")
            row["✓"] = "✅" if str(gold.get(name)) == row["value"] or (name == "position" and gold.get("_pos_ok")) else "❌"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})

    with st.expander("Why this severity? (feature contributions)"):
        ex = pd.DataFrame(pipe.explain_severity(rep)[:8], columns=["feature", "contribution"])
        ex["direction"] = np.where(ex.contribution > 0, "raises", "lowers")
        st.altair_chart(alt.Chart(ex).mark_bar(cornerRadiusEnd=4, height=14).encode(
            x=alt.X("contribution:Q", title="contribution to log-odds of HIGH or worse"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["raises", "lowers"], range=["#eb6834", "#2a78d6"])),
            tooltip=["feature", alt.Tooltip("contribution:Q", format="+.3f")],
        ).properties(height=230), width="stretch")
        st.caption("Linear model: contribution = weight × (value − training mean), the exact SHAP value for a linear model.")

    if "position" in j["fields"]:
        pos = j["fields"]["position"]["value"]
        st.markdown("##### Response planning")
        c1, c2 = st.columns([2, 1])
        with c2:
            hours = st.slider("Hours since call", 0.0, 12.0, 2.0, 0.5, key=f"h_{key}")
            cond = conditions(pos["lat"], pos["lon"])
            st.caption(f"Wind {cond['wind_kn']:.0f} kn from {cond['wind_from_deg']:.0f}°, "
                       f"current {cond.get('current_kn', 0):.1f} kn · source: {cond['source']}")
            target = "person_in_water" if "man_overboard" in j["fields"]["incident_types"]["value"] else \
                j["fields"].get("vessel_type", {}).get("value")
            parts = engine.drift_particles(pos["lat"], pos["lon"], hours, cond, target, pos["uncertainty_nm"])
            area = engine.search_area(parts)
            cal = engine.calibrated_radius90(hours)
            if cal:  # validated on real NOAA drifters: Monte Carlo alone was overconfident
                area["r90_nm"] = max(area["r90_nm"], cal + pos["uncertainty_nm"])
                area["area90_nm2"] = 3.1416 * area["r90_nm"] ** 2
            st.metric("90% search radius", f"{area['r90_nm']:.1f} nm",
                      help="Conformal radius calibrated on real NOAA drifter tracks (90% coverage on held-out buoys)")
            st.metric("Search area (90%)", f"{area['area90_nm2']:.0f} nm²")
            ranked = engine.rank_assets(assets(), area["lat"], area["lon"])
            st.dataframe(pd.DataFrame(ranked)[["asset", "eta_h", "distance_nm"]], hide_index=True, width="stretch")
        with c1:
            st.pydeck_chart(response_map(pos["lat"], pos["lon"], parts, area, ranked))
            st.caption("Red: reported position · blue cloud: drift particles · ring: 90% search area · green: nearest assets.")

        if st.button("➕ Add to incident board", key=f"add_{key}"):
            st.session_state.board.append({
                "id": f"INC-{len(st.session_state.board) + 1:03d}",
                "vessel": j["fields"].get("vessel_name", {}).get("value", "unknown"),
                "severity": rep.severity, "lat": area["lat"], "lon": area["lon"],
                "incident": ", ".join(j["fields"]["incident_types"]["value"]),
                "missing": ", ".join(j["missing_fields"]) or "—",
            })
            st.success("Added. See the Incident board tab.")

    with st.expander("Structured incident JSON"):
        st.json(j)


def gold_view(d: dict) -> dict:
    inc = d["incident"]
    g = {"vessel_name": inc["vessel"]["name"], "pob": str(inc["pob"]),
         "incident_types": ", ".join(sorted(inc["incident_types"])), "mmsi": inc["vessel"]["mmsi"] or "—",
         "vessel_type": inc["vessel"]["vessel_type"],
         "position": f"{inc['position']['lat']:.4f}, {inc['position']['lon']:.4f}"}
    rep = pipe.extract(d["transcript"]).fields.get("position")
    if rep:
        g["_pos_ok"] = haversine_nm(rep.value["lat"], rep.value["lon"], inc["position"]["lat"], inc["position"]["lon"]) <= max(1.0, rep.value["uncertainty_nm"])
    return g


# --- layout -------------------------------------------------------------------

st.title("⚓ Maritime Distress Call Intelligence")
st.caption("Prototype decision-support system on synthetic data. The operator verifies every field; "
           "the system never dispatches on its own.")

tabs = st.tabs(["📻 Live operations", "🗂 Incident board", "🎓 Training simulator", "📊 Model evaluation", "🧪 Dataset"])

with tabs[0]:
    test = load_split("test")
    audio = load_audio_manifest()
    src = st.radio("Input", ["Simulated incoming call", "Paste transcript", "Upload audio"], horizontal=True)

    if src == "Simulated incoming call":
        c1, c2, c3 = st.columns([1, 1, 2])
        lang_f = c1.selectbox("Language", ["any", "en", "hi", "ta", "ml"])
        only_audio = c2.checkbox("Only calls with audio", value=False, disabled=not audio)
        pool = [d for d in test if (lang_f == "any" or d["language"] == lang_f) and (not only_audio or d["id"] in audio)]
        if c3.button("📞 Next incoming call", type="primary") or st.session_state.call is None:
            st.session_state.call = random.choice(pool)["id"] if pool else None
        d = next((x for x in test if x["id"] == st.session_state.call), None)
        if d:
            st.caption(f"Call {d['id']} · language **{d['language']}** · caller style **{d['style']}**")
            transcript = d["transcript"]
            if d["id"] in audio:
                st.audio(audio[d["id"]]["wav"])
                st.caption(f"Simulated VHF channel, SNR ≈ {audio[d['id']]['call_snr_db']:.1f} dB")
                if st.toggle("Use Whisper ASR transcript instead of the reference text"):
                    from mdi.speech import transcribe
                    with st.spinner("Transcribing each turn with Whisper (CPU)…"):
                        transcript = "\n".join(
                            f"{t['speaker']}: {transcribe(t['wav'], Language(t['lang']))['text']}"
                            for t in audio[d["id"]]["turns"])
            run_report(transcript, Language(d["language"]), gold=gold_view(d), key=d["id"])

    elif src == "Paste transcript":
        lang = Language(st.selectbox("Caller language", [l.value for l in Language]))
        txt = st.text_area("Transcript (prefix lines with CALLER: / MRCC:)", height=160,
                           value="CALLER: MAYDAY MAYDAY MAYDAY, this is Sea Hawk. position one zero degrees zero five decimal two north, "
                                 "zero seven five degrees five eight decimal one east. we are taking water. six persons on board, over")
        if txt.strip():
            run_report(txt, lang, key="paste")

    else:
        lang = Language(st.selectbox("Caller language", [l.value for l in Language]))
        up = st.file_uploader("WAV/MP3 of the caller", type=["wav", "mp3", "m4a", "flac"])
        if up:
            import tempfile
            from mdi.speech import transcribe
            with tempfile.NamedTemporaryFile(suffix=Path(up.name).suffix, delete=False) as tmp:
                tmp.write(up.read())
            st.audio(tmp.name)
            with st.spinner("Whisper transcription (CPU)…"):
                r = transcribe(tmp.name, lang)
            st.caption(f"ASR confidence {r['confidence']:.2f} · detected language {r['language']} ({r['language_probability']:.2f})")
            run_report("CALLER: " + r["text"], lang, key="upload")

with tabs[1]:
    board = st.session_state.board
    if not board:
        st.info("No open incidents. Add some from the Live operations tab.")
    else:
        df = pd.DataFrame(board).sort_values("severity", ascending=False)
        df["severity"] = df.severity.map(lambda s: f"{SEV_ICON[SEVERITY_NAMES[s]]} {SEVERITY_NAMES[s]}")
        st.dataframe(df, hide_index=True, width="stretch")
        st.markdown("##### Optimal asset assignment")
        st.caption("Hungarian algorithm minimising Σ ETA × severity weight (LOW 1, MODERATE 2, HIGH 4, CRITICAL 8); "
                   "one asset per incident.")
        st.dataframe(pd.DataFrame(engine.assign(assets(), board)), hide_index=True, width="stretch")
        if st.button("Clear board"):
            st.session_state.board = []
            st.rerun()

with tabs[2]:
    st.markdown("Read the call as a trainee operator, fill in the incident, then compare with the ground truth and the model.")
    train_pool = [d for d in load_split("test") if d["language"] == "en"] or load_split("test")
    if st.button("New training scenario") or "drill" not in st.session_state:
        st.session_state.drill = random.choice(train_pool)
    d = st.session_state.drill
    st.code(d["transcript"], language=None)
    with st.form("drill_form"):
        c1, c2, c3 = st.columns(3)
        v = c1.text_input("Vessel name")
        p = c2.number_input("Persons on board", min_value=0, step=1)
        s = c3.selectbox("Severity", list(SEVERITY_NAMES.values()))
        inc = st.multiselect("Nature of distress", [t for t in pipe.incident_clf.mlb.classes])
        miss = st.multiselect("Which critical fields are still missing?", ["vessel_name", "position", "incident_types", "pob"])
        ok = st.form_submit_button("Submit")
    if ok:
        g = d["incident"]
        rep = pipe.extract(d["transcript"])
        res = pd.DataFrame([
            {"item": "vessel", "you": v, "model": rep.fields.get("vessel_name").value if "vessel_name" in rep.fields else "—", "truth": g["vessel"]["name"]},
            {"item": "persons on board", "you": p, "model": rep.fields["pob"].value if "pob" in rep.fields else "—", "truth": g["pob"]},
            {"item": "severity", "you": s, "model": SEVERITY_NAMES[rep.severity], "truth": SEVERITY_NAMES[g["severity"]]},
            {"item": "incident", "you": ", ".join(sorted(inc)), "model": ", ".join(sorted(rep.fields["incident_types"].value)), "truth": ", ".join(sorted(g["incident_types"]))},
            {"item": "missing fields", "you": ", ".join(sorted(miss)), "model": ", ".join(sorted(rep.missing_fields)), "truth": ", ".join(sorted(d["missing_fields"]))},
        ]).astype(str)
        res["you ✓"] = np.where(res.you.str.lower() == res.truth.str.lower(), "✅", "❌")
        res["model ✓"] = np.where(res.model.str.lower() == res.truth.str.lower(), "✅", "❌")
        st.dataframe(res, hide_index=True, width="stretch")
        st.metric("Your score", f"{(res['you ✓'] == '✅').sum()} / {len(res)}")

with tabs[3]:
    mp = ROOT / "reports/metrics.json"
    if not mp.exists():
        st.info("Run scripts/train_and_evaluate.py to produce reports/metrics.json.")
    else:
        m = json.loads(mp.read_text(encoding="utf-8"))
        e2e, sev = m["end_to_end"], m["severity"]
        k = st.columns(5)
        k[0].metric("NER micro-F1", f"{m['ner']['micro_f1']:.3f}", help=f"95% CI {m['ner']['micro_f1_ci95'][0]:.3f}–{m['ner']['micro_f1_ci95'][1]:.3f}")
        k[1].metric("Incident macro-F1", f"{m['incident']['tfidf_lr_macro_f1']:.3f}")
        chosen = {"OrdinalLogistic": "ordinal_frank_hall"}.get(sev["chosen_model"], "multinomial_temperature_scaled")
        k[2].metric("Severity QWK", f"{sev[chosen]['quadratic_kappa']:.3f}", help=f"model in use: {chosen}")
        k[3].metric("Position correct", f"{e2e.get('position_correct', 0):.1%}", help="within the format's precision (≥1 nm)")
        k[4].metric("Conformal coverage", f"{sev['conformal_90']['empirical_coverage']:.1%}", help="target 90%")
        st.caption(f"Test set: {m['sizes']['test']} dialogues with vessel names and phrasings partly unseen in training.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**NER F1 by entity**")
            df = pd.DataFrame(m["ner"]["per_label_f1"].items(), columns=["label", "F1"])
            st.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=SERIES[0]).encode(
                x=alt.X("F1:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("label:N", sort="-x", title=None),
                tooltip=["label", alt.Tooltip("F1:Q", format=".3f")]), width="stretch")
        with c2:
            st.markdown("**Incident classification: keyword baseline vs TF-IDF + LR**")
            rows = [{"phrasing": ph, "model": mo, "macro F1": m["incident"][f"{ph}_phrasing"][key]}
                    for ph in ("seen", "unseen") for mo, key in (("keyword baseline", "baseline"), ("TF-IDF + LR", "tfidf_lr"))]
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X("macro F1:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("model:N", title=None),
                color=alt.Color("model:N", scale=alt.Scale(range=SERIES[:2])), row=alt.Row("phrasing:N", title=None),
                tooltip=["phrasing", "model", alt.Tooltip("macro F1:Q", format=".3f")]).properties(height=60), width="stretch")

        st.markdown("**Robustness: extraction quality vs simulated ASR word error rate**")
        rc = pd.DataFrame(m["robustness_curve"]).melt(id_vars=["corruption_rate", "measured_wer"], var_name="metric", value_name="score")
        st.altair_chart(alt.Chart(rc).mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2).encode(
            x=alt.X("measured_wer:Q", title="word error rate on caller speech", axis=alt.Axis(format="%")),
            y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(range=SERIES)),
            tooltip=["metric", alt.Tooltip("measured_wer:Q", format=".1%"), alt.Tooltip("score:Q", format=".3f")],
        ).properties(height=300), width="stretch")

        st.markdown("**Severity models**")
        tbl = pd.DataFrame({k: v for k, v in sev.items() if isinstance(v, dict) and "accuracy" in v}).T
        st.dataframe(tbl.style.format("{:.3f}"), width="stretch")
        st.caption(f"Majority-class baseline accuracy {sev['majority_baseline_accuracy']:.3f}. "
                   f"ECE = expected calibration error (lower is better). Conformal 90% sets: coverage "
                   f"{sev['conformal_90']['empirical_coverage']:.1%}, mean set size {sev['conformal_90']['mean_set_size']:.2f}.")

        st.markdown("**Missing-information detection**")
        st.dataframe(pd.DataFrame(e2e["missing_info"]).T.style.format({"precision": "{:.3f}", "recall": "{:.3f}"}),
                     width="stretch")

        zs, fin = ROOT / "reports/martts_zeroshot.json", ROOT / "reports/martts_final.json"
        if zs.exists() and fin.exists():
            st.markdown("**External test: MARTTS** (DLR, 240 English radio dialogues, different generator and TTS)")
            z, f_ = json.loads(zs.read_text())["test_text"], json.loads(fin.read_text())["test_text"]
            rows = [{"metric": k.replace("_", " "), "stage": stage, "rate": src[k]["rate"]}
                    for k in f_ if k != "mean_severity_by_topic"
                    for stage, src in (("zero-shot", z), ("after dev-informed fixes", f_))]
            st.altair_chart(alt.Chart(pd.DataFrame(rows)).mark_bar(cornerRadiusEnd=4, height=10).encode(
                x=alt.X("rate:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
                y=alt.Y("stage:N", title=None, axis=None),
                color=alt.Color("stage:N", scale=alt.Scale(range=SERIES[:2]), legend=alt.Legend(orient="top", title=None)),
                row=alt.Row("metric:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
                tooltip=["metric", "stage", alt.Tooltip("rate:Q", format=".0%")],
            ).properties(height=26), width="stretch")
            st.caption("Fixes were developed on the MARTTS dev part only (~30% of base scenarios); the test part "
                       "was scored once before and once after. 'Routine flagged high or critical' is an error rate (lower is better).")

        for f in sorted((ROOT / "reports").glob("asr_*.json")):
            a = json.loads(f.read_text(encoding="utf-8"))
            st.markdown(f"**ASR: Whisper {a['model']}**: caller WER {a['caller_wer']:.1%}, CER {a['caller_cer']:.1%}")
            st.dataframe(pd.DataFrame(a["by_group"]).T, width="stretch")

with tabs[4]:
    sp = ROOT / "data/synthetic/stats.json"
    if sp.exists():
        s = json.loads(sp.read_text(encoding="utf-8"))
        st.metric("Dialogues", f"{s['n']:,}")
        cols = st.columns(3)
        for col, key in zip(cols * 2, ["language", "style", "severity", "incident_type", "position_format", "split"]):
            df = pd.DataFrame(s[key].items(), columns=[key, "count"])
            col.markdown(f"**{key.replace('_', ' ')}**")
            col.altair_chart(alt.Chart(df).mark_bar(cornerRadiusEnd=4, color=SERIES[0]).encode(
                x="count:Q", y=alt.Y(f"{key}:N", sort="-x", title=None), tooltip=[key, "count"]
            ).properties(height=170), width="stretch")
        d = random.choice(load_split("train"))
        st.markdown("**Random training example**")
        st.code(d["transcript"], language=None)
        st.json({"incident": d["incident"], "missing_fields": d["missing_fields"], "contradiction": d["contradiction"]})
