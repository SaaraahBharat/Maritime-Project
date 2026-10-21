/* Maritime Distress Intelligence: operator console (no build step). */
"use strict";

const SEV = {
  LOW: { c: "#0ca30c", n: 1 }, MODERATE: { c: "#fab219", n: 2 },
  HIGH: { c: "#ec835a", n: 3 }, CRITICAL: { c: "#d03b3b", n: 4 },
};
const SEV_BY_LEVEL = { 1: "LOW", 2: "MODERATE", 3: "HIGH", 4: "CRITICAL" };
const ENT = {
  VESSEL: "var(--ent-vessel)", VESSEL_TYPE: "var(--ent-type)", MMSI: "var(--ent-mmsi)",
  POSITION: "var(--ent-pos)", POB: "var(--ent-pob)", INCIDENT: "var(--ent-inc)",
};
const FIELD_COLOR = {
  vessel_name: ENT.VESSEL, vessel_type: ENT.VESSEL_TYPE, mmsi: ENT.MMSI, position: ENT.POSITION,
  pob: ENT.POB, incident_types: ENT.INCIDENT,
};
const LANG_NAME = { en: "English", hi: "Hindi", ta: "Tamil", ml: "Malayalam" };
const ASSET_ICON = { helicopter: "🚁", ship: "🛳", boat: "🚤", vessel_of_opportunity: "⚓" };

const state = {
  lang: "", audioOnly: false, calls: [], call: null, report: null, response: null, transcript: "",
  useAsr: false, useLlm: false, hours: 2, mlDrift: true, board: [], verified: new Set(), translation: null,
  health: {}, mapReady: false, playing: null,
};

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (x, d = 2) => (x == null || Number.isNaN(x) ? "–" : Number(x).toFixed(d));

async function api(path, opts = {}) {
  const r = await fetch(path, {
    headers: opts.body && !(opts.body instanceof FormData) ? { "Content-Type": "application/json" } : {},
    ...opts,
    body: opts.body && !(opts.body instanceof FormData) ? JSON.stringify(opts.body) : opts.body,
  });
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), ms);
}

/* ---------------- theme, clock, tabs ---------------- */
function initChrome() {
  let saved = null;
  try { saved = localStorage.getItem("mdi-theme"); } catch (e) { /* storage blocked */ }
  if (saved) document.documentElement.dataset.theme = saved;
  $("#theme-btn").onclick = () => {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("mdi-theme", next); } catch (e) { /* ignore */ }
    setBasemap();
  };
  const tick = () => ($("#clock").textContent = new Date().toISOString().slice(11, 19) + " UTC");
  tick();
  setInterval(tick, 1000);
  $$(".tab").forEach((b) => (b.onclick = () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
    $$(".view").forEach((v) => v.classList.toggle("active", v.id === "view-" + b.dataset.view));
    if (b.dataset.view === "perf") renderPerf();
    if (b.dataset.view === "ops" && map) setTimeout(() => map.resize(), 50);
  }));
}

async function initStatus() {
  try {
    state.health = await api("/api/health");
  } catch (e) {
    state.health = {};
  }
  const h = state.health;
  $("#status").innerHTML = [
    ["Extraction models", h.pipeline], ["Drift ML", h.drift_ml], ["Whisper ASR", h.asr_available], ["LLM", h.llm_available],
  ].map(([l, on]) => `<span class="pill ${on ? "on" : ""}">${l}</span>`).join("");
}

/* ---------------- map ---------------- */
let map, distressMarker;
const STYLES = {
  dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
};
const FALLBACK_STYLE = { version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#0a1826" } }] };

function initMap() {
  if (!window.maplibregl) return;
  map = new maplibregl.Map({
    container: "map", style: currentStyle(), center: [76.5, 13.5], zoom: 4.6, attributionControl: { compact: true },
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
  map.on("error", (e) => {
    if (!state.mapReady && String(e?.error?.message || "").match(/style|fetch|Failed/i)) map.setStyle(FALLBACK_STYLE);
  });
  map.on("style.load", () => {
    addLayers();
    state.mapReady = true;
    drawResponse();
  });
}

function currentStyle() {
  return document.documentElement.dataset.theme === "light" ? STYLES.light : STYLES.dark;
}

function setBasemap() {
  if (map) map.setStyle(currentStyle(), { diff: false });  // full reload so style.load re-adds our layers
}

const empty = () => ({ type: "FeatureCollection", features: [] });

function addLayers() {
  const src = (id) => map.getSource(id) || map.addSource(id, { type: "geojson", data: empty() });
  ["ring", "particles", "routes", "assets", "shift"].forEach(src);
  map.addLayer({ id: "ring-fill", type: "fill", source: "ring", paint: { "fill-color": "#38bdf8", "fill-opacity": 0.1 } });
  map.addLayer({ id: "ring-line", type: "line", source: "ring", paint: { "line-color": "#38bdf8", "line-width": 2, "line-dasharray": [2, 1.5] } });
  map.addLayer({
    id: "particles", type: "circle", source: "particles",
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 1.4, 10, 3.5], "circle-color": "#38bdf8", "circle-opacity": 0.55, "circle-blur": 0.4 },
  });
  map.addLayer({ id: "routes", type: "line", source: "routes", paint: { "line-color": "#1baf7a", "line-width": ["case", ["==", ["get", "rank"], 1], 2.6, 1.4], "line-opacity": 0.85, "line-dasharray": [1, 1.2] } });
  map.addLayer({ id: "shift", type: "line", source: "shift", paint: { "line-color": "#c084fc", "line-width": 3 } });
  map.addLayer({
    id: "assets", type: "circle", source: "assets",
    paint: { "circle-radius": ["case", ["==", ["get", "rank"], 1], 8, 6], "circle-color": "#1baf7a", "circle-stroke-color": "#fff", "circle-stroke-width": 1.5 },
  });
  map.addLayer({
    id: "asset-labels", type: "symbol", source: "assets",
    layout: { "text-field": ["get", "label"], "text-size": 11, "text-offset": [0, 1.4], "text-anchor": "top", "text-font": ["Open Sans Semibold", "Noto Sans Regular"] },
    paint: { "text-color": "#bff3dd", "text-halo-color": "#06121c", "text-halo-width": 1.4 },
  });
  map.on("click", "assets", (e) => {
    const p = e.features[0].properties;
    new maplibregl.Popup({ offset: 10 }).setLngLat(e.lngLat)
      .setHTML(`<b>${esc(p.name)}</b><br>${esc(p.kind.replaceAll("_", " "))}<br>ETA <b>${p.eta_h} h</b> · ${p.distance_nm} nm`).addTo(map);
  });
  map.on("mouseenter", "assets", () => (map.getCanvas().style.cursor = "pointer"));
  map.on("mouseleave", "assets", () => (map.getCanvas().style.cursor = ""));
}

function circle(lat, lon, rNm, n = 96) {
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const a = (i / n) * 2 * Math.PI;
    const dLat = (rNm / 60) * Math.cos(a);
    const dLon = ((rNm / 60) * Math.sin(a)) / Math.cos((lat * Math.PI) / 180);
    pts.push([lon + dLon, lat + dLat]);
  }
  return { type: "Feature", geometry: { type: "Polygon", coordinates: [pts] }, properties: {} };
}

function setData(id, fc) {
  const s = map && map.getSource(id);
  if (s) s.setData(fc);
}

function drawResponse(fit = false) {
  if (!map || !state.mapReady) return;
  const pos = state.report?.fields?.position?.value;
  const r = state.response;
  if (!pos || !r) {
    ["ring", "particles", "routes", "assets", "shift"].forEach((id) => setData(id, empty()));
    if (distressMarker) distressMarker.remove();
    return;
  }
  setData("ring", { type: "FeatureCollection", features: [circle(r.area.lat, r.area.lon, r.area.r90_nm)] });
  setData("particles", { type: "FeatureCollection", features: r.particles.map(([la, lo]) => ({ type: "Feature", geometry: { type: "Point", coordinates: [lo, la] }, properties: {} })) });
  setData("assets", {
    type: "FeatureCollection",
    features: r.assets.map((a, i) => ({ type: "Feature", geometry: { type: "Point", coordinates: [a.lon, a.lat] }, properties: { ...a, name: a.asset, rank: i + 1, label: i < 3 ? `${a.asset.replace(/ \(.*\)/, "")} · ${a.eta_h} h` : "" } })),
  });
  setData("routes", {
    type: "FeatureCollection",
    features: r.assets.slice(0, 3).map((a, i) => ({ type: "Feature", geometry: { type: "LineString", coordinates: [[a.lon, a.lat], [r.area.lon, r.area.lat]] }, properties: { rank: i + 1 } })),
  });
  setData("shift", r.ml_shift_nm ? {
    type: "FeatureCollection",
    features: [{ type: "Feature", geometry: { type: "LineString", coordinates: [[r.physics_centre.lon, r.physics_centre.lat], [r.area.lon, r.area.lat]] }, properties: {} }],
  } : empty());

  if (!distressMarker) {
    const el = document.createElement("div");
    el.className = "pulse-marker";
    distressMarker = new maplibregl.Marker({ element: el });
  }
  distressMarker.setLngLat([pos.lon, pos.lat]).addTo(map);
  if (fit) {
    const b = new maplibregl.LngLatBounds([pos.lon, pos.lat], [pos.lon, pos.lat]);
    b.extend([r.area.lon, r.area.lat]);
    r.assets.slice(0, 3).forEach((a) => b.extend([a.lon, a.lat]));
    const pad = r.area.r90_nm / 60;
    b.extend([r.area.lon + pad, r.area.lat + pad]);
    b.extend([r.area.lon - pad, r.area.lat - pad]);
    map.fitBounds(b, { padding: { top: 70, bottom: 110, left: 60, right: 60 }, duration: 1400, maxZoom: 10 });
  }
}

/* ---------------- queue ---------------- */
async function loadCalls() {
  const q = new URLSearchParams({ limit: 60 });
  if (state.lang) q.set("language", state.lang);
  if (state.audioOnly) q.set("audio_only", "true");
  state.calls = await api("/api/calls?" + q);
  renderCalls();
}

function renderCalls() {
  $("#call-list").innerHTML = state.calls.map((c) => {
    const done = state.board.find((b) => b.call === c.id);
    return `<div class="call ${state.call?.id === c.id ? "active" : ""}" data-id="${c.id}">
      <span class="ct ${c.call_type}">${c.call_type}</span>
      <span class="vn">${esc(c.vessel)}</span>
      ${done ? `<span class="sev-dot" style="background:${SEV[done.severity].c}"></span>` : `<span>${c.has_audio ? "🎙" : ""}</span>`}
      <span class="meta">${c.id} · ${c.style}</span>
      <span class="lang" style="justify-self:end">${c.language.toUpperCase()}</span>
    </div>`;
  }).join("") || `<p class="muted" style="padding:12px">No calls match.</p>`;
  $$(".call").forEach((el) => (el.onclick = () => openCall(el.dataset.id)));
}

function initQueue() {
  $$("#lang-filter .chip").forEach((b) => (b.onclick = () => {
    $$("#lang-filter .chip").forEach((x) => x.classList.toggle("active", x === b));
    state.lang = b.dataset.lang;
    loadCalls();
  }));
  $("#audio-only").onchange = (e) => { state.audioOnly = e.target.checked; loadCalls(); };
  $("#paste-btn").onclick = openPaste;
  $("#upload-btn").onclick = openUpload;
}

/* ---------------- pipeline stepper ---------------- */
function step(name, status, note = "") {
  const el = $(`.step[data-step="${name}"]`);
  el.classList.remove("run", "done", "skip");
  if (status) el.classList.add(status);
  $("em", el).textContent = note;
}
const resetSteps = () => $$(".step").forEach((s) => { s.classList.remove("run", "done", "skip"); $("em", s).textContent = ""; });

/* ---------------- analysis flow ---------------- */
async function openCall(id) {
  stopAudio();
  state.call = await api(`/api/calls/${id}`);
  state.verified = new Set();
  state.translation = null;
  state.useAsr = state.useAsr && !!state.call.audio;
  renderCalls();
  await analyse();
}

async function analyse() {
  const c = state.call;
  resetSteps();
  $("#map-empty").classList.add("hidden");
  renderSkeleton();
  step("audio", c.audio ? "done" : "skip", c.audio ? `${fmt(c.snr_db, 0)} dB SNR` : "text");

  state.transcript = c.transcript;
  state.asrInfo = null;
  if (state.useAsr && c.audio) {
    step("asr", "run", "Whisper…");
    try {
      const r = await api(`/api/calls/${c.id}/transcribe?indic=true`, { method: "POST" });
      state.transcript = r.transcript;
      state.asrInfo = r;
      step("asr", "done", `${r.seconds}s`);
    } catch (e) {
      step("asr", "skip", "failed");
      toast("ASR failed, using reference transcript");
    }
  } else {
    step("asr", "skip", "reference");
  }
  await runExtraction();
}

async function runExtraction() {
  const c = state.call;
  step("ner", "run");
  step("sev", "run");
  const t0 = performance.now();
  state.report = await api("/api/extract", { method: "POST", body: { transcript: state.transcript, language: c.language, llm: state.useLlm } });
  const ms = Math.round(performance.now() - t0);
  step("ner", "done", state.useLlm ? `${ms} ms +LLM` : `${ms} ms`);
  step("sev", "done", state.report.severity);
  await runResponse(true);
}

async function runResponse(fit = false) {
  const pos = state.report?.fields?.position?.value;
  renderIncident();
  if (!pos) {
    state.response = null;
    step("drift", "skip", "no position");
    step("dispatch", "skip");
    $("#timeline").classList.add("disabled");
    drawResponse();
    return;
  }
  $("#timeline").classList.remove("disabled");
  step("drift", "run");
  const types = state.report.fields.incident_types?.value || [];
  const target = types.includes("man_overboard") ? "person_in_water" : state.report.fields.vessel_type?.value || null;
  state.response = await api("/api/response", {
    method: "POST",
    body: { lat: pos.lat, lon: pos.lon, uncertainty_nm: pos.uncertainty_nm, hours: state.hours, target, ml_drift: state.mlDrift },
  });
  const r = state.response;
  step("drift", "done", `r90 ${fmt(r.area.r90_nm, 1)} nm`);
  step("dispatch", "done", `${r.assets[0]?.eta_h ?? "–"} h`);
  const cd = r.conditions;
  $("#conditions").innerHTML = `wind ${fmt(cd.wind_kn, 0)} kn ↗${fmt((cd.wind_from_deg + 180) % 360, 0)}°<br>current ${fmt(cd.current_kn, 1)} kn ↗${fmt(cd.current_to_deg, 0)}°<br><span style="color:var(--faint)">${esc(cd.source)}</span>`;
  drawResponse(fit);
  renderIncident();
}

/* ---------------- incident panel ---------------- */
function renderSkeleton() {
  $("#incident").innerHTML = `<div class="sec">${'<div class="shimmer"></div>'.repeat(3)}</div><div class="sec">${'<div class="shimmer"></div>'.repeat(6)}</div>`;
}

function highlight(text, ents) {
  let out = "", pos = 0;
  for (const e of [...ents].sort((a, b) => a.start - b.start)) {
    if (e.start < pos) continue;
    out += esc(text.slice(pos, e.start));
    out += `<mark class="ent" style="--c:${ENT[e.label] || "var(--accent)"}" title="${e.label} · confidence ${fmt(e.confidence)}">${esc(text.slice(e.start, e.end))}<sub>${e.label.replace("_", " ")}</sub></mark>`;
    pos = e.end;
  }
  out += esc(text.slice(pos));
  return out.replace(/^(CALLER|MRCC):/gm, (m, s) => `<span class="spk ${s === "CALLER" ? "caller" : ""}">${s}</span>`);
}

function fieldValue(k, v) {
  if (k === "position") return `${fmt(v.lat, 4)}°, ${fmt(v.lon, 4)}° <span class="muted" style="font-weight:400">±${v.uncertainty_nm} nm · ${v.format}</span>`;
  if (Array.isArray(v)) return v.length ? v.map((x) => x.replaceAll("_", " ")).join(", ") : "–";
  return esc(String(v).replaceAll("_", " "));
}

function goldCheck(k, f) {
  const g = state.call?.gold;
  if (!g || state.call.id.startsWith("custom")) return "";
  let gv, hit;
  if (k === "vessel_name") { gv = g.vessel.name; hit = String(f.value).toLowerCase() === gv.toLowerCase(); }
  else if (k === "mmsi") { gv = g.vessel.mmsi; if (!gv) return ""; hit = f.value === gv; }
  else if (k === "pob") { gv = g.pob; hit = Number(f.value) === gv; }
  else if (k === "vessel_type") { gv = g.vessel.vessel_type; hit = f.value === gv; }
  else if (k === "incident_types") { gv = g.incident_types.join(", "); hit = [...f.value].sort().join() === [...g.incident_types].sort().join(); }
  else if (k === "position") {
    const d = nm(f.value.lat, f.value.lon, g.position.lat, g.position.lon);
    gv = `${fmt(d, 2)} nm from truth`; hit = d <= Math.max(1, f.value.uncertainty_nm);
  } else return "";
  return `<div class="gold ${hit ? "hit" : "miss"}">${hit ? "✓ matches ground truth" : "✗ ground truth: " + esc(String(gv).replaceAll("_", " "))}</div>`;
}

function nm(a, b, c, d) {
  const R = 3440.065, t = Math.PI / 180;
  const x = Math.sin(((c - a) * t) / 2) ** 2 + Math.cos(a * t) * Math.cos(c * t) * Math.sin(((d - b) * t) / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(x));
}

function bar(label, v, color, max = 1, valText) {
  return `<div class="bar-row"><span class="lbl" title="${esc(label)}">${esc(label)}</span>
    <div class="bar-track"><div class="bar-fill" style="width:${Math.max(0, Math.min(100, (v / max) * 100))}%;background:${color}"></div></div>
    <span class="v">${valText ?? fmt(v)}</span></div>`;
}

function renderIncident() {
  const j = state.report, c = state.call, r = state.response;
  if (!j) return;
  const sev = j.severity, sc = SEV[sev].c;
  const fields = ["vessel_name", "vessel_type", "mmsi", "position", "pob", "incident_types"];
  const vessel = j.fields.vessel_name?.value || "Unknown vessel";
  const order = Object.keys(SEV);

  const fieldRows = fields.map((k) => {
    const f = j.fields[k];
    const missing = j.missing_fields.includes(k);
    if (!f) {
      return `<div class="field"><span class="k" style="--c:${FIELD_COLOR[k]}">${k.replaceAll("_", " ")}</span>
        <span class="val" style="color:${missing ? "var(--bad)" : "var(--faint)"}">${missing ? "missing" : "–"}</span><span></span></div>`;
    }
    const llm = String(f.evidence || "").startsWith("[LLM]");
    const ev = String(f.evidence || "").replace("[LLM] ", "");
    const ok = state.verified.has(k);
    return `<div class="field"><span class="k" style="--c:${FIELD_COLOR[k]}">${k.replaceAll("_", " ")}</span>
      <span class="val">${fieldValue(k, f.value)} ${llm ? '<span class="tag">LLM</span>' : ""}</span>
      <span class="conf"><span class="mini-track" title="confidence ${fmt(f.confidence)}"><span class="mini-fill" style="width:${f.confidence * 100}%;background:${f.confidence < 0.6 ? "var(--warn)" : "var(--ok)"}"></span></span>
        <button class="verify ${ok ? "ok" : ""}" data-k="${k}" title="Operator verified">✓</button></span>
      ${ev ? `<span class="ev">“${esc(ev)}”</span>` : ""}
      ${goldCheck(k, f)}</div>`;
  }).join("");

  const alerts = [
    ...j.missing_fields.map((f) => `<div class="alert miss">⚠<div><b>Missing:</b> ${f.replaceAll("_", " ")}</div></div>`),
    ...j.low_confidence.map((f) => `<div class="alert low">?<div><b>Verify:</b> ${f.replaceAll("_", " ")} has low confidence</div></div>`),
    ...j.contradictions.map((x) => `<div class="alert low">⇄<div><b>Contradiction:</b> ${esc(x)}</div></div>`),
    ...j.suggested_questions.map((q) => `<div class="alert ask">🎙<div><b>Ask the caller:</b> “${esc(q)}?”</div></div>`),
  ].join("");

  const incBars = Object.entries(j.incident_proba).slice(0, 5)
    .map(([k, v]) => bar(k.replaceAll("_", " "), v, "var(--ent-inc)")).join("");
  const maxC = Math.max(...j.explanation.map((e) => Math.abs(e.contribution)), 1e-6);
  const why = j.explanation.slice(0, 6).map((e) =>
    bar(e.feature.replace(/^p_/, "P(").replace(/^(P\(.*)$/, "$1)").replaceAll("_", " "), Math.abs(e.contribution),
      e.contribution > 0 ? "#eb6834" : "#2a78d6", maxC, (e.contribution > 0 ? "+" : "−") + fmt(Math.abs(e.contribution)))).join("");

  const assets = r ? r.assets.slice(0, 5).map((a, i) => `<div class="asset-row">
      <span class="asset-ic">${ASSET_ICON[a.kind] || "⚓"}</span>
      <div><div>${esc(a.asset)}${i === 0 ? ' <span class="tag" style="background:color-mix(in srgb,var(--ok) 25%,transparent);color:var(--ok)">FASTEST</span>' : ""}</div>
        <div class="sub">${a.kind.replaceAll("_", " ")} · ${a.distance_nm} nm</div></div>
      <span class="eta">${a.eta_h < 1 ? Math.round(a.eta_h * 60) + " min" : a.eta_h + " h"}</span></div>`).join("") : "";

  $("#incident").innerHTML = `
    <div class="sec">
      <div class="hero">
        <div class="sev-badge" style="background:${sc}">${sev}</div>
        <div class="hero-meta">
          <div class="hero-vessel">${esc(vessel)}</div>
          <div class="hero-sub">${LANG_NAME[c.language] || c.language} · ${c.style || "live"} · ${c.id}</div>
          <div class="set-chips" title="90% conformal prediction set: the true severity is in this set for ~90% of calls">
            <span class="muted" style="font-size:11px">90% set</span>
            ${order.map((s) => `<span class="set-chip" style="border-color:${SEV[s].c};${j.severity_90pct_set.includes(s) ? `background:${SEV[s].c};color:#111` : `color:${SEV[s].c};opacity:.45`}">${s}</span>`).join("")}
          </div>
        </div>
      </div>
      <div class="bars" style="margin-top:12px">${order.map((s) => bar(s, j.severity_proba[s] || 0, SEV[s].c)).join("")}</div>
    </div>
    <div class="sec">
      <div class="row-actions">
        <label class="switch-row compact"><input type="checkbox" id="use-asr" ${state.useAsr ? "checked" : ""} ${c.audio ? "" : "disabled"}> <span>Whisper ASR transcript${c.audio ? "" : " (no audio)"}</span></label>
        <label class="switch-row compact"><input type="checkbox" id="use-llm" ${state.useLlm ? "checked" : ""} ${state.health.llm_available ? "" : "disabled"}> <span>LLM fills gaps (Qwen 1.5B)</span></label>
      </div>
      ${state.asrInfo ? `<div class="muted" style="font-size:11.5px;margin-top:6px">ASR confidence ${fmt(state.asrInfo.asr_confidence)} · fine-tuned Indic Whisper for HI/TA/ML · ${state.asrInfo.seconds}s</div>` : ""}
    </div>
    ${alerts ? `<div class="sec"><div class="sec-title">Operator attention</div>${alerts}</div>` : ""}
    <div class="sec"><div class="sec-title">Extracted incident <span>${state.verified.size}/${Object.keys(j.fields).length} verified</span></div>${fieldRows}</div>
    ${r ? `<div class="sec"><div class="sec-title">Search & rescue plan <span>${fmt(state.hours, 1)} h after call</span></div>
      <div class="kv"><div><small>90% radius</small><b>${fmt(r.area.r90_nm, 1)}</b> nm</div><div><small>Area</small><b>${Math.round(r.area.area90_nm2)}</b> nm²</div><div><small>Drift</small><b>${fmt(r.drift_nm, 1)}</b> nm</div></div>
      ${r.ml_shift_nm != null ? `<div class="muted" style="font-size:11.5px;margin:8px 0 2px">LightGBM correction moved the search centre <b style="color:#c084fc">${fmt(r.ml_shift_nm, 2)} nm</b> (trained on real NOAA buoys).</div>` : ""}
      <div style="margin-top:8px">${assets}</div></div>` : ""}
    <div class="sec"><div class="sec-title">Incident type probabilities</div><div class="bars">${incBars}</div></div>
    <div class="sec"><div class="sec-title">Why ${sev}? <span>exact linear SHAP</span></div><div class="bars">${why}</div>
      <div class="muted" style="font-size:11px;margin-top:6px"><span style="color:#eb6834">■</span> raises severity · <span style="color:#2a78d6">■</span> lowers severity</div></div>
    <div class="sec"><div class="sec-title">Radio transcript ${state.asrInfo ? "<span>Whisper ASR</span>" : "<span>reference</span>"}</div>
      ${c.audio ? `<div class="audio-box"><button class="icon-btn" id="audio-play">▶</button><canvas id="wave" width="600" height="88"></canvas></div>` : ""}
      <div class="ent-legend">${Object.entries(ENT).map(([k, v]) => `<span style="--c:${v}">${k.replace("_", " ")}</span>`).join("")}</div>
      <div class="transcript">${highlight(state.transcript, j.entities)}</div>
      ${c.language !== "en" ? `<div style="margin-top:10px"><button class="btn sm" id="translate-btn">Translate caller to English (NLLB-200)</button></div>` : ""}
      ${state.translation ? `<div class="translation">${esc(state.translation)}</div><div class="muted" style="font-size:11px;margin-top:4px">Machine translation for context only; fields come from the original language.</div>` : ""}
    </div>
    <div class="sec"><details><summary class="muted" style="cursor:pointer">Structured incident JSON</summary>
      <pre class="mono" style="font-size:11px;white-space:pre-wrap;max-height:240px;overflow:auto">${esc(JSON.stringify({ fields: j.fields, severity: j.severity, severity_90pct_set: j.severity_90pct_set, missing_fields: j.missing_fields }, null, 2))}</pre></details></div>
    <div class="sticky-actions">
      <button class="btn primary" id="board-btn">Add to incident board</button>
    </div>`;

  $$(".verify").forEach((b) => (b.onclick = () => {
    const k = b.dataset.k;
    state.verified.has(k) ? state.verified.delete(k) : state.verified.add(k);
    renderIncident();
  }));
  $("#use-asr").onchange = (e) => { state.useAsr = e.target.checked; analyse(); };
  $("#use-llm").onchange = (e) => { state.useLlm = e.target.checked; runExtraction(); };
  $("#board-btn").onclick = addToBoard;
  const tb = $("#translate-btn");
  if (tb) tb.onclick = translate;
  if (c.audio) initWave(c.audio);
}

async function translate() {
  const b = $("#translate-btn");
  b.disabled = true;
  b.textContent = "Translating… (first run loads the model)";
  try {
    const r = await api("/api/translate", { method: "POST", body: { text: state.transcript, language: state.call.language } });
    state.translation = r.text;
  } catch (e) {
    toast("Translation model unavailable");
  }
  renderIncident();
}

/* ---------------- audio waveform ---------------- */
let audioEl = null, peaks = null, waveUrl = null, raf = 0;

async function initWave(url) {
  const cv = $("#wave");
  if (!cv) return;
  if (waveUrl !== url) {
    waveUrl = url;
    peaks = null;
    stopAudio();
    audioEl = new Audio(url);
    try {
      const buf = await (await fetch(url)).arrayBuffer();
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const data = (await ctx.decodeAudioData(buf)).getChannelData(0);
      const n = 150, step = Math.floor(data.length / n);
      peaks = Array.from({ length: n }, (_, i) => {
        let m = 0;
        for (let j = i * step; j < (i + 1) * step; j += 8) m = Math.max(m, Math.abs(data[j]));
        return m;
      });
      const mx = Math.max(...peaks, 1e-6);
      peaks = peaks.map((p) => p / mx);
      ctx.close();
    } catch (e) {
      peaks = null;
    }
  }
  drawWave();
  $("#audio-play").onclick = () => {
    if (audioEl.paused) { audioEl.play(); loopWave(); } else audioEl.pause();
    $("#audio-play").textContent = audioEl.paused ? "▶" : "❚❚";
  };
  audioEl.onended = () => { const b = $("#audio-play"); if (b) b.textContent = "▶"; drawWave(); };
  cv.onclick = (e) => {
    if (!audioEl.duration) return;
    audioEl.currentTime = ((e.offsetX / cv.clientWidth) * audioEl.duration);
    drawWave();
  };
}

function drawWave() {
  const cv = $("#wave");
  if (!cv) return;
  const g = cv.getContext("2d"), W = cv.width, H = cv.height;
  g.clearRect(0, 0, W, H);
  const prog = audioEl && audioEl.duration ? audioEl.currentTime / audioEl.duration : 0;
  const ps = peaks || Array.from({ length: 150 }, () => 0.15);
  const bw = W / ps.length;
  ps.forEach((p, i) => {
    const h = Math.max(2, p * (H - 10));
    g.fillStyle = i / ps.length < prog ? "#38bdf8" : "rgba(138,160,182,.45)";
    g.fillRect(i * bw + 1, (H - h) / 2, Math.max(1, bw - 2), h);
  });
}

function loopWave() {
  cancelAnimationFrame(raf);
  const f = () => { drawWave(); if (audioEl && !audioEl.paused) raf = requestAnimationFrame(f); };
  f();
}

function stopAudio() {
  if (audioEl) audioEl.pause();
  cancelAnimationFrame(raf);
}

/* ---------------- timeline ---------------- */
function initTimeline() {
  const s = $("#hours");
  let deb;
  s.oninput = () => {
    state.hours = Number(s.value);
    $("#hours-label").textContent = `${fmt(state.hours, 1)} h`;
    clearTimeout(deb);
    deb = setTimeout(() => state.report && runResponse(), 120);
  };
  $("#ml-drift").onchange = (e) => { state.mlDrift = e.target.checked; runResponse(); };
  $("#play-btn").onclick = async () => {
    if (state.playing) { state.playing = false; $("#play-btn").textContent = "▶"; return; }
    if (!state.report?.fields?.position) return;
    state.playing = true;
    $("#play-btn").textContent = "❚❚";
    for (let h = 0; h <= 24 && state.playing; h += 1) {
      s.value = h;
      state.hours = h;
      $("#hours-label").textContent = `${fmt(h, 1)} h`;
      await runResponse(h === 24);
      await new Promise((r) => setTimeout(r, 180));
    }
    state.playing = false;
    $("#play-btn").textContent = "▶";
  };
  $("#timeline").classList.add("disabled");
}

/* ---------------- custom input ---------------- */
function modal(title, html) {
  $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = html;
  $("#modal").hidden = false;
}
const closeModal = () => ($("#modal").hidden = true);

const langSelect = `<select id="m-lang">${Object.entries(LANG_NAME).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select>`;

function openPaste() {
  modal("Paste a radio transcript", `
    <p class="muted" style="margin:0">Prefix each line with <span class="mono">CALLER:</span> or <span class="mono">MRCC:</span>.</p>
    ${langSelect}
    <textarea id="m-text">CALLER: MAYDAY MAYDAY MAYDAY, this is Sea Hawk, MMSI four one nine one two three four five six. position one zero degrees zero five decimal two north, zero seven five degrees five eight decimal one east. engine room fire, we are taking water. six persons on board, over
MRCC: Sea Hawk, this is MRCC Mumbai, received your MAYDAY, over</textarea>
    <button class="btn primary" id="m-go">Analyse</button>`);
  $("#m-go").onclick = () => {
    const language = $("#m-lang").value, transcript = $("#m-text").value.trim();
    if (!transcript) return;
    closeModal();
    customCall(transcript, language, "pasted");
  };
}

function openUpload() {
  modal("Upload caller audio", `
    <p class="muted" style="margin:0">WAV / MP3 / M4A / FLAC. Hindi, Tamil and Malayalam use the fine-tuned Indic Whisper checkpoints.</p>
    ${langSelect}
    <input type="file" id="m-file" accept=".wav,.mp3,.m4a,.flac,audio/*">
    <button class="btn primary" id="m-go">Transcribe & analyse</button>`);
  $("#m-go").onclick = async () => {
    const file = $("#m-file").files[0];
    if (!file) return;
    const language = $("#m-lang").value;
    const fd = new FormData();
    fd.append("file", file);
    fd.append("language", language);
    fd.append("indic", "true");
    $("#m-go").disabled = true;
    $("#m-go").textContent = "Transcribing on CPU…";
    try {
      const r = await api("/api/transcribe", { method: "POST", body: fd });
      closeModal();
      customCall(r.transcript, language, "upload", r);
    } catch (e) {
      toast("Transcription failed: " + e.message.slice(0, 80));
      $("#m-go").disabled = false;
    }
  };
}

async function customCall(transcript, language, style, asr = null) {
  stopAudio();
  state.call = { id: `custom-${Date.now() % 100000}`, language, style, transcript, gold: null, audio: null };
  state.verified = new Set();
  state.translation = null;
  state.useAsr = false;
  renderCalls();
  resetSteps();
  $("#map-empty").classList.add("hidden");
  renderSkeleton();
  step("audio", asr ? "done" : "skip", asr ? "upload" : "text");
  step("asr", asr ? "done" : "skip", asr ? `${asr.seconds}s` : "typed");
  state.transcript = transcript;
  state.asrInfo = asr ? { asr_confidence: asr.asr_confidence, seconds: asr.seconds } : null;
  await runExtraction();
}

/* ---------------- board ---------------- */
function addToBoard() {
  const j = state.report, r = state.response;
  const pos = j.fields.position?.value;
  if (state.board.find((b) => b.call === state.call.id)) return toast("Already on the board");
  state.board.push({
    id: `INC-${String(state.board.length + 1).padStart(3, "0")}`, call: state.call.id,
    vessel: j.fields.vessel_name?.value || "unknown", severity: j.severity, level: j.severity_level,
    incident: (j.fields.incident_types?.value || []).join(", ").replaceAll("_", " ") || "–",
    lat: r ? r.area.lat : pos?.lat, lon: r ? r.area.lon : pos?.lon, pob: j.fields.pob?.value ?? "?",
    missing: j.missing_fields.join(", ").replaceAll("_", " ") || "–", asset: null, eta: null,
  });
  renderBoard();
  renderCalls();
  toast(`Added ${state.board.at(-1).id} to the incident board`);
}

function renderBoard() {
  $("#board-count").textContent = state.board.length;
  $("#board-empty").style.display = state.board.length ? "none" : "block";
  $("#board-table tbody").innerHTML = [...state.board].sort((a, b) => b.level - a.level).map((b) => `<tr>
    <td class="mono">${b.id}</td><td><b>${esc(b.vessel)}</b></td><td>${esc(b.incident)}</td>
    <td><span class="sev-pill" style="background:${SEV[b.severity].c}">${b.severity}</span></td>
    <td class="mono">${b.lat != null ? `${fmt(b.lat, 3)}, ${fmt(b.lon, 3)}` : "unknown"}</td><td>${b.pob}</td>
    <td style="color:${b.missing === "–" ? "var(--muted)" : "var(--bad)"}">${esc(b.missing)}</td>
    <td>${b.asset ? esc(b.asset) : '<span class="muted">unassigned</span>'}</td><td class="mono">${b.eta != null ? b.eta + " h" : "–"}</td></tr>`).join("");
}

async function optimiseAssignment() {
  const located = state.board.filter((b) => b.lat != null);
  if (!located.length) return toast("No incidents with a position yet");
  const res = await api("/api/assign", { method: "POST", body: { incidents: located.map((b) => ({ id: b.id, lat: b.lat, lon: b.lon, severity: b.level })) } });
  res.forEach((a) => Object.assign(state.board.find((b) => b.id === a.incident), { asset: a.asset, eta: a.eta_h }));
  renderBoard();
  toast("Assignment optimised: severity-weighted total ETA minimised");
}

/* ---------------- performance ---------------- */
async function renderPerf() {
  let cards = [];
  try { cards = await api("/api/metrics"); } catch (e) { /* offline */ }
  const groups = {};
  cards.forEach((c) => (groups[c.group] ||= []).push(c));
  const lowerBetter = (t) => /error|WER|ECE/i.test(t);
  const asPct = (c) => c.unit !== "nm" && !/F1|QWK|ECE/.test(c.title);
  const show = (c, v) => (v == null ? "–" : c.unit === "nm" ? fmt(v, 2) : asPct(c) ? fmt(v * 100, 1) : fmt(v, 3));
  $("#perf").innerHTML = Object.entries(groups).map(([g, cs]) => `<div class="perf-group"><h3>${esc(g)}</h3><div class="cards">${cs.map((c) => {
    const max = c.unit === "nm" ? Math.max(c.value, c.baseline || 0) * 1.25 : 1;
    let delta = "";
    if (c.baseline != null) {
      const better = lowerBetter(c.title) ? c.value < c.baseline : c.value > c.baseline;
      const d = c.value - c.baseline;
      delta = `<span class="delta ${better ? "up" : "down"}">${d > 0 ? "+" : "−"}${show(c, Math.abs(d))}${asPct(c) ? " pts" : ""} vs baseline ${show(c, c.baseline)}</span>`;
    }
    return `<div class="card"><div class="t">${esc(c.title)}</div>
      <div class="big">${show(c, c.value)}<small>${c.unit === "nm" ? " nm" : asPct(c) ? " %" : ""}</small></div>
      ${delta}
      ${c.ci ? `<div class="n">95% CI ${show(c, c.ci[0])} – ${show(c, c.ci[1])}</div>` : ""}
      <div class="cmp"><span class="m" style="width:${Math.min(100, (Math.max(0, c.value) / max) * 100)}%"></span>${c.baseline != null ? `<span class="b" style="left:${Math.min(100, (Math.max(0, c.baseline) / max) * 100)}%"></span>` : ""}</div>
      <div class="n">${esc(c.note)}</div></div>`;
  }).join("")}</div></div>`).join("") || `<p class="muted">No reports found. Run the evaluation scripts first.</p>`;
}

/* ---------------- demo mode ---------------- */
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function narrate(text) {
  let n = $("#narration");
  if (!n) {
    n = document.createElement("div");
    n.id = "narration";
    n.className = "narration glass";
    $(".stage").appendChild(n);
  }
  n.hidden = !text;
  n.textContent = text || "";
}

async function demo() {
  const btn = $("#demo-btn");
  btn.disabled = true;
  $$(".tab")[0].click();
  try {
    const pool = await api("/api/calls?audio_only=true&limit=60");
    const pick = pool.find((c) => c.call_type === "MAYDAY" && c.language !== "en") || pool.find((c) => c.call_type === "MAYDAY") || pool[0];
    if (!pick) return toast("No calls with audio found");
    narrate(`Incoming ${pick.call_type} on channel 16 · ${LANG_NAME[pick.language]} speaker`);
    await openCall(pick.id);
    if (audioEl) {
      narrate("Playing the radio call: simulated VHF channel with real background noise");
      $("#audio-play")?.click();
      await sleep(Math.min(9000, (audioEl.duration || 6) * 1000));
      audioEl.pause();
      const b = $("#audio-play");
      if (b) b.textContent = "▶";
    }
    narrate("Fields extracted with evidence spans · severity with a calibrated 90% set · missing info flagged");
    await sleep(3500);
    if (state.report?.fields?.position) {
      narrate("Drift forecast: physics + LightGBM correction learned from real NOAA buoys · conformal 90% search area");
      $("#play-btn").click();
      while (state.playing) await sleep(300);
      narrate("Fastest assets ranked by ETA · add to the board to optimise assignment across incidents");
      await sleep(4000);
    }
  } catch (e) {
    toast("Demo failed: " + e.message.slice(0, 80));
  } finally {
    narrate("");
    btn.disabled = false;
  }
}

/* ---------------- boot ---------------- */
$("#demo-btn").onclick = demo;
$("#modal-close").onclick = closeModal;
$("#modal").onclick = (e) => { if (e.target.id === "modal") closeModal(); };
$("#assign-btn").onclick = optimiseAssignment;
initChrome();
initMap();
initQueue();
initTimeline();
initStatus();
loadCalls().catch((e) => toast("API not reachable: " + e.message));
renderBoard();

// Deep links: #call=test-000015&hours=12&view=perf (handy for demos and screenshots).
(async () => {
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.get("hours")) {
    state.hours = Number(h.get("hours"));
    $("#hours").value = state.hours;
    $("#hours-label").textContent = `${fmt(state.hours, 1)} h`;
  }
  if (h.get("view")) $(`.tab[data-view="${h.get("view")}"]`)?.click();
  if (h.get("call")) await openCall(h.get("call"));
})();
