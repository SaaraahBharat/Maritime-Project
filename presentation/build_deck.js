// Builds presentation/Maritime_Distress_Intelligence.pptx from the live reports/*.json numbers.
// Run:  node presentation/build_deck.js   (re-run after any evaluation to refresh every chart)
"use strict";
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const ROOT = path.resolve(__dirname, "..");
const SKILL_THEME = process.env.APPLY_THEME_JS;
const OUT = path.join(__dirname, process.env.DECK_NAME || "Maritime_Distress_Intelligence.pptx");
const rep = (f) => { try { return JSON.parse(fs.readFileSync(path.join(ROOT, "reports", f), "utf8")); } catch (e) { return null; } };

// ---- data -------------------------------------------------------------------
const M = rep("metrics.json");
const DRIFT = rep("drift_ml.json");
const KB = rep("indic_asr_n8/report.json");
const LLM = rep("llm_extraction.json");
const MZ = rep("martts_zeroshot.json"), MF = rep("martts_final.json");
const SA = rep("seaalert.json");
const DIAG = rep("severity_diagnostics.json");
const ASR_B = rep("asr_small_indic_base.json"), ASR_F = rep("asr_small_indic_ft.json");
const r2 = (x) => Math.round(x * 100) / 100;
const pct = (x) => `${Math.round(x * 100)}%`;

// ---- theme ------------------------------------------------------------------
const THEME = {
  name: "Maritime Ivory",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1E2A36", lt1: "FAF7F0", dk2: "1B3A5C", lt2: "EFE7D6",
    accent1: "2E6F95", accent2: "3D8B74", accent3: "D9B45A", accent4: "A8C5B5",
    accent5: "8FB1CC", accent6: "C9B48A", hlink: "2E6F95", folHlink: "3D8B74",
  },
};
const HEX = { navy: "1B3A5C", ocean: "2E6F95", green: "3D8B74", gold: "D9B45A", sage: "A8C5B5", sky: "8FB1CC",
  sand: "EFE7D6", ivory: "FAF7F0", ink: "1E2A36", muted: "6B7785", grid: "E3DCCB", white: "FFFFFF" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "Maritime Distress Intelligence";
pres.author = "Data Science course project";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

// ---- layouts ----------------------------------------------------------------
pres.defineSlideMaster({
  title: "DARK",
  background: { color: HEX.navy },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.8, y: 2.35, w: 7.6, h: 1.6, fontFace: THEME.headFontFace,
      fontSize: 44, bold: false, color: C.background1, valign: "bottom", align: "left", margin: 0 }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.8, y: 4.15, w: 7.2, h: 1.2, fontSize: 18,
      color: C.accent4, valign: "top", margin: 0 }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: HEX.ivory },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: 0.7, y: 0.42, w: 9, h: 0.32, fontSize: 12, bold: true,
      color: C.accent2, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: 0.7, y: 0.72, w: 11.9, h: 0.8, fontFace: THEME.headFontFace,
      fontSize: 34, color: C.text2, valign: "top", align: "left", margin: 0 }, text: "" } },
    { text: { text: "Maritime Distress Intelligence", options: { x: 0.7, y: 7.02, w: 6, h: 0.3, fontSize: 10, color: C.accent1, margin: 0 } } },
  ],
  slideNumber: { x: 12.2, y: 7.02, w: 0.5, h: 0.3, fontSize: 10, color: HEX.muted, align: "right" },
});

// ---- helpers ----------------------------------------------------------------
const content = (section, kicker, title) => {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
};
const card = (s, name, x, y, w, h, fill = C.background2) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: fill }, line: { color: fill }, objectName: name });
const text = (s, t, o) => s.addText(t, { isTextBox: true, margin: 0, ...o });
const stat = (s, name, x, y, w, value, label, color = C.text2) => {
  text(s, value, { x, y, w, h: 0.9, fontFace: THEME.headFontFace, fontSize: 44, color, objectName: name + " value" });
  text(s, label, { x, y: y + 0.92, w, h: 0.6, fontSize: 14, color: C.text1, objectName: name + " label" });
};
const rings = (s, cx, cy, color, n = 4) => {
  for (let i = n; i >= 1; i--) {
    const r = i * 0.62;
    s.addShape(pres.shapes.OVAL, { x: cx - r, y: cy - r, w: 2 * r, h: 2 * r, fill: { color, transparency: 100 },
      line: { color, width: 1, transparency: 40 + i * 10 }, objectName: `radar ring ${i}` });
  }
  s.addShape(pres.shapes.OVAL, { x: cx - 0.12, y: cy - 0.12, w: 0.24, h: 0.24, fill: { color: HEX.gold }, line: { color: HEX.gold }, objectName: "radar blip" });
};
const CHART = {
  catAxisLabelColor: HEX.muted, valAxisLabelColor: HEX.muted, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  catAxisLabelFontSize: 12, valAxisLabelFontSize: 11, dataLabelFontFace: "+mn-lt", dataLabelFontSize: 11, dataLabelColor: HEX.ink,
  legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.ink, titleFontFace: "+mn-lt", titleColor: HEX.ink, titleFontSize: 13,
  valGridLine: { color: HEX.grid, size: 0.75 }, catGridLine: { style: "none" }, catAxisLineShow: false, valAxisLineShow: false,
};
const chartOpts = (o) => ({ ...CHART, valGridLine: { ...CHART.valGridLine }, catGridLine: { ...CHART.catGridLine }, ...o });

// ============================================================================
// 1. Title
pres.addSection({ title: "Introduction" });
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  rings(s, 10.6, 3.75, HEX.sage, 5);
  s.addText("Maritime Distress Intelligence", { placeholder: "title" });
  s.addText("From a noisy multilingual VHF call to a verified incident and a search plan", { placeholder: "body" });
  text(s, "Data Science course project  ·  ML-driven decision support for coast guard operators", {
    x: 0.8, y: 6.45, w: 8, h: 0.4, fontSize: 13, color: C.accent3, objectName: "course line" });
  s.addNotes("Open with the one-line promise: a distress call comes in on channel 16, possibly in Hindi, Tamil or Malayalam, over a noisy radio. The system turns it into a structured, explained incident and a search plan in seconds, and the operator verifies everything.");
}

// 2. Problem
{
  const s = content("Introduction", "The problem", "Distress calls are noisy, multilingual and urgent");
  const items = [
    ["4", "languages on the radio along the Indian coast: English, Hindi, Tamil, Malayalam", C.accent1],
    ["0", "public labelled VHF distress-call datasets for Indian waters", C.accent3],
    ["24 h", "a drifting casualty can move 10+ nm; every hour widens the search", C.accent2],
  ];
  items.forEach(([v, l, col], i) => {
    const x = 0.7 + i * 4.1;
    card(s, `problem card ${i + 1}`, x, 2.1, 3.75, 3.1, C.background2);
    text(s, v, { x: x + 0.4, y: 2.5, w: 3, h: 1.2, fontFace: THEME.headFontFace, fontSize: 60, color: col, objectName: `problem stat ${i + 1}` });
    text(s, l, { x: x + 0.4, y: 3.9, w: 3, h: 1.4, fontSize: 16, color: C.text1, valign: "top", objectName: `problem label ${i + 1}` });
  });
  text(s, "Operators must extract who, where, what and how many, under stress, from a few seconds of radio.", {
    x: 0.7, y: 5.75, w: 11.9, h: 0.5, fontSize: 16, italic: true, color: C.accent1, objectName: "problem takeaway" });
  s.addNotes("There is no public labelled corpus for this, which is why the project generates its own labelled data and validates on external datasets.");
}

// 3. Product
pres.addSection({ title: "Product" });
{
  const s = content("Product", "The product", "An operator console that listens, extracts and plans");
  s.addImage({ path: path.join(__dirname, "assets/console_ops.png"), x: 0.7, y: 1.75, w: 8.4, h: 4.725, objectName: "console screenshot" });
  const pts = [
    ["Listens", "Whisper ASR with fine-tuned Indic checkpoints"],
    ["Extracts", "Vessel, MMSI, position, persons on board, incident type, with evidence"],
    ["Judges", "Severity with calibrated probabilities and a 90% set"],
    ["Plans", "Drift forecast, search area and fastest assets"],
  ];
  pts.forEach(([h, b], i) => {
    const y = 1.8 + i * 1.18;
    s.addShape(pres.shapes.OVAL, { x: 9.5, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: `step dot ${i + 1}` });
    text(s, String(i + 1), { x: 9.5, y: y + 0.05, w: 0.42, h: 0.42, fontSize: 13, bold: true, color: C.background1, align: "center", valign: "middle", objectName: `step num ${i + 1}` });
    text(s, h, { x: 10.1, y, w: 2.6, h: 0.4, fontSize: 17, bold: true, color: C.text2, objectName: `step head ${i + 1}` });
    text(s, b, { x: 10.1, y: y + 0.4, w: 2.6, h: 0.7, fontSize: 12.5, color: C.text1, valign: "top", objectName: `step body ${i + 1}` });
  });
  text(s, "The operator verifies every field. The system never dispatches on its own.", {
    x: 0.7, y: 6.6, w: 8.4, h: 0.3, fontSize: 12, italic: true, color: HEX.muted, objectName: "human in loop note" });
}

// 4. Pipeline
{
  const s = content("Product", "How it works", "Seven stages, each one a model you can measure");
  const stages = [
    ["Audio", "VHF channel\nsimulation"], ["Speech", "Whisper +\nIndic fine-tunes"], ["Entities", "CRF sequence\nlabelling"],
    ["Incident", "TF-IDF +\nlogistic regression"], ["Severity", "Calibrated,\nconformal sets"], ["Drift", "Physics +\nLightGBM"], ["Assets", "Hungarian\nassignment"],
  ];
  const w = 1.55, gap = 0.18, x0 = 0.7, y = 2.55;
  stages.forEach(([h, b], i) => {
    const x = x0 + i * (w + gap);
    const fill = i % 2 ? C.accent1 : C.text2;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 1.25, rectRadius: 0.1, fill: { color: fill }, line: { color: fill }, objectName: `stage ${h}` });
    text(s, h, { x, y: y + 0.38, w, h: 0.5, fontSize: 17, bold: true, color: C.background1, align: "center", objectName: `stage label ${h}` });
    text(s, b, { x, y: y + 1.45, w, h: 0.75, fontSize: 12.5, color: C.text1, align: "center", valign: "top", objectName: `stage method ${h}` });
    if (i < stages.length - 1)
      s.addShape(pres.shapes.CHEVRON, { x: x + w + 0.03, y: y + 0.52, w: 0.12, h: 0.22, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: `arrow ${i + 1}` });
  });
  card(s, "llm band", 0.7, 5.0, 11.95, 1.15, C.background2);
  text(s, [
    { text: "Optional LLM layer  ", options: { bold: true, color: C.accent2 } },
    { text: "Qwen2.5-1.5B fills fields the CRF misses. Every LLM answer must quote the transcript, and our own parsers convert numbers, never the LLM." },
  ], { x: 1.0, y: 5.12, w: 11.4, h: 0.9, fontSize: 15, color: C.text1, valign: "middle", objectName: "llm band text" });
}

// 5. Data science depth (the emphasis slide)
pres.addSection({ title: "Data science" });
const METHODS = {
  "Learning models": ["Linear-chain CRF (NER)", "TF-IDF + logistic regression", "Ordinal vs multinomial regression", "LightGBM residual model", "Whisper transfer learning", "Grammar-constrained LLM"],
  "Uncertainty": ["Temperature scaling + ECE", "Split conformal prediction", "Conformal search radius", "Bootstrap confidence intervals", "Exact linear SHAP"],
  "Simulation": ["Synthetic data generator", "VHF signal processing", "Monte Carlo drift", "Hungarian assignment"],
  "Evaluation design": ["Leakage-controlled splits", "Buoy-grouped cross-validation", "Dev / test protocol", "Robustness vs ASR noise", "Learned CRF/LLM routing"],
};
const N_METHODS = Object.values(METHODS).reduce((a, b) => a + b.length, 0);
{
  const s = content("Data science", "Data science depth", `${N_METHODS} data science and ML techniques in one product`);
  const cols = Object.entries(METHODS);
  const fills = [C.text2, C.accent1, C.accent2, "7A6230"];
  cols.forEach(([head, items], i) => {
    const x = 0.7 + i * 3.05;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.85, w: 2.8, h: 0.62, rectRadius: 0.1, fill: { color: fills[i] }, line: { color: fills[i] }, objectName: `method head ${head}` });
    text(s, `${head}  ·  ${items.length}`, { x: x + 0.2, y: 1.85, w: 2.4, h: 0.62, fontSize: 14, bold: true, color: C.background1, valign: "middle", objectName: `method head text ${head}` });
    items.forEach((m, j) => {
      const y = 2.65 + j * 0.62;
      card(s, `method ${m}`, x, y, 2.8, 0.5, C.background2);
      text(s, m, { x: x + 0.2, y, w: 2.5, h: 0.5, fontSize: 13, color: C.text1, valign: "middle", objectName: `method text ${m}` });
    });
  });
  text(s, "Every technique is evaluated against a baseline on held-out or external data.", {
    x: 0.7, y: 6.55, w: 11.9, h: 0.35, fontSize: 14, italic: true, color: C.accent1, objectName: "depth takeaway" });
  s.addNotes(`${N_METHODS} distinct methods. Stress that each has a baseline and a held-out evaluation, not just an implementation.`);
}

// 6. Data
{
  const s = content("Data science", "Data", "Labels we generate, realism we borrow");
  card(s, "synthetic panel", 0.7, 1.85, 4.6, 4.6, C.text2);
  text(s, "20,000", { x: 1.05, y: 2.1, w: 4, h: 1.0, fontFace: THEME.headFontFace, fontSize: 54, color: C.background1, objectName: "synthetic count" });
  text(s, "synthetic distress dialogues", { x: 1.05, y: 3.1, w: 4, h: 0.4, fontSize: 16, color: C.accent4, objectName: "synthetic label" });
  text(s, [
    { text: "Labels sampled before text, so spans are exact", options: { bullet: true, breakLine: true } },
    { text: "4 languages × 3 caller styles", options: { bullet: true, breakLine: true } },
    { text: "Held-out vessels, landmarks and phrasings", options: { bullet: true } },
  ], { x: 1.05, y: 3.75, w: 4, h: 2.4, fontSize: 14, color: C.background1, paraSpaceAfter: 8, valign: "top", objectName: "synthetic bullets" });
  const ext = [
    ["MARTTS", "240 real radio dialogues", "test"], ["SeaAlert", "1,872 VHF messages + ASR", "test"],
    ["Kathbath", "real Indian speech", "test"], ["MUSAN", "real background noise", "train"],
    ["NOAA drifters", "real buoy tracks", "train/test"], ["ERA5 + Copernicus", "wind and currents", "input"],
    ["MarineCadastre AIS", "real vessel speeds", "input"], ["GEBCO", "bathymetry / land mask", "input"],
  ];
  ext.forEach(([n, d, use], i) => {
    const col = i % 2, row = Math.floor(i / 2);
    const x = 5.65 + col * 3.5, y = 1.85 + row * 1.17;
    card(s, `dataset ${n}`, x, y, 3.3, 1.0, C.background2);
    text(s, n, { x: x + 0.2, y: y + 0.12, w: 2.2, h: 0.4, fontSize: 15, bold: true, color: C.text2, objectName: `dataset name ${n}` });
    text(s, d, { x: x + 0.2, y: y + 0.52, w: 2.9, h: 0.36, fontSize: 12.5, color: C.text1, objectName: `dataset desc ${n}` });
    text(s, use, { x: x + 2.15, y: y + 0.14, w: 1.0, h: 0.3, fontSize: 10.5, bold: true, color: C.accent2, align: "right", objectName: `dataset use ${n}` });
  });
}

// 7. Language understanding
pres.addSection({ title: "Results" });
if (M) {
  const s = content("Results", "Language understanding", "Entities and incidents, on phrasings never seen in training");
  const labels = Object.keys(M.ner.per_label_f1);
  s.addChart(pres.charts.BAR, [{ name: "F1", labels: labels.map((l) => l.replace("_", " ")), values: labels.map((l) => r2(M.ner.per_label_f1[l])) }], chartOpts({
    x: 0.7, y: 1.8, w: 7.4, h: 4.7, barDir: "bar", chartColors: [HEX.ocean], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisMinVal: 0, valAxisMaxVal: 1.12, valAxisHidden: true, showTitle: true, title: "CRF entity F1 by type",
    showLegend: false, barGapWidthPct: 60, objectName: "ner f1 chart",
  }));
  stat(s, "ner", 8.7, 2.0, 4, M.ner.micro_f1.toFixed(2), "entity micro-F1 (95% CI " + M.ner.micro_f1_ci95.map((x) => x.toFixed(3)).join("–") + ")", C.accent1);
  stat(s, "incident", 8.7, 3.75, 4, M.incident.tfidf_lr_macro_f1.toFixed(2), `incident macro-F1, vs ${M.incident.keyword_baseline_macro_f1.toFixed(2)} for keyword rules`, C.accent2);
  text(s, "Unseen vessel names: a shape feature stops the CRF memorising names (66% → 99.6% exact).", {
    x: 8.7, y: 5.55, w: 4, h: 0.9, fontSize: 13, italic: true, color: HEX.muted, objectName: "cap token note" });
}

// 8. Severity & uncertainty
if (M) {
  const sev = M.severity, ts = sev.multinomial_temperature_scaled, un = sev.multinomial_uncalibrated;
  const s = content("Results", "Severity and trust", "Probabilities you can trust, sets that admit doubt");
  stat(s, "accuracy", 0.7, 2.0, 3.6, pct(ts.accuracy), `severity accuracy vs ${pct(sev.majority_baseline_accuracy)} majority baseline`, C.text2);
  stat(s, "coverage", 0.7, 3.85, 3.6, pct(sev.conformal_90.empirical_coverage), `conformal 90% sets hit the true severity (mean size ${sev.conformal_90.mean_set_size.toFixed(2)})`, C.accent2);
  s.addChart(pres.charts.BAR, [{ name: "ECE", labels: ["Uncalibrated", "Temperature-scaled"], values: [un.ece, ts.ece].map((x) => Math.round(x * 1000) / 1000) }], chartOpts({
    x: 4.8, y: 1.8, w: 4.0, h: 4.5, barDir: "col", chartColors: [HEX.sky, HEX.ocean], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.000", valAxisHidden: true, valAxisMinVal: 0, showTitle: true, title: "Calibration error (lower is better)",
    showLegend: false, barGapWidthPct: 70, objectName: "ece chart",
  }));
  card(s, "shap card", 9.3, 1.85, 3.35, 4.45, C.background2);
  text(s, "Explained", { x: 9.6, y: 2.1, w: 2.9, h: 0.45, fontSize: 18, bold: true, color: C.text2, objectName: "shap head" });
  const top = sev.top_features_high_or_worse.slice(0, 5);
  const mx = Math.max(...top.map((t) => Math.abs(t[1])));
  top.forEach(([f, v], i) => {
    const y = 2.75 + i * 0.62;
    text(s, f.replace(/^p_/, "").replaceAll("_", " "), { x: 9.6, y, w: 2.9, h: 0.28, fontSize: 12, color: C.text1, objectName: `shap label ${i}` });
    s.addShape(pres.shapes.RECTANGLE, { x: 9.6, y: y + 0.3, w: 2.7 * Math.abs(v) / mx, h: 0.16, fill: { color: v > 0 ? HEX.gold : HEX.sky }, line: { color: v > 0 ? HEX.gold : HEX.sky }, objectName: `shap bar ${i}` });
  });
  text(s, "Linear SHAP: weights that raise HIGH/CRITICAL", { x: 9.6, y: 5.85, w: 2.9, h: 0.3, fontSize: 10.5, color: HEX.muted, objectName: "shap note" });
}

// 8b. Severity diagnostics: confusion matrix + reliability diagram
if (DIAG) {
  const s = content("Results", "Severity diagnostics", "Where the severity model goes wrong");
  const L = DIAG.labels, cm = DIAG.severity_confusion, rn = DIAG.severity_confusion_row_normalised;
  const x0 = 1.75, y0 = 2.25, cw = 1.05, ch = 0.82;
  text(s, "Predicted", { x: x0, y: 1.72, w: cw * 4, h: 0.3, fontSize: 12, bold: true, color: HEX.muted, align: "center", objectName: "cm predicted axis" });
  text(s, "True", { x: 0.55, y: y0 + ch * 2 - 0.15, w: 0.6, h: 0.3, fontSize: 12, bold: true, color: HEX.muted, objectName: "cm true axis" });
  L.forEach((lab, j) => text(s, lab, { x: x0 + j * cw, y: 1.98, w: cw, h: 0.25, fontSize: 10.5, color: C.text1, align: "center", objectName: `cm col ${lab}` }));
  L.forEach((lab, i) => {
    text(s, lab, { x: 0.7, y: y0 + i * ch, w: 1.0, h: ch, fontSize: 10.5, color: C.text1, align: "right", valign: "middle", objectName: `cm row ${lab}` });
    L.forEach((_, j) => {
      const v = rn[i][j], diag = i === j;
      const fill = diag ? HEX.green : HEX.navy;
      s.addShape(pres.shapes.RECTANGLE, { x: x0 + j * cw + 0.03, y: y0 + i * ch + 0.03, w: cw - 0.06, h: ch - 0.06,
        fill: { color: fill, transparency: Math.round(100 - Math.max(6, v * 100)) }, line: { color: HEX.grid, width: 0.5 }, objectName: `cm cell ${i}${j}` });
      text(s, [{ text: String(cm[i][j]), options: { bold: true, breakLine: true } }, { text: `${Math.round(v * 100)}%`, options: { fontSize: 10 } }], {
        x: x0 + j * cw, y: y0 + i * ch, w: cw, h: ch, fontSize: 13, align: "center", valign: "middle",
        color: v > 0.45 ? C.background1 : C.text1, objectName: `cm text ${i}${j}` });
    });
  });
  const crit = rn[3][2];
  text(s, `${Math.round(crit * 100)}% of CRITICAL calls are rated HIGH: the costly error`, {
    x: 0.7, y: 5.65, w: 5.4, h: 0.6, fontSize: 14, italic: true, color: "7A6230", objectName: "cm takeaway" });
  if (DIAG.reliability_uncalibrated) {
    const bins = DIAG.reliability_calibrated.map((b) => b.bin.join("–"));
    const ru = DIAG.reliability_uncalibrated.map((b) => r2(b.accuracy)), rc = DIAG.reliability_calibrated.map((b) => r2(b.accuracy));
    const perfect = DIAG.reliability_calibrated.map((b) => r2((b.bin[0] + b.bin[1]) / 2));
    s.addChart(pres.charts.LINE, [
      { name: "Perfect calibration", labels: bins, values: perfect },
      { name: "Uncalibrated", labels: bins, values: ru },
      { name: "Temperature-scaled", labels: bins, values: rc },
    ], chartOpts({
      x: 6.5, y: 1.75, w: 6.2, h: 4.1, chartColors: ["C9B48A", HEX.sky, HEX.ocean], lineSize: 2.5, lineDataSymbol: "circle", lineDataSymbolSize: 7,
      valAxisMinVal: 0.3, valAxisMaxVal: 1.0, valAxisLabelFormatCode: "0.0", showLegend: true, legendPos: "b",
      showTitle: true, title: "Reliability diagram: accuracy per confidence bin", objectName: "reliability chart",
    }));
  }
  const cc = DIAG.conformal_coverage_by_true_class;
  text(s, `Conformal sets cover ${Math.round(cc.HIGH * 100)}% of HIGH but only ${Math.round(cc.CRITICAL * 100)}% of CRITICAL calls. Next: class-conditional conformal.`, {
    x: 6.5, y: 5.95, w: 6.2, h: 0.6, fontSize: 12.5, italic: true, color: HEX.muted, objectName: "coverage note" });
}

// 9. Speech
if (KB) {
  const s = content("Results", "Speech recognition", "Fine-tuned Whisper cuts noisy-speech errors in half");
  const langs = [["hi", "Hindi"], ["ta", "Tamil"], ["ml", "Malayalam"]];
  const get = (l, base) => KB.results.find((r) => r.language === l && r.condition === "vhf" && r.model && (base ? r.model.startsWith("faster") : !r.model.startsWith("faster")));
  s.addChart(pres.charts.BAR, [
    { name: "Whisper-small", labels: langs.map((l) => l[1]), values: langs.map(([l]) => r2(get(l, true)?.wer ?? 0)) },
    { name: "Fine-tuned Indic Whisper", labels: langs.map((l) => l[1]), values: langs.map(([l]) => r2(get(l, false)?.wer ?? 0)) },
  ], chartOpts({
    x: 0.7, y: 1.8, w: 7.6, h: 4.7, barDir: "col", chartColors: [HEX.sky, HEX.green], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.1, showLegend: true, legendPos: "b",
    showTitle: true, title: "Word error rate, real speech through a simulated VHF channel", barGapWidthPct: 60, objectName: "kathbath chart",
  }));
  const hi = get("hi", true), hf = get("hi", false);
  stat(s, "hindi drop", 8.9, 2.0, 3.8, `${Math.round(hi.wer * 100)}% → ${Math.round(hf.wer * 100)}%`, "Hindi word errors over radio noise", C.accent2);
  if (ASR_B && ASR_F) stat(s, "synthetic", 8.9, 3.75, 3.8, `${Math.round(ASR_B.caller_wer * 100)}% → ${Math.round(ASR_F.caller_wer * 100)}%`, `caller WER on our 24 synthetic TTS calls; CER ${Math.round(ASR_B.caller_cer * 100)}% → ${Math.round(ASR_F.caller_cer * 100)}%, so synthetic audio is the bottleneck`, C.accent1);
  text(s, `AI4Bharat Kathbath, ${get("hi", true).n} speakers per language (Malayalam ${get("ml", true).n}). No training run: published checkpoints, measured on our channel.`, {
    x: 8.9, y: 5.5, w: 3.8, h: 1.0, fontSize: 12, italic: true, color: HEX.muted, objectName: "kathbath note" });
}

// 10. LLM
if (LLM) {
  const best = LLM.routed ? "routed" : "hybrid";
  const s = content("Results", "LLM extraction", "A small open LLM, kept honest by a grounding guard");
  const f = [["vessel", "Vessel"], ["position", "Position"], ["pob", "Persons on board"], ["mmsi", "MMSI"]];
  const v = (sys, k) => r2(LLM[sys]?.seaalert?.asr_high?.[k]?.value ?? 0);
  s.addChart(pres.charts.BAR, [
    { name: "CRF only", labels: f.map((x) => x[1]), values: f.map(([k]) => v("crf_pipeline", k)) },
    { name: best === "routed" ? "CRF + LLM, routing learned on dev" : "CRF + LLM hybrid", labels: f.map((x) => x[1]), values: f.map(([k]) => v(best, k)) },
  ], chartOpts({
    x: 0.7, y: 1.8, w: 7.6, h: 4.7, barDir: "col", chartColors: [HEX.sky, HEX.ocean], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1, showLegend: true, legendPos: "b",
    showTitle: true, title: "Field accuracy on SeaAlert, noisiest ASR transcripts", barGapWidthPct: 60, objectName: "llm chart",
  }));
  const rej = Object.values(LLM[best]?.grounding_rejections || {}).reduce((a, b) => a + b, 0);
  stat(s, "rejections", 8.9, 2.0, 3.8, String(rej), "LLM answers blocked: quote not in the transcript or not a valid value", "7A6230");
  stat(s, "model", 8.9, 3.75, 3.8, "1.5B", "parameters, 4-bit, laptop CPU, output constrained to a JSON schema", C.accent1);
  if (LLM.routing) {
    const pol = Object.entries(LLM.routing.policy).map(([k, r]) => `${k.replace("_name", "").replace("pob", "POB")}: ${r === "crf_first" ? "CRF" : "LLM"}`).join(" · ");
    text(s, `Learned on dev data: ${pol}`, { x: 8.9, y: 5.5, w: 3.8, h: 1.0, fontSize: 12, italic: true, color: HEX.muted, objectName: "routing note" });
  }
}

// 11. Drift
if (DRIFT) {
  const s = content("Results", "Drift forecasting", "Physics plus machine learning, tested on real ocean buoys");
  const H = ["6", "12", "24"];
  const e = DRIFT.mean_error_nm;
  s.addChart(pres.charts.BAR, [
    { name: "Object stays put", labels: H.map((h) => `${h} h`), values: H.map((h) => r2(e.persistence[h])) },
    { name: "Physics", labels: H.map((h) => `${h} h`), values: H.map((h) => r2(e.physics[h])) },
    { name: "Physics + LightGBM", labels: H.map((h) => `${h} h`), values: H.map((h) => r2(e["physics+lgbm"][h])) },
  ], chartOpts({
    x: 0.7, y: 1.8, w: 7.6, h: 4.7, barDir: "col", chartColors: ["C9B48A", HEX.sky, HEX.green], showValue: true,
    dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", valAxisHidden: true, valAxisMinVal: 0, showLegend: true, legendPos: "b",
    showTitle: true, title: "Mean position error after drifting (nautical miles)", barGapWidthPct: 50, objectName: "drift chart",
  }));
  const g = DRIFT.lgbm_gain_24h;
  stat(s, "skill", 8.9, 2.0, 3.8, pct(DRIFT.skill_vs_persistence["physics+lgbm"]["24"]), "better than 'object stays put' at 24 h", C.accent2);
  stat(s, "gain", 8.9, 3.75, 3.8, `+${(g.mean * 100).toFixed(1)}%`, `LightGBM over physics at 24 h
95% CI ${(g.ci95[0] * 100).toFixed(1)}% to ${(g.ci95[1] * 100).toFixed(1)}%`, C.accent1);
  text(s, `${DRIFT.n_start_points} start points from ${DRIFT.n_buoys} NOAA buoys · cross-validated by buoy · conformal 90% search radius`, {
    x: 8.9, y: 5.5, w: 3.8, h: 1.0, fontSize: 12, italic: true, color: HEX.muted, objectName: "drift note" });
}

// 12. External validation
if (MZ && MF) {
  const s = content("Results", "External validation", "Trained on synthetic data, tested on real radio dialogues");
  const k = [["vessel_exact", "Vessel name"], ["mmsi_exact", "MMSI"], ["position_within_1nm", "Position ≤ 1 nm"]];
  s.addChart(pres.charts.BAR, [
    { name: "Zero-shot", labels: k.map((x) => x[1]), values: k.map(([x]) => r2(MZ.test_text[x].rate)) },
    { name: "After fixes found on the dev split", labels: k.map((x) => x[1]), values: k.map(([x]) => r2(MF.test_text[x].rate)) },
  ], chartOpts({
    x: 0.7, y: 1.8, w: 7.6, h: 4.7, barDir: "col", chartColors: [HEX.sky, HEX.navy], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFormatCode: "0.00", valAxisHidden: true, valAxisMinVal: 0, valAxisMaxVal: 1.1, showLegend: true, legendPos: "b",
    showTitle: true, title: "MARTTS held-out test dialogues (DLR, n = 177)", barGapWidthPct: 60, objectName: "martts chart",
  }));
  stat(s, "routine", 8.9, 2.0, 3.8, `${pct(MZ.test_text.routine_flagged_high_or_critical.rate)} → ${pct(MF.test_text.routine_flagged_high_or_critical.rate)}`, "routine calls wrongly rated HIGH or CRITICAL", C.accent2);
  if (SA) {
    const a = SA.A_message_category.ours.tfidf_lr_clean_training;
    stat(s, "seaalert", 8.9, 3.75, 3.8, a.asr_high.macro_f1.toFixed(2), "SeaAlert category F1 at heavy ASR noise, vs 0.42 published logistic regression", C.accent1);
  }
  text(s, "Fixes were made on a dev split only; test dialogues were scored, never inspected.", {
    x: 8.9, y: 5.5, w: 3.8, h: 1.0, fontSize: 12, italic: true, color: HEX.muted, objectName: "protocol note" });
}

// 13. Limits & next steps
pres.addSection({ title: "Close" });
{
  const s = content("Close", "Honest limits and next steps", "What a real deployment would still need");
  const left = ["Labels are synthetic; Indic phrasing needs native-speaker review", "Only 11 buoys with current data limit the drift model",
    "Small external samples (24–177 items) give wide intervals", "Kathbath may overlap the checkpoints' training data"];
  const right = ["Record real calls with consenting operators", "LoRA fine-tune Whisper on radio-channel audio",
    "LoRA fine-tune the LLM on our labelled dialogues", "Stream live calls through Kafka; real AIS feed"];
  [[left, "Limits", C.accent3, "7A6230"], [right, "Next steps", C.accent2, HEX.green]].forEach(([items, head, fill, ink], i) => {
    const x = 0.7 + i * 6.1;
    card(s, `${head} card`, x, 1.85, 5.85, 4.65, C.background2);
    text(s, head, { x: x + 0.35, y: 2.05, w: 5, h: 0.5, fontSize: 20, bold: true, color: ink, objectName: `${head} head` });
    items.forEach((t, j) => {
      const y = 2.75 + j * 0.9;
      s.addShape(pres.shapes.OVAL, { x: x + 0.35, y: y + 0.1, w: 0.2, h: 0.2, fill: { color: fill }, line: { color: fill }, objectName: `${head} dot ${j}` });
      text(s, t, { x: x + 0.75, y, w: 4.8, h: 0.75, fontSize: 15, color: C.text1, valign: "top", objectName: `${head} item ${j}` });
    });
  });
}

// 14. Closing
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Close" });
  rings(s, 10.6, 3.75, HEX.sage, 5);
  s.addText("Live demo", { placeholder: "title" });
  s.addText("One distress call, end to end: audio, transcript, extraction, severity, drift and dispatch", { placeholder: "body" });
  text(s, "Thank you  ·  questions welcome", { x: 0.8, y: 6.45, w: 8, h: 0.4, fontSize: 14, color: C.accent3, objectName: "thanks" });
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  if (SKILL_THEME) {
    const { applyTheme } = require(SKILL_THEME);
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT, `(${N_METHODS} methods)`);
})();
