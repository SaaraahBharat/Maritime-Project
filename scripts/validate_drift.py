"""Validate the drift model against REAL drifting buoys (NOAA Global Drifter Program).

For sampled start points on real 2021-22 drifter tracks in the Arabian Sea /
Bay of Bengal, predict the position 6/12/24 h later and compare with where the
buoy actually went:

  persistence   : object stays where it was reported (baseline)
  current_only  : advected by Copernicus surface current (via Open-Meteo)
  current+wind  : current + ERA5 wind leeway (our model's physics)
  monte_carlo   : our engine's 500-particle ensemble -> is the true position
                  inside the predicted 90% search radius? (uncertainty calibration)

Drogued drifters (sea anchor attached) follow the current; undrogued ones are
blown more by wind, so results are reported for both. GEBCO is used to drop
start points that are not in open water.

Usage:  python scripts/validate_drift.py --n 120
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mdi.data.spoken import destination, haversine_nm  # noqa: E402
from mdi.response import engine, geo  # noqa: E402

HORIZONS = (6, 12, 24)
LEEWAY = {"drogued": 0.0, "undrogued": 0.01}  # fraction of wind speed (undrogued SVP drifter ~1%)


def load_drifters() -> pd.DataFrame:
    frames = [pd.read_csv(p, skiprows=[1]) for p in sorted(Path("data/external/drifters").glob("drifters_*.csv"))]
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["drogue_lost_date"] = pd.to_datetime(df["drogue_lost_date"], utc=True, errors="coerce")
    df["drogued"] = df["drogue_lost_date"].isna() | (df["time"] < df["drogue_lost_date"])
    return df.sort_values(["ID", "time"]).reset_index(drop=True)


def sample_starts(df: pd.DataFrame, n: int, seed: int, year: int = 2022, stride: int = 24) -> list[dict]:
    """Start points with a continuous 24 h track ahead, in open water, spread over buoys.

    Only `year` is used: Open-Meteo's Copernicus current history starts in 2022.
    """
    rng = np.random.default_rng(seed)
    starts = []
    df = df[df["time"].dt.year == year]
    for _, g in df.groupby("ID"):
        g = g.set_index("time")
        times = g.index[:: stride]  # candidate every `stride` hours per buoy
        for t in times:
            ahead = [t + pd.Timedelta(hours=h) for h in HORIZONS]
            if all(a in g.index for a in ahead):
                row = g.loc[t]
                starts.append({"id": int(row["ID"]), "t0": t, "lat": float(row["latitude"]), "lon": float(row["longitude"]),
                               "drogued": bool(row["drogued"]),
                               "truth": {h: (float(g.loc[a, "latitude"]), float(g.loc[a, "longitude"])) for h, a in zip(HORIZONS, ahead)}})
    rng.shuffle(starts)
    if geo.gebco_available():
        starts = [s for s in starts if geo.elevation(s["lat"], s["lon"])[0] < -20]  # open water only
    return starts[:n]


def integrate(lat, lon, forcing, i0: int, hours: int, leeway: float, use_current=True,
              current_scale: float = 1.0) -> tuple[float, float]:
    """Hour-by-hour advection with time-varying forcing."""
    for k in range(hours):
        i = min(i0 + k, len(forcing["time"]) - 1)
        vx = vy = 0.0
        cur, cdir = forcing["current_kn"][i], forcing["current_to_deg"][i]
        if use_current and not math.isnan(cur):
            vx += current_scale * cur * math.sin(math.radians(cdir))
            vy += current_scale * cur * math.cos(math.radians(cdir))
        wind, wfrom = forcing["wind_kn"][i], forcing["wind_from_deg"][i]
        if leeway and not math.isnan(wind):
            wto = math.radians(wfrom + 180)
            vx += leeway * wind * math.sin(wto)
            vy += leeway * wind * math.cos(wto)
        speed = math.hypot(vx, vy)
        if speed > 0:
            lat, lon = destination(lat, lon, math.degrees(math.atan2(vx, vy)) % 360, speed)
    return lat, lon


def run(starts: list[dict], current_scale: float = 1.0, monte_carlo: bool = True) -> list[dict]:
    """Predict every start point at every horizon; one row per (start, horizon)."""
    rows = []
    for k, s in enumerate(starts):
        day0 = s["t0"].strftime("%Y-%m-%d")
        day1 = (s["t0"] + pd.Timedelta(hours=25)).strftime("%Y-%m-%d")
        try:
            f = geo.historical_forcing(s["lat"], s["lon"], day0, day1)
        except Exception as e:  # network hiccup: skip this start point
            print("  skip", e)
            continue
        i0 = f["time"].index(s["t0"].strftime("%Y-%m-%dT%H:00"))
        if math.isnan(f["current_kn"][i0]):
            continue
        group = "drogued" if s["drogued"] else "undrogued"
        for h in HORIZONS:
            tlat, tlon = s["truth"][h]
            preds = {
                "persistence": (s["lat"], s["lon"]),
                "current_only": integrate(s["lat"], s["lon"], f, i0, h, 0.0),
                "current+wind": integrate(s["lat"], s["lon"], f, i0, h, LEEWAY[group]),
                "calibrated": integrate(s["lat"], s["lon"], f, i0, h, LEEWAY[group], current_scale=current_scale),
            }
            cond = {"wind_kn": f["wind_kn"][i0], "wind_from_deg": f["wind_from_deg"][i0],
                    "current_kn": f["current_kn"][i0], "current_to_deg": f["current_to_deg"][i0]}
            inside = None
            if monte_carlo:
                area = engine.search_area(engine.drift_particles(s["lat"], s["lon"], h, cond, None, 0.5, n=300, seed=k))
                inside = haversine_nm(area["lat"], area["lon"], tlat, tlon) <= area["r90_nm"]
            rows.append({"id": s["id"], "group": group, "h": h,
                         "err": {m: haversine_nm(p[0], p[1], tlat, tlon) for m, p in preds.items()},
                         "mc_inside": inside})
    return rows


def summarise(rows: list[dict]) -> dict:
    out: dict = {"median_error_nm": {}, "skill_vs_persistence": {}, "uncalibrated_mc_90pct_coverage": {}}
    for g in ("all", "drogued", "undrogued"):
        sub = [r for r in rows if g == "all" or r["group"] == g]
        for h in HORIZONS:
            hs = [r for r in sub if r["h"] == h]
            if not hs:
                continue
            for m in hs[0]["err"]:
                out["median_error_nm"].setdefault(g, {}).setdefault(m, {})[str(h)] = float(np.median([r["err"][m] for r in hs]))
                if m != "persistence":
                    out["skill_vs_persistence"].setdefault(g, {}).setdefault(m, {})[str(h)] = float(
                        1 - np.mean([r["err"][m] for r in hs]) / np.mean([r["err"]["persistence"] for r in hs]))
            out["uncalibrated_mc_90pct_coverage"].setdefault(g, {})[str(h)] = float(np.mean([r["mc_inside"] for r in hs]))
        out.setdefault("n_rows", {})[g] = len(sub) // len(HORIZONS)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = load_drifters()
    starts = sample_starts(df, args.n, args.seed)
    ids = sorted({s["id"] for s in starts})
    np.random.default_rng(args.seed).shuffle(ids)
    cal_ids = set(ids[: len(ids) // 2])  # split by BUOY so calibration and test never share a track
    cal = [s for s in starts if s["id"] in cal_ids]
    test = [s for s in starts if s["id"] not in cal_ids]
    print(f"{df['ID'].nunique()} buoys total; 2022 start points: {len(starts)} "
          f"(calibration {len(cal)} from {len(cal_ids)} buoys, test {len(test)} from {len(ids) - len(cal_ids)} buoys)")

    # 1. Calibrate the current scale factor on the calibration buoys (grid search, 24 h error).
    best = (math.inf, 1.0)
    for a in np.arange(0.0, 1.55, 0.1):
        rows_a = run(cal, current_scale=float(a), monte_carlo=False)
        e = np.mean([r["err"]["calibrated"] for r in rows_a if r["h"] == 24])
        best = min(best, (e, float(a)))
    scale = best[1]
    # 2. Conformal search radius: 90th percentile of calibrated-model error per horizon.
    cal_rows = run(cal, current_scale=scale, monte_carlo=False)
    radius90 = {str(h): float(np.quantile([r["err"]["calibrated"] for r in cal_rows if r["h"] == h], 0.9)) for h in HORIZONS}

    # 3. Evaluate on held-out buoys.
    test_rows = run(test, current_scale=scale)
    report = {
        "n_buoys_total": int(df["ID"].nunique()),
        "calibration": {"buoys": len(cal_ids), "current_scale": scale, "radius90_nm": radius90},
        "test": summarise(test_rows),
        "test_conformal_90pct_coverage": {str(h): float(np.mean(
            [r["err"]["calibrated"] <= radius90[str(h)] for r in test_rows if r["h"] == h])) for h in HORIZONS},
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("reports/drift_calibration.json").write_text(json.dumps(report["calibration"], indent=2), encoding="utf-8")

    t = report["test"]
    print(f"calibrated current scale = {scale:.1f}; conformal 90% radius (nm) = {radius90}")
    print("TEST median error (nm):")
    for g, ms in t["median_error_nm"].items():
        for m, hs in ms.items():
            print(f"  {g:9s} {m:13s}", {h: round(x, 2) for h, x in hs.items()})
    print("TEST skill vs persistence:", json.dumps(t["skill_vs_persistence"]["all"], default=lambda x: round(x, 2)))
    print("TEST 90% coverage: uncalibrated Monte Carlo", t["uncalibrated_mc_90pct_coverage"]["all"],
          "-> conformal", report["test_conformal_90pct_coverage"])


if __name__ == "__main__":
    main()
