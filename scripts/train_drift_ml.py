"""Physics-informed ML drift correction, trained and tested on REAL drifting buoys.

The physics model (current x scale + wind leeway, see validate_drift.py) is a
strong prior but has systematic errors: Open-Meteo currents are a coarse
reanalysis, Stokes drift and inertial oscillations are missing, and leeway
varies. A gradient-boosted model (LightGBM) learns the RESIDUAL

    true displacement - physics displacement   (east, north, in nm)

from forcing features, so the final forecast is physics + learned correction
(hybrid / "grey-box" modelling: the ML part never has to learn the physics).

Evaluation is grouped K-fold BY BUOY: a buoy's track is never in both train
and test. Inside every fold:
  1. the physics current scale is re-fitted on the training buoys only,
  2. LightGBM is trained on 75 % of the training buoys,
  3. the remaining 25 % give a split-conformal 90 % search radius,
  4. the held-out buoys are scored.

Usage:  python scripts/train_drift_ml.py --n 400
Output: reports/drift_ml.json, models/drift_lgbm.joblib
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mdi.data.spoken import haversine_nm  # noqa: E402
from mdi.response import geo  # noqa: E402
from validate_drift import HORIZONS, LEEWAY, integrate, load_drifters, sample_starts  # noqa: E402

SCALES = np.arange(0.0, 1.55, 0.1)
LAMBDAS = np.round(np.arange(0.0, 1.01, 0.1), 1)
# No lat/lon/day-of-year: with ~11 buoys they act as buoy identifiers and the model memorises tracks.
FEATURES = ["h", "drogued",
            "cur_u0", "cur_v0", "cur_u_mean", "cur_v_mean", "wind_u_mean", "wind_v_mean",
            "wind_speed_mean", "cur_speed_std", "phys_dx", "phys_dy"]
LGB_PARAMS = dict(n_estimators=150, learning_rate=0.03, num_leaves=7, min_child_samples=40,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, verbose=-1)


def disp_nm(lat0, lon0, lat1, lon1) -> tuple[float, float]:
    """Local east/north displacement in nautical miles (flat-earth, fine for < 50 nm)."""
    return (lon1 - lon0) * 60 * math.cos(math.radians(lat0)), (lat1 - lat0) * 60


def uv(speed, to_deg):
    return speed * np.sin(np.radians(to_deg)), speed * np.cos(np.radians(to_deg))


def build_rows(starts: list[dict]) -> list[dict]:
    """One row per (start, horizon) with forcing features, truth, and physics runs for every scale."""
    rows = []
    for s in starts:
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
        doy = 2 * math.pi * s["t0"].dayofyear / 365.25
        cu, cv = uv(f["current_kn"], f["current_to_deg"])
        wu, wv = uv(f["wind_kn"], (f["wind_from_deg"] + 180) % 360)  # wind blowing TO
        for h in HORIZONS:
            w = slice(i0, i0 + h)
            tlat, tlon = s["truth"][h]
            phys = {round(a, 1): integrate(s["lat"], s["lon"], f, i0, h, LEEWAY[group], current_scale=float(a))
                    for a in SCALES}
            rows.append({
                "id": s["id"], "group": group, "h": h, "lat": s["lat"], "lon": s["lon"],
                "doy_sin": math.sin(doy), "doy_cos": math.cos(doy), "drogued": int(s["drogued"]),
                "cur_u0": cu[i0], "cur_v0": cv[i0],
                "cur_u_mean": np.nanmean(cu[w]), "cur_v_mean": np.nanmean(cv[w]),
                "wind_u_mean": np.nanmean(wu[w]), "wind_v_mean": np.nanmean(wv[w]),
                "wind_speed_mean": np.nanmean(f["wind_kn"][w]), "cur_speed_std": np.nanstd(f["current_kn"][w]),
                "true": disp_nm(s["lat"], s["lon"], tlat, tlon), "phys": phys,
            })
    return rows


def frame(rows: list[dict], scale: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature matrix and residual targets for a given physics current scale."""
    df = pd.DataFrame([{k: r[k] for k in FEATURES if k not in ("phys_dx", "phys_dy")} for r in rows])
    phys = np.array([disp_nm(r["lat"], r["lon"], *r["phys"][scale]) for r in rows])
    df["phys_dx"], df["phys_dy"] = phys[:, 0], phys[:, 1]
    true = np.array([r["true"] for r in rows])
    return df[FEATURES], true - phys


def fit_scale(rows: list[dict]) -> float:
    """Grid-search the current scale that minimises 24 h physics error (training buoys only)."""
    r24 = [r for r in rows if r["h"] == 24]
    errs = {a: np.mean([math.dist(r["true"], disp_nm(r["lat"], r["lon"], *r["phys"][a])) for r in r24])
            for a in r24[0]["phys"]}
    return min(errs, key=errs.get)


def fit_lgb(X: pd.DataFrame, y: np.ndarray) -> list[lgb.LGBMRegressor]:
    return [lgb.LGBMRegressor(**LGB_PARAMS, random_state=k).fit(X, y[:, k]) for k in (0, 1)]


def predict(models, X) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=24, help="hours between start points on one buoy track")
    args = ap.parse_args()

    starts = sample_starts(load_drifters(), args.n, args.seed, stride=args.stride)
    rows = build_rows(starts)
    ids = np.array([r["id"] for r in rows])
    hs = np.array([r["h"] for r in rows])
    print(f"{len(rows) // len(HORIZONS)} start points from {len(set(ids))} buoys")

    err = {m: np.full(len(rows), np.nan) for m in ("persistence", "physics", "physics+lgbm")}
    covered = {m: np.zeros(len(rows), bool) for m in ("physics", "physics+lgbm")}
    radius = {m: np.zeros(len(rows)) for m in ("physics", "physics+lgbm")}
    rng = np.random.default_rng(args.seed)
    lambdas = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=args.folds).split(rows, groups=ids)):
        train = [rows[i] for i in tr]
        scale = fit_scale(train)
        tr_ids = np.unique(ids[tr])
        cal_ids = set(rng.choice(tr_ids, size=max(1, len(tr_ids) // 4), replace=False))
        fit_i = [i for i in tr if ids[i] not in cal_ids]
        cal_i = [i for i in tr if ids[i] in cal_ids]

        X_fit, y_fit = frame([rows[i] for i in fit_i], scale)
        models = fit_lgb(X_fit, y_fit)

        def errors(idx, lam=None):
            X, resid = frame([rows[i] for i in idx], scale)
            corr = predict(models, X)
            if lam is not None:  # per-horizon shrinkage of the learned correction
                corr = corr * np.array([lam[h] for h in hs[idx]])[:, None]
            return np.linalg.norm(resid, axis=1), np.linalg.norm(resid - corr, axis=1)

        # Shrinkage lambda_h in [0, 1] chosen on the calibration buoys: lambda = 0 falls back to pure
        # physics, so the hybrid can only be as bad as physics on data like the calibration set.
        lam = {}
        for h in HORIZONS:
            ih = [i for i in cal_i if hs[i] == h]
            lam[h] = min(LAMBDAS, key=lambda a: errors(ih, {h: a})[1].mean())
        lambdas.append(lam)
        cal_phys, cal_ml = errors(cal_i, lam)
        te_phys, te_ml = errors(te, lam)
        for m, cal_e, te_e in (("physics", cal_phys, te_phys), ("physics+lgbm", cal_ml, te_ml)):
            q = {h: np.quantile(cal_e[hs[cal_i] == h], 0.9) for h in HORIZONS}  # split conformal per horizon
            radius[m][te] = [q[h] for h in hs[te]]
            covered[m][te] = te_e <= radius[m][te]
            err[m][te] = te_e
        err["persistence"][te] = [math.hypot(*rows[i]["true"]) for i in te]
        print(f"fold {k}: scale {scale}, test buoys {len(np.unique(ids[te]))}, "
              f"lambda {lam}, 24h mean err physics {te_phys[hs[te] == 24].mean():.2f} -> +lgbm {te_ml[hs[te] == 24].mean():.2f} nm")

    # Paired bootstrap BY BUOY of the 24 h improvement (resample whole buoys).
    m24 = hs == 24
    uniq = np.unique(ids[m24])
    gain = []
    for _ in range(2000):
        pick = rng.choice(uniq, size=len(uniq))
        mask = np.concatenate([np.where(m24 & (ids == b))[0] for b in pick])
        gain.append(1 - err["physics+lgbm"][mask].mean() / err["physics"][mask].mean())

    report = {"n_start_points": len(rows) // len(HORIZONS), "n_buoys": int(len(set(ids))), "folds": args.folds,
              "cv": "GroupKFold by buoy; scale + LightGBM + conformal radius fitted inside each fold",
              "mean_error_nm": {}, "median_error_nm": {}, "skill_vs_persistence": {},
              "conformal_90pct_coverage": {}, "mean_radius90_nm": {}}
    for h in HORIZONS:
        sel = hs == h
        for m in err:
            report["mean_error_nm"].setdefault(m, {})[str(h)] = float(err[m][sel].mean())
            report["median_error_nm"].setdefault(m, {})[str(h)] = float(np.median(err[m][sel]))
            if m != "persistence":
                report["skill_vs_persistence"].setdefault(m, {})[str(h)] = float(
                    1 - err[m][sel].mean() / err["persistence"][sel].mean())
        for m in covered:
            report["conformal_90pct_coverage"].setdefault(m, {})[str(h)] = float(covered[m][sel].mean())
            report["mean_radius90_nm"].setdefault(m, {})[str(h)] = float(radius[m][sel].mean())
    report["lgbm_gain_24h"] = {"mean": float(np.mean(gain)), "ci95": [float(np.quantile(gain, 0.025)),
                                                                      float(np.quantile(gain, 0.975))]}

    # Final model on all buoys (for the dashboard) + feature importance.
    scale = fit_scale(rows)
    X_all, y_all = frame(rows, scale)
    final = fit_lgb(X_all, y_all)
    imp = sum(m.booster_.feature_importance("gain") for m in final)
    lam_final = {str(h): float(np.mean([l[h] for l in lambdas])) for h in HORIZONS}
    report["final"] = {"current_scale": scale, "shrinkage_lambda": lam_final,
                       "feature_importance_gain": dict(sorted(zip(FEATURES, (imp / imp.sum()).round(4).tolist()),
                                                              key=lambda kv: -kv[1]))}
    Path("models").mkdir(exist_ok=True)
    joblib.dump({"models": final, "features": FEATURES, "current_scale": scale,
                 "shrinkage_lambda": lam_final}, "models/drift_lgbm.joblib")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift_ml.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("mean error (nm):", json.dumps(report["mean_error_nm"], default=lambda x: round(x, 2)))
    print("skill vs persistence:", json.dumps(report["skill_vs_persistence"], default=lambda x: round(x, 3)))
    print("90% coverage:", json.dumps(report["conformal_90pct_coverage"], default=lambda x: round(x, 2)))
    print("mean radius90 (nm):", json.dumps(report["mean_radius90_nm"], default=lambda x: round(x, 1)))
    print("24h LightGBM gain over physics:", report["lgbm_gain_24h"])
    print("top features:", list(report["final"]["feature_importance_gain"].items())[:6])


if __name__ == "__main__":
    main()
