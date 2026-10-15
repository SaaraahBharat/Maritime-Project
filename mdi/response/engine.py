"""Phase 5 response intelligence (prototype scale).

* Rescue assets: coast guard stations along the Indian coast plus simulated
  vessels of opportunity (stand-in for a live AIS feed).
* ETA: great-circle distance / asset speed.
* Assignment: Hungarian algorithm over severity-weighted ETAs when several
  incidents are open (scipy's exact solver for the assignment problem).
* Drift: Monte Carlo leeway model; the particle cloud gives a probabilistic
  search area (50% / 90% radius) that grows with time since the call.
* Weather: Open-Meteo forecast/marine APIs when online, sampled fallback otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path
import math
import urllib.request
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from mdi.data import gazetteer as gz
from mdi.data.spoken import destination, haversine_nm


@dataclass
class Asset:
    name: str
    kind: str  # ship | boat | helicopter | vessel_of_opportunity
    lat: float
    lon: float
    speed_kn: float


_KIND_SPEED = {"helicopter": 120.0, "ship": 22.0, "boat": 30.0, "vessel_of_opportunity": 11.0}


def coast_guard_assets() -> list[Asset]:
    assets = []
    for i, lm in enumerate(gz.LANDMARKS):
        assets.append(Asset(f"ICG boat ({lm.name})", "boat", lm.lat, lm.lon, _KIND_SPEED["boat"]))
        if i % 3 == 0:
            assets.append(Asset(f"ICG ship ({lm.name})", "ship", lm.lat, lm.lon, _KIND_SPEED["ship"]))
        if i % 5 == 0:
            assets.append(Asset(f"ICG helicopter ({lm.name})", "helicopter", lm.lat, lm.lon, _KIND_SPEED["helicopter"]))
    return assets


def simulated_ais(rng: np.random.Generator, n: int = 40) -> list[Asset]:
    """Merchant/fishing traffic offshore; replace with a real AIS snapshot when available."""
    # Speeds from real AIS (MarineCadastre, scripts/ais_stats.py) when available.
    speeds = {"cargo": (11.7, 2.5), "tanker": (11.7, 2.0), "fishing_boat": (3.2, 1.5)}
    stats_path = Path("data/external/ais/speed_by_type.json")
    if stats_path.exists():
        st = json.loads(stats_path.read_text())["speed_by_type"]
        speeds = {k: (v["median_kn"], (v["p90_kn"] - v["p10_kn"]) / 2.56) for k, v in st.items() if k in speeds}
    out = []
    for k in range(n):
        lm = gz.LANDMARKS[int(rng.integers(len(gz.LANDMARKS)))]
        lat, lon = destination(lm.lat, lm.lon, rng.uniform(*lm.seaward_bearing), rng.uniform(5, 60))
        kind = rng.choice(list(speeds))
        mu, sd = speeds[kind]
        out.append(Asset(f"{kind.replace('_', ' ')} {1000 + k} (AIS)", "vessel_of_opportunity", lat, lon,
                         float(np.clip(rng.normal(mu, sd), 2.0, 20.0))))
    return out


def eta_hours(asset: Asset, lat: float, lon: float) -> float:
    return haversine_nm(asset.lat, asset.lon, lat, lon) / asset.speed_kn


def rank_assets(assets: list[Asset], lat: float, lon: float, k: int = 5) -> list[dict]:
    rows = [{"asset": a.name, "kind": a.kind, "lat": a.lat, "lon": a.lon,
             "distance_nm": round(haversine_nm(a.lat, a.lon, lat, lon), 1),
             "eta_h": round(eta_hours(a, lat, lon), 2)} for a in assets]
    return sorted(rows, key=lambda r: r["eta_h"])[:k]


def assign(assets: list[Asset], incidents: list[dict]) -> list[dict]:
    """Optimal one-asset-per-incident assignment.

    Cost = ETA * severity weight, so a CRITICAL incident's minutes cost more than a
    LOW one's. Each incident dict needs lat, lon, severity (1-4) and id.
    """
    if not incidents:
        return []
    w = {1: 1.0, 2: 2.0, 3: 4.0, 4: 8.0}
    cost = np.array([[eta_hours(a, i["lat"], i["lon"]) * w[i["severity"]] for a in assets] for i in incidents])
    rows, cols = linear_sum_assignment(cost)
    return [{"incident": incidents[r]["id"], "asset": assets[c].name,
             "eta_h": round(eta_hours(assets[c], incidents[r]["lat"], incidents[r]["lon"]), 2)}
            for r, c in zip(rows, cols)]


# --- weather + drift -------------------------------------------------------

def fetch_conditions(lat: float, lon: float, timeout: float = 4.0) -> dict:
    """Current wind (and ocean current if available). Falls back to sampled values offline."""
    try:
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat:.3f}&longitude={lon:.3f}"
               "&current=wind_speed_10m,wind_direction_10m&wind_speed_unit=kn")
        with urllib.request.urlopen(url, timeout=timeout) as r:
            cur = json.load(r)["current"]
        cond = {"wind_kn": cur["wind_speed_10m"], "wind_from_deg": cur["wind_direction_10m"], "source": "open-meteo"}
        try:
            murl = (f"https://marine-api.open-meteo.com/v1/marine?latitude={lat:.3f}&longitude={lon:.3f}"
                    "&current=ocean_current_velocity,ocean_current_direction,wave_height")
            with urllib.request.urlopen(murl, timeout=timeout) as r:
                m = json.load(r)["current"]
            cond.update(current_kn=(m.get("ocean_current_velocity") or 0) / 1.852,  # km/h -> kn
                        current_to_deg=m.get("ocean_current_direction") or 0, wave_m=m.get("wave_height"))
        except Exception:
            cond.update(current_kn=0.5, current_to_deg=cond["wind_from_deg"] + 180)
        return cond
    except Exception:
        return {"wind_kn": 15.0, "wind_from_deg": 240.0, "current_kn": 0.6, "current_to_deg": 60.0,
                "wave_m": 1.5, "source": "fallback (offline)"}


# Leeway as a fraction of wind speed (simplified from published SAR leeway tables).
LEEWAY = {"person_in_water": 0.011, "fishing_boat": 0.04, "trawler": 0.035, "yacht": 0.045,
          "cargo": 0.025, "tanker": 0.02, "ferry": 0.03, "tug": 0.035, None: 0.035}


def drift_particles(lat: float, lon: float, hours: float, cond: dict, target: str | None,
                    position_uncertainty_nm: float = 0.5, n: int = 500, seed: int = 0) -> np.ndarray:
    """Monte Carlo drift: perturb start position, wind, current and leeway per particle."""
    rng = np.random.default_rng(seed)
    out = np.empty((n, 2))
    lw = LEEWAY.get(target, LEEWAY[None])
    for k in range(n):
        b0 = rng.uniform(0, 360)
        d0 = abs(rng.normal(0, max(position_uncertainty_nm, 0.1)))
        plat, plon = destination(lat, lon, b0, d0)
        wind = max(0.0, rng.normal(cond["wind_kn"], 0.2 * cond["wind_kn"] + 1))
        wind_to = cond["wind_from_deg"] + 180 + rng.normal(0, 15)
        cur = max(0.0, rng.normal(cond["current_kn"], 0.3))
        cur_to = cond["current_to_deg"] + rng.normal(0, 20)
        lwk = max(0.0, rng.normal(lw, 0.3 * lw))
        vx = wind * lwk * math.sin(math.radians(wind_to)) + cur * math.sin(math.radians(cur_to))
        vy = wind * lwk * math.cos(math.radians(wind_to)) + cur * math.cos(math.radians(cur_to))
        dist = math.hypot(vx, vy) * hours
        out[k] = destination(plat, plon, math.degrees(math.atan2(vx, vy)) % 360, dist)
    return out


def search_area(particles: np.ndarray) -> dict:
    """Centroid and 50/90% containment radii of the particle cloud."""
    clat, clon = particles.mean(0)
    d = np.array([haversine_nm(clat, clon, a, b) for a, b in particles])
    return {"lat": float(clat), "lon": float(clon),
            "r50_nm": float(np.quantile(d, 0.5)), "r90_nm": float(np.quantile(d, 0.9)),
            "area90_nm2": float(math.pi * np.quantile(d, 0.9) ** 2)}


def calibrated_radius90(hours: float) -> float | None:
    """Conformal 90% search radius (nm) from real-drifter validation (scripts/validate_drift.py)."""
    p = Path("reports/drift_calibration.json")
    if not p.exists():
        return None
    r = {int(k): v for k, v in json.loads(p.read_text())["radius90_nm"].items()}
    hs = sorted(r)
    return float(np.interp(hours, [0] + hs, [0.5] + [r[h] for h in hs]))
