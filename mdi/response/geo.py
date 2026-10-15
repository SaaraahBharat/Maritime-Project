"""Geospatial context from open data.

* GEBCO bathymetry (ERDDAP GEBCO_2020 subset, Indian waters, ~1 km): depth
  lookup, land mask for drift particles, shallow-water (grounding) risk.
* Historical forcing via Open-Meteo: ERA5 10 m wind (archive API) and
  Copernicus Marine surface currents (marine API). No accounts needed;
  responses are cached on disk.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

GEBCO_PATH = Path("data/external/gebco/gebco_india.nc")
CACHE = Path("data/external/cache")


@lru_cache(maxsize=1)
def _gebco():
    from scipy.io import netcdf_file

    with netcdf_file(GEBCO_PATH, "r", mmap=False) as nc:
        lat = nc.variables["latitude"][:].copy()
        lon = nc.variables["longitude"][:].copy()
        z = nc.variables["elevation"][:].copy().astype(np.float32)
    return lat, lon, z


def gebco_available() -> bool:
    return GEBCO_PATH.exists()


def elevation(lat: float | np.ndarray, lon: float | np.ndarray) -> np.ndarray:
    """GEBCO elevation in metres (negative = depth). NaN outside the subset."""
    glat, glon, z = _gebco()
    lat, lon = np.atleast_1d(lat), np.atleast_1d(lon)
    i = np.clip(np.searchsorted(glat, lat), 0, len(glat) - 1)
    j = np.clip(np.searchsorted(glon, lon), 0, len(glon) - 1)
    out = z[i, j]
    outside = (lat < glat[0]) | (lat > glat[-1]) | (lon < glon[0]) | (lon > glon[-1])
    return np.where(outside, np.nan, out)


def on_land(lat, lon) -> np.ndarray:
    return elevation(lat, lon) > 0


def _get_json(url: str) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / (hashlib.md5(url.encode()).hexdigest() + ".json")
    if f.exists():
        return json.loads(f.read_text())
    with urllib.request.urlopen(url, timeout=30) as r:
        data = json.load(r)
    f.write_text(json.dumps(data))
    return data


def historical_forcing(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    """Hourly ERA5 wind (kn, from-direction) and surface current (kn, to-direction)."""
    lat, lon = round(lat * 4) / 4, round(lon * 4) / 4  # 0.25 deg cells -> fewer API calls, better caching
    w = _get_json(
        f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
        f"&hourly=wind_speed_10m,wind_direction_10m&wind_speed_unit=kn&models=era5"
        f"&start_date={start_date}&end_date={end_date}")["hourly"]
    c = _get_json(
        f"https://marine-api.open-meteo.com/v1/marine?latitude={lat}&longitude={lon}"
        f"&hourly=ocean_current_velocity,ocean_current_direction"
        f"&start_date={start_date}&end_date={end_date}")["hourly"]
    to_kn = 1 / 1.852  # km/h -> kn
    return {
        "time": w["time"],
        "wind_kn": np.array(w["wind_speed_10m"], dtype=float),
        "wind_from_deg": np.array(w["wind_direction_10m"], dtype=float),
        "current_kn": np.array([v if v is not None else np.nan for v in c["ocean_current_velocity"]], dtype=float) * to_kn,
        "current_to_deg": np.array([v if v is not None else np.nan for v in c["ocean_current_direction"]], dtype=float),
    }
