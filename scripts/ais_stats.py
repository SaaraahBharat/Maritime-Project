"""Real vessel-speed statistics from one day of MarineCadastre AIS (US waters,
public domain), used to make the simulated vessels of opportunity realistic.

US traffic is not Indian traffic, but speeds per vessel class are physical
properties of the ships and transfer reasonably well.

Usage:  python scripts/ais_stats.py
Writes  data/external/ais/speed_by_type.json
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ZIP = Path("data/external/ais/AIS_2023_01_01.zip")
OUT = Path("data/external/ais/speed_by_type.json")

# AIS ship-type codes (ITU-R M.1371) -> our vessel classes
TYPE_GROUPS = {
    "fishing_boat": range(30, 31), "tug": list(range(31, 33)) + [52], "yacht": range(36, 38),
    "ferry": range(60, 70), "cargo": range(70, 80), "tanker": range(80, 90),
}


def main() -> None:
    with zipfile.ZipFile(ZIP) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name), usecols=["MMSI", "SOG", "VesselType"])
    print(f"{len(df):,} AIS position reports, {df['MMSI'].nunique():,} vessels")
    moving = df[(df["SOG"] > 1) & (df["SOG"] < 40)]
    stats = {}
    for cls, codes in TYPE_GROUPS.items():
        sog = moving.loc[moving["VesselType"].isin(list(codes)), "SOG"]
        # one value per vessel (median), so busy transponders don't dominate
        per_vessel = moving.loc[moving["VesselType"].isin(list(codes))].groupby("MMSI")["SOG"].median()
        stats[cls] = {"vessels": int(per_vessel.size), "reports": int(sog.size),
                      "median_kn": float(np.median(per_vessel)) if per_vessel.size else None,
                      "p10_kn": float(np.percentile(per_vessel, 10)) if per_vessel.size else None,
                      "p90_kn": float(np.percentile(per_vessel, 90)) if per_vessel.size else None}
        print(f"  {cls:13s} {stats[cls]}")
    OUT.write_text(json.dumps({"source": "MarineCadastre AIS 2023-01-01", "speed_by_type": stats}, indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
