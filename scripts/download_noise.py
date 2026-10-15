"""Download real background noise for the VHF simulator from ESC-50.

ESC-50 (Piczak 2015, CC BY-NC 3.0) has 2,000 labelled 5 s clips. We keep the
classes a boat radio would actually pick up. Download is ~600 MB once; only
~280 small WAVs are kept in data/noise/.

Usage:
    python scripts/download_noise.py
"""

from __future__ import annotations

import csv
import io
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
KEEP = {"engine", "sea_waves", "wind", "rain", "thunderstorm", "helicopter", "crackling_fire", "water_drops"}


def main(out: Path = Path("data/noise")) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "esc50.zip"
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, zpath)
        with zipfile.ZipFile(zpath) as z:
            meta = next(n for n in z.namelist() if n.endswith("meta/esc50.csv"))
            root = meta[: -len("meta/esc50.csv")]
            rows = list(csv.DictReader(io.TextIOWrapper(z.open(meta), encoding="utf-8")))
            kept = 0
            for r in rows:
                if r["category"] in KEEP:
                    with z.open(f"{root}audio/{r['filename']}") as src, open(out / f"{r['category']}_{r['filename']}", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    kept += 1
    print(f"kept {kept} clips in {out}")


if __name__ == "__main__":
    main()
