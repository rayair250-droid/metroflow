#!/usr/bin/env python
"""Download a real GTFS feed and point MetroFlow at it.

MetroFlow does NOT vendor large real feeds. This helper downloads one to a local
directory of your choosing; you then inspect it with ``metroflow gtfs-info`` and
simulate a chosen route/direction with ``metroflow simulate --gtfs ...``.

Cited open-data sources
-----------------------
* Île-de-France Mobilités (IDFM), via transport.data.gouv.fr:
  https://transport.data.gouv.fr/datasets/reseau-urbain-et-interurbain-dile-de-france-mobilites
* RATP open data:
  https://www.ratp.fr/en/ratp-and-open-data

The transport.data.gouv.fr dataset page exposes a stable "latest resource"
download URL for the GTFS zip. Because such URLs change over time and the feed is
large, this script takes the URL as an argument rather than hard-coding it.

Usage
-----
    # 1. Find the current GTFS .zip URL on the dataset page above, then:
    python scripts/fetch_gtfs.py <GTFS_ZIP_URL> --dest data/idfm_gtfs

    # 2. See what routes/stops it contains:
    metroflow gtfs-info data/idfm_gtfs

    # 3. Simulate one route/direction built from the feed:
    metroflow simulate --gtfs data/idfm_gtfs --route <route_id> --direction 0 --seed 42

Only the Python standard library is used (urllib + zipfile).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile


def fetch(url: str, dest: str) -> str:
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(dest, "gtfs.zip")
    print(f"Downloading {url}\n  -> {zip_path}")
    urllib.request.urlretrieve(url, zip_path)  # noqa: S310 - user-supplied URL
    print("Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    files = sorted(os.listdir(dest))
    print(f"Extracted {len(files)} entries into {dest}:")
    for f in files:
        print(f"  {f}")
    print(f"\nNext: metroflow gtfs-info {dest}")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("url", help="direct URL to a GTFS .zip (see the dataset page)")
    ap.add_argument("--dest", default="data/gtfs", help="output directory")
    args = ap.parse_args()
    try:
        fetch(args.url, args.dest)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"Fetch failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
