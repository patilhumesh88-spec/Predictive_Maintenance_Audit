#!/usr/bin/env python3
"""Download the UCI AI4I 2020 dataset with checksum validation.

Usage:  python scripts/download_data.py
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "ai4i2020.csv"
URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
EXPECTED_SHA256 = "dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e"


def download():
    if DEST.exists():
        h = hashlib.sha256(DEST.read_bytes()).hexdigest()
        if h == EXPECTED_SHA256:
            print(f"[OK] Dataset already present and verified: {DEST}")
            return
        print(f"[WARN] File exists but checksum mismatch (got {h[:16]}...). Re-downloading.")

    DEST.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading from UCI repository...")
    import zipfile, io
    try:
        resp = urllib.request.urlopen(URL)
        z = zipfile.ZipFile(io.BytesIO(resp.read()))
        # Find the CSV inside the zip
        csv_names = [n for n in z.namelist() if n.endswith('.csv')]
        if not csv_names:
            print("ERROR: No CSV found in archive", file=sys.stderr)
            sys.exit(1)
        data = z.read(csv_names[0])
        DEST.write_bytes(data)
    except Exception as e:
        print(f"ERROR downloading: {e}", file=sys.stderr)
        print("Manual download: visit https://doi.org/10.24432/C5HS5C")
        print(f"Place ai4i2020.csv in {DEST}")
        sys.exit(1)

    h = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if h != EXPECTED_SHA256:
        print("WARNING: Checksum mismatch after download.")
        print(f"  Expected: {EXPECTED_SHA256}")
        print(f"  Got:      {h}")
        print("  The file may still be usable but verify manually.")
    else:
        print(f"[OK] Downloaded and verified: {DEST}")


if __name__ == "__main__":
    download()
