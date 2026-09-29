"""Download the three original Elliptic Bitcoin CSV files.

Uses the public PyTorch Geometric mirror documented by the PyG Elliptic
dataset loader. Each file is provided as a separate ZIP archive.

No Kaggle credentials are required.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BASE = "https://data.pyg.org/datasets/elliptic"

FILES = [
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
]


def download_and_extract(filename: str) -> None:
    url = f"{BASE}/{filename}.zip"
    print(f"Downloading {filename} ...")

    with urlopen(url) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(RAW)

    print(f"Extracted {filename}")


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        target = RAW / filename
        if target.exists():
            print(f"Already present: {target}")
            continue
        download_and_extract(filename)

    print(f"Elliptic dataset ready in: {RAW}")


if __name__ == "__main__":
    main()
