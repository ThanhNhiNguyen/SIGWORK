"""Download the IEEE-CIS Fraud Detection competition data.

Prerequisites:
    pip install kaggle
    Configure a Kaggle API token with access to the competition.

The raw files are kept outside Git history under datasets/IEEE-CIS/raw/.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"


def main() -> None:
    kaggle = shutil.which("kaggle")
    if kaggle is None:
        raise SystemExit(
            "Kaggle CLI not found. Install it with: pip install kaggle"
        )

    RAW.mkdir(parents=True, exist_ok=True)

    cmd = [
        kaggle,
        "competitions",
        "download",
        "-c",
        "ieee-fraud-detection",
        "-p",
        str(RAW),
    ]

    print("Downloading IEEE-CIS from Kaggle...")
    subprocess.run(cmd, check=True)

    print(f"Download complete. Raw files are in: {RAW}")
    print("Inspect the zip contents before extracting if needed.")


if __name__ == "__main__":
    main()
