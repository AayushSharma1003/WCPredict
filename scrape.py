"""
scrape.py — Download fresh international match results.

Source: martj42/international_results on GitHub. This is the same dataset
used to train our Elo ratings and DC parameters, so we get consistency
between training and live updates.

Module usage:
    from scrape import fetch_remote_csv, validate_csv, hash_csv

Script usage:
    python scrape.py              # fetch and print summary
    python scrape.py --save       # fetch and overwrite data/results.csv
"""

import argparse
import hashlib
import io
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import requests

# Source URL (raw GitHub content, free, no API key required)
RESULTS_URL = ("https://raw.githubusercontent.com/"
               "martj42/international_results/master/results.csv")
LOCAL_PATH  = Path("data/results.csv")

REQUIRED_COLUMNS = {"date", "home_team", "away_team",
                    "home_score", "away_score",
                    "tournament", "city", "country", "neutral"}

TIMEOUT_SECONDS = 60
MIN_EXPECTED_ROWS = 40_000


# ---- Core functions ----

def fetch_remote_csv() -> bytes:
    """Download the raw CSV from martj42's GitHub repo. Raises on failure."""
    r = requests.get(RESULTS_URL, timeout=TIMEOUT_SECONDS)
    r.raise_for_status()
    return r.content


def validate_csv(content: bytes) -> pd.DataFrame:
    """Parse and sanity-check the downloaded CSV.

    Raises ValueError if the data looks malformed (wrong columns, too few
    rows, or stale dates). Returns the DataFrame if everything looks fine.
    """
    df = pd.read_csv(io.BytesIO(content))

    # Schema check
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Row count sanity
    if len(df) < MIN_EXPECTED_ROWS:
        raise ValueError(
            f"Suspiciously few rows: {len(df)} (expected >= {MIN_EXPECTED_ROWS})")

    # Date sanity
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        bad = df[df["date"].isna()].head(3)
        raise ValueError(f"Some rows have unparseable dates: {bad.to_dict()}")

    latest_date = df["date"].max()
    if latest_date < pd.Timestamp.now() - pd.Timedelta(days=365):
        raise ValueError(
            f"Latest match is over a year old: {latest_date}. "
            "Source data may be stale.")

    return df


def hash_csv(content: bytes) -> str:
    """SHA-256 hex digest for change detection."""
    return hashlib.sha256(content).hexdigest()


def save_atomically(content: bytes, path: Path) -> None:
    """Write to a temp file in the same dir, then rename atomically.

    Prevents leaving a half-written file if the process is killed mid-write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_", suffix=path.suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ---- CLI ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save the downloaded CSV to data/results.csv")
    args = parser.parse_args()

    print(f"Downloading {RESULTS_URL}")
    remote = fetch_remote_csv()
    df = validate_csv(remote)
    h = hash_csv(remote)

    print(f"  Bytes:        {len(remote):,}")
    print(f"  Matches:      {len(df):,}")
    print(f"  Date range:   {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  SHA-256:      {h[:16]}...")

    if args.save:
        save_atomically(remote, LOCAL_PATH)
        print(f"  Saved to:     {LOCAL_PATH}")
    else:
        print("  (pass --save to write to data/results.csv)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
