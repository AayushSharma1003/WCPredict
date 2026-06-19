"""
update.py — Live update pipeline for WCPredict.

End-to-end orchestration designed to run unattended via GitHub Actions cron:

  1. Fetch latest match results from martj42/international_results
  2. Hash-compare with our local copy
  3. If unchanged: exit early (no website update needed)
  4. If changed: re-run the full modeling pipeline:
       a. Save fresh results.csv (atomic write — never a half-written file)
       b. Re-compute Elo ratings
       c. Re-fit Dixon-Coles parameters
       d. Run forward simulations
       e. Run diagnostic outputs
  5. Validate every output file (existence, JSON parse, conservation)
  6. Exit 0 (GitHub Action will commit the changed data/ files)

Usage:
    python update.py             # standard cron run
    python update.py --force     # run pipeline even if data unchanged
    python update.py --dry-run   # check for new data, don't run sims
    python update.py --verbose   # echo subprocess stdout in real-time
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scrape import (
    fetch_remote_csv,
    validate_csv,
    hash_csv,
    save_atomically,
    LOCAL_PATH,
    RESULTS_URL,
)

ROOT      = Path(__file__).resolve().parent
DATA_DIR  = ROOT / "data"

# Files we expect after a full pipeline run
EXPECTED_OUTPUTS = {
    "data/elo_current.csv":           "csv",
    "data/elo_history.csv":           "csv",
    "data/dc_params.json":            "json",
    "data/tournament_simulation.csv": "csv",
    "data/bracket_view.json":         "json",
    "data/final_pairings.json":       "json",
    "data/champion_routes.json":      "json",
}

# Pipeline steps in order: (display_name, command)
PIPELINE_STEPS = [
    ("Re-compute Elo ratings",        ["python", "elo.py"]),
    ("Re-fit Dixon-Coles parameters", ["python", "dixon_coles.py"]),
    ("Run tournament simulations",    ["python", "simulate_tournament.py"]),
    ("Run diagnostic simulations",    ["python", "diagnose_tournament.py"]),
]


# ---- Logging ----

def log(msg: str = "") -> None:
    """Timestamped log line. Goes to stdout so GitHub Actions captures it."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"[{ts}] {msg}", flush=True)


# ---- Pipeline ----

def run_step(name: str, cmd: list, verbose: bool = False) -> None:
    """Run a subprocess; fail loudly if it returns non-zero.

    Captures output so it appears in the GitHub Action log in order, rather
    than interleaved. On non-zero exit, dumps stdout+stderr and exits.
    """
    log(f"→ {name}")
    t0 = time.time()
    if verbose:
        result = subprocess.run(cmd, cwd=ROOT)
    else:
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0

    if result.returncode != 0:
        log(f"  ✗ FAILED in {elapsed:.1f}s (exit {result.returncode})")
        if not verbose:
            log("  ---- stdout ----")
            for line in (result.stdout or "").splitlines():
                log(f"    {line}")
            log("  ---- stderr ----")
            for line in (result.stderr or "").splitlines():
                log(f"    {line}")
        sys.exit(1)

    log(f"  ✓ done in {elapsed:.1f}s")


def validate_outputs() -> None:
    """Verify every expected output exists, parses, and looks reasonable."""
    log("Validating outputs...")

    for path_str, kind in EXPECTED_OUTPUTS.items():
        p = ROOT / path_str
        if not p.exists():
            log(f"  ✗ Missing file: {path_str}")
            sys.exit(2)
        if p.stat().st_size == 0:
            log(f"  ✗ Empty file: {path_str}")
            sys.exit(2)
        if kind == "json":
            try:
                json.loads(p.read_text())
            except json.JSONDecodeError as e:
                log(f"  ✗ Invalid JSON in {path_str}: {e}")
                sys.exit(2)
        log(f"  ✓ {path_str}")

    # Conservation check on tournament_simulation.csv: sum P(Cup) ≈ 1
    sim_csv = ROOT / "data/tournament_simulation.csv"
    df = pd.read_csv(sim_csv)
    p_cup_sum = df["P(Cup)"].sum()
    if abs(p_cup_sum - 1.0) > 0.001:
        log(f"  ✗ Conservation broken: sum P(Cup) = {p_cup_sum:.4f}")
        sys.exit(2)
    log(f"  ✓ Conservation OK: sum P(Cup) = {p_cup_sum:.4f}")

    # Sanity: top champion is identifiable
    top = df.iloc[0]
    log(f"  Top champion: {top['team']}  (P(Cup) = {top['P(Cup)']:.2%})")


# ---- Main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Run full pipeline even if data unchanged.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check for new data but don't run sims.")
    parser.add_argument("--verbose", action="store_true",
                        help="Echo subprocess stdout in real-time.")
    args = parser.parse_args()

    log(f"WCPredict update pipeline starting")
    log(f"  ROOT: {ROOT}")
    log(f"  Options: force={args.force} dry_run={args.dry_run} verbose={args.verbose}")

    # ---- 1. Fetch ----
    log(f"Fetching latest data from {RESULTS_URL}")
    t0 = time.time()
    try:
        remote_bytes = fetch_remote_csv()
        df = validate_csv(remote_bytes)
    except Exception as e:
        log(f"  ✗ Failed to fetch/validate: {e}")
        sys.exit(3)
    remote_hash = hash_csv(remote_bytes)
    log(f"  ✓ {len(df):,} matches, latest={df['date'].max().date()}, "
        f"sha256={remote_hash[:16]}... ({time.time() - t0:.1f}s)")

    # ---- 2. Compare ----
    local_path = ROOT / LOCAL_PATH
    data_changed = True
    if local_path.exists():
        local_hash = hash_csv(local_path.read_bytes())
        if local_hash == remote_hash:
            data_changed = False
            log("No new match data since last run.")
        else:
            log(f"Data changed (local sha256={local_hash[:16]}...).")
    else:
        log("No local copy of results.csv yet, treating as first run.")

    if not data_changed and not args.force:
        log("Nothing to do. Exiting cleanly.")
        return 0

    if args.dry_run:
        log("Dry run: would have run pipeline. Exiting.")
        return 0

    # ---- 3. Save fresh CSV (atomic) ----
    log("Saving fresh results.csv (atomic write)")
    save_atomically(remote_bytes, local_path)

    # ---- 4-7. Pipeline ----
    for step_name, cmd in PIPELINE_STEPS:
        run_step(step_name, cmd, verbose=args.verbose)

    # ---- 8. Validate ----
    validate_outputs()

    log("✓ Update pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
