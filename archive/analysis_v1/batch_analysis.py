#!/usr/bin/env python3
"""batch_analysis.py — Run integral-analysis and protection-factor on all CSVs in data/.

This script finds all CSV files in the data/ directory and runs both
integral-analysis.py and protection-factor.py on each file. Output locations
are printed for each analysis.

Usage:
    python batch_analysis.py
"""

import subprocess
from pathlib import Path

DATA_DIR = Path("data")
CSV_FILES = sorted(DATA_DIR.glob("*.csv"))

if not CSV_FILES:
    print("No CSV files found in data/ directory.")
    exit(1)

for csv_file in CSV_FILES:
    print(f"\n=== Analyzing: {csv_file.name} ===")
    # Run integral-analysis.py
    try:
        result = subprocess.run([
            "python3", "integral-analysis.py", str(csv_file)
        ], capture_output=True, text=True, check=True)
        print("[integral-analysis]", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"[integral-analysis] ERROR: {e.stderr.strip()}")

    # Run protection-factor.py
    try:
        result = subprocess.run([
            "python3", "protection-factor.py", str(csv_file)
        ], capture_output=True, text=True, check=True)
        print("[protection-factor]", result.stdout.strip())
    except subprocess.CalledProcessError as e:
        print(f"[protection-factor] ERROR: {e.stderr.strip()}")

print("\nBatch analysis complete.")
