#!/usr/bin/env python3
"""
segment_breaths.py
------------------
Segment a pressure trace (Pa_Global) into individual breaths, write the
results back into an SQLite DB **and** auto‑generate a full‑length debug plot
showing the segmentation.

• Breath boundaries = falling zero‑crossing before each negative peak
  (trough) in the low‑pass filtered signal.
• Debug PNG saved next to the input CSV (or to --plot-file).

Example
~~~~~~~
python segment_breaths.py run01.csv --db breath_db.sqlite 
python segment_breaths.py run01.csv --prominence 1.5 --plot-file debug/run01_breaths.png
"""

import argparse, sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------

def low_pass(data: np.ndarray, fs: float, cutoff: float = 2.0, order: int = 4) -> np.ndarray:
    """Zero‑phase low‑pass Butterworth filter."""
    sos = butter(order, cutoff, fs=fs, btype="low", output="sos")
    return sosfiltfilt(sos, data)


def preceding_falling_zero(data: np.ndarray, idx: int) -> int:
    """Return index of nearest falling zero‑crossing before *idx*."""
    for i in range(idx, 0, -1):
        if data[i - 1] >= 0 and data[i] < 0:
            return i
    return 0


def segment_breaths(filtered: np.ndarray, prominence: float = 2.0):
    """List of (start_idx, end_idx) for each breath."""
    troughs, _ = find_peaks(-filtered, prominence=prominence)
    starts = sorted({preceding_falling_zero(filtered, t) for t in troughs})
    return list(zip(starts[:-1], starts[1:]))


def debug_plot(time_s: np.ndarray, signal: np.ndarray, segments, out_png: Path):
    """Save a PNG showing the full signal and breath segmentation."""
    plt.figure(figsize=(12, 4))
    plt.plot(time_s, signal, lw=0.8, label="Pa_Global (filtered)")
    ymax, ymin = signal.max(), signal.min()
    for s, e in segments:
        plt.axvspan(time_s[s], time_s[e], color="orange", alpha=0.15)
        plt.axvline(time_s[s], color="red", lw=0.5)
    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.title("Breath segmentation debug")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Debug plot saved ➜ {out_png}")


# ----------------------------------------------------------------------------------
# DB helpers
# ----------------------------------------------------------------------------------

def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def save_to_db(df: pd.DataFrame, db_path: Path, source_file: Path) -> None:
    conn = sqlite3.connect(db_path)
    table = "breath_data"
    if table_exists(conn, table):
        conn.execute(f"DELETE FROM {table} WHERE source_file=?", (str(source_file),))
    df["source_file"] = str(source_file)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Segment Pa_Global into breaths & save plot + DB")
    p.add_argument("csv", help="Input CSV file with columns t_us, Pa_Global, Pa_Horizontal, Pa_Vertical")
    p.add_argument("--db", default="breath_db.sqlite", help="SQLite database path")
    p.add_argument("--cutoff", type=float, default=2.0, help="Low‑pass cutoff frequency (Hz)")
    p.add_argument("--prominence", type=float, default=2.0, help="Peak prominence threshold (Pa)")
    p.add_argument("--plot-file", default=None, help="Optional output PNG path for debug plot")
    args = p.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if {"t_us", "Pa_Global"}.difference(df.columns):
        raise SystemExit("CSV missing required columns (t_us, Pa_Global)")

    time_s = df["t_us"] * 1e-6
    fs = 1.0 / np.median(np.diff(time_s))

    df["Pa_Global_Filtered"] = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=args.cutoff)
    segments = segment_breaths(df["Pa_Global_Filtered"].to_numpy(), prominence=args.prominence)

    # Assign breath index
    breath_idx = np.full(len(df), np.nan, dtype=float)
    for i, (s, e) in enumerate(segments, 1):
        breath_idx[s:e] = i
    df["breath"] = pd.Series(breath_idx, dtype="Int64")

    # Save plot covering full signal
    out_png = Path(args.plot_file) if args.plot_file else csv_path.with_name(f"{csv_path.stem}_breaths.png")
    debug_plot(time_s.to_numpy(), df["Pa_Global_Filtered"].to_numpy(), segments, out_png)

    # Write to DB
    save_to_db(df, Path(args.db), csv_path.name)
    print(f"Breath segmentation written to {args.db} (table: breath_data)")


if __name__ == "__main__":
    main()
