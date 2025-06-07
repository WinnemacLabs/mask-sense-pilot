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
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# pio rendering is set to 'browser' for interactive plots
pio.renderers.default = "browser"

def low_pass(data: np.ndarray, fs: float, cutoff: float = 3.0, order: int = 4) -> np.ndarray:
    """Low pass filter with zero-phase filtering."""
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
    parser = argparse.ArgumentParser(
        description="Segment Pa_Global into breaths and store in a database"
    )
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--db", default="breath_db.sqlite", help="SQLite database path")
    parser.add_argument("--cutoff", type=float, default=20.0, help="Low pass cutoff frequency (Hz)")
    parser.add_argument(
        "--prominence",
        type=float,
        default=10.0,
        help="Prominence for trough detection (Pa)",
    )
    parser.add_argument("--interactive", action="store_true", help="Plot peak prominences and set threshold interactively")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    if {"t_us", "Pa_Global"}.difference(df.columns):
        raise SystemExit("CSV missing required columns (t_us, Pa_Global)")

    time_s = df["t_us"] * 1e-6
    fs = 1.0 / np.median(np.diff(time_s))

    if args.interactive:
        temp_filtered = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=3.0)
        troughs, trough_props = find_peaks(-temp_filtered, prominence=1)
        peaks, peak_props = find_peaks(temp_filtered, prominence=1)

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Filtered Data", "Peak Prominences"))
        fig.add_trace(
            go.Scatter(x=list(range(len(temp_filtered))), y=temp_filtered, name="Filtered Data"),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=trough_props["prominences"], nbinsx=30, name="Troughs", marker_color="blue"),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=peak_props["prominences"], nbinsx=30, name="Peaks", marker_color="red"),
            row=1, col=2
        )
        fig.update_xaxes(title_text="Sample", row=1, col=1)
        fig.update_yaxes(title_text="Pa", row=1, col=1)
        fig.update_xaxes(title_text="Prominence", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.show()

        new_threshold = float(input("Enter new threshold for prominence: "))
        args.prominence = new_threshold

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

    analysis_dir = Path("output")
    analysis_dir.mkdir(parents=True, exist_ok=True)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=list(range(len(df["Pa_Global"]))),
            y=df["Pa_Global"],
            mode="lines",
            name="Filtered Data"
        )
    )
    for start, end in segments:
        fig2.add_vline(x=start, line_color="green", line_dash="dash")
        fig2.add_vline(x=end, line_color="red", line_dash="dash")
    fig2.update_layout(
        title="Segmented Breaths",
        xaxis_title="Sample",
        yaxis_title="Pa",
    )
    
    # write the figure to a PNG in the output directory, tagging it with the source file name
    fig2.write_image(analysis_dir / f"breath_segments_{Path(args.csv).stem}.png")
    # Show the figure in the browser
    fig2.show()


if __name__ == "__main__":
    main()
