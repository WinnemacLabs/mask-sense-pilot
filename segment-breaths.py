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


def plot_combined(time_s: np.ndarray, filtered_signal: np.ndarray, raw_signal: np.ndarray, segments, out_png: Path):
    """Save a PNG with two subplots: filtered+regions, and raw+segment lines."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    out_dir = Path("output") / "breath-segmentation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / out_png.name

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.12,
                        subplot_titles=("Filtered Signal with Breath Regions", "Raw Signal with Segment Boundaries"))

    # Subplot 1: Filtered signal with breath regions
    fig.add_trace(
        go.Scatter(x=time_s, y=filtered_signal, mode="lines", name="Pa_Global (filtered)"),
        row=1, col=1
    )
    for s, e in segments:
        fig.add_vrect(x0=time_s[s], x1=time_s[e], fillcolor="orange", opacity=0.15, line_width=0, row=1, col=1)
        fig.add_vline(x=time_s[s], line_color="red", line_dash="solid", row=1, col=1)

    fig.update_yaxes(title_text="Pressure (Pa)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)

    # Subplot 2: Raw signal with segment boundaries
    fig.add_trace(
        go.Scatter(x=list(range(len(raw_signal))), y=raw_signal, mode="lines", name="Pa_Global (raw)"),
        row=2, col=1
    )
    for start, end in segments:
        fig.add_vline(x=start, line_color="green", line_dash="dash", row=2, col=1)
        fig.add_vline(x=end, line_color="red", line_dash="dash", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=2, col=1)
    fig.update_xaxes(title_text="Sample", row=2, col=1)

    fig.update_layout(height=900, title="Breath Segmentation: Filtered & Raw Signal")
    fig.write_image(out_png)
    print(f"Combined plot saved ➜ {out_png}")


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

    # Save combined plot covering full signal and segment boundaries
    out_png = Path(f"breath_segments_{csv_path.stem}_combined.png")
    plot_combined(time_s.to_numpy(), df["Pa_Global_Filtered"].to_numpy(), df["Pa_Global"].to_numpy(), segments, out_png)

    # Write to DB
    save_to_db(df, Path(args.db), csv_path.name)
    print(f"Breath segmentation written to {args.db} (table: breath_data)")
    


if __name__ == "__main__":
    main()
