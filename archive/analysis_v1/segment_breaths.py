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


def segment_breaths(filtered: np.ndarray, distance_cutoff: float = 0.0, time_s: np.ndarray = None, peak_height: float = None, trough_height: float = None):
    """List of (start_idx, end_idx) for each breath, with optional minimum trough-to-trough distance in seconds. Only include segments containing both a trough and a peak above the specified heights. Each segment is from the last zero crossing before a valid trough to the last zero crossing before the next valid trough."""
    # Find troughs (negative peaks) with height only
    trough_kwargs = {}
    if trough_height is not None:
        trough_kwargs["height"] = trough_height
    troughs, trough_props = find_peaks(-filtered, **trough_kwargs)
    # Find peaks (positive peaks) with height only
    if peak_height is not None:
        peaks, peak_props = find_peaks(filtered, height=peak_height)
    else:
        peaks, peak_props = find_peaks(filtered)

    # Optionally filter troughs by distance
    if distance_cutoff > 0 and time_s is not None and len(troughs) > 1:
        keep = [0]
        for i in range(1, len(troughs)):
            if (time_s[troughs[i]] - time_s[troughs[keep[-1]]]) >= distance_cutoff:
                keep.append(i)
        troughs = troughs[keep]

    # For each valid trough, find the last zero crossing before it
    zero_crossings = [preceding_falling_zero(filtered, t) for t in troughs]

    # Only keep segments where both a valid trough and a valid peak are present
    valid_segments = []
    for i in range(len(troughs) - 1):
        s = zero_crossings[i]
        e = zero_crossings[i + 1]
        # Troughs in this segment
        seg_troughs = [t for t in troughs if s <= t < e]
        # Peaks in this segment
        seg_peaks = [p for p in peaks if s <= p < e]
        if seg_troughs and seg_peaks:
            valid_segments.append((s, e))
    return valid_segments


def plot_combined(time_s: np.ndarray, filtered_signal: np.ndarray, raw_signal: np.ndarray, segments, out_png: Path, valid_troughs=None, valid_peaks=None):
    """Save a PNG with two subplots: filtered+regions, and raw+segment lines. Optionally mark valid peaks and troughs."""
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
    # Mark valid troughs
    if valid_troughs is not None and len(valid_troughs) > 0:
        fig.add_trace(
            go.Scatter(
                x=time_s[valid_troughs],
                y=filtered_signal[valid_troughs],
                mode="markers",
                marker=dict(symbol="triangle-down", color="blue", size=10),
                name="Valid Troughs"
            ),
            row=1, col=1
        )
    # Mark valid peaks
    if valid_peaks is not None and len(valid_peaks) > 0:
        fig.add_trace(
            go.Scatter(
                x=time_s[valid_peaks],
                y=filtered_signal[valid_peaks],
                mode="markers",
                marker=dict(symbol="triangle-up", color="red", size=10),
                name="Valid Peaks"
            ),
            row=1, col=1
        )
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
    fig.show()


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
        "--trough-height",
        type=float,
        default=10.0,
        help="Height for trough detection (Pa)",
    )
    parser.add_argument("--peak-height", type=float, default=10.0, help="Height for peak detection (Pa)")
    parser.add_argument("--distance_cutoff", type=float, default=0.0, help="Minimum trough-to-trough distance (seconds) for histogram display and filtering")
    parser.add_argument("--interactive", action="store_true", help="Plot peak prominences and set threshold interactively")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path, comment="#")
    if {"t_us", "Pa_Global"}.difference(df.columns):
        raise SystemExit("CSV missing required columns (t_us, Pa_Global)")

    time_s = df["t_us"] * 1e-6
    fs = 1.0 / np.median(np.diff(time_s))

    if args.interactive:
        temp_filtered = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=3.0)
        troughs, trough_props = find_peaks(-temp_filtered, height=args.trough_height)
        peaks, peak_props = find_peaks(temp_filtered, height=args.peak_height)

        # Calculate trough-to-trough distances in seconds
        if len(troughs) > 1:
            trough_distances = np.diff(time_s[troughs])
        else:
            trough_distances = []

        # Optionally filter by distance-cutoff
        if args.distance_cutoff > 0 and len(troughs) > 1:
            keep = np.insert(trough_distances >= args.distance_cutoff, 0, True)
            troughs = troughs[keep]
            trough_props = {k: v[keep] for k, v in trough_props.items()}
            if len(troughs) > 1:
                trough_distances = np.diff(time_s[troughs])
            else:
                trough_distances = []

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=("Filtered Data", "Peak Heights", "Trough-to-Trough Distances (s)")
        )
        fig.add_trace(
            go.Scatter(x=list(range(len(temp_filtered))), y=temp_filtered, name="Filtered Data"),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=trough_props["peak_heights"], nbinsx=100, name="Troughs", marker_color="blue"),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=peak_props["peak_heights"], nbinsx=100, name="Peaks", marker_color="red"),
            row=1, col=2
        )
        fig.add_trace(
            go.Histogram(x=trough_distances, nbinsx=100, name="Trough-to-Trough (s)", marker_color="green"),
            row=1, col=3
        )
        fig.update_xaxes(title_text="Sample", row=1, col=1)
        fig.update_yaxes(title_text="Pa", row=1, col=1)
        fig.update_xaxes(title_text="Height", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Distance (s)", row=1, col=3)
        fig.update_yaxes(title_text="Count", row=1, col=3)
        fig.show()

        new_trough = float(input(f"Enter new threshold for trough height (current: {args.trough_height}): "))
        args.trough_height = new_trough
        new_peak = float(input(f"Enter new threshold for peak height (current: {args.peak_height}): "))
        args.peak_height = new_peak
        new_distance = float(input(f"Enter minimum trough-to-trough distance in seconds (current: {args.distance_cutoff}): "))
        args.distance_cutoff = new_distance

    df["Pa_Global_Filtered"] = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=args.cutoff)
    # Use both prominence and value for troughs, and value for peaks
    segments = segment_breaths(
        df["Pa_Global_Filtered"].to_numpy(),
        distance_cutoff=args.distance_cutoff,
        time_s=time_s.to_numpy(),
        peak_height=args.peak_height,
        trough_height=args.trough_height
    )

    # Assign breath index
    breath_idx = np.full(len(df), np.nan, dtype=float)
    for i, (s, e) in enumerate(segments, 1):
        breath_idx[s:e] = i
    df["breath"] = pd.Series(breath_idx, dtype="Int64")

    # Save combined plot covering full signal and segment boundaries
    out_png = Path(f"breath_segments_{csv_path.stem}_combined.png")
    # Find valid troughs and peaks for plotting (enforce both thresholds)
    filtered_arr = df["Pa_Global_Filtered"].to_numpy()
    valid_troughs, _ = find_peaks(-filtered_arr, height=args.trough_height)
    valid_peaks, _ = find_peaks(filtered_arr, height=args.peak_height)
    plot_combined(time_s.to_numpy(), filtered_arr, df["Pa_Global"].to_numpy(), segments, out_png, valid_troughs=valid_troughs, valid_peaks=valid_peaks)

    # Write to DB
    save_to_db(df, Path(args.db), csv_path.name)
    print(f"Breath segmentation written to {args.db} (table: breath_data)")
    


if __name__ == "__main__":
    main()
