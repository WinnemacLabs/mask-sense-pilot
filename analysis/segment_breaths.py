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


def filter_clustered_peaks(peaks: np.ndarray, peak_values: np.ndarray, min_distance: int = 10) -> np.ndarray:
    """
    Filter peaks to keep only the tallest peak within each cluster.
    
    Args:
        peaks: Array of peak indices
        peak_values: Array of peak values (heights)
        min_distance: Minimum distance between peaks (in samples)
        
    Returns:
        Array of filtered peak indices with only the tallest in each cluster
    """
    if len(peaks) == 0:
        return peaks
    
    # Sort peaks by their values (heights) in descending order
    sorted_indices = np.argsort(peak_values)[::-1]
    filtered_peaks = []
    
    for idx in sorted_indices:
        peak_pos = peaks[idx]
        # Check if this peak is far enough from already selected peaks
        is_isolated = True
        for selected_peak in filtered_peaks:
            if abs(peak_pos - selected_peak) < min_distance:
                is_isolated = False
                break
        
        if is_isolated:
            filtered_peaks.append(peak_pos)
    
    # Sort the filtered peaks by position
    return np.array(sorted(filtered_peaks))


def segment_breaths(filtered: np.ndarray, distance_cutoff: float = 0.0, time_s: np.ndarray = None, peak_height: float = None, trough_height: float = None, peak_min_distance: int = 50):
    """List of (start_idx, end_idx) for each breath, with optional minimum trough-to-trough distance in seconds. Only include segments containing both a trough and a peak above the specified heights. Each segment is from the last zero crossing before a valid trough to the last zero crossing before the next valid trough. The first breath cannot start until after the first positive pressure value."""
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
    
    # Filter peaks to keep only the tallest in each cluster
    if len(peaks) > 0:
        peaks = filter_clustered_peaks(peaks, filtered[peaks], peak_min_distance)

    # Optionally filter troughs by distance
    if distance_cutoff > 0 and time_s is not None and len(troughs) > 1:
        keep = [0]
        for i in range(1, len(troughs)):
            if (time_s[troughs[i]] - time_s[troughs[keep[-1]]]) >= distance_cutoff:
                keep.append(i)
        troughs = troughs[keep]

    # For each valid trough, find the last zero crossing before it
    zero_crossings = [preceding_falling_zero(filtered, t) for t in troughs]

    # --- ENFORCE: first breath cannot start until after first positive value ---
    # Find the first index where filtered > 0
    first_positive_idx = np.argmax(filtered > 0)
    # Only keep zero crossings that occur at or after this index
    zero_crossings = [zc for zc in zero_crossings if zc >= first_positive_idx]
    # Also, only keep troughs that correspond to these zero crossings
    troughs = [t for zc, t in zip(zero_crossings, troughs) if zc >= first_positive_idx]

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


def create_breath_segments_table(conn: sqlite3.Connection) -> None:
    """Create the breath_segments table if it doesn't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS breath_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            breath INTEGER NOT NULL,
            breath_start_us INTEGER NOT NULL,
            breath_end_us INTEGER NOT NULL,
            UNIQUE(source_file, breath)
        )
    """)


def create_settings_table(conn: sqlite3.Connection) -> None:
    conn.execute('''
        CREATE TABLE IF NOT EXISTS breath_segmentation_settings (
            source_file TEXT PRIMARY KEY,
            cutoff REAL,
            trough_height REAL,
            peak_height REAL,
            peak_min_distance INTEGER,
            distance_cutoff REAL
        )
    ''')


def load_settings_for_file(conn: sqlite3.Connection, source_file: str):
    cur = conn.execute('''
        SELECT cutoff, trough_height, peak_height, peak_min_distance, distance_cutoff
        FROM breath_segmentation_settings WHERE source_file=?
    ''', (source_file,))
    row = cur.fetchone()
    if row:
        return {
            'cutoff': row[0],
            'trough_height': row[1],
            'peak_height': row[2],
            'peak_min_distance': row[3],
            'distance_cutoff': row[4],
        }
    return None


def save_settings_for_file(conn: sqlite3.Connection, source_file: str, settings: dict):
    conn.execute('''
        INSERT INTO breath_segmentation_settings
        (source_file, cutoff, trough_height, peak_height, peak_min_distance, distance_cutoff)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_file) DO UPDATE SET
            cutoff=excluded.cutoff,
            trough_height=excluded.trough_height,
            peak_height=excluded.peak_height,
            peak_min_distance=excluded.peak_min_distance,
            distance_cutoff=excluded.distance_cutoff
    ''', (
        source_file,
        settings['cutoff'],
        settings['trough_height'],
        settings['peak_height'],
        settings['peak_min_distance'],
        settings['distance_cutoff'],
    ))
    conn.commit()


def save_to_db(df: pd.DataFrame, db_path: Path, source_file: Path, segments: list, time_s: np.ndarray) -> None:
    """Save raw data to main table and breath segments to separate table."""
    conn = sqlite3.connect(db_path)
    
    # Save raw data without breath column
    main_table = "breath_data"
    if table_exists(conn, main_table):
        conn.execute(f"DELETE FROM {main_table} WHERE source_file=?", (str(source_file),))
    
    # Remove breath column if it exists and add source_file
    df_clean = df.copy()
    if 'breath' in df_clean.columns:
        df_clean = df_clean.drop('breath', axis=1)
    df_clean["source_file"] = str(source_file)
    df_clean.to_sql(main_table, conn, if_exists="append", index=False)
    
    # Create and populate breath_segments table
    create_breath_segments_table(conn)
    
    # Delete existing segments for this file
    conn.execute("DELETE FROM breath_segments WHERE source_file=?", (str(source_file),))
    
    # Insert breath segments
    for i, (start_idx, end_idx) in enumerate(segments, 1):
        start_us = int(df.iloc[start_idx]['t_us'])
        end_us = int(df.iloc[end_idx-1]['t_us'])  # end_idx is exclusive
        
        conn.execute("""
            INSERT INTO breath_segments (source_file, breath, breath_start_us, breath_end_us)
            VALUES (?, ?, ?, ?)
        """, (str(source_file), i, start_us, end_us))
    
    conn.commit()
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
    # Set defaults to None for now; will override after loading from DB
    parser.add_argument("--cutoff", type=float, default=None, help="Low pass cutoff frequency (Hz)")
    parser.add_argument("--trough-height", type=float, default=None, help="Height for trough detection (Pa)")
    parser.add_argument("--peak-height", type=float, default=None, help="Height for peak detection (Pa)")
    parser.add_argument("--peak-min-distance", type=int, default=None, help="Minimum distance between peaks (samples) - keeps only tallest peak in each cluster")
    parser.add_argument("--distance_cutoff", type=float, default=None, help="Minimum trough-to-trough distance (seconds) for histogram display and filtering")
    parser.add_argument("--interactive", action="store_true", help="Plot peak prominences and set threshold interactively")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)
    source_file = csv_path.name
    
    # Open DB and ensure settings table exists
    conn = sqlite3.connect(db_path)
    create_settings_table(conn)
    
    # Try to load settings for this file
    prev_settings = load_settings_for_file(conn, source_file)
    # Set defaults if not present
    default_settings = {
        'cutoff': 20.0,
        'trough_height': 10.0,
        'peak_height': 10.0,
        'peak_min_distance': 50,
        'distance_cutoff': 0.0,
    }
    if prev_settings:
        print(f"Loaded stored segmentation settings for {source_file} from database.")
        default_settings.update({k: v for k, v in prev_settings.items() if v is not None})
    # Use argparse values if provided, else use loaded/default
    cutoff = args.cutoff if args.cutoff is not None else default_settings['cutoff']
    trough_height = args.trough_height if args.trough_height is not None else default_settings['trough_height']
    peak_height = args.peak_height if args.peak_height is not None else default_settings['peak_height']
    peak_min_distance = args.peak_min_distance if args.peak_min_distance is not None else default_settings['peak_min_distance']
    distance_cutoff = args.distance_cutoff if args.distance_cutoff is not None else default_settings['distance_cutoff']

    df = pd.read_csv(csv_path, comment="#")
    if {"t_us", "Pa_Global"}.difference(df.columns):
        raise SystemExit("CSV missing required columns (t_us, Pa_Global)")

    time_s = df["t_us"] * 1e-6
    fs = 1.0 / np.median(np.diff(time_s))

    if args.interactive:
        temp_filtered = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=3.0)
        troughs, trough_props = find_peaks(-temp_filtered, height=trough_height)
        peaks, peak_props = find_peaks(temp_filtered, height=peak_height)
        # Filter peaks to keep only the tallest in each cluster for display
        if len(peaks) > 0:
            peaks = filter_clustered_peaks(peaks, temp_filtered[peaks], peak_min_distance)
            peak_props = {"peak_heights": temp_filtered[peaks]}
        # Calculate trough-to-trough distances in seconds
        if len(troughs) > 1:
            trough_distances = np.diff(time_s[troughs])
        else:
            trough_distances = []
        # Optionally filter by distance-cutoff
        if distance_cutoff > 0 and len(troughs) > 1:
            keep = np.insert(trough_distances >= distance_cutoff, 0, True)
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
        new_trough = float(input(f"Enter new threshold for trough height (current: {trough_height}): "))
        trough_height = new_trough
        new_peak = float(input(f"Enter new threshold for peak height (current: {peak_height}): "))
        peak_height = new_peak
        new_distance = float(input(f"Enter minimum trough-to-trough distance in seconds (current: {distance_cutoff}): "))
        distance_cutoff = new_distance

    df["Pa_Global_Filtered"] = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=cutoff)
    segments = segment_breaths(
        df["Pa_Global_Filtered"].to_numpy(),
        distance_cutoff=distance_cutoff,
        time_s=time_s.to_numpy(),
        peak_height=peak_height,
        trough_height=trough_height,
        peak_min_distance=peak_min_distance
    )
    out_png = Path(f"breath_segments_{csv_path.stem}_combined.png")
    filtered_arr = df["Pa_Global_Filtered"].to_numpy()
    valid_troughs, _ = find_peaks(-filtered_arr, height=trough_height)
    valid_peaks, _ = find_peaks(filtered_arr, height=peak_height)
    plot_combined(time_s.to_numpy(), filtered_arr, df["Pa_Global"].to_numpy(), segments, out_png, valid_troughs=valid_troughs, valid_peaks=valid_peaks)
    save_to_db(df, db_path, csv_path.name, segments, time_s.to_numpy())
    # Save settings used for this file
    settings_to_save = {
        'cutoff': cutoff,
        'trough_height': trough_height,
        'peak_height': peak_height,
        'peak_min_distance': peak_min_distance,
        'distance_cutoff': distance_cutoff,
    }
    save_settings_for_file(conn, source_file, settings_to_save)
    print(f"Raw data written to {db_path} (table: breath_data)")
    print(f"Breath segments written to {db_path} (table: breath_segments)")
    print(f"Settings saved for {source_file} (table: breath_segmentation_settings)")
    print(f"Found {len(segments)} breath segments")
    conn.close()


if __name__ == "__main__":
    main()
