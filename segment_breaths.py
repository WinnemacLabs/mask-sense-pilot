import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks


def low_pass(data: np.ndarray, fs: float, cutoff: float = 2.0, order: int = 4) -> np.ndarray:
    """Low pass filter with zero-phase filtering."""
    sos = butter(order, cutoff, fs=fs, btype="low", output="sos")
    return sosfiltfilt(sos, data)


def preceding_falling_zero(data: np.ndarray, idx: int) -> int:
    """Return index of the nearest falling zero crossing before ``idx``."""
    for i in range(idx, 0, -1):
        if data[i - 1] >= 0 and data[i] < 0:
            return i
    return 0


def segment_breaths(filtered: np.ndarray, prominence: float = 2.0):
    """Segment breaths using negative peaks and surrounding zero crossings."""
    troughs, _ = find_peaks(-filtered, prominence=prominence)
    starts = [preceding_falling_zero(filtered, t) for t in troughs]
    starts = sorted(set(starts))
    segments = []
    for i in range(len(starts) - 1):
        segments.append((starts[i], starts[i + 1]))
    return segments


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment Pa_Global into breaths and store in a database"
    )
    parser.add_argument("csv", help="Input CSV file")
    parser.add_argument("--db", default="breath_db.sqlite", help="SQLite database path")
    parser.add_argument("--cutoff", type=float, default=2.0, help="Low pass cutoff frequency (Hz)")
    parser.add_argument(
        "--prominence",
        type=float,
        default=2.0,
        help="Prominence for trough detection (Pa)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if "t_us" not in df.columns or "Pa_Global" not in df.columns:
        raise SystemExit("CSV missing required columns")

    time_s = df["t_us"] / 1e6
    fs = 1 / np.median(np.diff(time_s))

    df["Pa_Global_Filtered"] = low_pass(df["Pa_Global"].to_numpy(), fs, cutoff=args.cutoff)
    segments = segment_breaths(df["Pa_Global_Filtered"].to_numpy(), prominence=args.prominence)

    breath_col = np.full(len(df), np.nan)
    for i, (start, end) in enumerate(segments, 1):
        breath_col[start:end] = i
    df["breath"] = pd.Series(breath_col, dtype="Int64")

    save_to_db(df, Path(args.db), Path(args.csv).name)


if __name__ == "__main__":
    main()
