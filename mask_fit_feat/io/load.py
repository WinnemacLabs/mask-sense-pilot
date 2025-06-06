from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


PRESSURE_COLS = [
    "Pa_Global",
    "Pa_Vertical",
    "Pa_Horizontal",
    "raw_Global",
    "raw_Vertical",
    "raw_Horizontal",
]
PARTICLE_COLS = ["mask_particles", "ambient_particles"]


def load_trial(path: str | Path) -> Dict[str, pd.Series]:
    """Load a trial CSV and return channels as Series.

    Parameters
    ----------
    path:
        Path to CSV file containing the required columns.

    Returns
    -------
    dict of pandas.Series
    """
    df = pd.read_csv(path)
    if "t_us" not in df.columns:
        raise ValueError("CSV missing t_us column")

    df.index = pd.to_datetime(df["t_us"], unit="us", utc=True)
    df.index.name = "time"

    out: Dict[str, pd.Series] = {}
    # Pressure channels resampled to 1000 Hz
    if len(df.index) > 1:
        start, end = df.index[0], df.index[-1]
        idx_1k = pd.date_range(start, end, freq="1ms", tz="UTC")
        pres = df[PRESSURE_COLS].astype(float)
        pres = pres.reindex(df.index).interpolate(method="time")
        pres = pres.reindex(idx_1k).interpolate(method="time")
        for c in PRESSURE_COLS:
            out[c] = pres[c]
    else:
        for c in PRESSURE_COLS:
            out[c] = pd.Series(dtype=float)

    # Particle channels left at 1 Hz
    part = df[PARTICLE_COLS].astype(float)
    idx_1hz = pd.date_range(df.index[0], df.index[-1], freq="1s", tz="UTC")
    part = part.resample("1s").mean()
    part = part.reindex(idx_1hz).interpolate(method="time")
    for c in PARTICLE_COLS:
        out[c] = part[c]

    return out
