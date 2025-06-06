from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def compute_spatial_features(
    P_vert: pd.Series,
    P_horiz: pd.Series,
    cycles: pd.DataFrame,
    fs: float = 1000.0,
):
    """Compute gradient and cross-channel lag over windows of 10 breaths."""
    gradient = P_horiz - P_vert
    lags = []
    for i in range(len(cycles)):
        start_idx = cycles.iloc[max(0, i - 9)].start
        end_idx = cycles.iloc[i].end
        seg_v = P_vert[start_idx:end_idx]
        seg_h = P_horiz[start_idx:end_idx]
        a = seg_h.to_numpy() - seg_h.mean()
        b = seg_v.to_numpy() - seg_v.mean()
        corr = signal.correlate(a, b, mode="full")
        lag_samples = signal.correlation_lags(len(a), len(b), mode="full")
        best_lag = lag_samples[np.argmax(corr)]
        lags.append(best_lag * 1000 / fs)
    lag_series = pd.Series(lags, index=cycles.index, name="lag_ms")
    return gradient, lag_series
