from __future__ import annotations

import numpy as np
import pandas as pd


def detect_breath_cycles(pressure: pd.Series, fs: float = 1000.0) -> pd.DataFrame:
    """Detect breath cycles using derivative zero crossings.

    Parameters
    ----------
    pressure:
        Pressure signal sampled at ``fs`` Hz.
    fs:
        Sampling frequency.

    Returns
    -------
    DataFrame with columns ``start``, ``end``, ``pip_idx``, ``pep_idx``.
    """
    data = pressure.to_numpy()
    d = np.gradient(data)
    sign = np.sign(d)
    zc = np.where(np.diff(sign))[0] + 1

    minima = [i for i in zc if d[i - 1] < 0 and d[i] > 0]
    maxima = [i for i in zc if d[i - 1] > 0 and d[i] < 0]

    records = []
    for i in range(len(minima) - 1):
        start = minima[i]
        end = minima[i + 1]
        if end - start < fs:
            continue
        peak = [m for m in maxima if start < m < end]
        pip = peak[0] if peak else int(start + (end - start) / 2)
        records.append(
            dict(
                start=pressure.index[start],
                end=pressure.index[end],
                pip_idx=pip,
                pep_idx=start,
            )
        )

    return pd.DataFrame(records)
