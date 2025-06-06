from __future__ import annotations

import pandas as pd
from scipy import signal


def bandpass(
    series: pd.Series,
    fs: float,
    low: float = 0.05,
    high: float = 50.0,
    order: int = 4,
    detrend: bool = True,
) -> pd.Series:
    """Bandpass filter with zero-phase Butterworth filter.

    Parameters
    ----------
    series:
        Input time series.
    fs:
        Sampling frequency in Hz.
    low, high:
        Cutoff frequencies in Hz.
    order:
        Filter order.
    detrend:
        Apply linear detrend before filtering.

    Returns
    -------
    pandas.Series with the same index as ``series``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> idx = pd.date_range('2020', periods=1000, freq='1ms')
    >>> s = pd.Series(np.r_[1, np.zeros(999)], index=idx)
    >>> f = bandpass(s, fs=1000, low=1, high=100, detrend=False)
    >>> int(np.argmax(f.values))
    0
    """
    data = series.to_numpy()
    if detrend:
        data = signal.detrend(data, type="linear")
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, data)
    return pd.Series(filtered, index=series.index, name=series.name)
