import numpy as np
import pandas as pd

from mask_leak_analysis import estimate_lag_seconds


def test_estimate_lag_seconds_synthetic():
    fs = 50.0
    duration = 20.0
    t = np.arange(0, duration, 1/fs)
    pressure = np.sin(2 * np.pi * 0.5 * t)
    dp = np.abs(np.gradient(pressure))
    lag_seconds = 0.4
    shift = int(round(lag_seconds * fs))
    particles = np.concatenate([np.zeros(shift), dp[:-shift]])
    df = pd.DataFrame({'pressure': pressure, 'particles': particles})
    est = estimate_lag_seconds(df, fs, 'pressure', 'particles', max_lag_s=1.0)
    assert abs(est - lag_seconds) <= 1.0 / fs
