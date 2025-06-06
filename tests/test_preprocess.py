import numpy as np
import pandas as pd

from mask_fit_feat.preprocess import bandpass


def test_bandpass_impulse():
    idx = pd.date_range('2020', periods=1000, freq='1ms')
    data = np.zeros(1000)
    data[500] = 1.0
    s = pd.Series(data, index=idx)
    f = bandpass(s, fs=1000, low=1, high=100, detrend=False)
    assert f.idxmax() == s.index[500]
