import pandas as pd
import numpy as np

from mask_fit_feat.breath import detect_breath_cycles


def test_detect_breath_cycles():
    idx = pd.date_range('2020', periods=20000, freq='1ms')
    t = np.linspace(0, 20, len(idx))  # include endpoint for clean minima
    s = pd.Series(np.sin(2 * np.pi * 0.2 * t), index=idx)
    cycles = detect_breath_cycles(s)
    assert len(cycles) >= 2
    assert all(cycles.end > cycles.start)
