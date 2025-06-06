from __future__ import annotations

import numpy as np
import pandas as pd


def extract_fft_features(segment: pd.Series, fs: float) -> np.ndarray:
    """Extract frequency-domain features from a segment."""
    data = segment.to_numpy()
    fft = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(len(data), d=1 / fs)
    power = np.abs(fft) ** 2
    total_power = power.sum()

    def band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        return power[mask].sum()

    p1 = band_power(0.1, 0.5)
    p2 = band_power(0.5, 5)
    p3 = band_power(5, 50)
    centroid = float((freqs * power).sum() / total_power)
    bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total_power))

    feats = np.array([
        np.log10(p1 + 1e-12),
        np.log10(p2 + 1e-12),
        np.log10(p3 + 1e-12),
        centroid,
        bandwidth,
    ])
    return feats
