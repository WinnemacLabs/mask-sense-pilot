from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression


def aggregate_features(
    df: pd.DataFrame, on: pd.Series, window: str = "60s"
) -> pd.DataFrame:
    """Aggregate features over fixed time windows."""
    df = df.copy()
    df["time"] = on

    def iqr(x: pd.Series) -> float:
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    agg = df.resample(window, on="time").agg(["mean", "std", "min", "max", iqr])
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg.drop(columns=[c for c in agg.columns if c.startswith("time")])


def select_top_n_features(df: pd.DataFrame, y: pd.Series, n: int = 25) -> pd.DataFrame:
    """Select top ``n`` features using mutual information."""
    X = df.fillna(0)
    mi = mutual_info_regression(X, y)
    order = np.argsort(mi)[::-1][:n]
    return X.iloc[:, order]
