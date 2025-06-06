from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score


def train_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Iterable,
    n_estimators: int = 100,
    random_state: int = 0,
    model_path: str | Path = "rf_model.joblib",
) -> Tuple[RandomForestRegressor, pd.DataFrame]:
    """Train RF with group-wise CV and save the model."""
    gkf = GroupKFold(n_splits=5)
    fold_results = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rmse = mean_squared_error(y.iloc[test_idx], pred, squared=False)
        r2 = r2_score(y.iloc[test_idx], pred)
        fold_results.append(dict(rmse=rmse, r2=r2))
    # Fit final model on all data
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    model.fit(X, y)
    joblib.dump(model, model_path)
    return model, pd.DataFrame(fold_results)
