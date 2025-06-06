#!/usr/bin/env python3
"""
train_pf_model.py
-----------------
Train a per‑breath Protection‑Factor regressor from the `breath_features` table
in an SQLite database.  Best‑practice pipeline:

1.  Load features and compute log10(PF).
2.  Drop non‑numeric and identifier columns.
3.  Use GroupKFold CV with `source_file` as the grouping key to prevent
    leakage across breaths from the same recording.
4.  HistGradientBoostingRegressor (fast, tree‑boosted) with sensible defaults.
5.  After CV, refit on the full data set and save `(model, feature_cols)`
    via joblib.
6.  Optional SHAP beeswarm plot for feature importance.

Usage examples
--------------
python train_pf_model.py --db breath_db.sqlite
python train_pf_model.py --db breath_db.sqlite --cv-splits 8 --max-depth 4 --shap
"""

from __future__ import annotations
import argparse, sqlite3, joblib, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

# Optional SHAP import handled later

def load_features(db_path: Path | str, table: str = "breath_features") -> pd.DataFrame:
    """Return breath_features table as DataFrame."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def build_dataset(df: pd.DataFrame):
    """Return (X, y, groups, feature_cols) ready for ML."""
    # Ensure PF exists and is positive
    df = df.dropna(subset=["protection_factor"])
    df = df[df["protection_factor"] > 0]

    df = df.copy()
    df["logPF"] = np.log10(df["protection_factor"].astype(float))

    # Features = all numeric except ID/target cols
    drop_cols = {"protection_factor", "logPF", "source_file", "breath", "t_start", "t_end"}
    num_df = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors="ignore")

    return num_df.values, df["logPF"].values, df["source_file"].values, num_df.columns.tolist()


def main():
    ap = argparse.ArgumentParser(description="Train PF regressor from breath_features table.")
    ap.add_argument("--db", required=True, help="SQLite database containing breath_features")
    ap.add_argument("--table", default="breath_features", help="Table name")
    ap.add_argument("--cv-splits", type=int, default=5, help="Number of GroupKFold splits (default 5)")
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--l2", type=float, default=1.0, help="L2 regularisation")
    ap.add_argument("--output", default="pf_regressor.joblib", help="Joblib output file")
    ap.add_argument("--shap", action="store_true", help="Save SHAP summary plot")
    args = ap.parse_args()

    df = load_features(args.db, args.table)
    X, y, groups, feat_cols = build_dataset(df)

    # ---------------- model
    model = HistGradientBoostingRegressor(
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        max_iter=args.max_iter,
        l2_regularization=args.l2,
        random_state=0,
    )

    gkf = GroupKFold(n_splits=args.cv_splits)
    scoring = {
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "r2": "r2",
    }

    cv_res = cross_validate(model, X, y, cv=gkf, groups=groups, scoring=scoring, return_train_score=False)
    mae_scores = -cv_res["test_mae"]  # neg sign removed
    r2_scores = cv_res["test_r2"]

    print(f"CV MAE(logPF): {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
    print(f"CV R²:          {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")

    # ---------------- fit final model
    model.fit(X, y)
    joblib.dump((model, feat_cols), args.output)
    print(f"Model + feature list saved to {args.output}")

    # ---------------- SHAP summary plot (optional)
    if args.shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X, check_additivity=False)
            shap.summary_plot(shap_values, features=X, feature_names=feat_cols, show=False)
            out_png = Path(args.output).with_suffix("")             # strip old suffix
            out_png = out_png.with_name(out_png.name + "_shap.png") # add custom tail
            import matplotlib.pyplot as plt
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"SHAP plot saved to {out_png}")
        except ImportError:
            warnings.warn("`shap` package not installed; skipping SHAP plot.")


if __name__ == "__main__":
    main()
