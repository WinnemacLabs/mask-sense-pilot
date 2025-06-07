#!/usr/bin/env python3
"""
correlation_heatmap.py
----------------------
Generate a correlation heat-map from the `breath_features` table.

Usage examples
--------------
# default: Pearson, save to heatmap.png
python correlation_heatmap.py --db breath_db.sqlite

# Spearman rank corr. and custom file name
python correlation_heatmap.py --db breath_db.sqlite --method spearman --out figure/corr.png
"""

import argparse, sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
def load_breath_features(db_path: Path | str, table: str = "breath_features") -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def main():
    ap = argparse.ArgumentParser(description="Correlation heat-map for breath features.")
    ap.add_argument("--db", required=True, help="SQLite database containing breath_features")
    ap.add_argument("--table", default="breath_features", help="Table name (default: breath_features)")
    ap.add_argument("--method", choices=["pearson", "spearman"], default="pearson",
                    help="Correlation method (default: pearson)")
    ap.add_argument("--out", default="heatmap.png", help="Output PNG file")
    args = ap.parse_args()

    df = load_breath_features(args.db, args.table)

    # --------------------------------------------------------------------- tidy up
    # Add log10(protection_factor) if PF column exists
    if "protection_factor" in df.columns:
        df = df.copy()
        df["log_protection_factor"] = np.log10(df["protection_factor"].replace(0, np.nan))

    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    # Compute correlation matrix
    corr = num_df.corr(method=args.method)

    # --------------------------------------------------------------------- plot
    plt.figure(figsize=(10, 10))
    im = plt.imshow(corr, aspect="equal", interpolation="nearest")  # no explicit colormap → default
    plt.colorbar(im, fraction=0.046, pad=0.04, label=f"{args.method.title()} correlation")

    # Tick labels
    labels = corr.columns
    plt.xticks(range(len(labels)), labels, rotation=90, fontsize=6)
    plt.yticks(range(len(labels)), labels, fontsize=6)

    plt.title(f"Correlation heat-map ({args.method.title()})", pad=20)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Heat-map saved to {args.out}")


if __name__ == "__main__":
    main()
