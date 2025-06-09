#!/usr/bin/env python3
"""integral_analysis.py — Analyze pressure CSV files and generate combined integral/AUC figures, including a 60‑second rolling ratio.

This script can be executed directly from the command line or imported as a module.

Usage (command‑line):
    python integral_analysis.py path/to/file.csv

Import (Python):
    from integral_analysis import analyze_csv
    asymptote, image_path = analyze_csv("data/myfile.csv")

Requirements:
    - pandas
    - numpy
    - scipy
    - plotly (plus the "kaleido" engine for static image export)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.integrate import cumulative_trapezoid

__all__ = ["analyze_csv"]

ROLLING_WINDOW_SEC = 60.0  # length of rolling window in seconds


def analyze_csv(csv_path: str | Path) -> Tuple[float, str]:
    """Process *csv_path* and return (auc_ratio_asymptote, image_path).

    The resulting figure is saved to
    ``output/integral-analysis/{source_basename}.png`` (created if absent).
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load data and prepare time axis
    df = pd.read_csv(csv_path)
    if "t_us" not in df.columns:
        raise ValueError("CSV must contain a 't_us' column representing microseconds")
    df["t_s"] = df["t_us"] * 1e-6  # seconds
    t = df["t_s"].values

    # Cumulative integrals for each pressure channel
    for ch in ("Pa_Global", "Pa_Vertical", "Pa_Horizontal"):
        if ch not in df.columns:
            raise ValueError(f"CSV missing required column: {ch}")
        df[f"int_{ch}"] = np.concatenate([[0], cumulative_trapezoid(df[ch].values, t)])

    # Positive/negative AUC ratio for Pa_Global (cumulative)
    pa = df["Pa_Global"].values
    pos_pa = np.where(pa > 0, pa, 0.0)
    neg_pa = np.where(pa < 0, pa, 0.0)

    df["pos_auc"] = np.concatenate([[0], cumulative_trapezoid(pos_pa, t)])
    df["neg_auc"] = np.concatenate([[0], cumulative_trapezoid(-neg_pa, t)])
    df["auc_ratio"] = df["pos_auc"] / np.where(df["neg_auc"] == 0, np.nan, df["neg_auc"])

    # 60‑second rolling AUC ratio
    pos_auc_total = df["pos_auc"].values
    neg_auc_total = df["neg_auc"].values
    idx_start = np.searchsorted(t, t - ROLLING_WINDOW_SEC)
    pos_auc_roll = pos_auc_total - pos_auc_total[idx_start]
    neg_auc_roll = neg_auc_total - neg_auc_total[idx_start]
    df["auc_ratio_roll"] = pos_auc_roll / np.where(neg_auc_roll == 0, np.nan, neg_auc_roll)

    # Asymptote = median of last 60 s of the cumulative ratio
    last_60 = df[df["t_s"] >= (df["t_s"].max() - ROLLING_WINDOW_SEC)]
    auc_ratio_asymptote = float(last_60["auc_ratio"].median())

    # Create combined figure with two stacked subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            "Cumulative Integral of Pressure Channels",
            "AUC Ratio (cumulative and 60‑s rolling)",
        ),
    )

    # Subplot 1 — Integrals
    fig.add_trace(
        go.Scatter(x=df["t_s"], y=df["int_Pa_Global"], name="Integral Global (Pa·s)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["t_s"], y=df["int_Pa_Vertical"], name="Integral Vertical (Pa·s)"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["t_s"], y=df["int_Pa_Horizontal"], name="Integral Horizontal (Pa·s)"),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Cumulative Integral (Pa·s)", row=1, col=1)

    # Subplot 2 — Ratio traces
    fig.add_trace(
        go.Scatter(x=df["t_s"], y=df["auc_ratio"], name="AUC Ratio (cumulative)"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=df["t_s"],
            y=df["auc_ratio_roll"],
            name=f"AUC Ratio (rolling {int(ROLLING_WINDOW_SEC)} s)",
            line=dict(dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=auc_ratio_asymptote,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Asymptote: {auc_ratio_asymptote:.3f}",
        annotation_position="top right",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="AUC Ratio", range=[0.5, 1.5], row=2, col=1)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)

    fig.update_layout(height=900, title_text="Integral Analysis with 60‑Second Rolling Ratio")

    # Save figure
    output_dir = Path("output/integral-analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / f"{csv_path.stem}.png"
    fig.write_image(str(image_path))
    print(f"[integral_analysis] Saved figure → {image_path}")

    return auc_ratio_asymptote, str(image_path)


# ---- CLI ----

def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate integral/AUC analysis figures from a pressure CSV file.",
    )
    parser.add_argument(
        "csv_file",
        type=str,
        help="Path to the CSV file generated by the data recorder.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover
    args = _parse_cli()
    asymptote, image_path = analyze_csv(args.csv_file)
    print(f"AUC Ratio Asymptote (cumulative): {asymptote:.6f}")
    print(f"Figure saved at: {image_path}")


if __name__ == "__main__":
    main()
