
"""protection_factor.py

Compute cumulative and 60‑second rolling protection factors from a CSV.

Usage
-----
python protection_factor.py your_data.csv

The CSV must contain at least two columns:

    mask_particles     – particle counts inside the mask
    ambient_particles  – ambient particle counts

Optionally, it may include a 'timestamp' column (any pandas‑parseable
datetime).  If present, a true 60‑second time window is used; otherwise the
script assumes the data are sampled once per second and uses a 30‑row window.

Outputs
-------
A PNG plot is saved to:

    output/protection-factor/<source_file_name>.png

containing two subplots:
1. Cumulative protection factor
2. 60‑second rolling protection factor

© 2025
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Tiny epsilon to avoid division by zero
EPSILON = 1e-9

def compute_protection_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Add cumulative and 60‑s rolling protection‑factor columns to *df*."""
    # Cumulative sums
    df["mask_cumsum"] = df["mask_particles"].cumsum()
    df["ambient_cumsum"] = df["ambient_particles"].cumsum()
    df["pf_cumulative"] = (df["ambient_cumsum"] + EPSILON) / (
        df["mask_cumsum"] + EPSILON
    )

    # Rolling (60 seconds, 1‑second step)
    if isinstance(df.index, pd.DatetimeIndex):
        # Time‑based window if a proper datetime index is present
        roll_mask = df["mask_particles"].rolling("60s", min_periods=1).sum()
        roll_ambient = df["ambient_particles"].rolling("60s", min_periods=1).sum()

    df["pf_rolling_60s"] = (roll_ambient + EPSILON) / (roll_mask + EPSILON)
    return df


def plot_protection_factors(df: pd.DataFrame, output_path: Path) -> None:
    """Plot cumulative and rolling PF into a two‑subplot figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # do not plot the first 10 seconds to avoid extreme values
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[df.index >= df.index[0] + pd.Timedelta(seconds=10)]
    else:
        df = df.iloc[10:]

    ax1.plot(df.index, df["pf_cumulative"])
    ax1.set_title("Cumulative Protection Factor")
    ax1.set_ylabel("Protection Factor")
    ax1.grid(True)

    ax2.plot(df.index, df["pf_rolling_60s"])
    ax2.set_title("60‑s Rolling Protection Factor")
    ax2.set_ylabel("Protection Factor")
    ax2.set_xlabel("Time" if isinstance(df.index, pd.DatetimeIndex) else "Sample")
    ax2.grid(True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cumulative and rolling protection factors from a CSV."
    )
    parser.add_argument(
        "csv",
        type=str,
        help="Path to CSV with 'mask_particles' and 'ambient_particles' columns (optionally 'timestamp').",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)

    # If there is a timestamp column, use it as the index to enable true 30‑s windows
    if "t_us" in df.columns:
        df["timestamp"] = pd.to_datetime(df["t_us"] * 1e-6, unit="s")
        df.set_index("timestamp", inplace=True)

    # Validate required columns
    required = {"mask_particles", "ambient_particles"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    # Compute PFs
    df = compute_protection_factors(df)

    # Plot
    out_file = Path("output/protection-factor") / f"{csv_path.stem}.png"
    plot_protection_factors(df, out_file)
    print(f"Plot saved to {out_file.resolve()}")  # noqa: T201 – CLI feedback only


if __name__ == "__main__":  # pragma: no cover
    main()
