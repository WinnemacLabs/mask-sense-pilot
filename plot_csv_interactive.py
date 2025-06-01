import argparse
import sys

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "browser"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize mask sense CSV data with an interactive Plotly graph"
    )
    parser.add_argument("csv", help="Input CSV file from the logger")

    args = parser.parse_args()

    try:
        df = pd.read_csv(args.csv)
    except FileNotFoundError:
        sys.exit(f"File not found: {args.csv}")

    if "t_us" not in df.columns:
        sys.exit("CSV missing required 't_us' column")

    df["time_s"] = df["t_us"] / 1e6

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Pressure (Pa)", "Particles"),
    )

    pressure_cols = [
        ("Pa_Global", "blue"),
        ("Pa_Vertical", "red"),
        ("Pa_Horizontal", "green"),
    ]
    for col, color in pressure_cols:
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=df["time_s"], y=df[col], mode="lines", name=col, line=dict(color=color)),
                row=1,
                col=1,
            )

    particle_cols = []
    if "mask_particles" in df.columns:
        particle_cols.append(("mask_particles", "purple"))
    if "ambient_particles" in df.columns:
        particle_cols.append(("ambient_particles", "orange"))

    for col, color in particle_cols:
        fig.add_trace(
            go.Scatter(x=df["time_s"], y=df[col], mode="lines", name=col, line=dict(color=color)),
            row=2,
            col=1,
        )

    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=1, col=1)
    fig.update_yaxes(title_text="Particles", row=2, col=1)

    fig.update_layout(height=600, legend_title_text="Signals")

    fig.show()


if __name__ == "__main__":
    main()
