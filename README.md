# Mask Sense Pilot

This repository contains a processing pipeline for analyzing respirator mask leak data.  The codebase is organized as a Python package `mask_fit_feat` which provides modules for loading data, preprocessing signals, detecting breaths, extracting features, computing protection factor (PF), training models, and generating diagnostic plots.

## Installation

Install the dependencies and the package itself:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm joblib
```

## Pipeline Overview

1. **Load a trial**: `mask_fit_feat.io.load_trial()` reads a CSV with timestamped pressure and particle measurements and returns each channel as a pandas Series.  Pressure channels are resampled to exactly 1000&nbsp;Hz; particle counts remain at 1&nbsp;Hz.
2. **Preprocess signals**: `mask_fit_feat.preprocess.bandpass()` applies a zero-phase Butterworth filter to remove drift and high-frequency noise.
3. **Detect breaths**: `mask_fit_feat.breath.detect_breath_cycles()` identifies individual breaths using zero crossings of the pressure derivative with a minimum duration filter.
4. **Extract features**: `mask_fit_feat.features` computes time‑domain, frequency‑domain, and spatial features for every breath.  `mask_fit_feat.pf.compute_pf()` appends per‑breath and rolling geometric mean PF values.
5. **Aggregate and select**: `mask_fit_feat.models.aggregate_features()` summarizes features over fixed windows and `mask_fit_feat.models.select_top_n_features()` selects informative features via mutual information.
6. **Train models**: `mask_fit_feat.models.train_random_forest()` performs group-wise cross‑validation and saves a fitted RandomForestRegressor.
7. **Visualize results**: `mask_fit_feat.viz` provides basic matplotlib plots for feature importance and prediction diagnostics.

## Command Line Interface

The repository includes a small CLI wrapper `mask_fit_run.py` which exposes the main pipeline steps.  Run it with:

```bash
python mask_fit_run.py <command> [options]
```

Available commands:

- `load <csv>` – load a trial and list available channels.
- `preprocess <csv>` – load and bandpass filter the global pressure channel.
- `extract <csv>` – run breath detection and feature extraction on a trial.
- `train <features> <labels> <groups>` – train a random forest model from prepared CSV files.

Use `python mask_fit_run.py --help` to see detailed usage for each command.

## Running Tests

Run the automated tests and code formatting checks with:

```bash
ruff check --fix mask_fit_feat tests
pytest -q
```

## Legacy Script

The older `calculate_fit_score.py` script is retained for reference.  It computes a simple protection factor from particle counts stored in `breath_db.sqlite`, shifting particle data by seven seconds to compensate for tubing lag.
