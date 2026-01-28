#!/usr/bin/env python3
"""
Pressure-to-Protection-Factor Analysis

Proof of concept: Determine whether pressure sensor measurements can predict
particle-based Protection Factor, establishing feasibility for pressure-only
mask fit assessment.

Usage:
    python analysis/pressure_pf_analysis.py

Outputs saved to analysis/outputs/
"""

import re
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.preprocessing import StandardScaler

# Constants
EPSILON = 1e-9
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Breathing protocols to analyze (exclude zeroing, frc_reset, etc.)
BREATHING_PROTOCOLS = {"quiet_breathing", "deep_breathing", "rainbow"}


def parse_metadata_from_header(filepath: Path) -> dict:
    """Parse metadata from aligned CSV comment headers."""
    metadata = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            match = re.match(r"#\s*(\w[\w\s]*?):\s*(.+)", line)
            if match:
                key = match.group(1).strip().lower().replace(" ", "_")
                metadata[key] = match.group(2).strip()
    return metadata


def parse_metadata_from_filename(filepath: Path) -> dict:
    """Parse metadata from filename as fallback.

    Expected format: rsc_P{participant}_{mask}[_leak]_{exercise}_{date}[_fixed]_aligned.csv
    """
    name = filepath.stem  # Remove .csv
    name = name.replace("_aligned", "").replace("_fixed", "")

    # Pattern: rsc_P{n}_{mask}[_leak]_{exercise}_{date}
    match = re.match(r"rsc_(P\d+)_([A-Z]+)(?:_(leak))?_(.+)_(\d{8})(?:_\d+)?", name)
    if not match:
        return {}

    participant, mask, leak, exercise, date = match.groups()

    return {
        "participant": participant,
        "mask_type": mask,
        "fit_condition": "leak" if leak else "no_leak",
        "exercise": exercise,
        "date": date,
    }


def load_aligned_csv(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """Load aligned CSV with metadata."""
    metadata = parse_metadata_from_header(filepath)
    df = pd.read_csv(filepath, comment="#")
    return df, metadata


def find_breathing_files() -> list[dict]:
    """Find all aligned breathing protocol files."""
    files = []
    for aligned_csv in DATA_DIR.glob("*/aligned/*_aligned.csv"):
        # Try header metadata first, fall back to filename parsing
        metadata = parse_metadata_from_header(aligned_csv)
        if not metadata.get("exercise"):
            metadata = parse_metadata_from_filename(aligned_csv)

        exercise = metadata.get("exercise", "")

        # Skip non-breathing protocols
        if exercise not in BREATHING_PROTOCOLS:
            continue

        # Skip _fixed duplicates if non-fixed version exists
        if "_fixed_aligned" in aligned_csv.name:
            non_fixed = aligned_csv.parent / aligned_csv.name.replace("_fixed_aligned", "_aligned")
            if non_fixed.exists():
                continue

        files.append({
            "filepath": aligned_csv,
            "participant": metadata.get("participant", ""),
            "mask": metadata.get("mask_type", ""),
            "condition": metadata.get("fit_condition", ""),
            "exercise": exercise,
        })

    return files


def extract_pressure_features(df: pd.DataFrame) -> dict:
    """Extract pressure features from a recording."""
    features = {}

    pressure_cols = ["Pa_Global", "Pa_Vertical", "Pa_Horizontal"]

    # Basic statistics per channel
    for col in pressure_cols:
        if col not in df.columns:
            continue
        data = df[col].dropna().values
        if len(data) == 0:
            continue

        features[f"{col}_mean"] = np.mean(data)
        features[f"{col}_std"] = np.std(data)
        features[f"{col}_min"] = np.min(data)
        features[f"{col}_max"] = np.max(data)
        features[f"{col}_range"] = np.max(data) - np.min(data)
        features[f"{col}_skew"] = stats.skew(data)
        features[f"{col}_kurtosis"] = stats.kurtosis(data)

        # Percentiles
        features[f"{col}_p10"] = np.percentile(data, 10)
        features[f"{col}_p90"] = np.percentile(data, 90)

    # Breathing dynamics from Pa_Global (primary signal)
    if "Pa_Global" in df.columns and "t_us_zero" in df.columns:
        pa = df["Pa_Global"].values
        t_us = df["t_us_zero"].values

        # Estimate sample rate
        dt = np.median(np.diff(t_us)) / 1e6  # seconds
        fs = 1 / dt if dt > 0 else 1000

        # Low-pass filter for breathing detection
        try:
            nyq = fs / 2
            cutoff = min(1.5, nyq * 0.9)  # 1.5 Hz - breathing is typically 0.1-0.5 Hz
            b, a = butter(4, cutoff / nyq, btype="low")
            pa_filtered = filtfilt(b, a, pa)

            # Adaptive prominence threshold based on signal range
            # Use interquartile range to be robust to outliers
            signal_iqr = np.percentile(pa_filtered, 75) - np.percentile(pa_filtered, 25)
            prominence_threshold = max(signal_iqr * 0.3, 2.0)  # At least 30% of IQR or 2 Pa

            # Min distance: assume breathing rate between 6-40 breaths/min
            # That's 1.5-10 seconds per breath, use 1.5s minimum
            min_distance = int(fs * 1.5)

            # Find negative peaks (inhalation creates negative pressure in mask)
            neg_peaks, neg_props = find_peaks(
                -pa_filtered,
                prominence=prominence_threshold,
                distance=min_distance
            )

            # Also find positive peaks (exhalation)
            pos_peaks, pos_props = find_peaks(
                pa_filtered,
                prominence=prominence_threshold,
                distance=min_distance
            )

            # Use the more reliable of the two (should be similar)
            n_breaths = max(len(neg_peaks), len(pos_peaks))

            if n_breaths > 1:
                duration_s = (t_us[-1] - t_us[0]) / 1e6
                features["breathing_rate_bpm"] = n_breaths / duration_s * 60

                # Breath amplitude: peak-to-trough for each breath cycle
                # Use negative peak prominences as they represent inhalation depth
                if len(neg_props.get("prominences", [])) > 0:
                    neg_amplitudes = neg_props["prominences"]
                    features["breath_amplitude_mean"] = np.mean(neg_amplitudes)
                    features["breath_amplitude_std"] = np.std(neg_amplitudes)
                    features["breath_amplitude_cv"] = np.std(neg_amplitudes) / (np.mean(neg_amplitudes) + EPSILON)

                # Full breath amplitude (peak to trough)
                if len(pos_peaks) > 0 and len(neg_peaks) > 0:
                    # Estimate full amplitude from signal range within breathing band
                    features["breath_full_amplitude"] = (
                        np.mean(pa_filtered[pos_peaks]) - np.mean(pa_filtered[neg_peaks])
                    )

        except Exception:
            pass  # Skip if filtering fails

        # Integral features
        features["auc_positive"] = np.sum(np.maximum(pa, 0)) * dt if 'dt' in dir() else 0
        features["auc_negative"] = np.sum(np.minimum(pa, 0)) * dt if 'dt' in dir() else 0
        features["auc_ratio"] = abs(features.get("auc_positive", 0)) / (abs(features.get("auc_negative", 0)) + EPSILON)

    # Multi-axis features
    if all(col in df.columns for col in pressure_cols):
        pa_g = df["Pa_Global"].values
        pa_v = df["Pa_Vertical"].values
        pa_h = df["Pa_Horizontal"].values

        # Correlations between channels
        features["corr_global_vertical"] = np.corrcoef(pa_g, pa_v)[0, 1]
        features["corr_global_horizontal"] = np.corrcoef(pa_g, pa_h)[0, 1]
        features["corr_vertical_horizontal"] = np.corrcoef(pa_v, pa_h)[0, 1]

        # RMS of 3D pressure vector
        features["pressure_rms"] = np.sqrt(np.mean(pa_g**2 + pa_v**2 + pa_h**2))

    return features


def compute_protection_factor(df: pd.DataFrame) -> Optional[float]:
    """Compute cumulative protection factor from particle data."""
    if "mask_particles" not in df.columns or "ambient_particles" not in df.columns:
        return None

    mask = df["mask_particles"].dropna()
    ambient = df["ambient_particles"].dropna()

    if len(mask) == 0 or len(ambient) == 0:
        return None

    mask_sum = mask.sum()
    ambient_sum = ambient.sum()

    # Skip if no meaningful particle data
    if mask_sum < EPSILON and ambient_sum < EPSILON:
        return None

    pf = (ambient_sum + EPSILON) / (mask_sum + EPSILON)
    return pf


def build_dataset() -> pd.DataFrame:
    """Build dataset with pressure features and protection factors."""
    print("Finding breathing protocol files...")
    files = find_breathing_files()
    print(f"Found {len(files)} files")

    records = []
    for file_info in files:
        filepath = file_info["filepath"]
        try:
            df, _ = load_aligned_csv(filepath)

            # Extract features
            features = extract_pressure_features(df)

            # Compute protection factor
            pf = compute_protection_factor(df)
            if pf is None:
                continue

            # Combine metadata, features, and target
            record = {
                "participant": file_info["participant"],
                "mask": file_info["mask"],
                "condition": file_info["condition"],
                "exercise": file_info["exercise"],
                "filepath": str(filepath),
                "protection_factor": pf,
                "log_pf": np.log10(pf + 1),  # Log-transformed PF
                **features,
            }
            records.append(record)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    print(f"Built dataset with {len(records)} valid recordings")
    return pd.DataFrame(records)


def analyze_correlations(df: pd.DataFrame) -> pd.Series:
    """Compute correlations between pressure features and protection factor."""
    # Get feature columns (exclude metadata and target)
    exclude_cols = {"participant", "mask", "condition", "exercise", "filepath",
                    "protection_factor", "log_pf"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    correlations = {}
    for col in feature_cols:
        valid = df[[col, "log_pf"]].dropna()
        if len(valid) > 5:
            r, p = stats.pearsonr(valid[col], valid["log_pf"])
            correlations[col] = r

    return pd.Series(correlations).sort_values(key=abs, ascending=False)


def plot_correlation_heatmap(df: pd.DataFrame, correlations: pd.Series, output_path: Path):
    """Plot correlation heatmap of top features."""
    # Select top 15 features by absolute correlation
    top_features = correlations.head(15).index.tolist()

    # Build correlation matrix
    cols_to_plot = top_features + ["log_pf"]
    corr_matrix = df[cols_to_plot].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax)
    ax.set_title("Correlation Matrix: Top Pressure Features vs Log(PF)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved correlation heatmap to {output_path}")


def plot_top_feature_scatter(df: pd.DataFrame, correlations: pd.Series, output_path: Path):
    """Plot scatter plots of top correlated features vs PF."""
    top_features = correlations.head(6).index.tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, feature in zip(axes, top_features):
        valid = df[[feature, "log_pf", "condition"]].dropna()

        for condition in valid["condition"].unique():
            subset = valid[valid["condition"] == condition]
            marker = "o" if condition == "no_leak" else "x"
            ax.scatter(subset[feature], subset["log_pf"], label=condition,
                      marker=marker, alpha=0.7)

        r = correlations[feature]
        ax.set_xlabel(feature)
        ax.set_ylabel("log10(PF)")
        ax.set_title(f"r = {r:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Top Correlated Pressure Features vs Protection Factor", y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature scatter plots to {output_path}")


def train_regression_model(df: pd.DataFrame) -> dict:
    """Train regression model with leave-one-participant-out cross-validation."""
    # Get feature columns
    exclude_cols = {"participant", "mask", "condition", "exercise", "filepath",
                    "protection_factor", "log_pf"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Add mask type encoding
    df_model = df.copy()
    df_model["mask_AURA"] = (df["mask"] == "AURA").astype(int)
    df_model["mask_MAKTEK"] = (df["mask"] == "MAKTEK").astype(int)

    # Selected features for optimized model (top discriminating + mask)
    selected_features = [
        "Pa_Global_p10", "breath_full_amplitude", "breath_amplitude_mean",
        "Pa_Global_std", "pressure_rms", "Pa_Global_max", "Pa_Global_p90",
        "Pa_Horizontal_p90", "corr_global_horizontal", "auc_positive",
        "mask_AURA", "mask_MAKTEK"
    ]
    # Filter to features that exist in the dataframe
    selected_features = [f for f in selected_features if f in df_model.columns]

    # Prepare data
    X = df_model[feature_cols].fillna(0).values
    X_selected = df_model[selected_features].fillna(0).values
    y = df_model["log_pf"].values
    groups = df_model["participant"].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_selected_scaled = scaler.fit_transform(X_selected)

    # Leave-one-participant-out CV
    logo = LeaveOneGroupOut()

    results = {}

    # Ridge regression (all features)
    ridge = Ridge(alpha=1.0)
    y_pred_ridge = cross_val_predict(ridge, X_scaled, y, cv=logo, groups=groups)

    ridge_r2 = 1 - np.sum((y - y_pred_ridge)**2) / np.sum((y - np.mean(y))**2)
    ridge_rmse = np.sqrt(np.mean((y - y_pred_ridge)**2))
    ridge_mae = np.mean(np.abs(y - y_pred_ridge))

    results["ridge"] = {
        "r2": ridge_r2,
        "rmse": ridge_rmse,
        "mae": ridge_mae,
        "y_pred": y_pred_ridge,
    }

    # Random Forest (all features)
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=logo, groups=groups)

    rf_r2 = 1 - np.sum((y - y_pred_rf)**2) / np.sum((y - np.mean(y))**2)
    rf_rmse = np.sqrt(np.mean((y - y_pred_rf)**2))
    rf_mae = np.mean(np.abs(y - y_pred_rf))

    results["rf"] = {
        "r2": rf_r2,
        "rmse": rf_rmse,
        "mae": rf_mae,
        "y_pred": y_pred_rf,
    }

    # Optimized GBM (selected features + mask type)
    gbm = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42
    )
    y_pred_gbm = cross_val_predict(gbm, X_selected_scaled, y, cv=logo, groups=groups)

    gbm_r2 = 1 - np.sum((y - y_pred_gbm)**2) / np.sum((y - np.mean(y))**2)
    gbm_rmse = np.sqrt(np.mean((y - y_pred_gbm)**2))
    gbm_mae = np.mean(np.abs(y - y_pred_gbm))

    results["gbm_optimized"] = {
        "r2": gbm_r2,
        "rmse": gbm_rmse,
        "mae": gbm_mae,
        "y_pred": y_pred_gbm,
    }

    # Fit GBM on all data for feature importance
    gbm.fit(X_selected_scaled, y)
    results["feature_importance"] = pd.Series(
        gbm.feature_importances_, index=selected_features
    ).sort_values(ascending=False)

    results["y_true"] = y
    results["condition"] = df["condition"].values
    results["feature_cols"] = feature_cols
    results["selected_features"] = selected_features

    return results


def plot_regression_results(df: pd.DataFrame, results: dict, output_path: Path):
    """Plot predicted vs actual protection factor."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    y_true = results["y_true"]
    condition = results.get("condition", ["unknown"] * len(y_true))

    models = [
        ("ridge", "Ridge Regression"),
        ("rf", "Random Forest"),
        ("gbm_optimized", "GBM Optimized")
    ]

    for ax, (name, label) in zip(axes, models):
        if name not in results:
            continue
        y_pred = results[name]["y_pred"]
        r2 = results[name]["r2"]

        # Color by condition
        colors = ["blue" if c == "no_leak" else "orange" for c in condition]
        ax.scatter(y_true, y_pred, c=colors, alpha=0.6)

        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", label="Perfect prediction")

        ax.set_xlabel("Actual log10(PF)")
        ax.set_ylabel("Predicted log10(PF)")
        ax.set_title(f"{label}\nR² = {r2:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Leave-One-Participant-Out Cross-Validation Results", y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved regression results to {output_path}")


def plot_feature_importance(results: dict, output_path: Path):
    """Plot feature importance from GBM."""
    importance = results["feature_importance"]

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Feature Importance (GBM Optimized)")
    ax.invert_yaxis()
    plt.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved feature importance to {output_path}")


def plot_condition_comparison(df: pd.DataFrame, output_path: Path):
    """Plot protection factor distribution by condition."""
    fig, ax = plt.subplots(figsize=(8, 5))

    df.boxplot(column="log_pf", by="condition", ax=ax)
    ax.set_xlabel("Fit Condition")
    ax.set_ylabel("log10(Protection Factor)")
    ax.set_title("Protection Factor by Fit Condition")
    plt.suptitle("")  # Remove automatic title

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved condition comparison to {output_path}")


def main():
    """Run the full analysis pipeline."""
    print("=" * 60)
    print("Pressure-to-Protection-Factor Analysis")
    print("=" * 60)

    # Build dataset
    df = build_dataset()

    if len(df) < 10:
        print("ERROR: Not enough valid recordings for analysis")
        return

    # Save dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "dataset.csv", index=False)
    print(f"Saved dataset to {OUTPUT_DIR / 'dataset.csv'}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total recordings: {len(df)}")
    print(f"Participants: {df['participant'].nunique()} ({', '.join(sorted(df['participant'].unique()))})")
    print(f"Masks: {', '.join(sorted(df['mask'].unique()))}")
    print(f"Conditions: {df['condition'].value_counts().to_dict()}")
    print(f"Exercises: {df['exercise'].value_counts().to_dict()}")
    print(f"\nProtection Factor range: {df['protection_factor'].min():.1f} - {df['protection_factor'].max():.1f}")
    print(f"Log10(PF) range: {df['log_pf'].min():.2f} - {df['log_pf'].max():.2f}")

    # Correlation analysis
    print("\n" + "=" * 60)
    print("Correlation Analysis")
    print("=" * 60)
    correlations = analyze_correlations(df)
    print("\nTop 10 features correlated with log(PF):")
    for feat, r in correlations.head(10).items():
        print(f"  {feat}: r = {r:.3f}")

    # Plot correlations
    plot_correlation_heatmap(df, correlations, OUTPUT_DIR / "correlation_heatmap.png")
    plot_top_feature_scatter(df, correlations, OUTPUT_DIR / "feature_scatter.png")
    plot_condition_comparison(df, OUTPUT_DIR / "condition_comparison.png")

    # Regression modeling
    print("\n" + "=" * 60)
    print("Regression Modeling (Leave-One-Participant-Out CV)")
    print("=" * 60)
    results = train_regression_model(df)

    print("\nRidge Regression (all features):")
    print(f"  R² = {results['ridge']['r2']:.3f}")
    print(f"  RMSE = {results['ridge']['rmse']:.3f}")
    print(f"  MAE = {results['ridge']['mae']:.3f}")

    print("\nRandom Forest (all features):")
    print(f"  R² = {results['rf']['r2']:.3f}")
    print(f"  RMSE = {results['rf']['rmse']:.3f}")
    print(f"  MAE = {results['rf']['mae']:.3f}")

    print("\nGBM Optimized (selected features + mask type):")
    print(f"  R² = {results['gbm_optimized']['r2']:.3f}")
    print(f"  RMSE = {results['gbm_optimized']['rmse']:.3f}")
    print(f"  MAE = {results['gbm_optimized']['mae']:.3f}")

    # Plot results
    plot_regression_results(df, results, OUTPUT_DIR / "regression_results.png")
    plot_feature_importance(results, OUTPUT_DIR / "feature_importance.png")

    # Interpretation
    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)

    best_r2 = max(results["ridge"]["r2"], results["rf"]["r2"], results["gbm_optimized"]["r2"])

    if best_r2 < 0.3:
        strength = "WEAK"
        interpretation = "Pressure features show limited predictive power for PF."
    elif best_r2 < 0.5:
        strength = "MODERATE"
        interpretation = "Pressure features show meaningful correlation with PF."
    else:
        strength = "STRONG"
        interpretation = "Pressure + mask type are strong predictors of PF."

    print(f"\nEvidence strength: {strength}")
    print(f"Best R² = {best_r2:.3f}")
    print(f"\n{interpretation}")

    # Prediction accuracy in real units
    y_pred = results["gbm_optimized"]["y_pred"]
    y_true = results["y_true"]
    pf_ratio = 10**y_pred / 10**y_true
    print(f"\nPrediction accuracy (GBM):")
    print(f"  Predicted/Actual PF ratio: median={np.median(pf_ratio):.2f}, IQR=[{np.percentile(pf_ratio, 25):.2f}, {np.percentile(pf_ratio, 75):.2f}]")

    print("\nTop 5 most important features (GBM):")
    for feat, imp in results["feature_importance"].head(5).items():
        print(f"  {feat}: {imp:.3f}")

    print("\n" + "=" * 60)
    print(f"Analysis complete. Outputs saved to {OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
