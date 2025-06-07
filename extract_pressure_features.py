#!/usr/bin/env python3
"""
breath_feature_extractor_v2.py
---------------------------------
Extends the original script with:
  • ΔH / ΔV / magnitude statistics
  • HF (5–15 Hz) band-power
  • Breath asymmetry metrics
  • H-V correlation
  • Heart-beat amplitude + HR at end-exhale

Author: Winnemac Labs – Alexander Curtiss & ChatGPT (2025-06-01)
"""

import argparse, sqlite3, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy import signal as sps       # NEW
import calculate_fit_score            # your existing module

# ------------------------------------------------------------------- helpers
def zero_crossings(arr: np.ndarray) -> int:
    arr = np.asarray(arr)
    return int(((arr[:-1] * arr[1:]) < 0).sum())


def band_power(arr: np.ndarray, fs: float, low: float, high: float) -> float:
    """Welch PSD integral between low–high Hz."""
    if len(arr) < 4:
        return np.nan
    f, Pxx = sps.welch(arr, fs=fs, nperseg=min(1024, len(arr)))
    band = (f >= low) & (f <= high)
    return float(np.trapezoid(Pxx[band], f[band]))


def cardiogenic_metrics(p_global: np.ndarray,
                        t: np.ndarray,
                        fs: float) -> tuple[float, float]:
    """Return (HB_amp_RMS, HR_bpm) from last 25 % of the breath."""
    if len(p_global) < 8:
        return np.nan, np.nan
    idx_start = int(0.75 * len(p_global))
    seg = p_global[idx_start:] - np.mean(p_global[idx_start:])
    # 0.8–3 Hz band-pass (Butterworth 2nd order)
    sos = sps.butter(2, [0.8, 3.0], btype="band", fs=fs, output="sos")
    hb = sps.sosfiltfilt(sos, seg)
    hb_amp = np.sqrt(np.mean(hb**2))
    # Peak-based HR
    peaks, _ = sps.find_peaks(hb, prominence=0.02 * np.std(hb))
    if len(peaks) >= 2:
        ibi = np.diff(t[idx_start:][peaks])          # inter-beat intervals (s)
        hr = 60.0 / np.median(ibi) if np.all(ibi > 0) else np.nan
    else:
        hr = np.nan
    return hb_amp, hr


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Extract per-breath pressure features (v2).")
    ap.add_argument("--db", default="breath_db.sqlite",
                    help="Input SQLite database path")
    ap.add_argument("--output", default=None,
                    help="Output SQLite database path (default: overwrite input)")
    args = ap.parse_args()

    db_path = args.db
    out_path = args.output or db_path

    # ---------------------------------------------------------------- ensure PF
    with sqlite3.connect(db_path) as conn:
        df_chk = pd.read_sql_query("SELECT * FROM breath_data LIMIT 1", conn)
        if "protection_factor" not in df_chk.columns:
            print("protection_factor missing → running calculate_fit_score.compute_and_store()")
            calculate_fit_score.compute_and_store(conn)

    # ---------------------------------------------------------------- load all
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM breath_data", conn)

    # ---------------------------------------------------------------- feature loop
    features = []
    for (src, breath_id), grp in df.groupby(["source_file", "breath"]):
        if pd.isna(breath_id):
            continue

        feat = {"source_file": src,
                "breath": int(breath_id),
                "n_samples": len(grp)}

        # timestamps & duration
        t = grp["t_us"].to_numpy(dtype=float) * 1e-6
        feat["t_start"], feat["t_end"] = float(t[0]), float(t[-1])
        fs = 1.0 / np.median(np.diff(t)) if len(t) > 1 else np.nan
        feat["duration_s"] = float(t[-1] - t[0]) if len(t) > 1 else np.nan

        # raw channels
        g = grp["Pa_Global"].to_numpy(float)
        h = grp["Pa_Horizontal"].to_numpy(float)
        v = grp["Pa_Vertical"].to_numpy(float)

        # per-channel classical stats
        for col, arr in zip(["Pa_Global", "Pa_Horizontal", "Pa_Vertical"],
                            [g, h, v]):
            feat[f"{col}_mean"]      = np.nanmean(arr)
            feat[f"{col}_std"]       = np.nanstd(arr)
            feat[f"{col}_ptp"]       = np.ptp(arr)
            feat[f"{col}_min"]       = np.nanmin(arr)
            feat[f"{col}_max"]       = np.nanmax(arr)
            feat[f"{col}_median"]    = np.nanmedian(arr)
            feat[f"{col}_skew"]      = skew(arr, nan_policy='omit')
            feat[f"{col}_kurtosis"]  = kurtosis(arr, nan_policy='omit')
            feat[f"{col}_auc"]       = np.trapezoid(arr, t) if len(arr) > 1 else np.nan
            feat[f"{col}_zero_cross"] = zero_crossings(arr)

        # ---------------- new derived signals
        dh = h - g
        dv = v - g
        mag = np.sqrt(dh**2 + dv**2)

        feat["dH_mean"]  = np.nanmean(dh)
        feat["dV_mean"]  = np.nanmean(dv)
        feat["mag_rms"]  = np.sqrt(np.mean(mag**2))
        feat["mag_ptp"]  = np.ptp(mag)
        feat["mag_skew"] = skew(mag, nan_policy='omit')
        feat["mag_kurt"] = kurtosis(mag, nan_policy='omit')

        # asymmetry (first vs second half RMS)
        mid = len(mag) // 2
        feat["mag_rms_first"]  = np.sqrt(np.mean(mag[:mid]**2))
        feat["mag_rms_second"] = np.sqrt(np.mean(mag[mid:]**2))
        feat["mag_rms_ratio"]  = (feat["mag_rms_first"] /
                                  feat["mag_rms_second"] if feat["mag_rms_second"] else np.nan)

        # high-frequency turbulence power (5–15 Hz) of magnitude
        feat["mag_pow_5_15"] = band_power(mag - np.mean(mag), fs, 5, 15) if fs else np.nan

        # correlation between H & V residuals
        if len(dh) > 3 and len(dv) > 3:
            feat["corr_dH_dV"] = np.corrcoef(dh, dv)[0, 1]
        else:
            feat["corr_dH_dV"] = np.nan

        # cardiogenic ripple metrics
        hb_amp, hr_bpm = cardiogenic_metrics(g, t, fs)
        feat["HB_amp"] = hb_amp
        feat["HR_bpm"] = hr_bpm

        # protection factor (first non-NaN)
        pf = grp["protection_factor"].dropna()
        feat["protection_factor"] = float(pf.iloc[0]) if not pf.empty else np.nan

        features.append(feat)

    features_df = pd.DataFrame(features)

    # ---------------------------------------------------------- write back
    with sqlite3.connect(out_path) as conn:
        features_df.to_sql("breath_features", conn, if_exists="replace", index=False)

    print(f"Wrote {len(features_df)} rows to {out_path}  (table: breath_features)")


if __name__ == "__main__":
    main()
