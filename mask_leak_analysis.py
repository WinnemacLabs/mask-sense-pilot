
"""mask_leak_analysis.py – Protection‑Factor regression (v1.1)

Changes from v1.0
-----------------
* **Optional --force-lag** CLI argument: skip auto cross‑correlation and use
  a fixed lag in seconds.  Useful when the tube delay is known or correlation
  is weak (e.g., perfect seal → tiny particle peaks).
* **Robust single‑file training** – if there is only one unique recording
  (group), fall back to 5‑fold K‑Fold cross‑validation inside that file.
* `np.trapz` → `np.trapezoid` to satisfy SciPy ≥1.11 deprecation.

Everything else (data ingest, feature extraction) unchanged.

Usage examples
--------------
# auto‑lag (default)
python mask_leak_analysis.py run1.csv run2.csv --participant alex --plot

# manual lag, single file
python mask_leak_analysis.py run1.csv --force-lag 2.4 --plot
"""

from __future__ import annotations
import argparse
import pathlib
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy import signal as sps, stats
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneGroupOut, KFold
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# 1. Data Loading (t_us → t_s)
# --------------------------------------------------------------------------------------

def load_data(path: str | pathlib.Path) -> tuple[pd.DataFrame, float]:
    df = pd.read_csv(path)
    if "t_us" not in df.columns:
        raise ValueError("Expected a 't_us' timestamp column in microseconds.")
    df["t_s"] = df["t_us"] * 1e-6
    fs = 1.0 / np.median(np.diff(df["t_s"]))
    return df, fs


# --------------------------------------------------------------------------------------
# 2. Lag Estimation
# --------------------------------------------------------------------------------------

def estimate_lag_seconds(df: pd.DataFrame,
                         fs: float,
                         pressure_col: str,
                         particle_col: str,
                         max_lag_s: float = 5.0) -> float:
    """Cross‑correlate |dP/dt| vs. particles; return lag in seconds."""
    dp = np.abs(np.gradient(df[pressure_col].to_numpy()))
    part = df[particle_col].to_numpy()
    # Light smoothing to reduce noise sensitivity
    if fs > 100:
        from scipy.signal import savgol_filter
        dp = savgol_filter(dp, int(fs*0.05)//2*2+1, 3)
        part = savgol_filter(part, int(fs*0.05)//2*2+1, 3)
    # z‑score
    dp = (dp - dp.mean()) / (dp.std() + 1e-9)
    part = (part - part.mean()) / (part.std() + 1e-9)
    corr = np.correlate(dp, part, mode="full")
    lags = np.arange(-len(dp)+1, len(dp))
    keep = (lags >= 0) & (lags <= int(max_lag_s*fs))
    if not np.any(keep):
        return 0.0
    best = lags[keep][np.argmax(corr[keep])]
    return best / fs


def shift_columns(df: pd.DataFrame, lag_s: float, cols: List[str]) -> pd.DataFrame:
    shifted = df.copy()
    dt = np.median(np.diff(df["t_s"]))
    shift = int(round(lag_s / dt))
    for c in cols:
        shifted[c] = shifted[c].shift(-shift).interpolate()
    return shifted


# --------------------------------------------------------------------------------------
# 3. Breath Segmentation
# --------------------------------------------------------------------------------------

def segment_breaths(df: pd.DataFrame,
                    fs: float,
                    signal_col: str = "Pa_Global",
                    lowcut: float = 0.05,
                    highcut: float = 1.0) -> List[Tuple[int,int]]:
    sos = sps.butter(4, [lowcut, highcut], btype="band", fs=fs, output="sos")
    filt = sps.sosfiltfilt(sos, df[signal_col].to_numpy())
    zero_cross = np.where(np.diff(np.signbit(np.gradient(filt))))[0]
    return [(zero_cross[i], zero_cross[i+1]) for i in range(len(zero_cross)-1)]


# --------------------------------------------------------------------------------------
# 4. Feature Engineering
# --------------------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(x)))

def extract_pressure_features(df: pd.DataFrame,
                              breaths: List[Tuple[int,int]],
                              h_col="Pa_Horizontal",
                              v_col="Pa_Vertical",
                              g_col="Pa_Global",
                              fs: float = 1.0) -> List[Dict[str,Any]]:
    feats = []
    h = df[h_col].to_numpy()
    v = df[v_col].to_numpy()
    g = df[g_col].to_numpy()
    for start, end in breaths:
        sl = slice(start, end)
        dh = h[sl] - g[sl]
        dv = v[sl] - g[sl]
        mag = np.sqrt(dh**2 + dv**2)
        d = {
            "rms_mag": _rms(mag),
            "ptp_mag": np.ptp(mag),
            "skew_mag": stats.skew(mag),
            "kurt_mag": stats.kurtosis(mag),
        }
        mid = start + (end-start)//2
        d["rms_first_half"] = _rms(mag[:mid-start])
        d["rms_second_half"] = _rms(mag[mid-start:])
        f, Pxx = sps.welch(mag, fs=fs, nperseg=min(1024, len(mag)))
        band = (f>=5) & (f<=15)
        d["power_5_15"] = np.trapezoid(Pxx[band], f[band]) if band.any() else 0.0
        feats.append(d)
    return feats


# --------------------------------------------------------------------------------------
# 5. Protection Factor computation
# --------------------------------------------------------------------------------------

def attach_pf(df: pd.DataFrame,
              breaths: List[Tuple[int,int]],
              in_col: str,
              out_col: str,
              search_window_s: float,
              fs: float) -> List[Dict[str,float]]:
    in_arr = df[in_col].to_numpy()
    out_arr = df[out_col].to_numpy()
    feats = []
    win = int(search_window_s*fs)
    for _, end in breaths:
        end_ = min(len(in_arr), end+win)
        pin = in_arr[end:end_].max()
        pout = out_arr[end:end_].max()
        pf = pout / max(pin, 1e-9)
        feats.append({"PF": pf, "logPF": np.log10(pf)})
    return feats


# --------------------------------------------------------------------------------------
# 6. DataFrame assembly
# --------------------------------------------------------------------------------------

def build_feature_dataframe(p_feats, pf_feats, participant, file_id):
    df = pd.concat([pd.DataFrame(p_feats), pd.DataFrame(pf_feats)], axis=1)
    df["participant"] = participant
    df["file"] = file_id
    return df


# --------------------------------------------------------------------------------------
# 7. Model training
# --------------------------------------------------------------------------------------

def train_regressor(df, feature_cols=None):
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {"PF","logPF","participant","file"}]
    X = df[feature_cols].to_numpy()
    y = df["logPF"].to_numpy()
    groups = df["file"].to_numpy()
    unique_groups = np.unique(groups)
    preds = np.zeros_like(y)
    if len(unique_groups) >= 2:
        splitter = LeaveOneGroupOut().split(X, y, groups)
    else:
        splitter = KFold(n_splits=5, shuffle=True, random_state=0).split(X, y)
    for train, test in splitter:
        model = make_pipeline(StandardScaler(),
                              GradientBoostingRegressor(random_state=0))
        model.fit(X[train], y[train])
        preds[test] = model.predict(X[test])
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"MAE(logPF): {mae:.3f}  |  R²: {r2:.3f}")
    return preds, mae, r2, feature_cols


# --------------------------------------------------------------------------------------
# 8. CLI
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Predict per‑breath Protection Factor from pressure.")
    ap.add_argument("csv", nargs="+", help="CSV recording(s)")
    ap.add_argument("--participant", default="anon")
    ap.add_argument("--in-col", default="Conc_In", help="in‑mask particle column name")
    ap.add_argument("--out-col", default="Conc_Out", help="out‑mask particle column name")
    ap.add_argument("--force-lag", type=float, default=None, help="manually set lag in seconds, bypass auto estimation")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    dfs = []
    for f in args.csv:
        df, fs = load_data(f)
        print(f"Loaded {f}  |  fs ≈ {fs:.1f} Hz")
        if args.force_lag is not None:
            lag = args.force_lag
            print(f"   using user‑supplied lag: {lag:.2f} s")
        else:
            lag = estimate_lag_seconds(df, fs, "Pa_Global", args.in_col)
            print(f"   estimated lag: {lag:.2f} s")
        df = shift_columns(df, lag, [args.in_col, args.out_col])
        breaths = segment_breaths(df, fs)
        p_feats = extract_pressure_features(df, breaths, fs=fs)
        pf_feats = attach_pf(df, breaths, args.in_col, args.out_col, 2.0, fs)
        fid = pathlib.Path(f).stem
        dfs.append(build_feature_dataframe(p_feats, pf_feats, args.participant, fid))

    df_all = pd.concat(dfs, ignore_index=True)
    preds, mae, r2, _ = train_regressor(df_all)

    outdir = pathlib.Path("results"); outdir.mkdir(exist_ok=True)
    df_all.to_parquet(outdir/"features.parquet")
    print("Saved features to results/features.parquet")

    if args.plot:
        plt.figure()
        plt.scatter(df_all["logPF"], preds, s=10, alpha=0.6)
        lims = [df_all["logPF"].min()-0.2, df_all["logPF"].max()+0.2]
        plt.plot(lims, lims, "--")
        plt.xlabel("Measured log10(PF)")
        plt.ylabel("Predicted log10(PF)")
        plt.title(f"logPF scatter  |  MAE={mae:.2f}  R²={r2:.2f}")
        plt.savefig(outdir/"scatter.png", dpi=150)
        print("Scatter plot saved to results/scatter.png")

if __name__ == "__main__":
    main()
