#!/usr/bin/env python3
"""
compare_peaks_troughs.py
------------------------
For each source file in the database, compare the timing of mask_particles peaks and Pa_Global troughs.
Applies a phase-preserving bandpass filter (1/30 Hz to 1/2 Hz, i.e., 0.033 to 0.5 Hz) to both signals.
Prints summary statistics and saves a plot for each file.
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

# Bandpass filter: 1/30 Hz to 1/2 Hz
LOW_CUT = 1/30  # Hz
HIGH_CUT = 1/2  # Hz
ORDER = 4

def bandpass_filter(data, fs, low=LOW_CUT, high=HIGH_CUT, order=ORDER):
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def main():
    db_path = "breath_db.sqlite"
    outdir = Path("output/peak-trough-comparison")
    outdir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source_file FROM breath_data")
    files = [row[0] for row in cur.fetchall()]
    print(f"Found {len(files)} files.")
    for file in files:
        df = pd.read_sql_query(
            "SELECT t_us, Pa_Global, mask_particles, t_us_particles FROM breath_data WHERE source_file=? ORDER BY t_us",
            conn, params=(file,)
        )
        if df.empty or df["mask_particles"].isnull().all() or df["Pa_Global"].isnull().all():
            print(f"Skipping {file} (no data)")
            continue
        # Use t_us for pressure, t_us_particles for mask_particles if available
        t_p = df["t_us"].values * 1e-6
        t_m = df["t_us_particles"].values * 1e-6 if "t_us_particles" in df.columns and not df["t_us_particles"].isnull().all() else t_p
        pa = df["Pa_Global"].values
        mp = df["mask_particles"].values
        # Remove NaNs
        mask = ~np.isnan(pa) & ~np.isnan(mp)
        t_p = t_p[mask]
        pa = pa[mask]
        t_m = t_m[mask]
        mp = mp[mask]
        # Estimate fs (use median diff)
        fs = 1.0 / np.median(np.diff(t_p))
        # Filter
        pa_filt = bandpass_filter(pa, fs)
        mp_filt = bandpass_filter(mp, fs)
        # Find troughs in pressure (minima)
        troughs, _ = find_peaks(-pa_filt)
        # Find peaks in mask_particles (maxima)
        peaks, _ = find_peaks(mp_filt)
        # For each mask_particles peak, find nearest pressure trough
        peak_times = t_m[peaks]
        trough_times = t_p[troughs]
        if len(peak_times) == 0 or len(trough_times) == 0:
            print(f"No peaks/troughs found for {file}")
            continue
        nearest_troughs = np.array([trough_times[np.argmin(np.abs(trough_times - pt))] for pt in peak_times])
        time_diffs = peak_times - nearest_troughs
        # Print stats
        print(f"{file}: N={len(time_diffs)} | mean={np.mean(time_diffs):.3f}s | std={np.std(time_diffs):.3f}s | median={np.median(time_diffs):.3f}s")
        # Plot
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(t_p, pa_filt, label="Pa_Global (filtered)", color="tab:blue")
        ax1.plot(t_p[troughs], pa_filt[troughs], "o", color="navy", label="Pa_Global Troughs")
        ax2 = ax1.twinx()
        ax2.plot(t_m, mp_filt, label="Mask Particles (filtered, aligned)", color="tab:red", alpha=0.7)
        ax2.plot(t_m[peaks], mp_filt[peaks], "o", color="darkred", label="Mask Particles Peaks (aligned)")
        # Also plot mask_particles using original t_us (unaligned) for comparison
        if "t_us_particles" in df.columns and not df["t_us_particles"].isnull().all():
            t_orig = df["t_us"].values * 1e-6
            ax2.plot(t_orig, mp_filt, label="Mask Particles (filtered, original t_us)", color="purple", alpha=0.4, linestyle='--')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pressure (Pa)", color="tab:blue")
        ax2.set_ylabel("Mask Particles", color="tab:red")
        ax1.set_title(f"{file}\nPeak-Trough Δt: mean={np.mean(time_diffs):.3f}s, std={np.std(time_diffs):.3f}s")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        fig.savefig(outdir / f"{Path(file).stem}_peak_trough_compare.png")
        plt.close(fig)
    conn.close()
    print(f"Plots and stats saved in {outdir}")

if __name__ == "__main__":
    main()
