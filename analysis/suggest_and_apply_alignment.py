#!/usr/bin/env python3
"""
suggest_and_apply_alignment.py
------------------------------
For each file in the database, suggest a new alignment for mask_particles:
- For each mask_particles sample, find the preceding Pa_Global trough (>2s before).
- Compute the lag as the difference between t_us_particles and the preceding trough.
- Show a plot for user confirmation.
- If confirmed, update t_us_particles in the database for that file.
"""
import sqlite3
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks
import matplotlib.pyplot as plt
from pathlib import Path

DB_PATH = "breath_db.sqlite"
TROUGH_MIN_GAP = 2.0  # seconds
ORDER = 4
LOW_CUT = 1/30
HIGH_CUT = 1/2

def bandpass_filter(data, fs, low=LOW_CUT, high=HIGH_CUT, order=ORDER):
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, data)

def get_preceding_troughs(t_us, trough_us, min_gap_s):
    # For each t_us, find the last trough_us that is at least min_gap_s before t_us
    trough_idx = np.searchsorted(trough_us, t_us, side='right') - 1
    valid = (trough_idx >= 0) & ((t_us - trough_us[trough_idx]) >= min_gap_s * 1e6)
    preceding_trough_us = np.where(valid, trough_us[trough_idx], np.nan)
    return preceding_trough_us

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT source_file FROM breath_data")
    files = [row[0] for row in cur.fetchall()]
    print(f"Found {len(files)} files.")
    for file in files:
        print(f"\nProcessing {file}")
        df = pd.read_sql_query(
            "SELECT rowid, t_us, t_us_particles, Pa_Global, mask_particles FROM breath_data WHERE source_file=? ORDER BY t_us",
            conn, params=(file,)
        )
        if df.empty or df["mask_particles"].isnull().all() or df["Pa_Global"].isnull().all():
            print(f"  Skipping (no data)")
            continue
        t_us = df["t_us"].values
        pa = df["Pa_Global"].values
        mp = df["mask_particles"].values
        fs = 1.0 / np.median(np.diff(t_us)) * 1e6  # Hz
        pa_filt = bandpass_filter(pa, fs)
        # Find troughs in filtered pressure
        troughs, _ = find_peaks(-pa_filt)
        trough_us = t_us[troughs]
        # Find peaks in mask_particles
        mp_peaks, _ = find_peaks(mp)
        mp_peak_us = t_us[mp_peaks]
        # Search for optimal lag (-10 to -2s, negative only) that minimizes sum of distances between shifted mask_particles peaks and nearest Pa_Global troughs
        lag_range = np.linspace(-10.0, -2.0, 161)  # 0.05s steps, negative lags only
        min_total_dist = None
        best_lag = None
        for lag in lag_range:
            shifted_peaks = mp_peak_us + lag * 1e6  # lag is negative, so this shifts peaks backward
            # For each shifted peak, find nearest trough
            idx = np.searchsorted(trough_us, shifted_peaks)
            dists = []
            for i, s in enumerate(shifted_peaks):
                # Check both left and right troughs
                left = trough_us[idx[i]-1] if idx[i] > 0 else trough_us[0]
                right = trough_us[idx[i]] if idx[i] < len(trough_us) else trough_us[-1]
                d = min(abs(s - left), abs(s - right))
                dists.append(d)
            total_dist = np.sum(dists)
            if (min_total_dist is None) or (total_dist < min_total_dist):
                min_total_dist = total_dist
                best_lag = lag
        print(f"  Optimal lag: {best_lag:.3f} s (sum of distances: {min_total_dist/1e6:.3f} s)")
        # Apply lag to all mask_particles timestamps (shift backward)
        new_t_us_particles = t_us + best_lag * 1e6
        # Plot for user
        t = (t_us - t_us[0]) * 1e-6
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(t, pa_filt, label="Pa_Global (filtered)", color="tab:blue")
        ax1.plot((trough_us - t_us[0]) * 1e-6, pa_filt[troughs], "o", color="navy", label="Pa_Global Troughs")
        ax2 = ax1.twinx()
        # Plot original mask_particles as a line
        ax2.plot(t, mp, label="Mask Particles (original)", color="tab:red", alpha=0.5)
        # Plot aligned mask_particles as a line (shifted by best_lag)
        t_aligned = (new_t_us_particles - t_us[0]) * 1e-6
        ax2.plot(t_aligned, mp, label=f"Mask Particles (lag={best_lag:.2f}s)", color="purple", alpha=0.7, linestyle="-")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Pressure (Pa)", color="tab:blue")
        ax2.set_ylabel("Mask Particles", color="tab:red")
        ax1.set_title(f"{file}\nProposed alignment: mask_particles lagged by {best_lag:.2f}s")
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()
        # Ask user to confirm or skip
        resp = input("Apply this alignment to the database? [y/N/s=skip file]: ").strip().lower()
        if resp == "y":
            print("  Applying alignment...")
            for rowid, new_val in zip(df["rowid"], new_t_us_particles):
                cur.execute("UPDATE breath_data SET t_us_particles=? WHERE rowid=?", (int(new_val), rowid))
            conn.commit()
            print("  Alignment applied.")
        elif resp == "s":
            print("  Skipped alignment for this file.")
            continue
        else:
            print("  No changes made.")
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()
