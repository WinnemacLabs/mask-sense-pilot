#!/usr/bin/env python3
"""
compute_breath_protection_factor.py
-----------------------------------
Compute per-breath protection factor and store it in the breath_segments table.

Protection factor for each breath is defined as:
    PF = mean(ambient_particles) / max(mask_particles)
for all samples within the breath's start and end timestamps.

Usage:
    python compute_breath_protection_factor.py --db breath_db.sqlite
"""
import sqlite3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_and_store_pf(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Check if PF column exists, add if not
    cur.execute("PRAGMA table_info(breath_segments)")
    columns = [row[1] for row in cur.fetchall()]
    if "protection_factor" not in columns:
        print("Adding protection_factor column to breath_segments table...")
        cur.execute("ALTER TABLE breath_segments ADD COLUMN protection_factor REAL")
        conn.commit()

    # Get all unique source_files
    cur.execute("SELECT DISTINCT source_file FROM breath_segments")
    source_files = [row[0] for row in cur.fetchall()]

    for source_file in source_files:
        print(f"Processing {source_file} ...")
        # Load all breath segments for this file
        cur.execute("""
            SELECT id, breath, breath_start_us, breath_end_us FROM breath_segments
            WHERE source_file=?
            ORDER BY breath
        """, (source_file,))
        segments = cur.fetchall()
        if not segments:
            print(f"  No breath segments found for {source_file}")
            continue
        # Load breath_data for this file
        df = pd.read_sql_query(
            "SELECT t_us, mask_particles, ambient_particles FROM breath_data WHERE source_file=?",
            conn, params=(source_file,)
        )
        if df.empty:
            print(f"  No breath_data found for {source_file}")
            continue
        # Debug: print info for first 3 breaths and collect for plotting
        debug_breaths = set([1, 2, 3])
        debug_breath_data = []
        for seg_id, breath_num, start_us, end_us in segments:
            breath_df = df[(df['t_us'] >= start_us) & (df['t_us'] < end_us)]
            if breath_df.empty:
                pf = None
            else:
                max_mask = breath_df['mask_particles'].max()
                mean_ambient = breath_df['ambient_particles'].mean()
                if max_mask > 0:
                    pf = mean_ambient / max_mask
                else:
                    pf = None
            # Debug output for first 3 breaths
            if breath_num in debug_breaths:
                print(f"  Breath {breath_num}: start_us={start_us}, end_us={end_us}, N={len(breath_df)}")
                if not breath_df.empty:
                    print(f"    max(mask_particles)={max_mask}")
                    print(f"    mean(ambient_particles)={mean_ambient}")
                    print(f"    protection_factor={pf}")
                    # Collect for plotting
                    debug_breath_data.append((breath_num, breath_df, pf))
                else:
                    print("    No data in this breath interval.")
            # Update the protection_factor in the table
            cur.execute(
                "UPDATE breath_segments SET protection_factor=? WHERE id=?",
                (pf, seg_id)
            )
        conn.commit()
        print(f"  Updated {len(segments)} breaths.")
        # Plot summary figure for first 3 breaths
        if debug_breath_data:
            fig, axes = plt.subplots(len(debug_breath_data), 1, figsize=(8, 3*len(debug_breath_data)), sharex=False)
            if len(debug_breath_data) == 1:
                axes = [axes]
            for ax, (breath_num, breath_df, pf) in zip(axes, debug_breath_data):
                t = (breath_df['t_us'] - breath_df['t_us'].iloc[0]) * 1e-6  # seconds from start of breath
                ax.plot(t, breath_df['mask_particles'], label='mask_particles', color='red')
                ax.plot(t, breath_df['ambient_particles'], label='ambient_particles', color='blue')
                ax.set_title(f"Breath {breath_num} | PF={pf:.2f}" if pf is not None else f"Breath {breath_num} | PF=None")
                ax.set_ylabel('Particle Count')
                ax.legend()
            axes[-1].set_xlabel('Time (s)')
            fig.suptitle(f"First 3 Breaths: {source_file}")
            outdir = Path('output') / 'protection-factor'
            outdir.mkdir(parents=True, exist_ok=True)
            outpng = outdir / f"{Path(source_file).stem}_breath_pf_debug.png"
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(outpng)
            print(f"  Saved debug figure: {outpng}")
            plt.close(fig)
    conn.close()
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Compute per-breath protection factor and store in DB.")
    parser.add_argument("--db", default="breath_db.sqlite", help="Path to SQLite database")
    args = parser.parse_args()
    compute_and_store_pf(args.db)

if __name__ == "__main__":
    main()
