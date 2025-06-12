#!/usr/bin/env python3
"""
verify_lag_correction.py
------------------------
Plot data from breath_db.sqlite to verify that lag correction was applied correctly.

This script:
1. Loads data from the database for a specific file
2. Shows pressure vs particle data alignment
3. Calculates protection factor over time
4. Highlights breathing periods from breath_segments table

Usage:
    python verify_lag_correction.py breath_db.sqlite
    python verify_lag_correction.py breath_db.sqlite --file "specific_file.csv"
    python verify_lag_correction.py breath_db.sqlite --participant P0 --mask AURA
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from pathlib import Path
import sys

# Set default renderer
pio.renderers.default = "browser"

def get_available_files(db_path: Path) -> pd.DataFrame:
    """Get list of files in the database with metadata."""
    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT DISTINCT source_file,
               COUNT(*) as data_points,
               MIN(t_us) as start_time,
               MAX(t_us) as end_time,
               (MAX(t_us) - MIN(t_us)) / 1e6 as duration_seconds
        FROM breath_data 
        GROUP BY source_file
        ORDER BY source_file
        """
        df = pd.read_sql_query(query, conn)
        
        # Extract metadata from filename
        df['filename'] = df['source_file'].apply(lambda x: Path(x).name)
        df['participant'] = df['filename'].str.extract(r'rsc_([^_]+)_')
        df['mask'] = df['filename'].str.extract(r'rsc_[^_]+_([^_]+)_')
        df['exercise'] = df['filename'].str.extract(r'rsc_[^_]+_[^_]+_(.+?)_\d{8}_\d{6}\.csv')
        
        return df
    finally:
        conn.close()

def get_breath_segments(db_path: Path, source_file: str) -> pd.DataFrame:
    """Get breath segments for a specific file."""
    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT breath, breath_start_us, breath_end_us,
               (breath_end_us - breath_start_us) / 1e6 as duration_seconds
        FROM breath_segments 
        WHERE source_file LIKE ?
        ORDER BY breath
        """
        return pd.read_sql_query(query, conn, params=(f'%{source_file}%',))
    finally:
        conn.close()

def load_file_data(db_path: Path, source_file: str) -> pd.DataFrame:
    """Load data for a specific file from the database."""
    conn = sqlite3.connect(db_path)
    try:
        query = """
        SELECT * FROM breath_data 
        WHERE source_file LIKE ?
        ORDER BY t_us
        """
        df = pd.read_sql_query(query, conn, params=(f'%{source_file}%',))
        
        if len(df) == 0:
            return None
        
        # Create time in seconds relative to start
        df['time_s'] = (df['t_us'] - df['t_us'].iloc[0]) / 1e6
        
        # Calculate protection factor
        df['protection_factor'] = np.maximum(df['ambient_particles'], 0.1) / np.maximum(df['mask_particles'], 0.1)
        df['log_pf'] = np.log10(df['protection_factor'])
        
        return df
    finally:
        conn.close()

def plot_verification(df: pd.DataFrame, breath_segments: pd.DataFrame, filename: str, 
                     time_range: tuple = None, show_breaths: bool = True):
    """Create verification plots showing pressure, particles, and protection factor."""
    
    # Filter data to time range if specified
    if time_range is not None:
        mask = (df['time_s'] >= time_range[0]) & (df['time_s'] <= time_range[1])
        df_plot = df[mask].copy()
        plot_title_suffix = f" (Time: {time_range[0]:.0f}-{time_range[1]:.0f}s)"
    else:
        df_plot = df.copy()
        plot_title_suffix = ""
    
    # Convert breath segments to relative time
    if not breath_segments.empty:
        start_time_us = df['t_us'].iloc[0]
        breath_segments = breath_segments.copy()
        breath_segments['start_time_s'] = (breath_segments['breath_start_us'] - start_time_us) / 1e6
        breath_segments['end_time_s'] = (breath_segments['breath_end_us'] - start_time_us) / 1e6
        
        # Filter breath segments to time range
        if time_range is not None:
            breath_mask = ((breath_segments['start_time_s'] <= time_range[1]) & 
                          (breath_segments['end_time_s'] >= time_range[0]))
            breath_segments = breath_segments[breath_mask]
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Pressure Data (Pa_Global)",
            "In-Mask Particle Concentrations", 
            "Normalized Signals (for visual correlation check)"
        ],
        vertical_spacing=0.08,
        row_heights=[0.33, 0.33, 0.34]
    )
    
    # Plot 1: Pressure data
    fig.add_trace(
        go.Scatter(
            x=df_plot['time_s'],
            y=df_plot['Pa_Global'],
            mode='markers',
            name='Pa_Global',
            marker=dict(color='blue', size=2)
        ),
        row=1, col=1
    )
    
    # Plot 2: In-mask particle concentrations only
    fig.add_trace(
        go.Scatter(
            x=df_plot['time_s'],
            y=df_plot['mask_particles'],
            mode='markers',
            name='Mask particles (corrected)',
            marker=dict(color='red', size=2)
        ),
        row=2, col=1
    )
    
    # Plot 3: Normalized signals for visual correlation check
    # Normalize pressure (to 0-1 range)
    pressure_norm = (df_plot['Pa_Global'] - df_plot['Pa_Global'].min()) / (df_plot['Pa_Global'].max() - df_plot['Pa_Global'].min())
    
    # Normalize mask particles (to 0-1 range) - no inversion needed if lag correction worked
    mask_particles_norm = (df_plot['mask_particles'] - df_plot['mask_particles'].min()) / (df_plot['mask_particles'].max() - df_plot['mask_particles'].min())
    
    fig.add_trace(
        go.Scatter(
            x=df_plot['time_s'],
            y=pressure_norm,
            mode='markers',
            name='Pressure (normalized)',
            marker=dict(color='blue', size=2)
        ),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_plot['time_s'],
            y=mask_particles_norm,
            mode='markers',
            name='Mask particles (normalized)',
            marker=dict(color='red', size=2)
        ),
        row=3, col=1
    )
    
    # Add breath segment regions if available and requested
    if show_breaths and not breath_segments.empty:
        for _, breath in breath_segments.iterrows():
            # Add vertical rectangles for each breath
            for row in range(1, 4):
                fig.add_vrect(
                    x0=breath['start_time_s'], 
                    x1=breath['end_time_s'],
                    fillcolor="lightblue", 
                    opacity=0.2, 
                    line_width=0,
                    row=row, col=1
                )
                
                # Add breath number annotation on the first plot
                if row == 1:
                    fig.add_annotation(
                        x=(breath['start_time_s'] + breath['end_time_s']) / 2,
                        y=df_plot['Pa_Global'].max() * 0.9,
                        text=f"B{int(breath['breath'])}",
                        showarrow=False,
                        font=dict(size=10, color="blue"),
                        row=1, col=1
                    )
    
    # Update layout
    fig.update_layout(
        title=f'Lag Correction Verification - {filename}{plot_title_suffix}',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_yaxes(title_text="Pressure (Pa)", row=1, col=1)
    fig.update_yaxes(title_text="Particle Count", row=2, col=1)
    fig.update_yaxes(title_text="Normalized", row=3, col=1)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description="Verify lag correction in breath database")
    parser.add_argument("database", help="SQLite database file")
    parser.add_argument("--file", help="Specific filename to plot (partial match)")
    parser.add_argument("--participant", help="Filter by participant (e.g., P0)")
    parser.add_argument("--mask", help="Filter by mask type (e.g., AURA, MAKTEK)")
    parser.add_argument("--exercise", help="Filter by exercise type")
    parser.add_argument("--time-range", nargs=2, type=float, metavar=('START', 'END'),
                       help="Time range in seconds (relative to start)")
    parser.add_argument("--no-breaths", action="store_true",
                       help="Don't show breath segment regions")
    parser.add_argument("--list-files", action="store_true",
                       help="List available files and exit")
    
    args = parser.parse_args()
    
    # Validate database
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database {db_path} does not exist")
        return 1
    
    # Get available files
    print(f"Loading file list from {db_path}...")
    available_files = get_available_files(db_path)
    
    if available_files.empty:
        print("No files found in database!")
        return 1
    
    # List files and exit if requested
    if args.list_files:
        print(f"\nFound {len(available_files)} files in database:")
        print("\nFilename | Participant | Mask | Exercise | Duration | Data Points")
        print("-" * 80)
        for _, row in available_files.iterrows():
            print(f"{row['filename'][:40]:<40} | {row['participant']:<11} | {row['mask']:<8} | {row['exercise']:<20} | {row['duration_seconds']:>6.1f}s | {row['data_points']:>8}")
        return 0
    
    # Filter files based on criteria
    filtered_files = available_files.copy()
    
    if args.participant:
        filtered_files = filtered_files[filtered_files['participant'] == args.participant]
    
    if args.mask:
        filtered_files = filtered_files[filtered_files['mask'] == args.mask]
    
    if args.exercise:
        filtered_files = filtered_files[filtered_files['exercise'].str.contains(args.exercise, na=False)]
    
    if args.file:
        filtered_files = filtered_files[filtered_files['filename'].str.contains(args.file, na=False)]
    
    if filtered_files.empty:
        print("No files match the specified criteria!")
        print("\nAvailable files:")
        for _, row in available_files.iterrows():
            print(f"  - {row['filename']} (P{row['participant']}, {row['mask']}, {row['exercise']})")
        return 1
    
    # Select file to plot
    if len(filtered_files) == 1:
        selected_file = filtered_files.iloc[0]
    else:
        print(f"\nFound {len(filtered_files)} matching files:")
        for i, (_, row) in enumerate(filtered_files.iterrows()):
            print(f"{i+1:2d}. {row['filename']} ({row['participant']}, {row['mask']}, {row['exercise']}, {row['duration_seconds']:.1f}s)")
        
        while True:
            try:
                choice = input(f"\nSelect file to plot (1-{len(filtered_files)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(filtered_files):
                    selected_file = filtered_files.iloc[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(filtered_files)}")
            except (ValueError, KeyboardInterrupt):
                print("Cancelled.")
                return 0
    
    filename = selected_file['filename']
    print(f"\nLoading data for: {filename}")
    
    # Load data and breath segments
    df = load_file_data(db_path, filename)
    if df is None:
        print(f"No data found for {filename}")
        return 1
    
    breath_segments = get_breath_segments(db_path, filename)
    
    print(f"Loaded {len(df)} data points")
    print(f"Duration: {df['time_s'].iloc[-1]:.1f} seconds")
    if not breath_segments.empty:
        print(f"Found {len(breath_segments)} breath segments")
        avg_breath_duration = breath_segments['duration_seconds'].mean()
        print(f"Average breath duration: {avg_breath_duration:.1f} seconds")
    
    # Create time range if not specified
    time_range = None
    if args.time_range:
        time_range = tuple(args.time_range)
    else:
        # Default to a good section for visualization (middle 2 minutes if available)
        total_duration = df['time_s'].iloc[-1]
        if total_duration > 120:
            mid_time = total_duration / 2
            time_range = (mid_time - 60, mid_time + 60)
    
    # Create and show plot
    print("Creating verification plot...")
    fig = plot_verification(df, breath_segments, filename, time_range, not args.no_breaths)
    fig.show()
    
    # Print some statistics
    print(f"\n=== Verification Statistics ===")
    print(f"File: {filename}")
    print(f"Data points: {len(df):,}")
    print(f"Total duration: {df['time_s'].iloc[-1]:.1f} seconds")
    
    # Particle statistics
    mask_mean = df['mask_particles'].mean()
    ambient_mean = df['ambient_particles'].mean()
    pf_median = df['protection_factor'].median()
    
    print(f"\nParticle concentrations:")
    print(f"  Mask particles (mean): {mask_mean:.1f}")
    print(f"  Ambient particles (mean): {ambient_mean:.1f}")
    print(f"  Protection factor (median): {pf_median:.1f}")
    
    if not breath_segments.empty:
        print(f"\nBreath segments:")
        print(f"  Number of breaths: {len(breath_segments)}")
        print(f"  Average breath duration: {breath_segments['duration_seconds'].mean():.1f}s")
        print(f"  Breath rate: {len(breath_segments) / (df['time_s'].iloc[-1] / 60):.1f} breaths/min")
    
    print(f"\nLag correction verification:")
    print(f"  Look at the bottom panel - the red dashed line (normalized mask particles)")
    print(f"  should correlate with the blue line (pressure) during breathing events.")
    print(f"  If they move together properly, the 7.5s lag correction was successful!")
    
    return 0

if __name__ == "__main__":
    exit(main())