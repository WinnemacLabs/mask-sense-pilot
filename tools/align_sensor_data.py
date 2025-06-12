#!/usr/bin/env python3
"""
align_sensor_data.py
--------------------
Align pressure and particle sensor data streams to compensate for measurement lag.

This script aligns particle data with pressure data by applying a specified lag
to synchronize the data streams.

The particle sensors (WRPAS) typically have a lag of ~7 seconds relative to pressure sensors
due to sampling tube transport time and measurement processing delays.

Usage:
    python align_sensor_data.py data_directory --lag 7.2
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


class SensorAligner:
    """Class to handle alignment of pressure and particle sensor data."""
    
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load CSV data with comment handling."""
        print(f"Loading data from {self.csv_path}")
        self.df = pd.read_csv(self.csv_path, comment="#")
        
        # Validate required columns
        required_pressure = ["t_us", "Pa_Global"]
        required_particles = ["mask_particles", "ambient_particles"]
        
        missing_cols = []
        for col in required_pressure + required_particles:
            if col not in self.df.columns:
                missing_cols.append(col)
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert time to seconds for easier processing
        self.df["time_s"] = self.df["t_us"] * 1e-6
        
        print(f"Loaded {len(self.df)} data points")
        print(f"Duration: {self.df['time_s'].iloc[-1] - self.df['time_s'].iloc[0]:.1f} seconds")
    
    def apply_lag_correction(self, lag_seconds):
        """
        Apply lag correction to the data.
        
        Args:
            lag_seconds: Lag in seconds (positive means particles lag behind pressure)
        
        Returns:
            DataFrame with corrected timestamps
        """
        print(f"Applying lag correction of {lag_seconds:.2f} seconds...")
        
        # Create corrected dataframe
        df_corrected = self.df.copy()
        
        if lag_seconds != 0:
            # Shift particle data timestamps
            lag_microseconds = int(lag_seconds * 1e6)
            
            # Create new time column for particles
            df_corrected["t_us_particles"] = df_corrected["t_us"] - lag_microseconds
            
            # For display purposes, also create time_s columns
            df_corrected["time_s_particles"] = df_corrected["t_us_particles"] * 1e-6

            print(f"Particle timestamps shifted by {lag_microseconds} microseconds")
        else:
            print("No lag correction applied (lag = 0)")
        
        return df_corrected
    
    def save_corrected_data(self, df_corrected, output_path):
        """Save the corrected data to a new CSV file."""
        output_path = Path(output_path)
        print(f"Saving corrected data to {output_path}")
        
        # Read original file header comments
        header_comments = []
        with open(self.csv_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    header_comments.append(line.strip())
                else:
                    break
        
        # Add alignment information to header
        header_comments.append("# Data alignment applied to particle measurements")
        
        # Write corrected data
        with open(output_path, 'w') as f:
            # Write header comments
            for comment in header_comments:
                f.write(comment + '\n')
            
            # Write corrected data (drop intermediate columns used for processing)
            columns_to_save = [col for col in df_corrected.columns 
                             if col not in ['time_s', 'time_s_particles_corrected']]
            df_corrected[columns_to_save].to_csv(f, index=False)
        
        print(f"Saved {len(df_corrected)} data points to {output_path}")


def process_directory(input_dir, lag_seconds):
    input_dir = Path(input_dir)
    aligned_dir = input_dir / "aligned"
    aligned_dir.mkdir(exist_ok=True)
    
    csv_files = list(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return 1
    
    for csv_path in csv_files:
        print(f"\nProcessing {csv_path.name} ...")
        try:
            aligner = SensorAligner(csv_path)
            df_corrected = aligner.apply_lag_correction(lag_seconds)
            output_path = aligned_dir / csv_path.name.replace('.csv', '_aligned.csv')
            aligner.save_corrected_data(df_corrected, output_path)
        except Exception as e:
            print(f"Error processing {csv_path.name}: {e}")
    print(f"\nAll files processed. Aligned files saved to: {aligned_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Align all CSVs in a directory using a manual lag.")
    parser.add_argument("directory", help="Directory containing CSV files to align")
    parser.add_argument("--lag", type=float, required=True, help="Manual lag in seconds (positive = particles lag behind)")
    args = parser.parse_args()
    
    return process_directory(args.directory, args.lag)


if __name__ == "__main__":
    exit(main())
