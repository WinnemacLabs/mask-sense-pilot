#!/usr/bin/env python3
"""
update_db_with_corrected_csvs.py
--------------------------------
Update the main database table with lag-corrected CSV data while preserving
the existing breath_segments table that contains manually curated breath boundaries.

This script:
1. Finds all CSV files in a directory 
2. Applies 7.5s lag correction to particle data
3. Updates the corresponding database entries in the main table
4. Leaves the breath_segments table completely untouched

Usage:
    python update_db_with_corrected_csvs.py database.sqlite data_directory/
    python update_db_with_corrected_csvs.py database.sqlite data_directory/ --lag 7.2
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def apply_particle_lag_correction(df: pd.DataFrame, lag_seconds: float = 7.5) -> pd.DataFrame:
    """
    Apply lag correction to particle data in a CSV DataFrame.
    
    Args:
        df: DataFrame with t_us, mask_particles, ambient_particles columns
        lag_seconds: Lag in seconds (positive means particles lag behind pressure)
    
    Returns:
        DataFrame with corrected particle data
    """
    if lag_seconds == 0:
        return df.copy()
    
    print(f"  Applying {lag_seconds}s lag correction to particle data...")
    
    df_corrected = df.copy()
    lag_microseconds = int(lag_seconds * 1e6)
    
    # Shift particle data by interpolating to earlier timestamps
    particle_cols = ["mask_particles", "ambient_particles"]
    
    for col in particle_cols:
        if col in df.columns:
            # Create shifted time array
            t_us_shifted = df["t_us"].values - lag_microseconds
            
            # Interpolate particle data to the shifted timestamps
            # This effectively moves the particle data backward in time
            df_corrected[col] = np.interp(
                df["t_us"].values,  # Original timestamps
                t_us_shifted,       # Shifted timestamps  
                df[col].values      # Original particle data
            )
    
    return df_corrected

def get_files_in_database(db_path: Path) -> set:
    """Get list of source files already in the database."""
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.execute("SELECT DISTINCT source_file FROM breath_data")
        db_files = {row[0] for row in cur.fetchall()}
        return db_files
    except sqlite3.OperationalError:
        # Table doesn't exist
        return set()
    finally:
        conn.close()

def update_database_with_csv(db_path: Path, csv_path: Path, lag_seconds: float = 7.5) -> bool:
    """
    Update database with corrected data from a CSV file.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Load and correct CSV data
        print(f"  Loading {csv_path.name}...")
        df = pd.read_csv(csv_path, comment="#")
        
        # Validate required columns
        required_cols = ["t_us", "Pa_Global", "mask_particles", "ambient_particles"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"  ✗ Missing columns: {missing_cols}")
            return False
        
        # Apply lag correction
        df_corrected = apply_particle_lag_correction(df, lag_seconds)
        
        # Add source_file column
        df_corrected["source_file"] = str(csv_path)
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        try:
            # Delete existing data for this file (check for any path variation)
            csv_filename = csv_path.name
            conn.execute("DELETE FROM breath_data WHERE source_file LIKE ?", (f'%{csv_filename}',))
            
            # Insert corrected data (store just the filename for consistency)
            df_corrected["source_file"] = csv_filename
            df_corrected.to_sql("breath_data", conn, if_exists="append", index=False)
            conn.commit()
            
            print(f"  ✓ Updated {len(df_corrected)} rows in database")
            return True
            
        finally:
            conn.close()
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update database with lag-corrected CSV data")
    parser.add_argument("database", help="SQLite database file")
    parser.add_argument("csv_directory", help="Directory containing CSV files")
    parser.add_argument("--lag", type=float, default=7.5, 
                       help="Lag correction in seconds (default: 7.5)")
    parser.add_argument("--skip-keywords", nargs="*", default=["init", "rainbow", "zero"],
                       help="Skip CSV files containing these keywords")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    parser.add_argument("--force", action="store_true",
                       help="Update all matching CSV files, even if not in database")
    
    args = parser.parse_args()
    
    # Validate inputs
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database {db_path} does not exist")
        return 1
    
    csv_dir = Path(args.csv_directory)
    if not csv_dir.is_dir():
        print(f"Error: Directory {csv_dir} does not exist")
        return 1
    
    print(f"Database: {db_path}")
    print(f"CSV directory: {csv_dir}")
    print(f"Lag correction: {args.lag} seconds")
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
    
    # Get files currently in database
    if not args.force:
        print("\nChecking which files are in the database...")
        db_files = get_files_in_database(db_path)
        print(f"Found {len(db_files)} files in database")
        if len(db_files) <= 10:
            print("Files in database:")
            for db_file in sorted(db_files):
                print(f"  - {db_file}")
        else:
            print("Sample files in database:")
            for db_file in sorted(list(db_files)[:5]):
                print(f"  - {db_file}")
            print(f"  ... and {len(db_files) - 5} more")
    else:
        db_files = None
    
    # Find CSV files to process
    all_csv_files = sorted(csv_dir.glob("*.csv"))
    
    # Filter out files with skip keywords
    filtered_files = []
    for f in all_csv_files:
        if not any(keyword in f.name.lower() for keyword in args.skip_keywords):
            filtered_files.append(f)
    
    print(f"\nFound {len(all_csv_files)} CSV files, {len(filtered_files)} after filtering")
    
    # Determine which files to process
    files_to_process = []
    files_not_in_db = []
    
    for csv_file in filtered_files:
        if args.force:
            files_to_process.append(csv_file)
        else:
            # Check if this file is in the database (match by filename)
            csv_filename = csv_file.name
            file_in_db = any(csv_filename in db_file for db_file in db_files)
            if file_in_db:
                files_to_process.append(csv_file)
            else:
                files_not_in_db.append(csv_file)
    
    print(f"\nFiles to process: {len(files_to_process)}")
    if files_not_in_db and not args.force:
        print(f"Files not in database (skipped): {len(files_not_in_db)}")
        if len(files_not_in_db) <= 5:
            for f in files_not_in_db:
                print(f"  - {f.name}")
        else:
            print(f"  (use --force to process all files)")
    
    if not files_to_process:
        print("No files to process!")
        return 0
    
    # Process files
    if args.dry_run:
        print("\nWould process these files:")
        for f in files_to_process:
            print(f"  - {f.name}")
        return 0
    
    print(f"\nProcessing {len(files_to_process)} files...")
    
    success_count = 0
    error_count = 0
    
    for csv_file in files_to_process:
        print(f"\nProcessing {csv_file.name}...")
        
        if update_database_with_csv(db_path, csv_file, args.lag):
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n=== Update Complete ===")
    print(f"Successfully updated: {success_count} files")
    print(f"Errors: {error_count} files")
    print(f"Lag correction applied: {args.lag} seconds")
    print("\nNote: breath_segments table was not modified - your manual segmentation work is preserved!")
    
    return 0 if error_count == 0 else 1

if __name__ == "__main__":
    exit(main())