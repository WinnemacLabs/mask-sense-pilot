#!/usr/bin/env python3
"""
Script to add a t_us_zero column to all CSV files in a directory.
The t_us_zero column will be t_us - t_us[0], starting from 0.
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path


def process_csv_file(file_path):
    """
    Process a single CSV file to add t_us_zero column.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Processing: {file_path}")
        
        # Read the file line by line to separate comments from data
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find where the header starts (first line that doesn't start with #)
        header_idx = 0
        comment_lines = []
        
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                comment_lines.append(line)
                header_idx = i + 1
            else:
                break
        
        # Read the CSV data starting from the header
        df = pd.read_csv(file_path, skiprows=header_idx)
        
        # Check if t_us column exists
        if 't_us' not in df.columns:
            print(f"Warning: No 't_us' column found in {file_path}. Skipping.")
            return False
        
        # Check if t_us_zero already exists
        if 't_us_zero' in df.columns:
            print(f"Warning: 't_us_zero' column already exists in {file_path}. Skipping.")
            return False
        
        # Calculate t_us_zero = t_us - t_us[0]
        t_us_start = df['t_us'].iloc[0]
        df['t_us_zero'] = df['t_us'] - t_us_start
        
        # Reorder columns to put t_us_zero right after t_us
        cols = df.columns.tolist()
        t_us_idx = cols.index('t_us')
        cols.insert(t_us_idx + 1, cols.pop(cols.index('t_us_zero')))
        df = df[cols]
        
        # Write the file back with comments preserved
        with open(file_path, 'w') as f:
            # Write comment lines
            for comment in comment_lines:
                f.write(comment)
            
            # Write the DataFrame
            df.to_csv(f, index=False)
        
        print(f"Successfully added t_us_zero column to {file_path}")
        print(f"  Original t_us range: {df['t_us'].min()} to {df['t_us'].max()}")
        print(f"  New t_us_zero range: {df['t_us_zero'].min()} to {df['t_us_zero'].max()}")
        return True
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def process_directory(directory_path, recursive=False):
    """
    Process all CSV files in a directory.
    
    Args:
        directory_path (str): Path to the directory
        recursive (bool): Whether to process subdirectories recursively
    
    Returns:
        tuple: (successful_count, total_count)
    """
    directory = Path(directory_path)
    
    if not directory.exists():
        print(f"Error: Directory {directory_path} does not exist.")
        return 0, 0
    
    if not directory.is_dir():
        print(f"Error: {directory_path} is not a directory.")
        return 0, 0
    
    # Find all CSV files
    if recursive:
        csv_files = list(directory.rglob("*.csv"))
    else:
        csv_files = list(directory.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return 0, 0
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    print("-" * 50)
    
    successful = 0
    total = len(csv_files)
    
    for csv_file in csv_files:
        if process_csv_file(str(csv_file)):
            successful += 1
        print("-" * 50)
    
    return successful, total


def main():
    parser = argparse.ArgumentParser(
        description="Add t_us_zero column to CSV files. t_us_zero = t_us - t_us[0]",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python add_t_us_zero_column.py data/P0
  python add_t_us_zero_column.py data/P0 --recursive
  python add_t_us_zero_column.py data --recursive
        """
    )
    
    parser.add_argument(
        "directory", 
        help="Directory containing CSV files to process"
    )
    
    parser.add_argument(
        "-r", "--recursive", 
        action="store_true",
        help="Process subdirectories recursively"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        directory = Path(args.directory)
        if args.recursive:
            csv_files = list(directory.rglob("*.csv"))
        else:
            csv_files = list(directory.glob("*.csv"))
        
        print(f"Would process {len(csv_files)} CSV files:")
        for csv_file in csv_files:
            print(f"  {csv_file}")
        return
    
    print(f"Processing CSV files in: {args.directory}")
    print(f"Recursive: {args.recursive}")
    print("=" * 60)
    
    successful, total = process_directory(args.directory, args.recursive)
    
    print("=" * 60)
    print(f"SUMMARY:")
    print(f"  Total files processed: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total - successful}")
    
    if successful == total:
        print("✅ All files processed successfully!")
    elif successful > 0:
        print("⚠️  Some files processed successfully, but there were errors.")
    else:
        print("❌ No files were processed successfully.")


if __name__ == "__main__":
    main()
