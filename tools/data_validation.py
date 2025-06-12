#!/usr/bin/env python3
"""
Data Validation Script for Pressure Fit CSV Files

This script validates CSV files in a directory by:
1. Calculating total duration of each file
2. Ensuring timestamp column (t_us) is monotonically increasing
3. Displaying validation results in the terminal

Expected CSV format:
- Comment lines starting with '#' at the top
- Header row: t_us,Pa_Global,Pa_Vertical,Pa_Horizontal,raw_Global,raw_Vertical,raw_Horizontal,mask_particles,ambient_particles
- Data rows with timestamps in microseconds
"""

import os
import sys
import pandas as pd
from pathlib import Path
import argparse
from typing import Tuple, Dict, Any
import numpy as np


class CSVValidator:
    """Validator class for pressure fit CSV files."""
    
    def __init__(self):
        self.results = []
        
    def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate a single CSV file.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing validation results
        """
        result = {
            'file': file_path.name,
            'path': str(file_path),
            'valid': True,
            'errors': [],
            'warnings': [],
            'total_duration_seconds': None,
            'total_duration_minutes': None,
            'num_data_points': 0,
            'timestamp_issues': []
        }
        
        try:
            # Read the file, skipping comment lines
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the header line (first line that doesn't start with #)
            header_idx = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith('#'):
                    header_idx = i
                    break
            
            # Read the CSV starting from the header
            df = pd.read_csv(file_path, skiprows=header_idx)
            
            # Validate that required columns exist
            required_columns = ['t_us', 'Pa_Global', 'Pa_Vertical', 'Pa_Horizontal', 
                              'raw_Global', 'raw_Vertical', 'raw_Horizontal', 
                              'mask_particles', 'ambient_particles']
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                result['valid'] = False
                result['errors'].append(f"Missing columns: {missing_columns}")
                return result
            
            # Check if file has data
            if len(df) == 0:
                result['valid'] = False
                result['errors'].append("File contains no data rows")
                return result
            
            result['num_data_points'] = len(df)
            
            # Validate timestamp column
            timestamps = df['t_us'].values
            
            # Check for NaN values in timestamps
            if pd.isna(timestamps).any():
                result['valid'] = False
                result['errors'].append("Timestamp column contains NaN values")
            
            # Check if timestamps are monotonically increasing
            if len(timestamps) > 1:
                diffs = np.diff(timestamps)
                non_increasing_indices = np.where(diffs <= 0)[0]
                
                if len(non_increasing_indices) > 0:
                    result['valid'] = False
                    for idx in non_increasing_indices[:5]:  # Show first 5 issues
                        result['timestamp_issues'].append({
                            'row': idx + 1,  # +1 because we're looking at differences
                            'current_timestamp': timestamps[idx + 1],
                            'previous_timestamp': timestamps[idx],
                            'difference': diffs[idx]
                        })
                    
                    if len(non_increasing_indices) > 5:
                        result['errors'].append(
                            f"Timestamps are not monotonically increasing. "
                            f"Found {len(non_increasing_indices)} issues. "
                            f"Showing first 5."
                        )
                    else:
                        result['errors'].append(
                            f"Timestamps are not monotonically increasing. "
                            f"Found {len(non_increasing_indices)} issues."
                        )
            
            # Calculate total duration
            if len(timestamps) > 1:
                duration_us = timestamps[-1] - timestamps[0]
                result['total_duration_seconds'] = duration_us / 1_000_000
                result['total_duration_minutes'] = result['total_duration_seconds'] / 60
            else:
                result['warnings'].append("Only one data point, cannot calculate duration")
            
            # Check for duplicate timestamps
            if len(timestamps) != len(set(timestamps)):
                duplicate_count = len(timestamps) - len(set(timestamps))
                result['warnings'].append(f"Found {duplicate_count} duplicate timestamps")
            
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Error reading file: {str(e)}")
        
        return result
    
    def validate_directory(self, directory_path: Path) -> None:
        """
        Validate all CSV files in a directory.
        
        Args:
            directory_path: Path to the directory containing CSV files
        """
        if not directory_path.exists():
            print(f"Error: Directory '{directory_path}' does not exist.")
            return
        
        # Find all CSV files
        csv_files = list(directory_path.glob("**/*.csv"))
        
        if not csv_files:
            print(f"No CSV files found in '{directory_path}'")
            return
        
        print(f"Found {len(csv_files)} CSV files in '{directory_path}'")
        print("=" * 80)
        
        valid_files = 0
        invalid_files = 0
        
        for csv_file in sorted(csv_files):
            print(f"\nValidating: {csv_file.name}")
            print("-" * 40)
            
            result = self.validate_file(csv_file)
            self.results.append(result)
            
            if result['valid']:
                valid_files += 1
                print("✅ VALID")
            else:
                invalid_files += 1
                print("❌ INVALID")
            
            # Display basic info
            print(f"Data points: {result['num_data_points']:,}")
            
            if result['total_duration_seconds'] is not None:
                print(f"Duration: {result['total_duration_seconds']:.2f} seconds "
                      f"({result['total_duration_minutes']:.2f} minutes)")
            
            # Display errors
            if result['errors']:
                print("Errors:")
                for error in result['errors']:
                    print(f"  • {error}")
            
            # Display warnings
            if result['warnings']:
                print("Warnings:")
                for warning in result['warnings']:
                    print(f"  • {warning}")
            
            # Display timestamp issues (first few)
            if result['timestamp_issues']:
                print("Timestamp Issues:")
                for issue in result['timestamp_issues'][:3]:
                    print(f"  • Row {issue['row']}: {issue['current_timestamp']} <= {issue['previous_timestamp']} "
                          f"(diff: {issue['difference']})")
        
        # Summary
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total files processed: {len(csv_files)}")
        print(f"Valid files: {valid_files}")
        print(f"Invalid files: {invalid_files}")
        
        if invalid_files > 0:
            print(f"\n❌ {invalid_files} files failed validation")
        else:
            print(f"\n✅ All files passed validation")
        
        # Duration statistics for valid files
        valid_durations = [r['total_duration_minutes'] for r in self.results 
                          if r['valid'] and r['total_duration_minutes'] is not None]
        
        if valid_durations:
            print(f"\nDuration Statistics (valid files):")
            print(f"  Average: {np.mean(valid_durations):.2f} minutes")
            print(f"  Min: {np.min(valid_durations):.2f} minutes")
            print(f"  Max: {np.max(valid_durations):.2f} minutes")
    
    def export_results(self, output_file: Path) -> None:
        """Export validation results to a CSV file."""
        if not self.results:
            print("No results to export")
            return
        
        # Flatten results for CSV export
        export_data = []
        for result in self.results:
            export_data.append({
                'file': result['file'],
                'path': result['path'],
                'valid': result['valid'],
                'num_data_points': result['num_data_points'],
                'duration_seconds': result['total_duration_seconds'],
                'duration_minutes': result['total_duration_minutes'],
                'errors': '; '.join(result['errors']),
                'warnings': '; '.join(result['warnings']),
                'timestamp_issues_count': len(result['timestamp_issues'])
            })
        
        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False)
        print(f"\nResults exported to: {output_file}")


def main():
    """Main function to run the validation script."""
    parser = argparse.ArgumentParser(
        description="Validate pressure fit CSV files for timestamp monotonicity and calculate durations"
    )
    parser.add_argument(
        "directory", 
        type=str, 
        help="Directory containing CSV files to validate"
    )
    parser.add_argument(
        "--export", 
        type=str, 
        help="Export results to CSV file (optional)"
    )
    
    args = parser.parse_args()
    
    directory_path = Path(args.directory)
    validator = CSVValidator()
    
    print("CSV Data Validation Tool")
    print("=" * 80)
    print(f"Target directory: {directory_path}")
    print()
    
    validator.validate_directory(directory_path)
    
    if args.export:
        validator.export_results(Path(args.export))


if __name__ == "__main__":
    main()
