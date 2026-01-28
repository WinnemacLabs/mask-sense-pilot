#!/usr/bin/env python3
"""
Timestamp Rollover Fix Script

This script detects and fixes timestamp rollovers in CSV files where timestamps
wrap around due to 32-bit unsigned integer overflow (rollover from ~4.3 billion back to 0).

Usage:
    python fix_timestamp_rollovers.py <input_file> [output_file]
    python fix_timestamp_rollovers.py <directory>  # Fix all CSV files in directory
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


class TimestampRolloverFixer:
    """Class to handle detection and fixing of timestamp rollovers in CSV data."""
    
    def __init__(self, rollover_threshold: int = 2**32):
        """
        Initialize the rollover fixer.
        
        Args:
            rollover_threshold: The value at which rollovers occur (default: 2^32 for 32-bit unsigned int)
        """
        self.rollover_threshold = rollover_threshold
        
    def detect_rollovers(self, timestamps: np.ndarray) -> List[int]:
        """
        Detect rollover points in timestamp data.
        
        Args:
            timestamps: Array of timestamp values
            
        Returns:
            List of indices where rollovers occur
        """
        if len(timestamps) < 2:
            return []
            
        # Calculate differences between consecutive timestamps
        diffs = timestamps[1:] - timestamps[:-1]
        
        # A rollover is indicated by a large negative difference
        # We use a threshold of -1000000000 (1 billion) to catch rollovers
        # while avoiding false positives from legitimate backwards jumps
        rollover_threshold = -1000000000
        rollover_indices = np.where(diffs < rollover_threshold)[0] + 1
        
        return rollover_indices.tolist()
    
    def fix_rollovers(self, timestamps: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        Fix timestamp rollovers by adding appropriate offsets.
        
        Args:
            timestamps: Array of timestamp values
            
        Returns:
            Tuple of (fixed_timestamps, rollover_indices)
        """
        if len(timestamps) < 2:
            return timestamps.copy(), []
            
        rollover_indices = self.detect_rollovers(timestamps)
        
        if not rollover_indices:
            return timestamps.copy(), []
            
        fixed_timestamps = timestamps.copy().astype(np.int64)
        
        # For each rollover, add the rollover threshold to all subsequent timestamps
        cumulative_offset = 0
        
        for rollover_idx in rollover_indices:
            # Calculate the offset needed
            # The offset is the rollover threshold value
            cumulative_offset += self.rollover_threshold
            
            # Apply offset to all timestamps from this rollover onwards
            fixed_timestamps[rollover_idx:] += cumulative_offset
            
        return fixed_timestamps, rollover_indices
    
    def fix_csv_file(self, input_file: str, output_file: Optional[str] = None, 
                     timestamp_column: str = 't_us') -> bool:
        """
        Fix timestamp rollovers in a CSV file.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file (if None, will add '_fixed' suffix)
            timestamp_column: Name of the timestamp column
            
        Returns:
            True if rollovers were found and fixed, False otherwise
        """
        try:
            # Read the CSV file
            print(f"Reading: {input_file}")
            df = pd.read_csv(input_file, comment='#')
            
            if timestamp_column not in df.columns:
                print(f"❌ Error: Column '{timestamp_column}' not found in {input_file}")
                print(f"Available columns: {df.columns.tolist()}")
                return False
            
            # Get original timestamps
            original_timestamps = df[timestamp_column].values
            
            # Fix rollovers
            fixed_timestamps, rollover_indices = self.fix_rollovers(original_timestamps)
            
            if not rollover_indices:
                print(f"✅ No rollovers detected in {input_file}")
                return False
            
            # Update the dataframe
            df[timestamp_column] = fixed_timestamps
            
            # Generate output filename if not provided
            if output_file is None:
                input_path = Path(input_file)
                output_file = str(input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}")
            
            # Read original file to preserve comments
            with open(input_file, 'r') as f:
                lines = f.readlines()
            
            # Find where data starts (after comments)
            data_start_line = 0
            for i, line in enumerate(lines):
                if not line.strip().startswith('#'):
                    data_start_line = i
                    break
            
            # Write fixed file
            with open(output_file, 'w') as f:
                # Write original comments
                for i in range(data_start_line):
                    f.write(lines[i])
                
                # Write fixed data
                df.to_csv(f, index=False)
            
            # Calculate duration improvement
            original_duration = (original_timestamps[-1] - original_timestamps[0]) / 1_000_000
            fixed_duration = (fixed_timestamps[-1] - fixed_timestamps[0]) / 1_000_000
            
            print(f"🔧 Fixed {len(rollover_indices)} rollover(s) in {input_file}")
            print(f"   Rollover positions: {rollover_indices}")
            print(f"   Original duration: {original_duration:.2f} seconds")
            print(f"   Fixed duration: {fixed_duration:.2f} seconds")
            print(f"   Output: {output_file}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error processing {input_file}: {str(e)}")
            return False
    
    def fix_directory(self, directory: str, timestamp_column: str = 't_us') -> Tuple[int, int]:
        """
        Fix timestamp rollovers in all CSV files in a directory.
        
        Args:
            directory: Path to directory containing CSV files
            timestamp_column: Name of the timestamp column
            
        Returns:
            Tuple of (total_files_processed, files_with_rollovers_fixed)
        """
        csv_files = list(Path(directory).glob('*.csv'))
        
        if not csv_files:
            print(f"No CSV files found in {directory}")
            return 0, 0
        
        print(f"Found {len(csv_files)} CSV files in '{directory}'")
        print("=" * 80)
        
        total_processed = 0
        files_fixed = 0
        
        for csv_file in csv_files:
            if '_fixed' in csv_file.name:
                print(f"⏭️  Skipping already fixed file: {csv_file.name}")
                continue
                
            total_processed += 1
            if self.fix_csv_file(str(csv_file), timestamp_column=timestamp_column):
                files_fixed += 1
            print("-" * 40)
        
        print("=" * 80)
        print(f"SUMMARY:")
        print(f"Files processed: {total_processed}")
        print(f"Files with rollovers fixed: {files_fixed}")
        
        return total_processed, files_fixed


def main():
    """Main function to handle command line arguments and run the fixer."""
    parser = argparse.ArgumentParser(
        description="Fix timestamp rollovers in CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fix_timestamp_rollovers.py data.csv
  python fix_timestamp_rollovers.py data.csv data_fixed.csv
  python fix_timestamp_rollovers.py ./data_directory/
  python fix_timestamp_rollovers.py ./data_directory/ --column time_us
        """
    )
    
    parser.add_argument('input', help='Input CSV file or directory')
    parser.add_argument('output', nargs='?', help='Output CSV file (for single file mode)')
    parser.add_argument('--column', '-c', default='t_us', 
                        help='Name of timestamp column (default: t_us)')
    parser.add_argument('--threshold', '-t', type=int, default=2**32,
                        help='Rollover threshold value (default: 2^32)')
    
    args = parser.parse_args()
    
    # Create fixer instance
    fixer = TimestampRolloverFixer(rollover_threshold=args.threshold)
    
    # Check if input is a file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single file mode
        if not input_path.suffix.lower() == '.csv':
            print(f"❌ Error: {args.input} is not a CSV file")
            sys.exit(1)
        
        success = fixer.fix_csv_file(args.input, args.output, args.column)
        sys.exit(0 if success else 1)
        
    elif input_path.is_dir():
        # Directory mode
        if args.output:
            print("❌ Error: Output file cannot be specified in directory mode")
            sys.exit(1)
        
        total, fixed = fixer.fix_directory(args.input, args.column)
        sys.exit(0)
        
    else:
        print(f"❌ Error: {args.input} is not a valid file or directory")
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print(__doc__)
        sys.exit(0)
    
    main()
