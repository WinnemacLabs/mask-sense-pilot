#!/usr/bin/env python3
"""
Quick validation script for pressure fit CSV files.
Usage: python quick_validate.py [directory_path]
"""

import sys
from pathlib import Path
from data_validation import CSVValidator


def main():
    """Quick validation for a directory."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Default to the data directory
        directory = "data"
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory '{directory}' not found.")
        print("Available directories:")
        current_dir = Path(".")
        for item in current_dir.iterdir():
            if item.is_dir():
                print(f"  {item.name}/")
        return
    
    validator = CSVValidator()
    validator.validate_directory(directory_path)


if __name__ == "__main__":
    main()
