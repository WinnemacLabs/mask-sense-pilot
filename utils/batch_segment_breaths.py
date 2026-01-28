import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch process CSV files with segment_breaths.py.")
    parser.add_argument("directory", type=str, help="Directory containing CSV files to process.")
    parser.add_argument("--extra-args", type=str, default="", help="Extra arguments to pass to segment_breaths.py (optional)")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory.")
        sys.exit(1)

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        sys.exit(0)

    script_path = Path(__file__).parent / "analysis" / "segment_breaths.py"
    if not script_path.exists():
        print(f"segment_breaths.py not found at {script_path}")
        sys.exit(1)

    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        cmd = [sys.executable, str(script_path), str(csv_file), "--interactive"]
        if args.extra_args:
            cmd.extend(args.extra_args.split())
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"segment_breaths.py failed for {csv_file.name}")
        else:
            print(f"Done: {csv_file.name}")

if __name__ == "__main__":
    main()
