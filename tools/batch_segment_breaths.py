import os
import sys
import argparse
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Batch process CSV files with segment_breaths.py.")
    parser.add_argument("directory", type=str, help="Directory containing CSV files to process.")
    parser.add_argument("--extra-args", type=str, default="", help="Extra arguments to pass to segment_breaths.py (optional)")
    parser.add_argument("--no-confirm", action="store_true", help="Skip user confirmation after each file (old behavior)")
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"Error: {directory} is not a directory.")
        sys.exit(1)

    all_csv_files = sorted(directory.glob("*.csv"))
    # Filter out files containing "init", "rainbow", or "zero" in their names
    skip_keywords = ["init", "rainbow", "zero"]
    filtered_files = [f for f in all_csv_files if not any(keyword in f.name.lower() for keyword in skip_keywords)]
    
    # Handle duplicate files where one has "_fixed" suffix - prefer the "_fixed" version
    final_files = []
    duplicate_skipped = []
    
    # Create a mapping of base names to files
    file_groups = {}
    for f in filtered_files:
        if f.name.endswith("_fixed.csv"):
            base_name = f.name[:-10] + ".csv"  # Remove "_fixed.csv" and add ".csv"
        else:
            base_name = f.name
        
        if base_name not in file_groups:
            file_groups[base_name] = []
        file_groups[base_name].append(f)
    
    # For each group, prefer the "_fixed" version if it exists
    for base_name, files in file_groups.items():
        if len(files) == 1:
            final_files.append(files[0])
        else:
            # Multiple files with same base name - prefer "_fixed" version
            fixed_files = [f for f in files if f.name.endswith("_fixed.csv")]
            non_fixed_files = [f for f in files if not f.name.endswith("_fixed.csv")]
            
            if fixed_files:
                # Use the "_fixed" version
                final_files.append(fixed_files[0])  # Should only be one
                duplicate_skipped.extend(non_fixed_files)
                if len(fixed_files) > 1:
                    duplicate_skipped.extend(fixed_files[1:])  # Skip extra "_fixed" files
            else:
                # No "_fixed" version, use the first one
                final_files.append(files[0])
                duplicate_skipped.extend(files[1:])
    
    csv_files = sorted(final_files)
    skipped_files = [f for f in all_csv_files if any(keyword in f.name.lower() for keyword in skip_keywords)]
    if skipped_files:
        print(f"Skipping {len(skipped_files)} files containing 'init', 'rainbow', or 'zero':")
        for f in skipped_files:
            print(f"  - {f.name}")
    
    if duplicate_skipped:
        print(f"\nSkipping {len(duplicate_skipped)} files that have '_fixed' versions:")
        for f in duplicate_skipped:
            print(f"  - {f.name}")
    
    if not csv_files:
        print(f"No CSV files found in {directory} after filtering")
        sys.exit(0)

    print(f"\nProcessing {len(csv_files)} CSV files...")

    script_path = Path(__file__).parent / ".." / "analysis" / "segment_breaths.py"
    if not script_path.exists():
        print(f"segment_breaths.py not found at {script_path}")
        sys.exit(1)

    processed_count = 0
    failed_count = 0
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        # First attempt: non-interactive mode
        cmd = [sys.executable, str(script_path), str(csv_file), "--peak-height", "6.0", "--trough-height", "6.0", "--distance_cutoff", "3.0", "--peak-min-distance", "500"]
        if args.extra_args:
            cmd.extend(args.extra_args.split())
        
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"segment_breaths.py failed for {csv_file.name}")
            failed_count += 1
            continue
        
        # Ask for user confirmation unless --no-confirm is specified
        if args.no_confirm:
            print(f"Done: {csv_file.name}")
            processed_count += 1
        else:
            # Ask for user confirmation
            while True:
                user_choice = input(f"Accept results for {csv_file.name}? (y/n/q for quit): ").lower().strip()
                
                if user_choice in ['y', 'yes']:
                    print(f"Accepted: {csv_file.name}")
                    processed_count += 1
                    break
                elif user_choice in ['n', 'no']:
                    print(f"Reprocessing {csv_file.name} in interactive mode...")
                    # Reprocess in interactive mode
                    interactive_cmd = cmd + ["--interactive"]
                    interactive_result = subprocess.run(interactive_cmd)
                    if interactive_result.returncode != 0:
                        print(f"Interactive reprocessing failed for {csv_file.name}")
                        failed_count += 1
                    else:
                        print(f"Interactive processing completed: {csv_file.name}")
                        processed_count += 1
                    break
                elif user_choice in ['q', 'quit']:
                    print("Batch processing cancelled by user.")
                    return
                else:
                    print("Please enter 'y' (yes), 'n' (no), or 'q' (quit)")

    print(f"\n=== Batch Processing Complete ===")
    print(f"Successfully processed: {processed_count} files")
    if failed_count > 0:
        print(f"Failed: {failed_count} files")
    if skipped_files:
        print(f"Skipped (keywords): {len(skipped_files)} files")
    if duplicate_skipped:
        print(f"Skipped (duplicates): {len(duplicate_skipped)} files")

if __name__ == "__main__":
    main()
