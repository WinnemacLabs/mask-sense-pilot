#!/usr/bin/env python3
"""
analysis_pipeline.py
--------------------
Comprehensive pipeline to process respiratory sensor data through all analysis steps.

This script orchestrates the following processing steps in order:
1. add_headers - Add metadata headers to CSV files
2. fix_timestamp_rollovers - Fix timestamp discontinuities
3. add_t_us_zero_column - Add relative timestamp column
4. align_sensor_data - Apply sensor lag correction
5. batch_segment_breaths - Segment breathing data and store in database

Usage:
    python analysis_pipeline.py data/P0/ --database breath_data.sqlite
    python analysis_pipeline.py data/ --recursive --database breath_data.sqlite
    python analysis_pipeline.py data/P0/ --skip-steps align_sensor_data --database breath_data.sqlite
"""

import argparse
import subprocess
import sys
from pathlib import Path
import time
from typing import List, Optional
import shutil

class AnalysisPipeline:
    """Main pipeline orchestrator for respiratory data analysis."""
    
    def __init__(self, data_directory: Path, database_path: Optional[Path] = None, 
                 recursive: bool = False, lag_correction: float = 7.5):
        self.data_directory = data_directory
        self.database_path = database_path
        self.recursive = recursive
        self.lag_correction = lag_correction
        
        # Find script paths relative to this file
        self.script_dir = Path(__file__).parent
        self.scripts = {
            'add_headers': self.script_dir / 'tools' / 'add_headers.py',
            'fix_timestamp_rollovers': self.script_dir / 'tools' / 'fix_timestamp_rollovers.py',
            'add_t_us_zero_column': self.script_dir / 'tools' / 'add_t_us_zero_column.py',
            'align_sensor_data': self.script_dir / 'tools' / 'align_sensor_data.py',
            'batch_segment_breaths': self.script_dir / 'tools' / 'batch_segment_breaths.py'
        }
        
        # Check if all scripts exist
        self.validate_scripts()
        
        # Pipeline configuration
        self.steps = [
            'add_headers',
            'fix_timestamp_rollovers', 
            'add_t_us_zero_column',
            'align_sensor_data',
            'batch_segment_breaths'
        ]
        
        # Step descriptions
        self.step_descriptions = {
            'add_headers': 'Adding metadata headers to CSV files',
            'fix_timestamp_rollovers': 'Fixing timestamp discontinuities',
            'add_t_us_zero_column': 'Adding relative timestamp columns',
            'align_sensor_data': 'Applying sensor lag correction',
            'batch_segment_breaths': 'Segmenting breathing data and storing in database'
        }
    
    def validate_scripts(self):
        """Check that all required scripts exist."""
        missing_scripts = []
        for name, path in self.scripts.items():
            if not path.exists():
                missing_scripts.append(f"{name}: {path}")
        
        if missing_scripts:
            print("❌ Missing required scripts:")
            for script in missing_scripts:
                print(f"   {script}")
            sys.exit(1)
    
    def validate_inputs(self):
        """Validate input directory and database path."""
        if not self.data_directory.exists():
            print(f"❌ Data directory does not exist: {self.data_directory}")
            return False
        
        if not self.data_directory.is_dir():
            print(f"❌ Path is not a directory: {self.data_directory}")
            return False
        
        # Check for CSV files
        if self.recursive:
            csv_files = list(self.data_directory.rglob("*.csv"))
        else:
            csv_files = list(self.data_directory.glob("*.csv"))
        
        if not csv_files:
            print(f"❌ No CSV files found in {self.data_directory}")
            return False
        
        print(f"✅ Found {len(csv_files)} CSV files to process")
        
        # Validate database path for steps that need it
        if self.database_path and not self.database_path.parent.exists():
            print(f"❌ Database directory does not exist: {self.database_path.parent}")
            return False
        
        return True
    
    def run_script(self, script_name: str, args: List[str], description: str) -> bool:
        """Run a single script with the given arguments."""
        script_path = self.scripts[script_name]
        cmd = [sys.executable, str(script_path)] + args
        
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"Command: {' '.join(cmd)}")
        print('='*60)
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, check=True, capture_output=False)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"\n✅ {description} completed successfully ({duration:.1f}s)")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {description} failed with return code {e.returncode}")
            return False
        except Exception as e:
            print(f"\n❌ {description} failed with error: {e}")
            return False
    
    def step_add_headers(self) -> bool:
        """Step 1: Add headers to CSV files."""
        args = [str(self.data_directory)]
        if self.recursive:
            args.append('--recursive')
        
        return self.run_script('add_headers', args, self.step_descriptions['add_headers'])
    
    def step_fix_timestamp_rollovers(self) -> bool:
        """Step 2: Fix timestamp rollovers."""
        args = [str(self.data_directory)]
        if self.recursive:
            args.append('--recursive')
        
        return self.run_script('fix_timestamp_rollovers', args, 
                              self.step_descriptions['fix_timestamp_rollovers'])
    
    def step_add_t_us_zero_column(self) -> bool:
        """Step 3: Add t_us_zero column."""
        args = [str(self.data_directory)]
        if self.recursive:
            args.append('--recursive')
        
        return self.run_script('add_t_us_zero_column', args,
                              self.step_descriptions['add_t_us_zero_column'])
    
    def step_align_sensor_data(self) -> bool:
        """Step 4: Apply sensor lag correction to all CSV files."""
        # Find all CSV files to process
        if self.recursive:
            csv_files = list(self.data_directory.rglob("*.csv"))
        else:
            csv_files = list(self.data_directory.glob("*.csv"))
        
        # Filter out files we want to skip
        skip_keywords = ['init', 'rainbow', 'zero', '_aligned']
        filtered_files = []
        for f in csv_files:
            if not any(keyword in f.name.lower() for keyword in skip_keywords):
                filtered_files.append(f)
        
        print(f"Processing {len(filtered_files)} files for sensor alignment...")
        
        success_count = 0
        for csv_file in filtered_files:
            # Create aligned filename
            aligned_file = csv_file.parent / f"{csv_file.stem}_aligned{csv_file.suffix}"
            
            args = [
                str(csv_file),
                '--manual-lag', str(self.lag_correction),
                '--output', str(aligned_file)
            ]
            
            print(f"\nAligning {csv_file.name}...")
            cmd = [sys.executable, str(self.scripts['align_sensor_data'])] + args
            
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"  ✅ Created {aligned_file.name}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Failed to align {csv_file.name}: {e}")
                if e.stdout:
                    print(f"     stdout: {e.stdout}")
                if e.stderr:
                    print(f"     stderr: {e.stderr}")
        
        print(f"\n✅ Sensor alignment completed: {success_count}/{len(filtered_files)} files")
        return success_count > 0
    
    def step_batch_segment_breaths(self) -> bool:
        """Step 5: Batch segment breathing data."""
        if not self.database_path:
            print("❌ Database path required for breath segmentation step")
            return False
        
        args = [
            str(self.data_directory),
            '--database', str(self.database_path),
            '--no-confirm'  # Run in batch mode
        ]
        
        return self.run_script('batch_segment_breaths', args,
                              self.step_descriptions['batch_segment_breaths'])
    
    def create_backup(self) -> Optional[Path]:
        """Create a backup of the data directory before processing."""
        backup_dir = self.data_directory.parent / f"{self.data_directory.name}_backup_{int(time.time())}"
        
        try:
            print(f"Creating backup: {backup_dir}")
            shutil.copytree(self.data_directory, backup_dir)
            print(f"✅ Backup created successfully")
            return backup_dir
        except Exception as e:
            print(f"⚠️  Backup failed: {e}")
            return None
    
    def run_pipeline(self, skip_steps: List[str] = None, backup: bool = False, 
                    dry_run: bool = False) -> bool:
        """Run the complete analysis pipeline."""
        if skip_steps is None:
            skip_steps = []
        
        print(f"🚀 Starting Analysis Pipeline")
        print(f"📁 Data directory: {self.data_directory}")
        if self.database_path:
            print(f"🗄️  Database: {self.database_path}")
        print(f"🔄 Recursive: {self.recursive}")
        print(f"⏱️  Lag correction: {self.lag_correction}s")
        
        if skip_steps:
            print(f"⏭️  Skipping steps: {', '.join(skip_steps)}")
        
        # Validate inputs
        if not self.validate_inputs():
            return False
        
        if dry_run:
            print(f"\n🧪 DRY RUN MODE - No changes will be made")
            print(f"Would execute these steps:")
            for step in self.steps:
                if step not in skip_steps:
                    print(f"  ✓ {step}: {self.step_descriptions[step]}")
                else:
                    print(f"  ⏭️  {step}: {self.step_descriptions[step]} (SKIPPED)")
            return True
        
        # Create backup if requested
        backup_path = None
        if backup:
            backup_path = self.create_backup()
        
        # Run pipeline steps
        start_time = time.time()
        failed_steps = []
        
        step_methods = {
            'add_headers': self.step_add_headers,
            'fix_timestamp_rollovers': self.step_fix_timestamp_rollovers,
            'add_t_us_zero_column': self.step_add_t_us_zero_column,
            'align_sensor_data': self.step_align_sensor_data,
            'batch_segment_breaths': self.step_batch_segment_breaths
        }
        
        for step in self.steps:
            if step in skip_steps:
                print(f"\n⏭️  Skipping step: {step}")
                continue
            
            success = step_methods[step]()
            if not success:
                failed_steps.append(step)
                
                # Ask user if they want to continue
                response = input(f"\n❓ Step '{step}' failed. Continue with remaining steps? (y/n): ")
                if response.lower() not in ['y', 'yes']:
                    break
        
        # Pipeline summary
        end_time = time.time()
        total_duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print(f"🏁 PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"Total duration: {total_duration/60:.1f} minutes")
        
        if failed_steps:
            print(f"❌ Failed steps: {', '.join(failed_steps)}")
            if backup_path:
                print(f"💾 Backup available at: {backup_path}")
        else:
            print(f"✅ All steps completed successfully!")
        
        print(f"📁 Results in: {self.data_directory}")
        if self.database_path:
            print(f"🗄️  Database: {self.database_path}")
        
        return len(failed_steps) == 0

def main():
    parser = argparse.ArgumentParser(
        description="Complete analysis pipeline for respiratory sensor data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps (in order):
  1. add_headers          - Add metadata headers to CSV files
  2. fix_timestamp_rollovers - Fix timestamp discontinuities  
  3. add_t_us_zero_column - Add relative timestamp columns
  4. align_sensor_data    - Apply sensor lag correction
  5. batch_segment_breaths - Segment breathing data and store in database

Examples:
  # Process single participant
  python analysis_pipeline.py data/P0/ --database breath_data.sqlite
  
  # Process all participants recursively
  python analysis_pipeline.py data/ --recursive --database breath_data.sqlite
  
  # Skip sensor alignment step
  python analysis_pipeline.py data/P0/ --skip-steps align_sensor_data --database breath_data.sqlite
  
  # Dry run to see what would be processed
  python analysis_pipeline.py data/P0/ --dry-run
        """
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing CSV files to process"
    )
    
    parser.add_argument(
        "--database",
        help="SQLite database file for storing breath segmentation results"
    )
    
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process subdirectories recursively"
    )
    
    parser.add_argument(
        "--lag-correction",
        type=float,
        default=7.5,
        help="Sensor lag correction in seconds (default: 7.5)"
    )
    
    parser.add_argument(
        "--skip-steps",
        nargs="*",
        choices=['add_headers', 'fix_timestamp_rollovers', 'add_t_us_zero_column', 
                'align_sensor_data', 'batch_segment_breaths'],
        default=[],
        help="Skip specific pipeline steps"
    )
    
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup of data directory before processing"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without making changes"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    data_directory = Path(args.directory)
    database_path = Path(args.database) if args.database else None
    
    # Create pipeline
    pipeline = AnalysisPipeline(
        data_directory=data_directory,
        database_path=database_path,
        recursive=args.recursive,
        lag_correction=args.lag_correction
    )
    
    # Run pipeline
    success = pipeline.run_pipeline(
        skip_steps=args.skip_steps,
        backup=args.backup,
        dry_run=args.dry_run
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()