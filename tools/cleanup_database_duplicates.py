#!/usr/bin/env python3
"""
cleanup_database_duplicates.py
------------------------------
Remove duplicate entries from breath_db.sqlite, keeping only the most recent version
of each file based on when it was last processed.

Usage:
    python cleanup_database_duplicates.py breath_db.sqlite
    python cleanup_database_duplicates.py breath_db.sqlite --dry-run
    python cleanup_database_duplicates.py breath_db.sqlite --backup
"""

import argparse
import sqlite3
import pandas as pd
import shutil
from pathlib import Path
from collections import Counter

def backup_database(db_path: Path) -> Path:
    """Create a backup of the database."""
    backup_path = db_path.with_suffix(f"{db_path.suffix}.backup")
    shutil.copy2(db_path, backup_path)
    print(f"Created backup: {backup_path}")
    return backup_path

def find_duplicates(db_path: Path):
    """Find all duplicate files in the database."""
    conn = sqlite3.connect(db_path)
    try:
        # Get all source files with their data counts
        query = """
        SELECT source_file, 
               COUNT(*) as row_count,
               MIN(rowid) as first_rowid,
               MAX(rowid) as last_rowid
        FROM breath_data 
        GROUP BY source_file
        ORDER BY source_file
        """
        df = pd.read_sql_query(query, conn)
        
        # Extract filenames
        df['filename'] = df['source_file'].apply(lambda x: Path(x).name)
        
        # Find files with duplicate filenames
        filename_counts = Counter(df['filename'])
        duplicate_filenames = {name: count for name, count in filename_counts.items() if count > 1}
        
        duplicate_groups = {}
        for filename in duplicate_filenames.keys():
            # Get all entries for this filename
            file_entries = df[df['filename'] == filename].copy()
            file_entries = file_entries.sort_values('last_rowid')  # Sort by most recent insertion
            duplicate_groups[filename] = file_entries
        
        return duplicate_groups
        
    finally:
        conn.close()

def cleanup_duplicates(db_path: Path, duplicate_groups: dict, dry_run: bool = False):
    """Remove duplicate entries, keeping the most recent version of each file."""
    conn = sqlite3.connect(db_path)
    try:
        total_rows_to_delete = 0
        files_to_clean = 0
        
        print("\n=== Cleanup Plan ===")
        
        for filename, file_entries in duplicate_groups.items():
            print(f"\nFile: {filename}")
            print(f"  Found {len(file_entries)} versions:")
            
            # Keep the most recent version (highest rowid)
            keep_entry = file_entries.iloc[-1]  # Last entry (highest rowid)
            delete_entries = file_entries.iloc[:-1]  # All but the last
            
            print(f"  KEEP:   {keep_entry['source_file']} ({keep_entry['row_count']:,} rows)")
            
            for _, entry in delete_entries.iterrows():
                print(f"  DELETE: {entry['source_file']} ({entry['row_count']:,} rows)")
                total_rows_to_delete += entry['row_count']
            
            files_to_clean += 1
        
        print(f"\nSummary:")
        print(f"  Files to clean: {files_to_clean}")
        print(f"  Total rows to delete: {total_rows_to_delete:,}")
        
        if dry_run:
            print("\nDRY RUN - No changes made")
            return
        
        # Perform the cleanup
        print(f"\nPerforming cleanup...")
        
        deleted_rows = 0
        for filename, file_entries in duplicate_groups.items():
            delete_entries = file_entries.iloc[:-1]  # All but the most recent
            
            for _, entry in delete_entries.iterrows():
                source_file = entry['source_file']
                print(f"  Deleting {entry['row_count']:,} rows from {source_file}")
                
                # Delete from breath_data
                cursor = conn.execute("DELETE FROM breath_data WHERE source_file = ?", (source_file,))
                rows_deleted = cursor.rowcount
                deleted_rows += rows_deleted
                
                # Delete from breath_segments (if exists)
                try:
                    cursor = conn.execute("DELETE FROM breath_segments WHERE source_file = ?", (source_file,))
                    segments_deleted = cursor.rowcount
                    if segments_deleted > 0:
                        print(f"    Also deleted {segments_deleted} breath segments")
                except sqlite3.OperationalError:
                    pass  # breath_segments table might not exist
        
        conn.commit()
        print(f"\n✓ Cleanup complete!")
        print(f"  Deleted {deleted_rows:,} rows from breath_data")
        
        # Show final statistics
        final_files_query = "SELECT COUNT(DISTINCT source_file) as count FROM breath_data"
        final_files = pd.read_sql_query(final_files_query, conn)['count'].iloc[0]
        
        final_rows_query = "SELECT COUNT(*) as count FROM breath_data"
        final_rows = pd.read_sql_query(final_rows_query, conn)['count'].iloc[0]
        
        print(f"  Final database: {final_files} unique files, {final_rows:,} total rows")
        
    finally:
        conn.close()

def verify_cleanup(db_path: Path):
    """Verify that cleanup was successful."""
    conn = sqlite3.connect(db_path)
    try:
        print(f"\n=== Verification ===")
        
        # Check for remaining duplicates
        query = "SELECT DISTINCT source_file FROM breath_data ORDER BY source_file"
        df = pd.read_sql_query(query, conn)
        df['filename'] = df['source_file'].apply(lambda x: Path(x).name)
        
        filename_counts = Counter(df['filename'])
        remaining_duplicates = {name: count for name, count in filename_counts.items() if count > 1}
        
        if remaining_duplicates:
            print(f"⚠️  Still found {len(remaining_duplicates)} files with duplicates:")
            for filename, count in remaining_duplicates.items():
                print(f"    {filename}: {count} versions")
        else:
            print("✓ No duplicate filenames found - cleanup successful!")
        
        # Show database size reduction
        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"✓ Database size: {db_size_mb:.1f} MB")
        
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate entries in breath database")
    parser.add_argument("database", help="SQLite database file")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be done without making changes")
    parser.add_argument("--backup", action="store_true",
                       help="Create a backup before cleanup")
    args = parser.parse_args()
    
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database {db_path} does not exist")
        return 1
    
    print(f"Analyzing database: {db_path}")
    
    # Create backup if requested
    if args.backup and not args.dry_run:
        backup_database(db_path)
    
    # Find duplicates
    print(f"\nScanning for duplicate files...")
    duplicate_groups = find_duplicates(db_path)
    
    if not duplicate_groups:
        print("✓ No duplicate files found - database is clean!")
        return 0
    
    print(f"Found {len(duplicate_groups)} files with duplicates")
    
    # Clean up duplicates
    cleanup_duplicates(db_path, duplicate_groups, args.dry_run)
    
    # Verify cleanup (if not dry run)
    if not args.dry_run:
        verify_cleanup(db_path)
        
        print(f"\n💡 Tip: You can now re-run the verification script and it should")
        print(f"   only show one version of each file!")
    
    return 0

if __name__ == "__main__":
    exit(main())