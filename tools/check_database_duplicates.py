#!/usr/bin/env python3
"""
check_database_duplicates.py
----------------------------
Diagnostic script to check for duplicate data in breath_db.sqlite

This script will help identify if there are:
1. Duplicate source_file entries with different paths
2. Duplicate data rows for the same file
3. Multiple versions of the same file in the database

Usage:
    python check_database_duplicates.py breath_db.sqlite
"""

import argparse
import sqlite3
import pandas as pd
from pathlib import Path
from collections import Counter

def check_source_files(db_path: Path):
    """Check for duplicate source files and path variations."""
    conn = sqlite3.connect(db_path)
    try:
        print("=== Source File Analysis ===")
        
        # Get all unique source_file paths
        query = "SELECT DISTINCT source_file FROM breath_data ORDER BY source_file"
        df = pd.read_sql_query(query, conn)
        
        print(f"Total unique source_file entries: {len(df)}")
        
        # Extract filenames from paths
        df['filename'] = df['source_file'].apply(lambda x: Path(x).name)
        
        # Check for duplicate filenames with different paths
        filename_counts = Counter(df['filename'])
        duplicates = {name: count for name, count in filename_counts.items() if count > 1}
        
        if duplicates:
            print(f"\nFound {len(duplicates)} filenames with multiple path entries:")
            for filename, count in duplicates.items():
                print(f"\n  {filename} appears {count} times:")
                matching_paths = df[df['filename'] == filename]['source_file'].tolist()
                for i, path in enumerate(matching_paths, 1):
                    print(f"    {i}. {path}")
        else:
            print("\n✓ No duplicate filenames found - each file has a unique path")
        
        return df, duplicates
        
    finally:
        conn.close()

def check_data_duplicates(db_path: Path, source_files_df):
    """Check for duplicate data rows within files."""
    conn = sqlite3.connect(db_path)
    try:
        print("\n=== Data Row Analysis ===")
        
        total_duplicate_rows = 0
        files_with_duplicates = 0
        
        for _, row in source_files_df.iterrows():
            source_file = row['source_file']
            filename = row['filename']
            
            # Count total rows for this file
            total_query = "SELECT COUNT(*) as count FROM breath_data WHERE source_file = ?"
            total_count = pd.read_sql_query(total_query, conn, params=(source_file,))['count'].iloc[0]
            
            # Count unique rows based on timestamp
            unique_query = "SELECT COUNT(DISTINCT t_us) as count FROM breath_data WHERE source_file = ?"
            unique_count = pd.read_sql_query(unique_query, conn, params=(source_file,))['count'].iloc[0]
            
            if total_count != unique_count:
                duplicate_rows = total_count - unique_count
                total_duplicate_rows += duplicate_rows
                files_with_duplicates += 1
                print(f"  {filename}: {total_count} total rows, {unique_count} unique timestamps → {duplicate_rows} duplicates")
        
        if files_with_duplicates == 0:
            print("✓ No duplicate data rows found within files")
        else:
            print(f"\nFound {total_duplicate_rows} duplicate rows across {files_with_duplicates} files")
        
    finally:
        conn.close()

def check_breath_segments(db_path: Path):
    """Check breath_segments table for consistency."""
    conn = sqlite3.connect(db_path)
    try:
        print("\n=== Breath Segments Analysis ===")
        
        # Check if breath_segments table exists
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='breath_segments'")
        if not cursor.fetchone():
            print("⚠️  breath_segments table does not exist")
            return
        
        # Get breath segments info
        query = """
        SELECT source_file, 
               COUNT(*) as segment_count,
               MIN(breath) as min_breath,
               MAX(breath) as max_breath
        FROM breath_segments 
        GROUP BY source_file
        ORDER BY source_file
        """
        df = pd.read_sql_query(query, conn)
        
        print(f"Files with breath segments: {len(df)}")
        
        # Check for files in breath_data but not in breath_segments
        data_files_query = "SELECT DISTINCT source_file FROM breath_data"
        data_files = pd.read_sql_query(data_files_query, conn)['source_file'].tolist()
        
        segment_files = df['source_file'].tolist()
        missing_segments = [f for f in data_files if f not in segment_files]
        
        if missing_segments:
            print(f"\n⚠️  {len(missing_segments)} files have data but no breath segments:")
            for file in missing_segments[:5]:  # Show first 5
                filename = Path(file).name
                print(f"    - {filename}")
            if len(missing_segments) > 5:
                print(f"    ... and {len(missing_segments) - 5} more")
        else:
            print("✓ All data files have corresponding breath segments")
        
        # Show sample of breath segments
        if len(df) > 0:
            print(f"\nSample breath segment counts:")
            for _, row in df.head(5).iterrows():
                filename = Path(row['source_file']).name
                print(f"  {filename}: {row['segment_count']} segments (breath {row['min_breath']}-{row['max_breath']})")
        
    finally:
        conn.close()

def show_database_summary(db_path: Path):
    """Show overall database statistics."""
    conn = sqlite3.connect(db_path)
    try:
        print(f"\n=== Database Summary ===")
        print(f"Database: {db_path}")
        
        # Table sizes
        tables = ['breath_data', 'breath_segments']
        for table in tables:
            try:
                query = f"SELECT COUNT(*) as count FROM {table}"
                count = pd.read_sql_query(query, conn)['count'].iloc[0]
                print(f"{table}: {count:,} rows")
            except:
                print(f"{table}: table does not exist")
        
        # Database file size
        db_size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"Database size: {db_size_mb:.1f} MB")
        
    finally:
        conn.close()

def main():
    parser = argparse.ArgumentParser(description="Check for duplicates in breath database")
    parser.add_argument("database", help="SQLite database file")
    args = parser.parse_args()
    
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database {db_path} does not exist")
        return 1
    
    print(f"Analyzing database: {db_path}\n")
    
    # Check database summary
    show_database_summary(db_path)
    
    # Check for duplicate source files
    source_files_df, duplicates = check_source_files(db_path)
    
    # Check for duplicate data rows
    check_data_duplicates(db_path, source_files_df)
    
    # Check breath segments
    check_breath_segments(db_path)
    
    # Recommendations
    print(f"\n=== Recommendations ===")
    if duplicates:
        print("🔧 You have duplicate filenames with different paths.")
        print("   This likely happened from processing files multiple times or from different directories.")
        print("   Consider cleaning up the database to keep only the most recent entries.")
    
    print(f"\n💡 If you see duplicate entries in the verification script, it's likely due to")
    print(f"   the filename matching logic picking up multiple path variations of the same file.")
    
    return 0

if __name__ == "__main__":
    exit(main())