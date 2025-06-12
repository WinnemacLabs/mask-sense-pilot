#!/usr/bin/env python3
"""
migrate_database.py
-------------------
Migrate existing breath_db.sqlite database from the old format (with breath column
in main table) to the new normalized format (separate breath_segments table).

This script will:
1. Create a new breath_segments table
2. Extract breath segment data from existing breath_data table
3. Remove the breath column from the breath_data table
4. Preserve all existing raw data

Usage:
    python migrate_database.py breath_db.sqlite
    python migrate_database.py breath_db.sqlite --backup
"""

import argparse
import sqlite3
import shutil
from pathlib import Path
import pandas as pd
import numpy as np


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Check if a table exists in the database."""
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    )
    return cur.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Check if a column exists in a table."""
    cur = conn.execute(f"PRAGMA table_info({table})")
    columns = [row[1] for row in cur.fetchall()]
    return column in columns


def create_breath_segments_table(conn: sqlite3.Connection) -> None:
    """Create the breath_segments table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS breath_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            breath INTEGER NOT NULL,
            breath_start_us INTEGER NOT NULL,
            breath_end_us INTEGER NOT NULL,
            UNIQUE(source_file, breath)
        )
    """)
    print("Created breath_segments table")


def extract_breath_segments(conn: sqlite3.Connection) -> None:
    """Extract breath segment data from the main table."""
    # Check if breath column exists
    if not column_exists(conn, "breath_data", "breath"):
        print("No 'breath' column found in breath_data table - migration not needed")
        return
    
    print("Extracting breath segments from breath_data table...")
    
    # Get all files that have breath data
    cur = conn.execute("SELECT DISTINCT source_file FROM breath_data WHERE breath IS NOT NULL")
    source_files = [row[0] for row in cur.fetchall()]
    
    segments_inserted = 0
    
    for source_file in source_files:
        print(f"Processing {source_file}...")
        
        # Get data for this file ordered by time
        df = pd.read_sql_query("""
            SELECT t_us, breath FROM breath_data 
            WHERE source_file = ? AND breath IS NOT NULL 
            ORDER BY t_us
        """, conn, params=(source_file,))
        
        if df.empty:
            continue
        
        # Find breath boundaries
        breath_changes = df['breath'].diff() != 0
        breath_starts = df[breath_changes].copy()
        
        # Process each breath
        current_breath = None
        breath_start_us = None
        
        for idx, row in df.iterrows():
            if current_breath != row['breath']:
                # End previous breath if it exists
                if current_breath is not None and breath_start_us is not None:
                    # Find the last timestamp of the previous breath
                    prev_breath_data = df[df['breath'] == current_breath]
                    breath_end_us = int(prev_breath_data['t_us'].iloc[-1])
                    
                    # Insert the breath segment
                    conn.execute("""
                        INSERT OR REPLACE INTO breath_segments 
                        (source_file, breath, breath_start_us, breath_end_us)
                        VALUES (?, ?, ?, ?)
                    """, (source_file, int(current_breath), breath_start_us, breath_end_us))
                    segments_inserted += 1
                
                # Start new breath
                current_breath = row['breath']
                breath_start_us = int(row['t_us'])
        
        # Handle the last breath
        if current_breath is not None and breath_start_us is not None:
            breath_end_us = int(df['t_us'].iloc[-1])
            conn.execute("""
                INSERT OR REPLACE INTO breath_segments 
                (source_file, breath, breath_start_us, breath_end_us)
                VALUES (?, ?, ?, ?)
            """, (source_file, int(current_breath), breath_start_us, breath_end_us))
            segments_inserted += 1
    
    conn.commit()
    print(f"Extracted {segments_inserted} breath segments")


def remove_breath_column(conn: sqlite3.Connection) -> None:
    """Remove the breath column from the breath_data table."""
    if not column_exists(conn, "breath_data", "breath"):
        print("No 'breath' column found - already migrated")
        return
    
    print("Removing breath column from breath_data table...")
    
    # SQLite doesn't support DROP COLUMN directly, so we need to:
    # 1. Create a new table without the breath column
    # 2. Copy data to the new table
    # 3. Drop the old table
    # 4. Rename the new table
    
    # Get the current table schema (excluding the breath column)
    cur = conn.execute("PRAGMA table_info(breath_data)")
    columns = []
    for row in cur.fetchall():
        col_name = row[1]
        col_type = row[2]
        if col_name != 'breath':  # Skip the breath column
            columns.append(f"{col_name} {col_type}")
    
    # Create new table
    new_table_sql = f"""
        CREATE TABLE breath_data_new (
            {', '.join(columns)}
        )
    """
    conn.execute(new_table_sql)
    
    # Copy data (excluding breath column)
    column_names = [col.split()[0] for col in columns]
    conn.execute(f"""
        INSERT INTO breath_data_new ({', '.join(column_names)})
        SELECT {', '.join(column_names)} FROM breath_data
    """)
    
    # Drop old table and rename new one
    conn.execute("DROP TABLE breath_data")
    conn.execute("ALTER TABLE breath_data_new RENAME TO breath_data")
    
    conn.commit()
    print("Removed breath column from breath_data table")


def verify_migration(conn: sqlite3.Connection) -> None:
    """Verify the migration was successful."""
    print("\nVerifying migration...")
    
    # Check tables exist
    if not table_exists(conn, "breath_data"):
        print("ERROR: breath_data table missing!")
        return
    
    if not table_exists(conn, "breath_segments"):
        print("ERROR: breath_segments table missing!")
        return
    
    # Check breath column is gone
    if column_exists(conn, "breath_data", "breath"):
        print("WARNING: breath column still exists in breath_data table")
    else:
        print("✓ breath column successfully removed from breath_data table")
    
    # Count records
    cur = conn.execute("SELECT COUNT(*) FROM breath_data")
    data_count = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(*) FROM breath_segments")
    segments_count = cur.fetchone()[0]
    
    cur = conn.execute("SELECT COUNT(DISTINCT source_file) FROM breath_segments")
    files_count = cur.fetchone()[0]
    
    print(f"✓ breath_data table: {data_count:,} records")
    print(f"✓ breath_segments table: {segments_count} segments from {files_count} files")
    
    # Show sample breath segments
    if segments_count > 0:
        print("\nSample breath segments:")
        cur = conn.execute("""
            SELECT source_file, breath, breath_start_us, breath_end_us 
            FROM breath_segments 
            ORDER BY source_file, breath 
            LIMIT 5
        """)
        for row in cur.fetchall():
            source_file, breath, start_us, end_us = row
            duration_s = (end_us - start_us) / 1e6
            print(f"  {source_file}: breath {breath}, {duration_s:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Migrate breath database to normalized structure")
    parser.add_argument("database", help="Path to the SQLite database file")
    parser.add_argument("--backup", action="store_true", help="Create a backup before migration")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()
    
    db_path = Path(args.database)
    if not db_path.exists():
        print(f"Error: Database file {db_path} does not exist")
        return 1
    
    # Create backup if requested
    if args.backup:
        backup_path = db_path.with_suffix(f"{db_path.suffix}.backup")
        shutil.copy2(db_path, backup_path)
        print(f"Created backup: {backup_path}")
    
    if args.dry_run:
        print("DRY RUN - No changes will be made")
        print(f"Would migrate database: {db_path}")
        return 0
    
    print(f"Migrating database: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Check if migration is needed
        if not table_exists(conn, "breath_data"):
            print("Error: breath_data table does not exist")
            return 1
        
        if table_exists(conn, "breath_segments") and not column_exists(conn, "breath_data", "breath"):
            print("Database appears to already be migrated")
            verify_migration(conn)
            return 0
        
        # Perform migration
        create_breath_segments_table(conn)
        extract_breath_segments(conn)
        remove_breath_column(conn)
        
        # Verify results
        verify_migration(conn)
        
        conn.close()
        print("\nMigration completed successfully!")
        
    except Exception as e:
        print(f"Error during migration: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
