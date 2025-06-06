import subprocess
from pathlib import Path

def process_all_csvs(data_dir, db_path, segment_script):
    data_path = Path(data_dir)
    csv_files = list(data_path.glob('*.csv'))
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        subprocess.run([
            'python', segment_script,
            str(csv_file),
            '--db', db_path,
            '--prominence', '15.0',
        ], check=True)
    print("All files processed.")

if __name__ == "__main__":
    DATA_DIR = "data"
    DB_PATH = "breath_db.sqlite"
    SEGMENT_SCRIPT = "segment_breaths.py"
    process_all_csvs(DATA_DIR, DB_PATH, SEGMENT_SCRIPT)
