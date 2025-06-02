"""Calculate a per-breath protection factor for entries in breath_db.sqlite."""

import argparse
import sqlite3

import pandas as pd


SHIFT_SECONDS = 7.0


def ensure_column(conn: sqlite3.Connection, column: str) -> None:
    cur = conn.execute("PRAGMA table_info(breath_data)")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        conn.execute(f"ALTER TABLE breath_data ADD COLUMN {column} REAL")
        conn.commit()


def shift_particles(df: pd.DataFrame, seconds: float) -> pd.DataFrame:
    """Shift particle columns backwards by the given number of seconds."""
    if seconds <= 0:
        return df
    df = df.copy()
    dt = df["t_us"].diff().median() / 1e6
    if pd.isna(dt) or dt == 0:
        return df
    steps = int(round(seconds / dt))
    df["mask_particles"] = df["mask_particles"].shift(-steps)
    df["ambient_particles"] = df["ambient_particles"].shift(-steps)
    return df


def compute_and_store(conn: sqlite3.Connection) -> None:
    ensure_column(conn, "protection_factor")
    df = pd.read_sql_query("SELECT * FROM breath_data ORDER BY source_file, t_us", conn)

    out = []
    for source, df_file in df.groupby("source_file"):
        df_file = shift_particles(df_file, SHIFT_SECONDS)
        for b, df_breath in df_file.groupby("breath"):
            if pd.isna(b):
                continue
            max_mask = df_breath["mask_particles"].max()
            mean_amb = df_breath["ambient_particles"].mean()
            pf = None
            if pd.notna(max_mask) and pd.notna(mean_amb) and mean_amb != 0:
                pf = float(mean_amb) / float(max_mask)
            mask = (df["source_file"] == source) & (df["breath"] == b)
            df.loc[mask, "protection_factor"] = pf
    df.to_sql("breath_data", conn, if_exists="replace", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute protection factor for each breath")
    parser.add_argument("--db", default="breath_db.sqlite", help="Path to SQLite database")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    compute_and_store(conn)
    conn.close()


if __name__ == "__main__":
    main()
