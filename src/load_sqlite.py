
from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATABASE_DIR = PROJECT_ROOT / "database"

DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DB_FILE = DATABASE_DIR / "network_traffic.db"


def main():
    # Load the prepared data
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv")

    # Reattach the target for SQL analysis
    train_df = X_train.copy()
    train_df["label"] = y_train["label"]

    # Create or open the SQLite database
    with sqlite3.connect(DB_FILE) as connection:
        train_df.to_sql(
            "network_traffic",
            connection,
            if_exists="replace",
            index=False
        )

        print("Database created:", DB_FILE)
        print("Table created: network_traffic")
        print("Rows loaded:", len(train_df))

        # Query 1: Normal versus attack traffic
        print("\n1. TRAFFIC CLASS COUNTS")
        query = """
        SELECT
            CASE
                WHEN label = 0 THEN 'Normal'
                ELSE 'Attack'
            END AS traffic_type,
            COUNT(*) AS total_records
        FROM network_traffic
        GROUP BY label
        ORDER BY label;
        """
        print(pd.read_sql_query(query, connection))

        # Query 2: Top 10 protocols
        print("\n2. TOP 10 PROTOCOLS")
        query = """
        SELECT proto, COUNT(*) AS total_records
        FROM network_traffic
        GROUP BY proto
        ORDER BY total_records DESC
        LIMIT 10;
        """
        print(pd.read_sql_query(query, connection))

        # Query 3: Average traffic rate by class
        print("\n3. AVERAGE RATE BY TRAFFIC CLASS")
        query = """
        SELECT
            CASE
                WHEN label = 0 THEN 'Normal'
                ELSE 'Attack'
            END AS traffic_type,
            ROUND(AVG(rate), 3) AS average_rate
        FROM network_traffic
        GROUP BY label
        ORDER BY label;
        """
        print(pd.read_sql_query(query, connection))

    print("\nSQL loading and analysis completed.")


if __name__ == "__main__":
    main()
