
from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_ROOT / "database" / "network_traffic.db"
SQL_FILE = PROJECT_ROOT / "sql" / "analysis_queries.sql"


def main():
    sql_text = SQL_FILE.read_text(encoding="utf-8")

    with sqlite3.connect(DB_FILE) as connection:
        queries = [
            query.strip()
            for query in sql_text.split(";")
            if query.strip()
        ]

        for number, query in enumerate(queries, start=1):
            print(f"\n{'=' * 50}")
            print(f"SQL QUERY {number}")
            print("=" * 50)

            result = pd.read_sql_query(query, connection)
            print(result.to_string(index=False))

    print("\nAll SQL analysis queries completed.")


if __name__ == "__main__":
    main()
