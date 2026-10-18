
from pathlib import Path
import sqlite3

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_ROOT / "database" / "network_traffic.db"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)


def main():
    with sqlite3.connect(DB_FILE) as connection:
        # 1. Overall traffic distribution
        traffic_summary = pd.read_sql_query(
            """
            SELECT
                CASE WHEN label = 0 THEN 'Normal'
                     ELSE 'Attack'
                END AS traffic_type,
                COUNT(*) AS total_records,
                ROUND(
                    COUNT(*) * 100.0 /
                    (SELECT COUNT(*) FROM network_traffic),
                    2
                ) AS percentage
            FROM network_traffic
            GROUP BY label
            ORDER BY label;
            """,
            connection
        )

        traffic_summary.to_csv(
            DASHBOARD_DIR / "traffic_summary.csv",
            index=False
        )

        # 2. Protocol summary
        protocol_summary = pd.read_sql_query(
            """
            SELECT
                proto,
                COUNT(*) AS total_records,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)
                    AS attack_records,
                ROUND(
                    SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END)
                    * 100.0 / COUNT(*),
                    2
                ) AS attack_percentage
            FROM network_traffic
            GROUP BY proto
            ORDER BY total_records DESC;
            """,
            connection
        )

        protocol_summary.to_csv(
            DASHBOARD_DIR / "protocol_summary.csv",
            index=False
        )

        # 3. Average traffic measurements by class
        feature_summary = pd.read_sql_query(
            """
            SELECT
                CASE WHEN label = 0 THEN 'Normal'
                     ELSE 'Attack'
                END AS traffic_type,
                ROUND(AVG(rate), 3) AS average_rate,
                ROUND(AVG(sbytes), 2) AS average_source_bytes,
                ROUND(AVG(dbytes), 2) AS average_destination_bytes,
                ROUND(AVG(spkts), 2) AS average_source_packets,
                ROUND(AVG(dpkts), 2) AS average_destination_packets
            FROM network_traffic
            GROUP BY label
            ORDER BY label;
            """,
            connection
        )

        feature_summary.to_csv(
            DASHBOARD_DIR / "feature_summary.csv",
            index=False
        )

    print("Dashboard data exported successfully.")
    print("\nFiles created:")
    for path in sorted(DASHBOARD_DIR.glob("*.csv")):
        print(path.name)


if __name__ == "__main__":
    main()
