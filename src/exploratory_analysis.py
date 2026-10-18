
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

X_TRAIN_FILE = PROCESSED_DIR / "X_train.csv"
Y_TRAIN_FILE = PROCESSED_DIR / "y_train.csv"


def main():
    # Load the prepared training data
    X = pd.read_csv(X_TRAIN_FILE, low_memory=False)
    y = pd.read_csv(Y_TRAIN_FILE)["label"]

    print("=" * 60)
    print("NETWORK TRAFFIC EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # 1. Dataset size
    print("\n1. DATASET SIZE")
    print("Rows:", X.shape[0])
    print("Features:", X.shape[1])

    # 2. Target distribution
    print("\n2. NORMAL VS ATTACK TRAFFIC")
    label_counts = y.value_counts().sort_index()
    print(label_counts.rename(index={0: "Normal", 1: "Attack"}))

    print("\nPercentage distribution:")
    print(
        (y.value_counts(normalize=True).sort_index() * 100)
        .rename(index={0: "Normal", 1: "Attack"})
        .round(2)
    )

    # 3. Most frequent network protocols
    print("\n3. TOP 10 NETWORK PROTOCOLS")
    print(X["proto"].value_counts().head(10))

    # 4. Most frequent services
    print("\n4. NETWORK SERVICE DISTRIBUTION")
    print(X["service"].value_counts().head(10))

    # 5. Compare selected numerical features
    print("\n5. SELECTED FEATURES BY TRAFFIC LABEL")
    analysis_df = X.copy()
    analysis_df["label"] = y

    selected_features = ["dur", "spkts", "dpkts", "sbytes", "dbytes", "rate"]

    print(
        analysis_df.groupby("label")[selected_features]
        .mean()
        .round(3)
        .rename(index={0: "Normal", 1: "Attack"})
        .T
    )

    # 6. Confirm that features and target align
    print("\n6. DATA ALIGNMENT")
    print("Feature rows:", len(X))
    print("Target rows:", len(y))
    print("Rows aligned:", len(X) == len(y))

    print("\nEDA summary completed.")


if __name__ == "__main__":
    main()
