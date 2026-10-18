
from pathlib import Path

import pandas as pd

# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = RAW_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DIR / "UNSW_NB15_testing-set.csv"


def prepare_dataset(file_path):
    """Prepare one dataset for binary attack classification."""

    print(f"\nPreparing: {file_path.name}")

    df = pd.read_csv(file_path, low_memory=False)

    # Remove the identifier and attack category.
    # Keep 'label' as the target column.
    df = df.drop(columns=["id", "attack_cat"])

    # Separate input features (X) and prediction target (y)
    X = df.drop(columns=["label"])
    y = df["label"]

    print("Feature rows and columns:", X.shape)
    print("Target rows:", len(y))
    print("Target distribution:")
    print(y.value_counts().sort_index())

    # Check for missing values after preparation
    print("Missing values:", df.isnull().sum().sum())

    return X, y


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = prepare_dataset(TRAIN_FILE)
    X_test, y_test = prepare_dataset(TEST_FILE)

    # Ensure training and testing features match
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and testing feature columns do not match.")

    # Save features and targets separately
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)

    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    print("\nPreparation completed successfully.")
    print("Number of input features:", X_train.shape[1])
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Saved files to:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
