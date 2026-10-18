
from pathlib import Path

import pandas as pd

# Find the project folder and dataset files
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

TRAIN_FILE = RAW_DATA_DIR / "UNSW_NB15_training-set.csv"
TEST_FILE = RAW_DATA_DIR / "UNSW_NB15_testing-set.csv"


def inspect_dataset(file_path):
    """Display basic information about a network traffic CSV file."""

    print("\n" + "=" * 60)
    print(f"FILE: {file_path.name}")
    print("=" * 60)

    # Stop early if the dataset file is missing
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        return

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, low_memory=False)

    print(f"\nNumber of rows: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    # Inspect labels if these columns exist
    for column in ["label", "attack_cat"]:
        if column in df.columns:
            print(f"\nValue counts for '{column}':")
            print(df[column].value_counts(dropna=False))


if __name__ == "__main__":
    inspect_dataset(TRAIN_FILE)
    inspect_dataset(TEST_FILE)
