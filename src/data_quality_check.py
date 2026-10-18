
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

FILES = [
    RAW_DATA_DIR / "UNSW_NB15_training-set.csv",
    RAW_DATA_DIR / "UNSW_NB15_testing-set.csv",
]


def check_data_quality(file_path):
    print("\n" + "=" * 60)
    print(f"DATA QUALITY CHECK: {file_path.name}")
    print("=" * 60)

    df = pd.read_csv(file_path, low_memory=False)

    # 1. Check missing values
    print("\n1. Missing values:")
    print("Total missing cells:", df.isnull().sum().sum())

    # 2. Check duplicate rows
    print("\n2. Duplicate rows:")
    print("Total duplicates:", df.duplicated().sum())

    # 3. Check infinite values in numerical columns
    numeric_df = df.select_dtypes(include="number")
    infinite_counts = pd.Series(
        np.isinf(numeric_df).sum(),
        index=numeric_df.columns
    )
    print("\n3. Infinite values by column:")
    print(infinite_counts[infinite_counts > 0])

    if infinite_counts.sum() == 0:
        print("No infinite numerical values found.")

    # 4. Check numerical ranges
    print("\n4. Numerical summary:")
    print(numeric_df.describe().T)

    # 5. Check categorical values
    print("\n5. Unique values in categorical columns:")
    categorical_df = df.select_dtypes(
        include=["object", "str", "category"]
    )

    for column in categorical_df.columns:
        print(f"{column}: {df[column].nunique()} unique values")

    # 6. Verify target labels
    print("\n6. Label validation:")
    print(df["label"].value_counts(dropna=False).sort_index())

    unexpected_labels = set(df["label"].unique()) - {0, 1}
    print("Unexpected label values:", unexpected_labels)

    # 7. Check identifiers
    print("\n7. Identifier check:")
    print("Unique IDs:", df["id"].nunique())
    print("Total rows:", len(df))


if __name__ == "__main__":
    for file_path in FILES:
        check_data_quality(file_path)
