
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

train = pd.read_csv(
    RAW_DIR / "UNSW_NB15_training-set.csv",
    low_memory=False
)
test = pd.read_csv(
    RAW_DIR / "UNSW_NB15_testing-set.csv",
    low_memory=False
)

# 1. Compare the dataset columns
print("1. COLUMN CHECK")
print("Same columns:", set(train.columns) == set(test.columns))
print("Training columns:", len(train.columns))
print("Testing columns:", len(test.columns))

# 2. Verify that labels match attack categories
print("\n2. LABEL AND ATTACK CATEGORY CHECK")
print(pd.crosstab(train["attack_cat"], train["label"]))

print("\nTesting data:")
print(pd.crosstab(test["attack_cat"], test["label"]))

# 3. Find negative numerical values
print("\n3. NEGATIVE NUMERICAL VALUES")

numeric_columns = train.select_dtypes(include="number").columns
negative_counts = (train[numeric_columns] < 0).sum()
print("Training data:")
print(negative_counts[negative_counts > 0])

numeric_columns_test = test.select_dtypes(include="number").columns
negative_counts_test = (test[numeric_columns_test] < 0).sum()
print("\nTesting data:")
print(negative_counts_test[negative_counts_test > 0])

# 4. Check whether IDs overlap across files
print("\n4. ID CHECK")
print("IDs overlap:", train["id"].isin(test["id"]).any())

print("\nValidation checks completed.")

# 5. Investigate overlapping IDs
print("\n5. OVERLAPPING ID INVESTIGATION")

overlap_ids = set(train["id"]) & set(test["id"])

print("Number of overlapping IDs:", len(overlap_ids))

if overlap_ids:
    train_overlap = train[train["id"].isin(overlap_ids)]
    test_overlap = test[test["id"].isin(overlap_ids)]

    # Compare complete rows, excluding the ID column
    train_records = train_overlap.drop(columns=["id"])
    test_records = test_overlap.drop(columns=["id"])

    train_records = train_records.reset_index(drop=True)
    test_records = test_records.reset_index(drop=True)

    print("Overlapping rows in training:", len(train_overlap))
    print("Overlapping rows in testing:", len(test_overlap))

    # Check whether any shared IDs have identical feature and label values
    train_by_id = train_overlap.set_index("id").sort_index()
    test_by_id = test_overlap.set_index("id").sort_index()

    common_ids = train_by_id.index.intersection(test_by_id.index)

    same_rows = (
        train_by_id.loc[common_ids]
        .sort_index()
        .equals(
            test_by_id.loc[common_ids].sort_index()
        )
    )

    print("All shared-ID rows identical:", same_rows)

    print("\nFirst 10 overlapping IDs:")
    print(sorted(overlap_ids)[:10])
else:
    print("No overlapping IDs found.")

print("\n6. ID RANGE AND ROW COMPARISON")

print(
    "Training ID range:",
    train["id"].min(),
    "to",
    train["id"].max()
)

print(
    "Testing ID range:",
    test["id"].min(),
    "to",
    test["id"].max()
)

print(
    "Training IDs unique:",
    train["id"].is_unique
)

print(
    "Testing IDs unique:",
    test["id"].is_unique
)

# Compare the actual records for the first shared ID
shared_id = sorted(set(train["id"]) & set(test["id"]))[0]

train_row = (
    train.loc[train["id"] == shared_id]
    .drop(columns=["id"])
    .reset_index(drop=True)
)

test_row = (
    test.loc[test["id"] == shared_id]
    .drop(columns=["id"])
    .reset_index(drop=True)
)

print("\nFirst shared ID:", shared_id)
print("Rows identical excluding ID:", train_row.equals(test_row))

if not train_row.equals(test_row):
    print("\nDifferences for the first shared ID:")
    comparison = pd.DataFrame({
        "training": train_row.iloc[0],
        "testing": test_row.iloc[0]
    })
    print(comparison[comparison["training"] != comparison["testing"]])
