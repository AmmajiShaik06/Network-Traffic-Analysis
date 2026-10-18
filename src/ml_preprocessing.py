
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(X_train):
    """Build preprocessing steps using training features only."""

    # Identify numerical and categorical columns.
    numeric_columns = X_train.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X_train.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    # Numerical features: fill missing values and scale.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Categorical features: fill missing values and encode.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # Apply the appropriate processing to each column type.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ]
    )

    return preprocessor, numeric_columns, categorical_columns


if __name__ == "__main__":
    X_train = pd.read_csv("data/processed/X_train.csv")

    preprocessor, numeric_columns, categorical_columns = (
        create_preprocessor(X_train)
    )

    print("=" * 50)
    print("MACHINE LEARNING PREPROCESSING CHECK")
    print("=" * 50)

    print("Total feature columns:", len(X_train.columns))
    print("Numerical columns:", len(numeric_columns))
    print("Categorical columns:", len(categorical_columns))

    print("\nCategorical features:")
    print(categorical_columns)

    print("\nNumerical features:")
    print(numeric_columns)

    print("\nPreprocessing pipeline created successfully.")
    print("Original dataset has not been modified.")
