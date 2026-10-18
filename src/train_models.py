
import os
import warnings

import joblib
import pandas as pd

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.pipeline import Pipeline

from ml_preprocessing import create_preprocessor


# 1. Load the prepared training and testing data.
print("Loading datasets...")

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")["label"]

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")["label"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 2. Create the preprocessing pipeline.
preprocessor, numeric_columns, categorical_columns = (
    create_preprocessor(X_train)
)

# 3. Combine preprocessing and Logistic Regression.
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                solver="saga",
                random_state=42,
                tol=1e-3,
            ),
        ),
    ]
)


# 4. Train the model.
print("\nTraining Logistic Regression...")
print("This may take a few minutes.")

with warnings.catch_warnings(record=True) as caught_warnings:
    warnings.simplefilter("always", ConvergenceWarning)
    model.fit(X_train, y_train)

convergence_warnings = [
    warning
    for warning in caught_warnings
    if issubclass(warning.category, ConvergenceWarning)
]

print("Training completed.")

if convergence_warnings:
    print(
        "WARNING: The model may not have fully converged. "
        "We will review this before relying on the results."
    )


# 5. Predict labels for the test data.
print("\nEvaluating on test data...")
y_pred = model.predict(X_test)


# 6. Calculate evaluation metrics.
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Attack"],
        zero_division=0,
    )
)

print("Confusion Matrix (rows=actual, columns=predicted):")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))


# 7. Save the trained pipeline.
os.makedirs("models", exist_ok=True)

model_path = "models/logistic_regression_pipeline.joblib"
joblib.dump(model, model_path)

print("\nTrained pipeline saved to:", model_path)
print("Training script finished.")
