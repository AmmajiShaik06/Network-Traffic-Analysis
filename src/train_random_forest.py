
import os

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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


# 1. Load the prepared data.
print("Loading datasets...")

X_train = pd.read_csv("data/processed/X_train.csv")
y_train = pd.read_csv("data/processed/y_train.csv")["label"]

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")["label"]

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# 2. Create preprocessing and model pipelines.
preprocessor, _, _ = create_preprocessor(X_train)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)


# 3. Train Random Forest.
print("\nTraining Random Forest...")
print("This may take several minutes.")

model.fit(X_train, y_train)

print("Training completed.")


# 4. Evaluate on the same test data.
print("\nEvaluating on test data...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
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


# 5. Save the trained pipeline.
os.makedirs("models", exist_ok=True)

model_path = "models/random_forest_pipeline.joblib"
joblib.dump(model, model_path)

print("\nTrained pipeline saved to:", model_path)
print("Training script finished.")
