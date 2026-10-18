
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. Load test data
print("Loading test data...")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")["label"]

# 2. Load the saved XGBoost model
print("Loading XGBoost model...")

model = joblib.load("models/xgboost_pipeline.joblib")

# 3. Generate predictions on the full test set
print("Evaluating the full test dataset...")

y_pred = model.predict(X_test)

# 4. Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = matrix.ravel()

# 5. Create the output folder
os.makedirs("outputs/reports", exist_ok=True)

# 6. Save evaluation metrics
metrics = {
    "Metric": [
        "Test records",
        "Accuracy",
        "Attack precision",
        "Attack recall",
        "Attack F1-score",
        "Correctly identified normal records",
        "False alarms",
        "Missed attacks",
        "Correctly detected attacks",
    ],
    "Value": [
        len(y_test),
        round(accuracy, 4),
        round(precision, 4),
        round(recall, 4),
        round(f1, 4),
        int(tn),
        int(fp),
        int(fn),
        int(tp),
    ],
}

metrics_df = pd.DataFrame(metrics)

csv_path = "outputs/reports/final_evaluation_metrics.csv"
metrics_df.to_csv(csv_path, index=False)

# 7. Save the detailed classification report
report = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Normal", "Attack"],
    zero_division=0,
)

report_path = "outputs/reports/final_classification_report.txt"

with open(report_path, "w", encoding="utf-8") as file:
    file.write("NETWORK TRAFFIC ANALYSIS\n")
    file.write("FINAL MODEL EVALUATION\n")
    file.write("=" * 45 + "\n\n")
    file.write("Model: XGBoost\n")
    file.write("Dataset: UNSW-NB15 test dataset\n")
    file.write(f"Test records: {len(y_test)}\n\n")
    file.write(report)
    file.write("\nConfusion matrix:\n")
    file.write(str(matrix))
    file.write("\n\nRows = actual [Normal, Attack]\n")
    file.write("Columns = predicted [Normal, Attack]\n")

# 8. Display the results
print("\nFINAL XGBOOST EVALUATION")
print("=" * 45)
print(f"Test records: {len(y_test)}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Attack precision: {precision:.2%}")
print(f"Attack recall: {recall:.2%}")
print(f"Attack F1-score: {f1:.2%}")

print("\nConfusion matrix:")
print(matrix)

print("\nReports saved:")
print(csv_path)
print(report_path)

print("\nFinal evaluation completed successfully.")
