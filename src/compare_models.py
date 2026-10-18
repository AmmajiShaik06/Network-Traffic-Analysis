
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# 1. Load the prepared test data.
print("Loading test data...")

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")["label"]

print("Test records:", len(X_test))


# 2. Define the saved models.
model_paths = {
    "Logistic Regression":
        "models/logistic_regression_pipeline.joblib",
    "Random Forest":
        "models/random_forest_pipeline.joblib",
    "XGBoost":
        "models/xgboost_pipeline.joblib",
}


# 3. Create output folders.
os.makedirs("outputs/reports", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)


# 4. Evaluate every saved model.
results = []
matrix_lines = []

for model_name, model_path in model_paths.items():
    print(f"\nEvaluating {model_name}...")

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Attack_Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Attack_Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "Attack_F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
    })

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    matrix_lines.append(
        f"{model_name}\n"
        "Rows = actual [Normal, Attack]\n"
        "Columns = predicted [Normal, Attack]\n"
        f"{matrix}\n"
    )


# 5. Save the comparison table.
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy", ascending=False
)

csv_path = "outputs/reports/model_comparison.csv"
results_df.to_csv(csv_path, index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False, float_format="%.4f"))


# 6. Save all confusion matrices.
matrix_path = "outputs/reports/confusion_matrices.txt"

with open(matrix_path, "w", encoding="utf-8") as file:
    file.write("\n".join(matrix_lines))


# 7. Create a comparison chart.
chart_data = results_df.set_index("Model")[
    ["Accuracy", "Attack_Precision", "Attack_Recall",
     "Attack_F1_Score"]
]

ax = chart_data.plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0,
)

ax.set_title("Network Traffic Classification Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Model")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

chart_path = "outputs/figures/model_comparison.png"
plt.savefig(chart_path, dpi=150)
plt.close()

print("\nReports saved:")
print(csv_path)
print(matrix_path)
print(chart_path)
print("\nModel comparison completed successfully.")
