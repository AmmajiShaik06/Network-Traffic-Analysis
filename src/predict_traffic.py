
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load the trained model
print("Loading XGBoost model...")
model = joblib.load("models/xgboost_pipeline.joblib")

# 2. Load the test dataset
print("Loading test data...")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv")["label"]


# 3. Randomly select 50 records from each class
normal_indices = y_test[y_test == 0].sample(
    n=50, random_state=42
).index

attack_indices = y_test[y_test == 1].sample(
    n=50, random_state=42
).index

sample_indices = normal_indices.append(attack_indices)

X_sample = X_test.loc[sample_indices]
y_actual = y_test.loc[sample_indices]


# 4. Generate predictions
y_predicted = model.predict(X_sample)

# 5. Evaluate this sample
accuracy = accuracy_score(y_actual, y_predicted)
matrix = confusion_matrix(
    y_actual, y_predicted, labels=[0, 1]
)

tn, fp, fn, tp = matrix.ravel()

print("\nREPRESENTATIVE SAMPLE EVALUATION")
print("-" * 40)
print("Normal records:", len(normal_indices))
print("Attack records:", len(attack_indices))
print("Total records:", len(sample_indices))
print(f"Sample accuracy: {accuracy:.2%}")
print("Correctly identified normal records:", tn)
print("Normal records incorrectly flagged:", fp)
print("Missed attacks:", fn)
print("Correctly detected attacks:", tp)

print("\nConfusion matrix:")
print(matrix)
print("\nRows = actual [Normal, Attack]")
print("Columns = predicted [Normal, Attack]")
