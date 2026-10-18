# Network Traffic Analysis and Anomaly Detection

## Project Overview

Network Traffic Analysis and Anomaly Detection is a data analytics and machine learning project that analyzes network traffic records to distinguish normal traffic from potentially malicious traffic.

The project uses the **UNSW-NB15 dataset** and combines Python, SQL, Power BI, and machine learning to perform data preprocessing, exploratory data analysis, database analysis, dashboard development, model training, and evaluation.

The current implementation classifies records from a benchmark dataset. It does not yet monitor live network traffic.

## Problem Statement

Network traffic datasets contain large volumes of records with different protocols, services, connection states, and traffic characteristics. Manually analyzing these records can be time-consuming.

This project explores how data analytics and machine learning can help identify attack-related traffic patterns, summarize network activity, and evaluate the effectiveness of traffic classification models.

## Objectives

- Analyze normal and attack traffic using a real-world benchmark dataset.
- Inspect data quality and prepare features for machine learning.
- Perform exploratory data analysis using Python.
- Store and analyze network traffic records using SQLite and SQL.
- Develop a Power BI dashboard to visualize traffic statistics.
- Train and compare Logistic Regression, Random Forest, and XGBoost.
- Evaluate model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Generate reusable evaluation reports and a prediction demonstration.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data processing, analysis, and machine learning |
| Pandas and NumPy | Data manipulation and numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Preprocessing, pipelines, and model evaluation |
| XGBoost | Gradient-boosted classification |
| Joblib | Saving and loading trained pipelines |
| SQLite | Storing and querying traffic records |
| SQL | Traffic summaries and analytical queries |
| Power BI | Interactive dashboard and visualization |
| Git and GitHub | Version control and project hosting |

## Dataset

**Dataset name:** UNSW-NB15

The dataset contains network traffic records labelled as normal or attack traffic.

The project uses separate training and testing CSV files.

| Dataset | Records | Original columns |
|---|---:|---:|
| Training set | 175,341 | 45 |
| Testing set | 82,332 | 45 |

### Target labels

- `0` — Normal traffic
- `1` — Attack traffic

The project removes the `id` and `attack_cat` columns from the model features and uses `label` as the binary classification target. The resulting model input contains 42 features, including numeric and categorical features.

The training and testing datasets are kept separate for model development and evaluation.

**Dataset source:** [UNSW-NB15 — UNSW Canberra Cyber](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Download the training and testing CSV files from the dataset source and place them in `data/raw/`.

## Project Workflow

1. **Dataset collection:** Obtain the UNSW-NB15 training and testing datasets.
2. **Data inspection:** Examine dimensions, column types, labels, missing values, duplicates, and data-quality issues.
3. **Preprocessing:** Separate features and labels, exclude unsuitable identifier and category columns, and prepare the model inputs.
4. **Exploratory data analysis:** Study traffic class distribution, protocols, services, and feature patterns.
5. **Visualization:** Generate charts to communicate the main dataset characteristics.
6. **Database development:** Load training records into SQLite.
7. **SQL analysis:** Calculate traffic counts, protocol distributions, feature summaries, and attack percentages.
8. **Power BI dashboard:** Visualize the dataset's traffic statistics.
9. **Machine learning:** Train Logistic Regression, Random Forest, and XGBoost pipelines.
10. **Model evaluation:** Compare classification metrics and confusion matrices.
11. **Prediction demonstration:** Run the saved XGBoost pipeline on selected test records.
12. **Reporting:** Save comparison results and final evaluation reports.

## Exploratory Data Analysis

The project explores:

- Normal versus attack traffic distribution.
- Network protocol distribution.
- Service distribution.
- Comparisons of selected numerical features.
- Relationships between protocol types and attack labels.

Generated visualizations are saved in `outputs/figures/`.

## SQL Analysis

SQLite is used to store the training dataset in the `network_traffic` table.

The SQL analysis includes:

- Total records and traffic class counts.
- Protocol-wise traffic distribution.
- Feature summary statistics.
- Attack percentages across protocols.

Database file: `database/network_traffic.db`

SQL queries: `sql/analysis_queries.sql`

## Power BI Dashboard

The Power BI dashboard summarizes the training dataset and presents key traffic statistics.

### Dashboard KPIs

| KPI | Value |
|---|---:|
| Total records | 175,341 |
| Normal records | 56,000 |
| Attack records | 119,341 |
| Attack percentage | 68.06% |

The dashboard includes traffic class distribution and additional traffic summaries.

**Dashboard file:** `dashboard/Network_Traffic_Analysis.pbix`

The dashboard describes the benchmark dataset; it does not represent a live network monitoring feed.

## Machine Learning Models

Three binary classification models were trained using preprocessing pipelines.

### 1. Logistic Regression

A baseline classification model used to establish an initial performance benchmark.

### 2. Random Forest

An ensemble model that combines multiple decision trees to classify traffic records.

### 3. XGBoost

A gradient-boosting model that builds trees sequentially to improve classification performance.

The pipelines include numerical and categorical preprocessing. Saved models can be loaded for subsequent predictions without manually repeating the preprocessing steps.

## Model Comparison

The models were evaluated on the 82,332-record testing dataset.

| Metric | Logistic Regression | Random Forest | XGBoost |
|---|---:|---:|---:|
| Accuracy | 80.57% | 86.27% | **87.78%** |
| Attack precision | 74.93% | 80.53% | **82.58%** |
| Attack recall | 97.27% | **99.00%** | 98.61% |
| Attack F1-score | 84.65% | 88.81% | **89.89%** |

### Selected model

XGBoost achieved the highest overall accuracy and attack F1-score among the three evaluated models.

Random Forest achieved slightly higher attack recall. Model selection therefore involves a trade-off between overall performance, missed attacks, and false alarms.

### XGBoost confusion matrix

Rows represent actual classes; columns represent predicted classes.

| Actual class | Predicted Normal | Predicted Attack |
|---|---:|---:|
| Normal | 27,571 | 9,429 |
| Attack | 631 | 44,701 |

Key observations:

- Correctly identified normal records: 27,571.
- Normal records incorrectly flagged as attacks: 9,429.
- Attacks incorrectly classified as normal: 631.
- Correctly detected attacks: 44,701.

These results show why recall, precision, and confusion matrix analysis are important in addition to accuracy.

## Prediction Demonstration

The saved XGBoost pipeline is used to classify selected records from the prepared test dataset.

The demonstration displays:

- Actual traffic class.
- Predicted traffic class.
- Correct or incorrect prediction.
- Sample-level evaluation statistics.

The current demonstration uses a balanced sample of 50 normal and 50 attack records. Its result is separate from the full-test evaluation.

## Generated Reports and Artifacts

| File | Description |
|---|---|
| `outputs/figures/traffic_class_distribution.png` | Normal and attack class distribution |
| `outputs/figures/protocol_distribution.png` | Protocol distribution |
| `outputs/figures/service_distribution.png` | Service distribution |
| `outputs/figures/feature_comparison.png` | Feature comparison visualization |
| `outputs/figures/model_comparison.png` | Model performance comparison chart |
| `outputs/reports/model_comparison.csv` | Comparison metrics for all three models |
| `outputs/reports/confusion_matrices.txt` | Confusion matrices for the models |
| `outputs/reports/final_evaluation_metrics.csv` | Final XGBoost evaluation metrics |
| `outputs/reports/final_classification_report.txt` | Detailed classification report |

## Project Structure

```text
Network-Traffic-Analysis/
├── data/
│   ├── raw/
│   └── processed/
├── database/
│   └── network_traffic.db
├── dashboard/
│   └── Network_Traffic_Analysis.pbix
├── models/
│   ├── logistic_regression_pipeline.joblib
│   ├── random_forest_pipeline.joblib
│   └── xgboost_pipeline.joblib
├── notebooks/
├── outputs/
│   ├── figures/
│   └── reports/
├── sql/
│   └── analysis_queries.sql
├── src/
│   ├── ml_preprocessing.py
│   ├── train_models.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── compare_models.py
│   ├── predict_traffic.py
│   └── final_evaluation_report.py
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Execution

### Prerequisites

- Python installed on your system.
- Git.
- Power BI Desktop for opening the dashboard.
- The UNSW-NB15 training and testing CSV files.

### 1. Clone the repository

```powershell
git clone https://github.com/AmmajiShaik06/Network-Traffic-Analysis.git
cd Network-Traffic-Analysis
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add the datasets

Place the downloaded CSV files inside `data/raw/`.

The raw and processed data folders are excluded from Git by the project's `.gitignore` rules, so downloading or preparing the data may be necessary after cloning.

### 5. Run model comparison

After the processed datasets and trained model files are available:

```powershell
python .\src\compare_models.py
```

### 6. Run the prediction demonstration

```powershell
python .\src\predict_traffic.py
```

### 7. Generate the final evaluation report

```powershell
python .\src\final_evaluation_report.py
```

The scripts above require their corresponding prepared datasets and saved model artifacts. Training scripts can be run when those artifacts need to be recreated.

## Limitations

- The project uses a benchmark dataset rather than live network traffic.
- Binary classification distinguishes Normal and Attack labels but does not provide a complete multiclass attack identification system.
- The test results reflect performance on UNSW-NB15 and may not generalize to other networks or newer attack patterns.
- A data-integrity audit found repeated feature combinations across the training and testing sets, as well as some identical feature combinations associated with different labels. This may affect evaluation on previously unseen traffic and highlights limitations in the supplied dataset labels.
- The false-positive count indicates that some normal records are incorrectly flagged as attacks.
- Real-world deployment would require live traffic collection, operational monitoring, security validation, and additional testing.

## Future Enhancements

- Add live or streamed network traffic ingestion.
- Extend the classifier to identify individual attack categories.
- Add explainability to help analysts understand model predictions.
- Build an alerting interface with configurable thresholds.
- Evaluate performance on additional datasets and changing traffic patterns.
- Add automated tests and model monitoring.

## Conclusion

This project combines data analytics, SQL, Power BI, and machine learning to analyze network traffic and classify normal and attack records. Comparing three models demonstrated the value of evaluating several performance metrics rather than relying on accuracy alone.

XGBoost achieved 87.78% accuracy and an 89.89% attack F1-score on the full UNSW-NB15 test dataset. The project provides a foundation for further work in network security analytics and intrusion detection.

