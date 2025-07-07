import os
import json
import re
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset
from main import DB_CONFIG, TABLE_NAME, load_data_from_postgres, preprocess_data, model_registry
from mlflowcode import mlflow_run_with_random_search

# Drift threshold value used to decide if retraining is needed
DRIFT_THRESHOLD = 0.3

# JSON file to store whether drift is detected
DRIFT_FLAG_PATH = os.path.join(os.getcwd(), "drift_flag.json")

# Clean metric names to be MLflow-friendly by removing unsupported characters
def clean_metric_name(name):
    return re.sub(r"[^\w\-/\. ]", "_", name)

# This function runs Evidently data drift and summary reports,
# logs them to MLflow, and detects if drift exceeds threshold
def log_evidently_reports(train_df, test_df, new_df, historical_df):
    # Define dataset comparisons for drift check
    report_pairs = [
        ("train_vs_test", train_df, test_df),
        ("historical_vs_new", historical_df, new_df)
    ]
    # Define types of reports to generate
    report_configs = [
        ("drift", DataDriftPreset(method='psi')),  # PSI method for drift
        ("summary", DataSummaryPreset())           # Summary report
    ]

    drift_found = False  # Flag to track drift detection

    for name, ref_df, curr_df in report_pairs:
        for report_type, preset in report_configs:
            report = Report([preset], include_tests=True)

            # Use common columns between reference and current dataset
            common_cols = ref_df.columns.intersection(curr_df.columns)
            ref_df_clean = ref_df[common_cols].copy()
            curr_df_clean = curr_df[common_cols].copy()

            try:
                # Run the Evidently report
                result = report.run(reference_data=ref_df_clean, current_data=curr_df_clean)

                # Save report as HTML and log to MLflow
                html_path = f"evidently_{name}_{report_type}.html"
                result.save_html(html_path)
                mlflow.log_artifact(html_path)

                # Convert report to JSON to extract drift metrics
                json_data = json.loads(result.json())

                # Loop through all reported metrics
                for metric in json_data.get("metrics", []):
                    metric_id = metric.get("metric_id") or metric.get("metric", "")
                    value = metric.get("value", None)

                    # If metric value is a dictionary (contains sub-metrics)
                    if isinstance(value, dict):
                        for sub_name, sub_val in value.items():
                            if isinstance(sub_val, (int, float)):
                                metric_name = clean_metric_name(f"{name}_{report_type}_{metric_id}_{sub_name}")
                                mlflow.log_metric(metric_name, sub_val)
                                # Check if drift is found
                                if "drift" in sub_name and sub_val > DRIFT_THRESHOLD:
                                    drift_found = True

                    # If metric is a single float or int
                    elif isinstance(value, (int, float)):
                        metric_name = clean_metric_name(f"{name}_{report_type}_{metric_id}")
                        mlflow.log_metric(metric_name, value)
                        if "drift" in metric_name and value > DRIFT_THRESHOLD:
                            drift_found = True

                    # Special handling for column-wise drift
                    elif "ValueDrift(column=" in metric_id:
                        try:
                            col_name = metric_id.split("ValueDrift(column=")[1].split(",")[0]
                            metric_name = clean_metric_name(f"{name}_{report_type}_{col_name}")
                            mlflow.log_metric(metric_name, value)
                            if isinstance(value, float) and value > DRIFT_THRESHOLD:
                                drift_found = True
                        except Exception as e:
                            print(f"Could not parse drift metric: {metric_id} — {e}")
            except Exception as e:
                print(f"Evidently report failed for {name} - {report_type}: {e}")

    # Write final drift detection result to a local file
    with open(DRIFT_FLAG_PATH, "w") as f:
        json.dump({"drift_detected": drift_found}, f)

    return drift_found  # Return True if drift was found


def retrain_and_register_model():
    print("Starting drift check + possible retraining...")

    # Load data from PostgreSQL database
    df = load_data_from_postgres(DB_CONFIG, TABLE_NAME)
    
    # Preprocess the data (cleaning, encoding, scaling, etc.)
    processed_df = preprocess_data(df)

    # Split preprocessed data into features and target
    X = processed_df.drop(columns=["adjusted_total_usd"])
    y = processed_df["adjusted_total_usd"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create training dataframe with target column for drift comparison
    train_df = X_train.copy()
    train_df["adjusted_total_usd"] = y_train

    # Create testing dataframe with target column
    test_df = X_test.copy()
    test_df["adjusted_total_usd"] = y_test

    # Historical data used as reference for drift detection
    historical_df = train_df.copy()

    # New incoming data (simulated by sampling from processed_df)
    new_df = processed_df.sample(frac=0.2, random_state=7)

    # Start an MLflow run to track drift detection and retraining
    with mlflow.start_run(run_name="Drift + Retrain Check"):

        # Run Evidently drift detection on different dataset pairs
        drift = log_evidently_reports(train_df, test_df, new_df, historical_df)

        # If drift is detected, retrain and register the best model using MLflow
        if drift:
            print("Drift detected! Starting retraining...")
            mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry)
            print("Model retrained and logged to MLflow.")
        else:
            # If no drift, skip retraining
            print("No drift. Skipping retraining.")






# import os
# import json
# import re
# import mlflow
# import numpy as np
# from sklearn.model_selection import train_test_split
# from evidently import Report
# from evidently.presets import DataDriftPreset, DataSummaryPreset
# from main import DB_CONFIG, TABLE_NAME, load_data_from_postgres, preprocess_data
# from mlflowcode import mlflow_run_with_random_search

# DRIFT_THRESHOLD = 0.3


# def clean_metric_name(name):
#     return re.sub(r"[^\w\-/\. ]", "_", name)

# def log_evidently_reports(train_df, test_df, new_df, historical_df):
#     report_pairs = [
#         ("train_vs_test", train_df, test_df),
#         ("historical_vs_new", historical_df, new_df)
#     ]
#     report_configs = [
#         ("drift", DataDriftPreset(method='psi')),
#         ("summary", DataSummaryPreset())
#     ]
#     drift_found = False

#     for name, ref_df, curr_df in report_pairs:
#         for report_type, preset in report_configs:
#             report = Report([preset], include_tests=True)
#             common_cols = ref_df.columns.intersection(curr_df.columns)
#             ref_df_clean = ref_df[common_cols].copy()
#             curr_df_clean = curr_df[common_cols].copy()

#             try:
#                 result = report.run(reference_data=ref_df_clean, current_data=curr_df_clean)
#                 html_path = f"evidently_{name}_{report_type}.html"
#                 result.save_html(html_path)
#                 mlflow.log_artifact(html_path)

#                 json_data = json.loads(result.json())
#                 for metric in json_data.get("metrics", []):
#                     metric_id = metric.get("metric_id") or metric.get("metric", "")
#                     value = metric.get("value", None)

#                     if isinstance(value, dict):
#                         for sub_name, sub_val in value.items():
#                             if isinstance(sub_val, (int, float)):
#                                 metric_name = clean_metric_name(f"{name}_{report_type}_{metric_id}_{sub_name}")
#                                 mlflow.log_metric(metric_name, sub_val)
#                                 if "drift" in sub_name and sub_val > DRIFT_THRESHOLD:
#                                     drift_found = True

#                     elif isinstance(value, (int, float)):
#                         metric_name = clean_metric_name(f"{name}_{report_type}_{metric_id}")
#                         mlflow.log_metric(metric_name, value)
#                         if "drift" in metric_name and value > DRIFT_THRESHOLD:
#                             drift_found = True

#                     elif "ValueDrift(column=" in metric_id:
#                         try:
#                             col_name = metric_id.split("ValueDrift(column=")[1].split(",")[0]
#                             metric_name = clean_metric_name(f"{name}_{report_type}_{col_name}")
#                             mlflow.log_metric(metric_name, value)
#                             if isinstance(value, float) and value > DRIFT_THRESHOLD:
#                                 drift_found = True
#                         except Exception as e:
#                             print(f"⚠️ Could not parse drift metric: {metric_id} — {e}")
#             except Exception as e:
#                 print(f"❌ Evidently report failed for {name} - {report_type}: {e}")

#     with open("/opt/airflow/dags/drift_flag.json", "w") as f:
#             json.dump({"drift_detected": drift_found}, f)
#     return drift_found


