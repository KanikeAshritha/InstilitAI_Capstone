# import os
# import joblib
# import mlflow
# import pandas as pd
# import json
# from evidently import Report
# from evidently.presets import DataDriftPreset
# from mlflow.tracking import MlflowClient
# from mlflow.exceptions import MlflowException
# from sklearn.model_selection import train_test_split
# import sys

# sys.path.append("/opt/airflow/instilit_capstoneproj")
# from mlflowcode import mlflow_run_with_random_search  # <- Your model training + MLflow logging module
# from main import preprocess_data, model_registry       # <- Your custom pipeline and model dict


# # --------------------------
# # Configuration
# # --------------------------
# MLFLOW_URI = "http://host.docker.internal:5002"
# mlflow.set_tracking_uri(MLFLOW_URI)
# mlflow.set_experiment("Instilit_Regression")


# def load_latest_production_model(model_name):
#     client = MlflowClient()
#     try:
#         prod_model = client.get_latest_versions(name=model_name, stages=["Production"])[0]
#         model_uri = prod_model.source
#         run_id = prod_model.run_id
#         print(f"✅ Loaded model from: {model_uri}")
#         model = mlflow.sklearn.load_model(model_uri=f"runs:/{run_id}/model")
#         return model
#     except Exception as e:
#         print(f"❌ Failed to load production model: {e}")
#         return None


# def check_for_drift(reference_df, current_df):
#     drift_report = Report(metrics=[DataDriftPreset()])
#     drift_report.run(reference_data=reference_df, current_data=current_df)

#     report_dict = drift_report.as_dict()
#     drift_detected = report_dict["metrics"][0]["result"]["drift_detected"]

#     # Save and log HTML drift report
#     drift_report.save_html("drift_report.html")
#     mlflow.log_artifact("drift_report.html")

#     print(f"📊 Drift Detected: {drift_detected}")
#     return drift_detected


# def retrain_best_model(df):
#     print("🚀 Starting model retraining...")

#     X, y = preprocess_data(df)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     best_model_info = mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry)

#     # Save latest model info
#     with open("latest_model_uri.txt", "w") as f:
#         f.write(best_model_info["model_uri"])
#     with open("latest_run_id.txt", "w") as f:
#         f.write(best_model_info["run_id"])

#     print("✅ Retraining complete. Best model registered:")
#     print(best_model_info)
#     return best_model_info


# def main():
#     # Simulate drift flag from JSON
#     with open("/opt/airflow/dags/drift_flag.json") as f:
#         drift_status = json.load(f)

#     if drift_status.get("drift_detected", False):
#         print("⚠️ Drift detected! Triggering retraining...")

#         # Load new data and retrain
#         df = pd.read_csv("/opt/airflow/data/full_data.csv")
#         df = preprocess_data(df)
#         X = df.drop(columns=["adjusted_total_usd"])
#         y = df["adjusted_total_usd"]

#         from sklearn.model_selection import train_test_split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#         mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry)

#     else:
#         print("✅ No drift detected. Skipping retraining.")


# if __name__ == "__main__":
#     main()
