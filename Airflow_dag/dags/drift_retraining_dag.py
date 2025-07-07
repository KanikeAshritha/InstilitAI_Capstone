from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
import sys
import os
import pandas as pd

# Import retraining function
# from drift_retraining import main as run_drift_retraining

# Add project path for imports
sys.path.append("/opt/airflow/instilit_capstoneproj")

# Import custom pipeline functions
from main import (
    DB_CONFIG,
    TABLE_NAME,
    load_data_from_postgres,
    perform_eda,
    preprocess_data,
    model_registry
)
from mlflowcode import mlflow_run_with_random_search

# Default DAG settings
default_args = {
    "owner": "ash",
    "start_date": datetime(2024, 1, 1),
    "retries": 0
}

dag = DAG(
    dag_id="full_salary_model_pipeline",
    default_args=default_args,
    description="Extract, EDA, preprocess, train, evaluate, and register model with MLflow.",
    schedule_interval=None,
    catchup=False,
    tags=["training", "mlflow", "eda"]
)

# Global paths for temp data exchange
DATA_PATH = "/opt/airflow/data/full_data.csv"
PREPROCESSED_PATH = "/opt/airflow/data/preprocessed_data.csv"

# Task 1: Extract from PostgreSQL
def extract_data():
    df = load_data_from_postgres(TABLE_NAME, DB_CONFIG)
    df.to_csv(DATA_PATH, index=False)

# Task 2: EDA
def run_eda():
    df = pd.read_csv(DATA_PATH)
    perform_eda(df)

# Task 3: Preprocessing
def run_preprocessing():
    df = pd.read_csv(DATA_PATH)
    processed = preprocess_data(df)
    processed.to_csv(PREPROCESSED_PATH, index=False)

# Task 4: Model training and registration
def train_and_register_model():
    df = pd.read_csv(PREPROCESSED_PATH)
    X = df.drop(columns=["adjusted_total_usd"])
    y = df["adjusted_total_usd"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry)

# Task 5: Completion
def notify_complete():
    print("✅ End-to-end ML pipeline complete.")

# # Task 6: Retrain if drift is detected
# def retrain_and_register_model():
#     run_drift_retraining()

# Define DAG tasks
with dag:
    start = EmptyOperator(task_id="start")

    extract = PythonOperator(
        task_id="extract_data",
        python_callable=extract_data
    )

    eda = PythonOperator(
        task_id="perform_eda",
        python_callable=run_eda
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing
    )

    train = PythonOperator(
        task_id="train_and_register",
        python_callable=train_and_register_model
    )

    end = PythonOperator(
        task_id="pipeline_done",
        python_callable=notify_complete
    )

    # retrain_if_drift = PythonOperator(
    #     task_id='retrain_if_drift_detected',
    #     python_callable=retrain_and_register_model
    # )

    # DAG structure
    start >> extract >> eda >> preprocess >> train >> end 





# from airflow import DAG
# from airflow.sensors.filesystem import FileSensor
# from airflow.operators.python import PythonOperator, BranchPythonOperator
# from airflow.operators.empty import EmptyOperator
# from datetime import datetime
# import os
# import json
# import sys

# # Add project root to sys.path for module imports
# sys.path.append("/opt/airflow/instilit_capstoneproj")

# from evidently_ai import retrain_and_register_model

# # Path to the drift flag file
# DRIFT_FLAG_PATH = "/opt/airflow/dags/drift_flag.json"

# # Default DAG arguments
# default_args = {
#     'owner': 'ash',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 1, 1),
#     'retries': 0,
# }

# # Define the DAG
# dag = DAG(
#     dag_id='realtime_drift_monitor',
#     default_args=default_args,
#     description='Triggers retraining when real-time drift is detected.',
#     schedule_interval='*/30 * * * *',  # Run every 30 minutes
#     catchup=False,
#     tags=["drift-monitoring", "mlflow", "evidently"]
# )

# # Utility function to check drift from JSON file
# def is_drift_true():
#     if not os.path.exists(DRIFT_FLAG_PATH):
#         return False
#     try:
#         with open(DRIFT_FLAG_PATH, "r") as f:
#             return json.load(f).get("drift_detected", False)
#     except Exception as e:
#         print(f"Error reading drift flag: {e}")
#         return False

# # Branch task logic
# def check_drift_flag():
#     try:
#         print(">>> [check_drift_flag] Checking drift flag JSON path...")
#         print(f">>> Current Working Directory: {os.getcwd()}")
#         print(f">>> DRIFT_FLAG_PATH: {DRIFT_FLAG_PATH}")

#         if not os.path.exists(DRIFT_FLAG_PATH):
#             print(">>> Drift flag file not found.")
#             return "skip_retraining"

#         with open(DRIFT_FLAG_PATH, "r") as f:
#             content = f.read()
#             print(f">>> Contents of drift_flag.json: {content}")
#             drift_data = json.loads(content)

#         drift_detected = drift_data.get("drift_detected", False)
#         print(f">>> Drift Detected = {drift_detected}")

#         if drift_detected:
#             print(">>> Drift detected — returning retrain task id.")
#             return "retrain_model_on_drift"
#         else:
#             print(">>> No drift detected — returning skip task id.")
#             return "skip_retraining"

#     except Exception as e:
#         print(f">>> Error in check_drift_flag: {e}")
#         return "skip_retraining"


# # Retraining task
# def safe_retrain_and_register():
#     try:
#         print("Starting model retraining...")
#         retrain_and_register_model()
#         print("Model retraining and registration complete.")
#     except Exception as e:
#         print(f"Retraining failed: {str(e)}")
#         raise  # Let Airflow mark the task as failed

# # Define the DAG tasks
# with dag:
#     wait_for_drift_flag = FileSensor(
#         task_id='wait_for_drift_flag',
#         filepath='drift_flag.json',
#         fs_conn_id='fs_default',
#         poke_interval=30,
#         timeout=1800,
#         mode='reschedule',
#         soft_fail=False,
#     )

#     check_flag = BranchPythonOperator(
#         task_id="check_drift_flag",
#         python_callable=check_drift_flag,
#     )

#     retrain_model = PythonOperator(
#         task_id="retrain_model_on_drift",
#         python_callable=safe_retrain_and_register,
#     )

#     skip_task = EmptyOperator(
#         task_id="skip_retraining"
#     )

#     # Set DAG dependencies
#     wait_for_drift_flag >> check_flag
#     check_flag >> [retrain_model, skip_task]




# from airflow import DAG
# from airflow.sensors.filesystem import FileSensor
# from airflow.operators.python import PythonOperator
# from datetime import datetime
# import json
# import os
# import sys

# # ✅ Append your project directory so Airflow can import your Python files
# sys.path.append("/opt/airflow/instilit_capstoneproj")

# from evidently_ai import retrain_and_register_model

# # ✅ Default DAG arguments
# default_args = {
#     'owner': 'ash',
#     'depends_on_past': False,
#     'start_date': datetime(2024, 1, 1),
#     'retries': 0,
# }

# # ✅ Define DAG
# dag = DAG(
#     dag_id='realtime_drift_monitor',
#     default_args=default_args,
#     description='Triggers retraining when real-time drift is detected.',
#     schedule_interval='*/30 * * * *',  # every 30 mins
#     catchup=False
# )

# DRIFT_FLAG_PATH = "/opt/airflow/dags/drift_flag.json"

# # ✅ Python function to check drift flag
# def check_drift_flag():
#     import json
#     import os

#     flag_path = "/opt/airflow/dags/drift_flag.json"
#     if not os.path.exists(flag_path):
#         raise FileNotFoundError(f"{flag_path} not found.")

#     with open(flag_path, "r") as f:
#         drift_data = json.load(f)

#     if drift_data.get("drift_detected"):
#         print("Drift detected, proceed to retrain.")
#         return True
#     else:
#         print("No drift detected. Skipping retrain.")
#         # Instead of raising, just exit cleanly
#         return False

# # ✅ Define tasks inside DAG context
# with dag:
#     wait_for_drift_flag = FileSensor(
#         task_id='wait_for_drift_flag',
#         filepath='/opt/airflow/dags/drift_flag.json',
#         fs_conn_id='fs_default',
#         poke_interval=30,     # every 30 seconds
#         timeout=1800,         # wait max 30 minutes
#         mode='reschedule',    # release worker
#         soft_fail=False,
#     )

#     check_flag = PythonOperator(
#         task_id="check_drift_flag",
#         python_callable=check_drift_flag,
#         trigger_rule="all_success"
#     )

#     retrain_model = PythonOperator(
#         task_id="retrain_model_on_drift",
#         python_callable=retrain_and_register_model
#     )

#     # ✅ Correct task dependency chaining
#     wait_for_drift_flag >> check_flag >> retrain_model
