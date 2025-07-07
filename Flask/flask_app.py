from flask import Flask, render_template, request
import pandas as pd
import os
import mlflow
import json
from main import load_data_from_postgres,DB_CONFIG,TABLE_NAME

# Importing necessary functions from the main pipeline
from main import (
    preprocess_data,
    load_data_from_postgres,
    DB_CONFIG,
    TABLE_NAME,
    store_predictions_in_db 
)

from evidently import Report
from evidently.presets import DataDriftPreset

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"  # Folder to store uploaded CSVs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not present

# Load the latest ML model from MLflow registry using the stored URI
def load_model_from_latest_run():
    with open("latest_model_uri.txt", "r") as f:
        model_uri = f.read().strip()
    return mlflow.pyfunc.load_model(model_uri)

# Function to check for data drift between reference and new data using Evidently
def check_data_drift(reference_df, current_df, threshold=0.3):
    report = Report(metrics = [DataDriftPreset()])
    res = report.run(reference_data=reference_df, current_data=current_df)

    # Save drift report as HTML so user can view it
    html_path = "static/drift_report.html"
    res.save_html(html_path)

    # Parse the JSON report to check if drift is detected
    report_json = json.loads(res.json())
    metrics = report_json.get("metrics", [])
    for metric in metrics:
        if metric.get("metric") == "DataDriftTable":
            return metric["result"].get("dataset_drift", False)
    return False


# Home page — renders upload form
@app.route("/")
def home():
    return render_template("upload.html")


# Endpoint for handling predictions from uploaded CSV
@app.route("/predict", methods=["POST"])
def predict():
    # Check if file is included in form submission
    if "file" not in request.files:
        return "No file uploaded."

    file = request.files["file"]
    if file.filename == "":
        return "No selected file."

    # Save uploaded file to disk
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Read uploaded CSV into DataFrame
        new_df = pd.read_csv(file_path)

        # Convert numeric columns safely and fill missing values with 0
        numeric_cols = ["base_salary", "bonus", "stock_options", "conversion_rate"]
        for col in numeric_cols:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce").fillna(0)

        # Manually calculate adjusted salary for reference
        new_df["adjusted_total_usd"] = (
            new_df["base_salary"] + new_df["bonus"] + new_df["stock_options"]
        ) * new_df["conversion_rate"]

        # Load reference data (historical training data) from PostgreSQL
        reference_df = load_data_from_postgres(TABLE_NAME, DB_CONFIG).sample(1000, random_state=42)

        # Preprocess both new and reference datasets using same function
        reference_preprocessed = preprocess_data(reference_df)
        new_df_preprocessed = preprocess_data(new_df)

        # Check for data drift
        drift_found = check_data_drift(reference_preprocessed, new_df_preprocessed)

        # Drop target column before prediction if it exists
        if "adjusted_total_usd" in new_df_preprocessed.columns:
            new_df_preprocessed = new_df_preprocessed.drop(columns=["adjusted_total_usd"])

        # Load model from MLflow and make predictions
        model = load_model_from_latest_run()
        predictions = model.predict(new_df_preprocessed)
        new_df["Predicted_Adjusted_Salary"] = predictions.round(2)

        # Save predictions back to PostgreSQL
        store_predictions_in_db(new_df, table_name="salary_predictions", db_config=DB_CONFIG)

        # Convert DataFrame to HTML table for UI display
        html_table = new_df.to_html(classes="table table-bordered", index=False)

        # Show drift status above table
        drift_note = "<p><b> Data Drift Detected!</b></p>" if drift_found else "<p>✅ No Data Drift.</p>"

        # Render final result page with predictions and drift message
        return render_template("result.html", table=drift_note + html_table)

    except Exception as e:
        return f" Error during prediction: {str(e)}"


# Start the Flask web server on port 5001
if __name__ == "__main__":
    app.run(debug=True, port=5001)

