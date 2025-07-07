# %% Imports
import pandas as pd
import numpy as np
import psycopg2
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
import shap

from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

import joblib
import mlflow
from sqlalchemy import create_engine

import os

# Dynamically choose the DB host
DB_HOST = os.getenv("DB_HOST", "localhost")  # default is localhost for local testing



# Database configuration
DB_CONFIG = {
    "dbname": "instilit",
    "user": "kanikeashritha",
    "password": "ash",
    "host": DB_HOST,
    "port": "5432"
}

TABLE_NAME = "instilit_salary_data"
CSV_PATH = "Software_Salaries.csv"

# Load CSV and insert into PostgreSQL after removing duplicates
def load_csv_to_postgres(csv_path, table_name, db_config):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.replace({pd.NA: None, pd.NaT: None})
    df = df.drop_duplicates()

    data = df.values.tolist()
    
    conn = psycopg2.connect(**db_config)
    cur = conn.cursor()

    cur.execute(f"DELETE FROM {table_name}")  # Clear old data before inserting new

    # Insert all cleaned rows
    insert_query = f"""
    INSERT INTO {table_name} (
        job_title, experience_level, employment_type, company_size, company_location,
        remote_ratio, salary_currency, years_experience, base_salary, bonus,
        stock_options, total_salary, salary_in_usd, currency, education, skills,
        conversion_rate, adjusted_total_usd
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cur.executemany(insert_query, data)
    conn.commit()
    cur.close()
    conn.close()
    print("Duplicates removed & data inserted into PostgreSQL.")

# Load data from PostgreSQL
def load_data_from_postgres(table_name, db_config):
    conn = psycopg2.connect(**db_config)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", con=conn)
    conn.close()
    print("Data loaded successfully from PostgreSQL!")
    return df

# Perform exploratory data analysis
def perform_eda(df):
    df = df.copy()

    print("Data Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDescriptive Statistics:")
    print(df.describe())

    # Plot histograms for numeric features
    num_cols = ['base_salary', 'bonus', 'stock_options', 'adjusted_total_usd']
    for col in num_cols:
        if col in df.columns:
            plt.figure()
            sns.histplot(df[col], bins=30, kde=True)
            plt.title(f"Histogram: {col}")
            plt.tight_layout()
            plt.show()

    # Plot correlation heatmap
    corr = df.corr(numeric_only=True)
    if not corr.empty:
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        plt.show()

    # Plot boxplots for important salary columns
    for col in ['base_salary', 'adjusted_total_usd', 'salary_in_usd']:
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        plt.show()

    # Countplot for categorical columns
    cat_cols = ['company_size', 'remote_ratio', 'salary_currency', 'currency',
                'employment_type', 'experience_level', 'company_location', 'job_title']
    for col in cat_cols:
        if df[col].nunique() > 20:
            top_values = df[col].value_counts().nlargest(10).index
            df_plot = df[df[col].isin(top_values)]
        else:
            df_plot = df

        plt.figure(figsize=(8, 3))
        sns.countplot(data=df_plot, x=col, order=df_plot[col].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f"Distribution of {col} (Top 10 if >20 unique)")
        plt.tight_layout()
        plt.show()

    # Scatterplots of features vs adjusted salary
    for col in ['years_experience', 'base_salary', 'bonus', 'stock_options']:
        sns.scatterplot(x=col, y='adjusted_total_usd', data=df)
        plt.title(f"{col} vs Adjusted Salary")
        plt.show()

# Cap outliers using the IQR method
def cap_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df.loc[df[column] < lower, column] = lower
    df.loc[df[column] > upper, column] = upper
    return df

# Preprocess the data (clean, encode, scale)
def preprocess_data(df):
    df = df.copy()

    if 'base_salary' not in df.columns:
        raise ValueError("'base_salary' column is missing from the input data.")

    # Remove rows where base salary is missing or zero
    df['base_salary'] = pd.to_numeric(df['base_salary'], errors='coerce')
    df = df[df['base_salary'].notna() & (df['base_salary'] > 0)]


    def normalize_job_title(title):
        title = title.lower().strip()
        
        # Fix common variants of Data Scientist
        if title in ["Data Scienist", "Data Scntist", "Dt Scientist"]:
            return "Data Scientist"
        if title in ["ML Enginer","ML Engr","Machine Learning Engr"]:
            return "Machine Learning Engineer"
        if title in ["Softwre Engineer","Software Engr","Sofware Engneer"]:
            return "Software Engineer"

        
        # Add more mappings as needed
        return title.title()  # Capitalize each word

# Apply to column
    df["job_title"] = df["job_title"].apply(normalize_job_title)


    # Fill missing values for key categorical fields
    df['experience_level'] = df.get('experience_level', 'Unknown').fillna('Unknown')
    df['employment_type'] = df.get('employment_type', 'Unknown').fillna('Unknown')

    # Drop unused or less relevant columns
    df.drop(columns=['skills', 'education'], inplace=True, errors='ignore')

    # Clean numeric fields and fill missing with 0
    numeric_cols = ['base_salary', 'salary_in_usd', 'conversion_rate', 'bonus', 'stock_options', 'total_salary']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Cap outliers for salary fields
    cap_cols = ['base_salary', 'salary_in_usd']
    if 'adjusted_total_usd' in df.columns:
        cap_cols.append('adjusted_total_usd')
    for col in cap_cols:
        df = cap_outliers_iqr(df, col)

    # Apply transformation to the target variable
    if 'adjusted_total_usd' in df.columns:
        pt = PowerTransformer(method='yeo-johnson')
        df['transformed_salary'] = pt.fit_transform(df[['adjusted_total_usd']])
    else:
        df['transformed_salary'] = np.log1p(df['base_salary'])  # fallback

    df['log_base_salary'] = np.log1p(df['base_salary'])

    return df

# Custom accuracy metric: percentage of predictions within ±10% error
def regression_accuracy(y_true, y_pred, tolerance=0.1):
    rel_error = np.abs((y_pred - y_true) / y_true)
    return np.mean(rel_error <= tolerance)

# Print evaluation metrics for regression models
def evaluate_regression(y_true, y_pred, model_name):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    acc = regression_accuracy(y_true, y_pred)

    print(f"\n{model_name} Evaluation:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"Accuracy within ±10%: {acc * 100:.2f}%")

# Define pipeline with preprocessing and regression model
def get_pipeline(model):
    num_features = ['remote_ratio', 'years_experience', 'base_salary', 'bonus', 'stock_options',
                    'total_salary', 'conversion_rate']
    cat_features = ['company_size', 'salary_currency', 'currency',
                    'employment_type', 'experience_level', 'company_location', 'job_title']

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_features),
        ("cat", categorical_pipeline, cat_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

# Generate SHAP summary plot for XGBoost model
def explain_with_shap(pipeline, X_train):
    print("\nSHAP Explanation for XGB Regressor")
    model = pipeline.named_steps["regressor"]
    transformer = pipeline.named_steps["preprocessor"]
    X_trans = transformer.transform(X_train)

    explainer = shap.Explainer(model)
    shap_values = explainer(X_trans)
    shap.summary_plot(shap_values, X_trans, show=True)

# Model configurations to train
model_registry = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10, min_samples_split=10, min_samples_leaf=5,
        ccp_alpha=0.01, random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=4,
        max_features='sqrt', random_state=42, n_jobs=-1
    ),
    "XGB Regressor": XGBRegressor(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
    "LightGBM Regressor": LGBMRegressor(
        n_estimators=100, learning_rate=0.1, max_depth=10,
        num_leaves=31, random_state=42
    )
}

# Train and evaluate all models
def train_all_models(df_clean, model_registry):
    for name, model in model_registry.items():
        print(f"\nTraining {name}")
        pipeline = get_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        evaluate_regression(y_test, y_pred, name)

        if "XGB" in name:
            explain_with_shap(pipeline, X_train)

    joblib.dump(pipeline, "model.pkl")  # Save the last model pipeline
    print("Final trained model pipeline saved as 'model.pkl'")
    return X_train, X_test, y_train, y_test

# Store final predictions to PostgreSQL
def store_predictions_in_db(df, table_name, db_config):
    engine = create_engine(
        f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['dbname']}"
    )
    df.to_sql(table_name, engine, if_exists='append', index=False)
    print(f"Stored predictions to {table_name} table.")

# Main entry point
if __name__ == "__main__":
    from mlflowcode import mlflow_run_with_random_search
    from evidently_ai import log_evidently_reports

    # Load and insert data into PostgreSQL
    load_csv_to_postgres(CSV_PATH, TABLE_NAME, DB_CONFIG)
    
    # Load data from PostgreSQL
    df = load_data_from_postgres(TABLE_NAME, DB_CONFIG)
    
    # Preprocess data
    df_clean = preprocess_data(df)

    # Prepare train-test split
    X = df_clean.drop(columns=["adjusted_total_usd"])
    y = df_clean["adjusted_total_usd"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Prepare drift inputs
    train_df = X_train.copy()
    train_df["adjusted_total_usd"] = y_train
    test_df = X_test.copy()
    test_df["adjusted_total_usd"] = y_test
    new_df = test_df.sample(frac=1.0).reset_index(drop=True)  # simulate new incoming data

    # Run MLflow training with randomized search
    flow = mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry)

    # Drift check using Evidently
    with mlflow.start_run(run_name="Evidently Drift Check"):
        drift_detected = log_evidently_reports(train_df, test_df, new_df, df_clean)

    # Retrain model if drift is detected
    if drift_detected:
        print("Drift Detected! Triggering retraining...")
        mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry)
    else:
        print("No drift detected. Skipping retrain.")
