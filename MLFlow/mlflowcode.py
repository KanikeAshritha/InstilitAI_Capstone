import os
import tempfile
import joblib
import mlflow
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from main import get_pipeline


def regression_accuracy(y_true, y_pred, tolerance=0.1):
    rel_error = np.abs((y_pred - y_true) / y_true)
    return np.mean(rel_error <= tolerance)


def perform_random_search(pipeline, X_train, y_train, param_distributions={}, n_iter=25):
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=3,
        scoring='r2',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)
    return search


def log_and_register_model(model, model_name):
    artifact_path = "model"

    # End any existing MLflow run
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=f"{model_name}_Register") as run:
        run_id = run.info.run_id
        experiment = mlflow.get_experiment(run.info.experiment_id)

        # ✅ Save model to a proper tmp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_model_path = os.path.join(tmp_dir, "model.pkl")
            joblib.dump(model, local_model_path)

            # ✅ log_model with `path=...` pointing to actual file
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
            )


        # ✅ Register the model using MLflow client
        model_uri = f"runs:/{run_id}/{artifact_path}"
        client = MlflowClient()

        try:
            client.get_registered_model(model_name)
        except MlflowException:
            client.create_registered_model(model_name)

        version = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        ).version

        # Promote to Production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )

        # Save locally
        unique_model_name = f"{model_name}_v{version}"
        joblib.dump(model, f"{unique_model_name}.pkl")

        # Save run/model URIs
        with open("latest_model_uri.txt", "w") as f:
            f.write(model_uri)
        with open("latest_run_id.txt", "w") as f:
            f.write(run_id)

        return {
            "model_uri": model_uri,
            "model_name": model_name,
            "version": version,
            "run_id": run_id,
            "experiment_name": experiment.name,
            "unique_model_name": unique_model_name
        }


def mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry):
    best_model_info = {}
    best_r2 = float('-inf')
    best_model = None
    best_model_name = ""
    best_pipeline = None

    mlflow.set_tracking_uri("http://host.docker.internal:5002")  # or localhost:5002 if on host
    mlflow.set_experiment("Instilit_Regression")

    for model_name, model in model_registry.items():
        with mlflow.start_run(run_name=f"{model_name}_Regressor", nested=True):
            pipeline = get_pipeline(model)
            search = perform_random_search(pipeline, X_train, y_train)

            current_best_model = search.best_estimator_
            y_pred = current_best_model.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            acc = regression_accuracy(y_test, y_pred)

            mlflow.log_params(search.best_params_)
            mlflow.log_metrics({
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Accuracy_within_10pct": acc
            })

            print(f"\n✅ {model_name} — R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy ±10%: {acc * 100:.2f}%")

            if r2 > best_r2:
                best_r2 = r2
                best_model = current_best_model
                best_model_name = model_name
                best_pipeline = pipeline

    # ✅ After comparing all models — Register only the BEST one
    if best_model:
        with mlflow.start_run(run_name=f"{best_model_name}_Register"):
            best_model_info = log_and_register_model(best_model, best_model_name)

        with open("latest_run_id.txt", "w") as f:
            f.write(best_model_info["run_id"])

        print("\n🏆 FINAL BEST MODEL")
        print(f"Model Name        : {best_model_info['model_name']}")
        print(f"Version           : {best_model_info['version']}")
        print(f"Run ID            : {best_model_info['run_id']}")
        print(f"Experiment Name   : {best_model_info['experiment_name']}")
        print(f"Saved as          : {best_model_info['unique_model_name']}.pkl")
    else:
        print("⚠️ No best model selected.")

    return best_model_info






















# import joblib
# import mlflow
# import numpy as np
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from mlflow.tracking import MlflowClient
# from mlflow.exceptions import MlflowException
# from main import get_pipeline  # Function that returns pipeline with preprocessor + model

# # Custom metric to calculate how many predictions fall within ±10% of true value
# def regression_accuracy(y_true, y_pred, tolerance=0.1):
#     rel_error = np.abs((y_pred - y_true) / y_true)
#     return np.mean(rel_error <= tolerance)

# # Perform RandomizedSearchCV with given pipeline and parameter grid
# def perform_random_search(pipeline, X_train, y_train, param_distributions={}, n_iter=25):
#     search = RandomizedSearchCV(
#         pipeline,
#         param_distributions=param_distributions,
#         n_iter=n_iter,
#         cv=3,
#         scoring='r2',
#         verbose=1,
#         n_jobs=-1,
#         random_state=42
#     )
#     search.fit(X_train, y_train)  # Fit search object on training data
#     return search  # Return best model and results

# # Log the model to MLflow, register it, promote to production, and save locally
# def log_and_register_model(model, model_name):
#     run = mlflow.active_run()
#     run_id = run.info.run_id
#     artifact_path = f"{model_name}_model"  # MLflow model folder name

#     # Log the model artifact
#     mlflow.sklearn.log_model(model, artifact_path=artifact_path)

#     # Construct full model URI for registry
#     model_uri = f"runs:/{run_id}/{artifact_path}"

#     # Initialize MLflow client
#     client = MlflowClient()

#     # Create registered model entry if not already created
#     try:
#         client.get_registered_model(model_name)
#     except MlflowException:
#         client.create_registered_model(model_name)

#     # Register new version of the model
#     version = client.create_model_version(
#         name=model_name,
#         source=model_uri,
#         run_id=run_id
#     ).version

#     # Promote new version to "Production" stage
#     client.transition_model_version_stage(
#         name=model_name,
#         version=version,
#         stage="Production"
#     )

#     # Save the model to disk as .pkl file with versioned name
#     unique_model_name = f"{model_name}_v{version}"
#     joblib.dump(model, f"{unique_model_name}.pkl")

#     # Save the latest model URI and run ID to local text files
#     with open("latest_model_uri.txt", "w") as f:
#         f.write(model_uri)

#     with open("latest_run_id.txt", "w") as f:
#         f.write(run_id)

#     # Return all important info
#     return {
#         "model_uri": model_uri,
#         "model_name": model_name,
#         "version": version,
#         "run_id": run_id,
#         "unique_model_name": unique_model_name
#     }

# # Main function to loop through models, run random search, log results to MLflow, and register best model
# def mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry):
#     best_model_info = {}     # Dictionary to store info about best model
#     best_r2 = float("-inf")  # Initialize best R² score as very low

#     mlflow.set_tracking_uri("http://host.docker.internal:5000") # <--- valid local path inside container
#     mlflow.set_experiment("Instilit_Regression")# Set experiment name for MLflow

#     # Loop through each model in model_registry dictionary
#     for model_name, model in model_registry.items():
#         with mlflow.start_run(run_name=f"{model_name}_Regressor", nested=True):
#             pipeline = get_pipeline(model)  # Get full pipeline with preprocessing + model

#             # Perform hyperparameter tuning
#             search = perform_random_search(pipeline, X_train, y_train)

#             best_model = search.best_estimator_  # Get best model after tuning
#             y_pred = best_model.predict(X_test)  # Predict on test data

#             # Evaluate metrics
#             r2 = r2_score(y_test, y_pred)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             mae = mean_absolute_error(y_test, y_pred)
#             acc = regression_accuracy(y_test, y_pred)

#             # Log best parameters and R² to MLflow
#             mlflow.log_params(search.best_params_)
#             mlflow.log_metrics({"R2": r2})

#             # Print evaluation summary
#             print(f"\n{model_name} — R²: {r2:.4f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, Accuracy ±10%: {acc:.2%}")

#             # Track best performing model for registration
#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model_info = log_and_register_model(best_model, model_name)

#     # Summary print after all models are evaluated
#     if best_model_info:
#         print("\nBest model registered:")
#         print(f"Model Name        : {best_model_info['model_name']}")
#         print(f"Version           : {best_model_info['version']}")
#         print(f"Run ID            : {best_model_info['run_id']}")
#         print(f"Saved as          : {best_model_info['unique_model_name']}.pkl")
#     else:
#         print("No model met the criteria.")

#     return best_model_info  # Return final best model info for use






# # mlflowcode.py
# import mlflow
# import numpy as np
# import joblib
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from mlflow.tracking import MlflowClient
# from main import get_pipeline
# from mlflow.exceptions import MlflowException


# def regression_accuracy(y_true, y_pred, tolerance=0.1):
#     rel_error = abs((y_pred - y_true) / y_true)
#     return (rel_error <= tolerance).mean()


# def perform_random_search(pipeline, X_train, y_train, param_distributions={}, n_iter=25):
#     search = RandomizedSearchCV(
#         pipeline,
#         param_distributions=param_distributions,
#         n_iter=n_iter,
#         cv=3,
#         scoring='r2',
#         verbose=2,
#         n_jobs=-1,
#         random_state=42
#     )
#     search.fit(X_train, y_train)
#     return search

# def log_and_register_model(model, model_name):
#     run = mlflow.active_run()
#     run_id = run.info.run_id
#     experiment = mlflow.get_experiment(run.info.experiment_id)
#     artifact_path = f"{model_name}_model"

#     mlflow.sklearn.log_model(sk_model=model, artifact_path=artifact_path)
#     client = MlflowClient()
#     model_uri = f"runs:/{run_id}/{artifact_path}"

#     # Register model
#     try:
#         client.get_registered_model(model_name)
#     except MlflowException:
#         client.create_registered_model(model_name)

#     version = client.create_model_version(
#         name=model_name,
#         source=model_uri,
#         run_id=run_id
#     ).version

#     # Transition to Staging
#     client.transition_model_version_stage(
#         name=model_name,
#         version=version,
#         stage="Production"
#     )

#     unique_model_name = f"{model_name}_v{version}"
#     joblib.dump(model, f"{unique_model_name}.pkl")

#     # ✅ Save full URI to a text file for Flask prediction
#     with open("latest_model_uri.txt", "w") as f:
#         f.write(model_uri)

#     # Optional: also save run ID
#     with open("latest_run_id.txt", "w") as f:
#         f.write(run_id)

#     return {
#     "model_uri": model_uri,
#     "model_name": model_name,
#     "version": version,
#     "run_id": run_id,
#     "experiment_name": experiment.name,
#     "unique_model_name": unique_model_name
#     }




# def mlflow_run_with_random_search(X_train, X_test, y_train, y_test, model_registry):
#     best_model_info = {}
#     best_r2 = float('-inf')
#     best_model = None
#     best_model_name = ""
#     best_pipeline = None

#     mlflow.set_experiment("Instilit_Regression")

#     for model_name, model in model_registry.items():
#         with mlflow.start_run(run_name=f"{model_name}_Regressor", nested=True):
#             pipeline = get_pipeline(model)
#             search = perform_random_search(pipeline, X_train, y_train)

#             current_best_model = search.best_estimator_
#             y_pred = current_best_model.predict(X_test)

#             r2 = r2_score(y_test, y_pred)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             mae = mean_absolute_error(y_test, y_pred)
#             acc = regression_accuracy(y_test, y_pred)

#             mlflow.log_params(search.best_params_)
#             mlflow.log_metrics({
#                 "R2": r2,
#                 "RMSE": rmse,
#                 "MAE": mae,
#                 "Accuracy_within_10pct": acc
#             })

#             print(f"\n✅ {model_name} — R²: {r2:.4f}, RMSE: {rmse:.2f},  MAE: {mae:.2f}, Accuracy ±10%: {acc * 100:.2f}%")

#             if r2 > best_r2:
#                 best_r2 = r2
#                 best_model = current_best_model
#                 best_model_name = model_name
#                 best_pipeline = pipeline

#     # ✅ Register only the best model
#     if best_model:
#         with mlflow.start_run(run_name=f"{best_model_name}_Register"):
#             best_model_info = log_and_register_model(best_model, best_model_name)

#         # ✅ Save run_id for Flask
#         with open("latest_run_id.txt", "w") as f:
#             f.write(best_model_info["run_id"])


#         print("\n🏆 FINAL BEST MODEL")
#         print(f"Model Name        : {best_model_info['model_name']}")
#         print(f"Version           : {best_model_info['version']}")
#         print(f"Run ID            : {best_model_info['run_id']}")
#         print(f"Experiment Name   : {best_model_info['experiment_name']}")
#         print(f"Saved as          : {best_model_info['unique_model_name']}.pkl")
#     else:
#         print("⚠️ No best model selected.")

#     return best_model_info
