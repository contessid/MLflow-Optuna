import os

import joblib
import mlflow
import mlflow.models
import mlflow.sklearn
import optuna
from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlflow_utils import get_or_create_experiment
from optuna_utils import champion_callback, logistic_regression_error

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

client = mlflow.tracking.MlflowClient()

iris = load_iris()

# Split the data into a training set and a test set
X_train, X_valid, y_train, y_valid = train_test_split(
    iris.data, iris.target, test_size=0.3
)

# Set the current active MLflow experiment
# run_name = "fourth_attempt"
experiment_id = get_or_create_experiment("Iris Classification")
# set the experiment id for ALL the runs
mlflow.set_experiment(experiment_id=experiment_id)

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(
    nested=True
):  # ,run_name=run_name, experiment_id=experiment_id, ):
    # Create an Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name="Iris Classification",
        # storage="sqlite:///iris.db",
        load_if_exists=True,
    )

    # Optimize the study
    study.optimize(
        lambda trial: logistic_regression_error(
            trial, X_train, X_valid, y_train, y_valid
        ),
        n_trials=10,
        # callbacks=[champion_callback],
    )

    # Load the best model of the study from the file
    if os.path.exists("best_model.pkl"):
        best_model = joblib.load("best_model.pkl")
    # Otherwise, train the best model from scratch
    else:
        best_model = LogisticRegression(**study.best_params)
        best_model.fit(X_train, y_train)

    signature = mlflow.models.infer_signature(X_train, y_train)

    artifact_path = "model"
    # get the best model's run name
    best_run_name = study.best_trial.user_attrs["trial_name"]
    # change the current run name to the best run name
    # Set the model tags
    mlflow.set_tags(
        tags={
            "project": "Iris Classification",
            "mlflow.runName": best_run_name,
            "optimizer_engine": "optuna",
            "model_family": "logistic_regression",
            "feature_set_version": 1,
        }
    )
    mlflow.log_params(study.best_params)
    mlflow.log_metric("accuracy", study.best_value)

    # Log the model as an artifact
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path=artifact_path,
        input_example=X_train[0].reshape(1, -1),
        signature=signature,
        registered_model_name="iris_classification_model",
    )

    model_version = client.get_latest_versions("iris_classification_model")[0].version
    client.set_model_version_tag(
        name="iris_classification_model",
        version=model_version,
        key="validation_status",
        value="pending",
    )

    # remove the best model file
    os.remove("best_model.pkl")
    # Get the logged model uri so that we can load it from the artifact store
    # model_uri = mlflow.get_artifact_uri(artifact_path)

# print(model_uri)
