import mlflow
import optuna
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlflow_utils import get_or_create_experiment
from optuna_utils import champion_callback, logistic_regression_error

mlflow.set_tracking_uri("http://localhost:8080")

iris = load_iris()

# Split the data into a training set and a test set
X_train, X_valid, y_train, y_valid = train_test_split(
    iris.data, iris.target, test_size=0.25
)

# Set the current active MLflow experiment
run_name = "first_attempt"
experiment_id = get_or_create_experiment("Iris Classification")

# Initiate the parent run and call the hyperparameter tuning child run logic
with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
    # Create an Optuna study
    study = optuna.create_study(
        direction="minimize",
        study_name="Iris Classification",
        storage="sqlite:///iris.db",
        load_if_exists=True,
    )

    # Optimize the study
    study.optimize(
        lambda trial: logistic_regression_error(
            trial, X_train, X_valid, y_train, y_valid
        ),
        n_trials=100,
        callbacks=[champion_callback],
    )

    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_rmse", study.best_value)

    # Log tags
    mlflow.set_tags(
        tags={
            "project": "Iris Classification",
            "optimizer_engine": "optuna",
            "model_family": "logistic_regression",
            "feature_set_version": 1,
        }
    )

    # Log a fit model instance
    model = LogisticRegression(**study.best_params)
    model.fit(X_train, y_train)

    artifact_path = "model"

    # Log the model as an artifact
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        input_example=X_train[0].reshape(1, -1),
        registered_model_name="Iris Classification Model",
    )

    # Get the logged model uri so that we can load it from the artifact store
    model_uri = mlflow.get_artifact_uri(artifact_path)

print(model_uri)
