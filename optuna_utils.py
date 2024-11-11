import joblib
import mlflow
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import root_mean_squared_error

# override Optuna's default logging to ERROR only
optuna.logging.set_verbosity(optuna.logging.ERROR)

# Global variable to store the best model and score
best_model = None
best_score = float("-inf")


def logistic_regression_error(trial, X_train, X_valid, y_train, y_valid):
    global best_model, best_score
    with mlflow.start_run(nested=True):
        # set the name of the optuna trial as the run name
        trial.set_user_attr("trial_name", mlflow.active_run().info.run_name)

        # Define fixed hyperparameters
        params = {}

        # define hyperparameters to optimize
        params["C"] = trial.suggest_float("C", 1e-5, 1, log=True)
        params["solver"] = trial.suggest_categorical(
            "solver", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
        )

        # Train a logistic regression model
        logreg = LogisticRegression(**params)
        logreg.fit(X_train, y_train)
        preds = logreg.predict(X_valid)
        error = root_mean_squared_error(y_valid, preds)
        accuracy = logreg.score(X_valid, y_valid)

        # Check if this is the best score so far
        if accuracy > best_score:
            best_score = accuracy
            best_model = logreg  # Save the current model
            # Save the model to a file (optional)
            joblib.dump(best_model, "best_model.pkl")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", error)
        mlflow.log_metric("accuracy", accuracy)

    return accuracy


# define a logging callback that will report on only new challenger parameter configurations if a
# trial has usurped the state of 'best conditions'
def champion_callback(study, frozen_trial):
    """
    Logging callback that will report when a new trial iteration improves upon existing
    best trial values.

    Note: This callback is not intended for use in distributed computing systems such as Spark
    or Ray due to the micro-batch iterative implementation for distributing trials to a cluster's
    workers or agents.
    The race conditions with file system state management for distributed trials will render
    inconsistent values with this callback.
    """

    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            improvement_percent = (
                abs(winner - study.best_value) / study.best_value
            ) * 100
            print(
                f"Trial {frozen_trial.number} achieved value: {frozen_trial.value} with "
                f"{improvement_percent: .4f}% improvement"
            )
        else:
            print(
                f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}"
            )
