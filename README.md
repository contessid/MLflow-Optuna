# Iris Classification with MLflow and Optuna

This project demonstrates how to perform hyperparameter tuning for a logistic regression model using the Iris dataset. The code leverages MLflow for experiment tracking and Optuna for hyperparameter optimization.

## Project Structure

- `iris_classification.py`: Main script for training and tuning the logistic regression model.
- `mlflow_utils.py`: Utility functions for MLflow.
- `optuna_utils.py`: Utility functions for Optuna.
- `blob_storage_deploy.py`: Script for uploading files to Azure Blob Storage.
- `start_mlflow_server.sh`: Script to start the MLflow server.
- `mlartifacts/`: Directory containing MLflow artifacts.
- `mlruns/`: Directory containing MLflow run data.
- `models/`: Directory containing registered models.

## Requirements

- Python 3.x
- MLflow
- Optuna
- scikit-learn
- Azure Storage Blob

## Setup

1. Install the required Python packages:
    ```sh
    pip install mlflow optuna scikit-learn azure-storage-blob
    ```

2. Start the MLflow server:
    ```sh
    ./start_mlflow_server.sh
    ```

## Running the Scripts

### Iris Classification

To run the `iris_classification.py` script, execute the following command:

```sh 
python iris_classification.py
```

### Blob Storage Deployment

To run the `blob_storage_deploy.py` script, execute the following command:

```sh
python blob_storage_deploy.py
```

## Script Explanations

### `iris_classification.py`

The `iris_classification.py` script performs the following steps:

1. **Import Libraries**: Imports necessary libraries including MLflow, Optuna, and scikit-learn.
2. **Load Dataset**: Loads the Iris dataset using scikit-learn's `load_iris` function.
3. **Split Data**: Splits the dataset into training and validation sets.
4. **Set MLflow Experiment**: Sets the current active MLflow experiment using the `get_or_create_experiment` function from `mlflow_utils.py`.
5. **Start MLflow Run**: Initiates an MLflow run and creates an Optuna study for hyperparameter tuning.
6. **Optimize Hyperparameters**: Uses Optuna to optimize the hyperparameters of a logistic regression model. The `objective` function from `optuna_utils.py` is used as the objective function.
7. **Log Parameters and Metrics**: Logs the best hyperparameters and the best accuracy to MLflow.
8. **Set Tags**: Logs tags related to the project, optimizer engine, model family, and feature set version.
9. **Train Model**: Trains a logistic regression model using the best hyperparameters found by Optuna.
10. **Log Model**: Logs the trained model as an artifact in MLflow.
11. **Print Model URI**: Prints the URI of the logged model.
12. **Evaluate Model**: Evaluates the model on the validation set and logs the evaluation metrics to MLflow.

### `blob_storage_deploy.py`

The `blob_storage_deploy.py` script performs the following steps:

1. **Import Libraries**: Imports necessary libraries including `os` and `azure.storage.blob`.
2. **Define Function**: Defines the `upload_directory_to_blob` function to upload a directory to Azure Blob Storage.
3. **Connect to Azure Blob Storage**: Connects to Azure Blob Storage using the connection string from the environment variable `AZURE_STORAGE_CONNECTION_STRING`.
4. **Recursively Upload Files**: Recursively uploads each file in the specified local directory to Azure Blob Storage, excluding certain files like `requirements.txt`, `python_env.yaml`, and `conda.yaml`.
5. **Get Blob Client**: Gets the blob client for each file and uploads the file to the specified container and blob name.
6. **Handle Exceptions**: Handles any exceptions that occur during the upload process and prints an error message.

## Utility Functions

- `mlflow_utils.py`: Contains the `get_or_create_experiment` function to manage MLflow experiments.
- `optuna_utils.py`: Contains the `champion_callback` and `logistic_regression_error` functions for Optuna optimization.

## Output in MLflow Dashboard

When you run the `iris_classification.py` script, the following will be logged and displayed in the MLflow dashboard:

1. **Experiment Creation/Loading**: An experiment will be created or loaded in MLflow. This experiment will contain all the runs related to the Iris classification project.

2. **Nested Runs**: A nested run will be logged within the experiment. This parent run will encapsulate several child runs, each corresponding to a training run performed during the Optuna hyperparameter tuning process.

3. **Training Runs**: At the lowest level, there will be multiple training runs logged by Optuna. Each run will represent a different set of hyperparameters evaluated during the tuning process. Metrics such as accuracy and loss will be logged for each run.

4. **Best Parameters Logging**: Once Optuna identifies the best hyperparameters, these parameters will be logged in the parent run. This includes the best hyperparameters and the corresponding performance metrics.

5. **Model Registration**: The best model, trained with the optimal hyperparameters, will be registered as a new artifact in MLflow. The model will have a default validation status set to "pending".

6. **Tags and Metrics**: Various tags and metrics will be logged for both the parent and child runs, providing detailed information about the experiment, optimizer engine, model family, and feature set version.

By navigating to the MLflow dashboard, you can visualize and analyze the experiment, nested runs, and the performance of different hyperparameter configurations. The registered model can be accessed and managed through the MLflow model registry.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
