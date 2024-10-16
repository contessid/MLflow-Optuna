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
6. **Optimize Hyperparameters**: Uses Optuna to optimize the hyperparameters of a logistic regression model. The `logistic_regression_error` function from `optuna_utils.py` is used as the objective function.
7. **Log Parameters and Metrics**: Logs the best hyperparameters and the best root mean square error (RMSE) to MLflow.
8. **Set Tags**: Logs tags related to the project, optimizer engine, model family, and feature set version.
9. **Train Model**: Trains a logistic regression model using the best hyperparameters found by Optuna.
10. **Log Model**: Logs the trained model as an artifact in MLflow.
11. **Print Model URI**: Prints the URI of the logged model.

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
