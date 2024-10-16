import os
import tempfile

import mlflow
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

# Load environment variables from .env file
load_dotenv()

# Azure Blob Storage configuration
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AzureWebJobsStorage")
AZURE_CONTAINER_NAME = os.getenv("ContainerName")

# Initialize the MLflow Client
client = MlflowClient()


# # Function to download model artifacts and upload to Azure Blob Storage
# def push_model_to_blob(model_name, model_version, local_path, blob_name):
#     try:
#         # Connect to Azure Blob Storage
#         blob_service_client = BlobServiceClient.from_connection_string(
#             AZURE_STORAGE_CONNECTION_STRING
#         )
#         blob_client = blob_service_client.get_blob_client(
#             container=AZURE_CONTAINER_NAME, blob=blob_name
#         )

#         # Upload the model file to Blob Storage
#         with open(local_path, "rb") as data:
#             blob_client.upload_blob(data, overwrite=True)

#         print(
#             f"Model {model_name} version {model_version} successfully uploaded to blob storage: {blob_name}"
#         )
#     except Exception as e:
#         print(f"Error uploading {model_name} version {model_version} to blob: {e}")


# Function to upload files in the local model directory to Azure Blob Storage
def upload_directory_to_blob(local_dir, blob_prefix):
    try:
        # Connect to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_STORAGE_CONNECTION_STRING
        )

        # Recursively upload each file in the directory to Blob Storage
        for root, _, files in os.walk(local_dir):
            for file_name in files:
                if file_name not in [
                    "requirements.txt",
                    "python_env.yaml",
                    "conda.yaml",
                ]:
                    file_path = os.path.join(root, file_name)
                    relative_path = os.path.relpath(
                        file_path, local_dir
                    )  # Get relative path for blob storage structure
                    blob_name = os.path.join(blob_prefix, relative_path).replace(
                        "\\", "/"
                    )  # Ensure forward slashes for blob

                    # Get blob client and upload the file
                    blob_client = blob_service_client.get_blob_client(
                        container=AZURE_CONTAINER_NAME, blob=blob_name
                    )
                    with open(file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)
                    print(f"Uploaded {file_name} to Azure Blob: {blob_name}")
    except Exception as e:
        print(f"Error uploading files to blob: {e}")


# Get all registered models
registered_models = client.search_registered_models()

# Temporary directory for downloading artifacts
with tempfile.TemporaryDirectory() as tmp_dir:
    for model in registered_models:
        model_name = model.name
        print(f"Checking model: {model_name}")

        # Iterate through all versions of the model
        for version in model.latest_versions:
            model_version = version.version
            model_tags = version.tags

            # Check if the model has the required tag
            if model_tags.get("validation_status") == "approved":
                print(f"Model {model_name} version {model_version} is approved.")

                # Download the model artifact to the temporary directory
                local_model_path = os.path.join(
                    tmp_dir, f"{model_name}_v{model_version}"
                )

                # Use the source attribute to download the correct model artifact
                model_source = version.source
                mlflow.artifacts.download_artifacts(
                    model_source, dst_path=local_model_path
                )

                # Define blob prefix (folder structure in Blob storage)
                blob_prefix = f"{model_name}_v{model_version}"

                # Upload the entire directory to Blob Storage
                upload_directory_to_blob(local_model_path, blob_prefix)
            else:
                print(
                    f"Model {model_name} version {model_version} does not have the 'staging_status:approved' tag."
                )
