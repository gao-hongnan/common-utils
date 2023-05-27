#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker to run this script."
    exit
fi

#  \
# -e MLFLOW_TRACKING_URI=file:///gaohn/outputs/mlruns

# Build the Docker image
echo "Building Docker image..."
export GOOGLE_APPLICATION_CREDENTIALS=gcp-storage-service-account.json
docker build \
    -t gaohn-e2e-imdb:v1 \
    -f serve.Dockerfile \
    .

# Run the Docker container
echo "Running Docker container..."
docker run \
    --rm \
    -p 8000:8000 \
    --name gaohn-e2e-imdb \
    gaohn-e2e-imdb:v1

docker run \
    --rm \
    -p 8000:8000 \
    -e GOOGLE_APPLICATION_CREDENTIALS=gcp-storage-service-account.json \
    -e SERVICE_ACCOUNT_KEY_JSON=gcp-storage-service-account.json \
    -e PROJECT_ID=gao-hongnan \
    --name gaohn-e2e-imdb \
    gaohn-e2e-imdb:v1

docker run \
    --rm \
    -p 8000:8000 \
    -e GOOGLE_SERVICE_ACCOUNT_BASE64="$(cat gcp-storage-service-account-base64.txt)" \
    --name gaohn-e2e-imdb \
    gaohn-e2e-imdb:v1

