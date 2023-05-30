#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker to run this script."
    exit
fi

# Build the Docker image
echo "Building Docker image..."
docker build \
    -t gaohn-e2e-imdb:v1 \
    -f docker/serve.Dockerfile \
    .

# Run the Docker container
echo "Running Docker container..."

docker run \
    --rm \
    -p 8000:8000 \
    -e GOOGLE_SERVICE_ACCOUNT_BASE64="$(cat gcp-storage-service-account-base64.txt)" \
    --name gaohn-e2e-imdb \
    gaohn-e2e-imdb:v1

