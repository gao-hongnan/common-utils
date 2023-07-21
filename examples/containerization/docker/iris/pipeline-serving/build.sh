#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker to run this script."
    exit
fi

docker volume create iris-stores

# Build the Docker image
echo "Building Docker image..."
docker build \
    -t iris-app:v1 \
    .

# Run the Docker container
echo "Running Docker container..."
docker run \
    --rm \
    -p 8501:8501 \
    -v iris-stores:/pipeline-serving/stores \
    --env STORES_DIR=/pipeline-serving/stores \
    --name iris-app \
    iris-app:v1
