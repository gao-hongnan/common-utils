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
    -t iris-training:v1 \
    .

# Run the Docker container
echo "Running Docker container..."
docker run -v \
    --rm \
    iris-stores:/pipeline-training/stores \
    --name iris-training \
    iris-training:v1
