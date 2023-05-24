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
    -t streamlit-app:v1 \
    .

# Run the Docker container
echo "Running Docker container..."
docker run \
    --rm \
    -p 8501:8501 \
    --name streamlit-app \
    streamlit-app:v1
