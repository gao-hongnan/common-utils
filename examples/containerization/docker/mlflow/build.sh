#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker to run this script."
    exit
else
    echo "Docker is installed."
fi

# Build the Docker image
echo "Building Docker image..."
docker-compose up -d