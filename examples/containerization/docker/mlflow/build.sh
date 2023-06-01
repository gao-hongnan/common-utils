#!/bin/bash

# Check if docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Please install Docker to run this script."
    exit
fi

gcloud compute ssh \
    --project=gao-hongnan \
    --zone=us-west2-a \
    imdb

touch .env
# echo into env

# Build the Docker image
echo "Building Docker image..."
docker-compose up -d