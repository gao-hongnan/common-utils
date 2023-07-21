# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /pipeline-training
WORKDIR /pipeline-training

# Copy only requirements.txt first to leverage Docker cache
COPY ./requirements.txt /pipeline-training/requirements.txt

# Run pip to install dependencies as root
RUN python3 -m pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application
COPY . /pipeline-training

# Declare Volume?
VOLUME /pipeline-training/stores

# Run app.py when the container launches
CMD ["python", "train.py"]

