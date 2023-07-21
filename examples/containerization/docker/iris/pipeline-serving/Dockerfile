# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /pipeline-serving
WORKDIR /pipeline-serving

# Copy only requirements.txt first to leverage Docker cache
COPY ./requirements.txt /pipeline-serving/requirements.txt

# Run pip to install dependencies as root
RUN python3 -m pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application
COPY . /pipeline-serving

# Declare Volume
VOLUME /pipeline-serving/stores

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]
