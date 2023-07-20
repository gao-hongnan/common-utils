# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY ./requirements.txt /app/requirements.txt

# Run pip to install dependencies as root
RUN python3 -m pip install -r requirements.txt --no-cache-dir

# Copy the rest of the application
COPY . /app

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Create the data directory in the container
RUN mkdir -p /app/data

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py"]