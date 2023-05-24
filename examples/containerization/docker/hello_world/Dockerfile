FROM ubuntu:22.04

# Update and install Python3
RUN apt-get update -y && \
    apt-get install -y python3 \
    python3-pip

# Copy requirements.txt to the container
COPY requirements.txt .
COPY hello_world.py .

# Run pip to install dependencies
RUN python3 -m pip install -r requirements.txt --no-cache-dir

CMD ["python3", "hello_world.py"]