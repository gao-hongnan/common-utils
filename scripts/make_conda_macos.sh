#!/bin/bash

# Initialize variables for environment name, Python version, and requirements files
env_name=""
python_version=""
requirements_file="requirements.txt"
requirements_dev_file="requirements_dev.txt"
install_dev=false

# Print usage instructions
print_usage() {
  echo "Usage: $0 -n environment_name -p python_version [-r requirements_file] [-d]"
}

# Parse command-line arguments
while getopts ":n:p:r:d" opt; do
  case $opt in
    n)
      env_name="$OPTARG"
      ;;
    p)
      python_version="$OPTARG"
      ;;
    r)
      requirements_file="$OPTARG"
      ;;
    d)
      install_dev=true
      ;;
    \?)
      print_usage
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check if environment name and Python version are provided
if [[ -z "$env_name" || -z "$python_version" ]]; then
  echo "Error: Both environment name and Python version must be provided."
  print_usage
  exit 1
fi

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Error: Conda is not installed or not in your PATH. Please install Conda first."
    exit 1
fi

# Check if the requirements file exists
if [ ! -f "$requirements_file" ]
then
    echo "Error: Requirements file '$requirements_file' not found."
    exit 1
fi

# Create the new Conda environment
echo "Creating a new Conda environment '$env_name' with Python $python_version..."
conda create -y -n "$env_name" python="$python_version"

# Activate the environment
echo "Activating the Conda environment '$env_name'..."
conda init bash
source ~/.bash_profile
# source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$env_name"

# Install pip in the new environment
echo "Installing pip in the Conda environment '$env_name'..."
conda install -y pip

# Install packages from the requirements file using pip
echo "Installing packages from '$requirements_file' in the Conda environment '$env_name'..."
pip install -r "$requirements_file"

# Install packages from the requirements_dev file using pip if -d option is specified
if $install_dev; then
  if [ ! -f "$requirements_dev_file" ]; then
      echo "Error: Development requirements file '$requirements_dev_file' not found."
      exit 1
  fi

  echo "Installing development packages from '$requirements_dev_file' in the Conda environment '$env_name'..."
  pip install -r "$requirements_dev_file"
fi

echo "Done. The Conda environment '$env_name' with Python $python_version is now set up and activated."