#!/bin/bash
# curl -o scripts/clean.sh https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/clean.sh

# This script cleans the project directory by removing unwanted files and directories.
# These include: *.DS_Store, __pycache__, *.pyc, *.pyo, .pytest_cache, .ipynb_checkpoints, .trash, and .coverage files.

function clean_project {
    # Set text color to yellow
    yellow='\033[1;33m'
    # Reset text color to default
    reset='\033[0m'

    # Display the "WARNING" text in yellow followed by the prompt
    printf "${yellow}WARNING${reset}: Are you sure you want to clean the project directory? [y/N] "
    read confirm

    confirm=$(echo "$confirm" | tr '[:upper:]' '[:lower:]')  # tolower
    if [[ $confirm =~ ^(yes|y)$ ]]; then
        echo "Cleaning project..."

        echo "Cleaning *.DS_Store, pycache, .pyc, .pyo, .pytest_cache, .ipynb_checkpoints, .trash files and .coverage file..."
        find . \( \
            -name "*.DS_Store" -o \
            -name "__pycache__" -o \
            -name "*.pyc" -o \
            -name "*.pyo" -o \
            -name ".pytest_cache" -o \
            -name ".ipynb_checkpoints" -o \
            -name ".trash" \
        \) -exec rm -rf {} \;

        echo "Removing .coverage file..."
        rm -f .coverage

        echo "Cleaning complete."
    else
        echo "Cleaning cancelled."
    fi
}

clean_project
