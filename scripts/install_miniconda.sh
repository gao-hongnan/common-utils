#!/bin/bash

# Check if Homebrew is installed, and install it if necessary
if ! command -v brew &> /dev/null; then
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "Homebrew found."
fi

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Install Miniconda
echo "Installing Miniconda..."
brew install miniconda

# Initialize Miniconda
echo "Initializing Miniconda..."
conda init "$(basename "${SHELL}")"

# Determine the shell and add conda to the appropriate configuration file
if [ "$(basename "${SHELL}")" = "bash" ]; then
    echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
    source ~/.bashrc
elif [ "$(basename "${SHELL}")" = "zsh" ]; then
    echo 'eval "$(conda shell.zsh hook)"' >> ~/.zshrc
    source ~/.zshrc
else
    echo "Unsupported shell. Please add conda initialization to your shell configuration file manually."
fi

# Verify Miniconda installation
echo "Verifying Miniconda installation..."
conda info
