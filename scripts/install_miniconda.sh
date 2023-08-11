#!/bin/bash

install_homebrew() {
    echo "Homebrew not found. Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
}

update_homebrew() {
    echo "Updating Homebrew..."
    brew update
}

install_miniconda() {
    echo "Installing Miniconda..."
    brew install miniconda
}

initialize_miniconda() {
    echo "Initializing Miniconda..."
    conda init "$(basename "${SHELL}")"
}

configure_shell() {
    local shell_name="$(basename "${SHELL}")"
    if [ "${shell_name}" = "bash" ]; then
        echo 'eval "$(conda shell.bash hook)"' >> ~/.bashrc
        source ~/.bashrc
    elif [ "${shell_name}" = "zsh" ]; then
        echo 'eval "$(conda shell.zsh hook)"' >> ~/.zshrc
        source ~/.zshrc
    else
        echo "Unsupported shell. Please add conda initialization to your shell configuration file manually."
    fi
}

verify_miniconda_installation() {
    echo "Verifying Miniconda installation..."
    conda info
}

# Main script execution
if ! command -v brew &> /dev/null; then
    install_homebrew
else
    echo "Homebrew found."
fi

update_homebrew
install_miniconda
initialize_miniconda
configure_shell
verify_miniconda_installation
