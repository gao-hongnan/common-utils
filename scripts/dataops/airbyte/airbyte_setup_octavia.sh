#!/bin/bash
# curl -o airbyte_setup_octavia.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/airbyte/airbyte_setup_octavia.sh

set -e # Exit immediately if a command exits with a non-zero status
set -o pipefail # Fail a pipe if any sub-command fails

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."

usage() {
    logger "INFO" "Usage: $0"
    logger "INFO" "This script automates the deployment of Airbyte Open Source."
    logger "INFO" ""
    logger "INFO" "The script will perform the following steps:"
    logger "INFO" "  1. Install Octavia CLI."
    logger "INFO" "  2. Prompt user for Airbyte credentials."
    logger "INFO" "  3. Initialize Octavia in the specified directory."
    logger "INFO" ""
    logger "INFO" "Please make sure Docker is installed and running before executing this script."
    logger "INFO" ""
    exit 1
}

# Check if the user asked for help
if check_for_help "$@"; then
    usage
fi

# Prompt user for Airbyte credentials
prompt_for_credentials() {
    logger "INFO" "Please enter your Airbyte username:"
    read -r username
    logger "INFO" "Please enter your Airbyte password:"
    read -r -s password
    airbyte_username="$username"
    airbyte_password="$password"
}

# Install Octavia CLI and add username and password
install_octavia_cli() {
    logger "INFO" "Installing Octavia CLI..."
    curl -s -o- https://raw.githubusercontent.com/airbytehq/airbyte/master/octavia-cli/install.sh | bash
    printf "AIRBYTE_USERNAME=%s\n" "$airbyte_username" >> ~/.octavia
    printf "AIRBYTE_PASSWORD=%s\n" "$airbyte_password" >> ~/.octavia
    logger "INFO" "Octavia CLI installed and credentials added to ~/.octavia"
}

# Prompt user for project directory
prompt_for_directory() {
    logger "INFO" "Please enter the directory for the Octavia project:"
    read -r project_dir
    octavia_project_dir="$project_dir"
}

# Initialize Octavia
initialize_octavia() {
    mkdir -p "$octavia_project_dir" && cd "$octavia_project_dir" || exit
    logger "INFO" "Initializing Octavia in ${octavia_project_dir}..."

    # Automatically get the Octavia version
    OCTAVIA_VERSION=$(docker images | grep 'airbyte/octavia-cli' | awk '{print $2}' | head -1)
    if [ -z "$OCTAVIA_VERSION" ]; then
        logger "ERROR" "No Octavia version found."
        exit 1
    fi

    # Define OCTAVIA_ENV_FILE dynamically
    OCTAVIA_ENV_FILE="$HOME/.octavia"
    docker run -i --rm \
        -v "$(pwd)":/home/octavia-project \
        --network host \
        --env-file "${OCTAVIA_ENV_FILE}" \
        --user "$(id -u)":"$(id -g)" \
        "airbyte/octavia-cli:${OCTAVIA_VERSION}" init

    logger "INFO" "Octavia has been initialized in ${octavia_project_dir}"
}

# Main function to call the functions
main() {
    prompt_for_credentials
    install_octavia_cli
    prompt_for_directory
    initialize_octavia
}

# Run the main function
main