#!/bin/bash
# curl -o airbyte_setup.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/airbyte/airbyte_setup.sh


set -e # Exit immediately if a command exits with a non-zero status
set -o pipefail # Fail a pipe if any sub-command fails

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

# Default values
USERNAME="airbyte"
PASSWORD="password"

# Usage
usage() {
    echo "Usage: $0 [-u username] [-p password]"
    echo "This script automates the deployment of Airbyte Open Source."
    echo
    echo "Options:"
    echo "  -u    Specify the username for basic authentication in the Airbyte application. Default is 'airbyte'."
    echo "  -p    Specify the password for basic authentication in the Airbyte application. Default is 'password'."
    echo "  -h    Display this help and exit."
    echo
    echo "The script will perform the following steps:"
    echo "  1. Clone the Airbyte GitHub repository."
    echo "  2. Update the .env file in the cloned repository with the provided (or default) username and password."
    echo "  3. Run the Airbyte platform."
    echo
    echo "Please make sure Docker is installed and running before executing this script."
    echo
    exit 1;
}

# Usage
usage() {
    logger "INFO" "Usage: $0 [-u username] [-p password]"
    logger "INFO" "This script automates the deployment of Airbyte Open Source."
    empty_line

    logger "INFO" "Options:"
    logger "INFO" "  -u    Specify the username for basic authentication in the Airbyte application. Default is 'airbyte'."
    logger "INFO" "  -p    Specify the password for basic authentication in the Airbyte application. Default is 'password'."
    logger "INFO" "  -h    Display this help and exit."
    empty_line

    logger "INFO" "The script will perform the following steps:"
    logger "INFO" "  1. Clone the Airbyte GitHub repository."
    logger "INFO" "  2. Update the .env file in the cloned repository with the provided (or default) username and password."
    logger "INFO" "  3. Run the Airbyte platform."
    logger "INFO" ""
    logger "INFO" "Please make sure Docker is installed and running before executing this script."
    logger "INFO" ""
    exit 1
}

# Check if the user asked for help
if check_for_help "$@"; then
    usage
fi

# Parse command line arguments
while getopts "u:p:" flag; do
    case "${flag}" in
        u)
            USERNAME=${OPTARG}
            ;;
        p)
            PASSWORD=${OPTARG}
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done
shift $((OPTIND-1))

# If in VM, we do minimal build
is_vm() {
    logger "WARN" "If you are running this script in a VM, we will do a minimal build."
    read -r -p "Are you running this script in a VM? [y/N] " response
    case "$response" in
        [yY][eE][sS]|[yY])
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

# Function to clone Airbyte repository
clone_airbyte() {
    logger "INFO" "Cloning Airbyte repository..."
    if [ ! -d "airbyte" ]; then
        git clone https://github.com/airbytehq/airbyte.git
    fi
    cd airbyte
    logger "INFO" "Airbyte repository cloned."
}

# Function to run Airbyte
run_airbyte() {
    logger "INFO" "Running Airbyte platform..."
    ./run-ab-platform.sh
    logger "INFO" "Airbyte platform is running."
}

# Function to update .env file with provided username and password
update_env() {
    logger "INFO" "Updating .env file with provided username and password..."
    sed -i -e "s/BASIC_AUTH_USERNAME=airbyte/BASIC_AUTH_USERNAME=${USERNAME}/" .env
    sed -i -e "s/BASIC_AUTH_PASSWORD=password/BASIC_AUTH_PASSWORD=${PASSWORD}/" .env
    logger "INFO" ".env file updated."
}

# Main function to call the functions
main() {
    if is_vm; then
        logger "INFO" "Running in VM, performing minimal build..."
        mkdir airbyte && cd airbyte
        curl -sOOO https://raw.githubusercontent.com/airbytehq/airbyte-platform/main/{.env,flags.yml,docker-compose.yaml}
        docker compose up -d
        logger "INFO" "Minimal build completed."
    else
        logger "INFO" "Not running in VM, performing full build..."
        clone_airbyte
        run_airbyte
        update_env
        logger "INFO" "Full build completed."
    fi
}

# Run the main function
main

