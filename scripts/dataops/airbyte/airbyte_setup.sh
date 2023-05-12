#!/bin/bash
# curl -o scripts/airbyte_setup.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/airbyte/airbyte_setup.sh


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

# Function to clone Airbyte repository
clone_airbyte() {
    git clone https://github.com/airbytehq/airbyte.git
    cd airbyte
}

# Function to update .env file with provided username and password
update_env() {
    sed -i -e "s/BASIC_AUTH_USERNAME=airbyte/BASIC_AUTH_USERNAME=${USERNAME}/" .env
    sed -i -e "s/BASIC_AUTH_PASSWORD=password/BASIC_AUTH_PASSWORD=${PASSWORD}/" .env
}

# Function to run Airbyte
run_airbyte() {
    ./run-ab-platform.sh
}

# Main script execution
clone_airbyte
update_env
run_airbyte


