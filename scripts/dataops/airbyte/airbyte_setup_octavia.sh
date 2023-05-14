#!/bin/bash
# curl -o scripts/airbyte_setup_octavia.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/airbyte/airbyte_setup_octavia.sh

# Define colors
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
# Reset text color to default
RESET='\033[0m'

usage() {
    echo "Usage: $0"
    echo "This script automates the deployment of Airbyte Open Source."
    echo
    echo "The script will perform the following steps:"
    echo "  1. Install Octavia CLI."
    echo "  2. Prompt user for Airbyte credentials."
    echo "  3. Initialize Octavia in the specified directory."
    echo
    echo "Please make sure Docker is installed and running before executing this script."
    echo
    exit 1;
}

# Check if the user asked for help
if [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
fi

# Function to prompt user for Airbyte credentials
prompt_for_credentials() {
    echo -e "${YELLOW}Please enter your Airbyte username:${RESET}"
    read username
    echo -e "${YELLOW}Please enter your Airbyte password:${RESET}"
    read -s password
    USERNAME=$username
    PASSWORD=$password
}

# Function to install Octavia CLI and add username and password
install_octavia_cli() {
    echo -e "${YELLOW}Installing Octavia CLI...${RESET}"
    curl -s -o- https://raw.githubusercontent.com/airbytehq/airbyte/master/octavia-cli/install.sh | bash
    echo "AIRBYTE_USERNAME=${USERNAME}" >> ~/.octavia
    echo "AIRBYTE_PASSWORD=${PASSWORD}" >> ~/.octavia
    echo -e "${GREEN}Octavia CLI installed and credentials added to ~/.octavia${RESET}"
}

# Function to prompt user for project directory
prompt_for_directory() {
    echo -e "${YELLOW}Please enter the directory for the Octavia project:${RESET}"
    read project_dir
    PROJECT_DIR=$project_dir
}

# Function to initialize Octavia
initialize_octavia() {
    mkdir -p "$PROJECT_DIR" && cd "$PROJECT_DIR"
    echo -e "${YELLOW}Initializing Octavia in ${PROJECT_DIR}...${RESET}"
    octavia init
    echo -e "${GREEN}Octavia has been initialized in ${PROJECT_DIR}${RESET}"
}

# Call the functions
prompt_for_credentials
install_octavia_cli
prompt_for_directory
initialize_octavia