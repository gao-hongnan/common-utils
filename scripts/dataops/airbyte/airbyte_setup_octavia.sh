#!/bin/bash
# curl -o scripts/airbyte_setup_octavia.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/airbyte/airbyte_setup_octavia.sh

# Define colors
yellow='\033[1;33m'
green='\033[0;32m'
# Reset text color to default
reset='\033[0m'

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
    echo -e "${yellow}Please enter your Airbyte username:${reset}"
    read username
    echo -e "${yellow}Please enter your Airbyte password:${reset}"
    read -s password
    echo "$username"
    echo "$password"
}

# Function to install Octavia CLI and add username and password
install_octavia_cli() {
    local username=$1
    local password=$2
    echo -e "${yellow}Installing Octavia CLI...${reset}"
    curl -s -o- https://raw.githubusercontent.com/airbytehq/airbyte/master/octavia-cli/install.sh | bash
    echo "AIRBYTE_USERNAME=${username}" >> ~/.octavia
    echo "AIRBYTE_PASSWORD=${password}" >> ~/.octavia
    echo -e "${green}Octavia CLI installed and credentials added to ~/.octavia${reset}"
}

# Function to prompt user for project directory
prompt_for_directory() {
    echo -e "${yellow}Please enter the directory for the Octavia project:${reset}"
    read project_dir
    echo "$project_dir"
}

# Function to initialize Octavia
initialize_octavia() {
    local project_dir=$1
    mkdir -p "$project_dir" && cd "$project_dir"
    echo -e "${yellow}Initializing Octavia in ${project_dir}...${reset}"
    octavia init
    echo -e "${green}Octavia has been initialized in ${project_dir}${reset}"
}

# Call the functions
credentials=($(prompt_for_credentials))
install_octavia_cli "${credentials[0]}" "${credentials[1]}"
project_dir=$(prompt_for_directory)
initialize_octavia "$project_dir"