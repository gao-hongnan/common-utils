#!/bin/bash

# Define colors
yellow='\033[1;33m'
green='\033[0;32m'
# Reset text color to default
reset='\033[0m'

# Function to initialize Octavia
initialize_octavia() {
    local project_dir=$1
    mkdir -p "$project_dir" && cd "$project_dir"
    echo -e "${yellow}Initializing Octavia in ${project_dir}...${reset}"
    octavia init
    echo -e "${green}Octavia has been initialized in ${project_dir}${reset}"
}

# Prompt user for project directory
echo -e "${yellow}Please enter the directory for the Octavia project:${reset}"
read project_dir

# Call the function with user input
initialize_octavia "$project_dir"
