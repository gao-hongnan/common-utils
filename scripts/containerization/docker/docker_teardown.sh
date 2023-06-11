#!/bin/bash
# curl -o docker_teardown.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/containerization/docker/docker_teardown.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."

# Function to stop all Docker containers related to a given name
stop_containers() {
    local name=$1
    logger "INFO" "Stopping all Docker containers related to ${name}..."
    docker ps -a -f "name=${name}" --format "{{.ID}}" | xargs docker stop
    logger "INFO" "All Docker containers related to ${name} have been stopped."
}

# Function to remove all Docker containers related to a given name
remove_containers() {
    local name=$1
    logger "INFO" "Removing all Docker containers related to ${name}..."
    docker ps -a -f "name=${name}" --format "{{.ID}}" | xargs docker rm
    logger "INFO" "All Docker containers related to ${name} have been removed."
}

# Function to remove all Docker images related to a given name
remove_images() {
    local name=$1
    logger "INFO" "Removing all Docker images related to ${name}..."
    docker images -a -f "reference=${name}/*" --format "{{.ID}}" | xargs docker rmi -f
    logger "INFO" "All Docker images related to ${name} have been removed."
}

# Function to remove all Docker volumes related to a given name
remove_volumes() {
    local name=$1
    logger "INFO" "Removing all Docker volumes related to ${name}..."
    docker volume ls -f "name=${name}" --format "{{.Name}}" | xargs docker volume rm
    logger "INFO" "All Docker volumes related to ${name} have been removed."
}

# Function to remove all Docker networks related to a given name
remove_networks() {
    local name=$1
    logger "INFO" "Removing all Docker networks related to ${name}..."
    docker network ls -f "name=${name}" --format "{{.Name}}" | xargs docker network rm
    logger "INFO" "All Docker networks related to ${name} have been removed."
}

# Main function to call the functions
main() {
    local name=$1
    stop_containers "$name"
    remove_containers "$name"
    remove_images "$name"
    remove_volumes "$name"
    remove_networks "$name"
}

# Check if a name was provided as an argument
if [[ -z $1 ]]; then
    logger "WARN" "No name argument provided. Exiting."
    exit 1
fi

# Run the main function with the first command line argument as the name
main "$1"
