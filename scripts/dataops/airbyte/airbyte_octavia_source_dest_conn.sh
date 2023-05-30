#!/bin/bash

# This script provides functions to use Octavia CLI for managing sources,
# destinations and connections.
# Functions must be called with necessary arguments.

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

# Function to list all sources
list_sources() {
    logger "INFO" "Listing all sources..."
    octavia list connectors sources
}

# Function to list all destinations
list_destinations() {
    logger "INFO" "Listing all destinations..."
    octavia list connectors destinations
}

# Function to get the definition id of a source
get_source_definition_id() {
    local source_name="$1"
    logger "INFO" "Getting definition id for source: $source_name"
    octavia list connectors sources | grep "$source_name" | awk '{print $4}'
}

# Function to generate source configuration
generate_source() {
    local definition_id="$1"
    local source_name="$2"
    logger "INFO" "Generating source configuration for: $source_name with definition_id: $definition_id"
    logger "INFO" "The source_name argument can be any custom name you want to give to the source."
    octavia generate source "$definition_id" "$source_name"
}

# Function to apply source configuration
apply_source() {
    local source_config_path="$1"
    logger "INFO" "Applying source configuration from: $source_config_path"
    octavia apply --file "$source_config_path"
}

# Function to get the definition id of a destination
get_destination_definition_id() {
    local destination_name="$1"
    logger "INFO" "Getting definition id for destination: $destination_name"
    octavia list connectors destinations | grep "$destination_name" | awk '{print $4}'
}

# Function to generate destination configuration
generate_destination() {
    local definition_id="$1"
    local destination_name="$2"
    logger "INFO" "Generating destination configuration for: $destination_name with definition_id: $definition_id"
    logger "INFO" "The destination_name argument can be any custom name you want to give to the destination."
    octavia generate destination "$definition_id" "$destination_name"
}

# Function to apply destination configuration
apply_destination() {
    local destination_config_path="$1"
    logger "INFO" "Applying destination configuration from: $destination_config_path"
    octavia apply --file "$destination_config_path"
}

# Function to generate connection between source and destination
generate_connection() {
    local source_config_path="$1"
    local destination_config_path="$2"
    local connection_name="$3"
    logger "INFO" "Generating connection: $connection_name"
    octavia generate connection --source "$source_config_path" --destination "$destination_config_path" "$connection_name"
}

# Function to apply connection configuration
apply_connection() {
    local connection_config_path="$1"
    logger "INFO" "Applying connection configuration from: $connection_config_path"
    octavia apply --file "$connection_config_path"
}

# Uncomment and call functions as needed with the required arguments
#list_sources
#list_destinations
#get_source_definition_id "<source_name>"
#generate_source "<definition_id>" "<source_name>"
#apply_source "<path-to-source-configuration>.yaml"
#get_destination_definition_id "<destination_name>"
#generate_destination "<definition_id>" "<destination_name>"
#apply_destination "<path-to-destination-configuration>.yaml"
#generate_connection "<path-to-source-configuration>.yaml" "<path-to-destination-configuration>.yaml" "<connection_name>"
#apply_connection "<path-to-connection-configuration>.yaml"
