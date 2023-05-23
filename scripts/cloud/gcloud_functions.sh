#!/bin/bash
# This script details various cli functions for Google Cloud SDK.

# Google Cloud SDK 431.0.0
# bq 2.0.92
# core 2023.05.12
# gcloud-crc32c 1.0.0
# gsutil 5.23

# Define the color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
WHITE='\033[1;37m'
RESET='\033[0m' # No Color

# Create a VM instance from a public image

# Function to create a Google Cloud VM instance
usage() {
    echo "Usage: source <this-script>.sh"
    echo "Call functions inside this script by running: function_name"
}





empty_line() {
    printf "\n"
}

# Define a logger function
logger() {
    local level=$1
    shift
    local message=$@

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    PADDED_LOG_LEVEL=$(printf '%-5s' "$level")

    case $level in
        "INFO")
            color=$GREEN
            ;;
        "WARN")
            color=$YELLOW
            ;;
        "ERROR")
            color=$RED
            ;;
        "CODE")
            color=$CYAN
            ;;
        "CODE_MULTI")
            color=$CYAN
            printf "${color}$TIMESTAMP [$PADDED_LOG_LEVEL]:\n    ${message}${RESET}\n"
            return
            ;;
        "TIP")
            color=$PURPLE
            ;;
        "LINK")
            color=$BLUE
            ;;
        *)
            color=$RESET
            ;;
    esac


    printf "${color}$TIMESTAMP [$PADDED_LOG_LEVEL]: ${message}${RESET}\n" "$level"
}

list_all_service_accounts() {
    logger "INFO" "This function lists all service accounts in the current project.\n"
    logger "INFO" "For more detailed information, please refer to the official Google Cloud documentation:"
    logger "LINK" "https://cloud.google.com/sdk/gcloud/reference/iam/service-accounts/list"
    empty_line

    logger "TIP" "You can set gcloud global flags as well. For example, to set the project ID, run the following command:"
    logger "CODE" "$ gcloud iam service-accounts list --project <PROJECT_ID>"
    empty_line

    logger "INFO" "Now, let's proceed with listing all service accounts..."
    gcloud iam service-accounts list
}

# # Function to SSH into a Google Cloud VM instance

ssh_to_gcp_instance() {
    logger "INFO" "This function SSHes into a Google Cloud VM instance."
    logger "INFO" "Here's a sample command:"
    empty_line

    logger "CODE_MULTI" \
        "$ gcloud compute ssh \\
            --project=<PROJECT_ID> \\
            --zone=<ZONE> \\
            <VM_NAME>"

    empty_line
    logger "INFO" "For more detailed information, please refer to the official Google Cloud documentation:"
    logger "LINK" "https://cloud.google.com/compute/docs/connect/standard-ssh#connect_to_vms"
}

