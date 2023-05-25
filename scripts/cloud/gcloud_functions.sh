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

# Function to check for --help flag
check_for_help() {
    for arg in "$@"
    do
        if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]
        then
            return 0
        fi
    done
    return 1
}

log_gcloud_global_flags_info() {
    # Combine the command components into a single string
    local command="$*"
    logger "TIP" "You can set gcloud global flags as well. For example, to set the project ID, run the following command:"
    logger "CODE" "$ ${command} --project <PROJECT_ID>"
    empty_line
}

log_command_info() {
    local description="$1"
    local documentation_link="$2"

    logger "INFO" "${description}"
    logger "INFO" "For more detailed information, please refer to the official Google Cloud documentation:"
    logger "LINK" "${documentation_link}"
    empty_line
}

list_all_service_accounts() {
    # We define the full command as an array
    local command=("gcloud" "iam" "service-accounts" "list")

    # Check if the first argument is --help
    if check_for_help "$@"; then
        # We use "${command[@]}" to correctly expand the array as a command and its arguments
        "${command[@]}" --help
        return
    fi

    log_command_info "This function lists all service accounts in the current project." \
                     "https://cloud.google.com/sdk/gcloud/reference/iam/service-accounts/list"

    log_gcloud_global_flags_info "${command[@]}"

    logger "INFO" "Now, let's proceed with listing all service accounts..."
    "${command[@]}"
}

list_all_docker_images() {
    # We define the full command as an array
    local command=("gcloud" "container" "images" "list")

    # Check if the first argument is --help
    if check_for_help "$@"; then
        # We use "${command[@]}" to correctly expand the array as a command and its arguments
        "${command[@]}" --help
        return
    fi

    log_command_info "This function lists all Docker images in the current project." \
                     "https://cloud.google.com/sdk/gcloud/reference/container/images/list"

    log_gcloud_global_flags_info "${command[@]}"

    logger "INFO" "Now, let's proceed with listing all Docker images..."
    "${command[@]}"
}

list_all_vms() {
    # We define the full command as an array
    local command=("gcloud" "compute" "instances" "list")

    # Check if the first argument is --help
    if check_for_help "$@" "${command[@]}"; then
        return
    fi

    log_command_info "This function lists all VMs in the current project." \
                     "https://cloud.google.com/sdk/gcloud/reference/compute/instances/list"

    provide_gcloud_global_flags_info "${command[@]}"

    logger "INFO" "Now, let's proceed with listing all VMs..."
    "${command[@]}"
}


############# Functions for VM instance #############

# Function to create a Google Cloud VM instance
create_vm() {
    # Define default parameters
    local instance_name="my-instance"
    local machine_type="e2-medium"
    local zone="us-west2-a"
    local boot_disk_size="10GB"
    local image="ubuntu-1804-bionic-v20230510"
    local image_project="ubuntu-os-cloud"
    local project_id=""
    local service_account=""
    local scopes="https://www.googleapis.com/auth/cloud-platform"
    local description="A VM instance"

    # Process user-provided parameters
    while (( "$#" )); do
        case "$1" in
            --instance-name)
                instance_name="$2"
                shift 2
                ;;
            --machine-type)
                machine_type="$2"
                shift 2
                ;;
            --zone)
                zone="$2"
                shift 2
                ;;
            --boot-disk-size)
                boot_disk_size="$2"
                shift 2
                ;;
            --image)
                image="$2"
                shift 2
                ;;
            --image-project)
                image_project="$2"
                shift 2
                ;;
            --project)
                project_id="$2"
                shift 2
                ;;
            --service-account)
                service_account="$2"
                shift 2
                ;;
            --scopes)
                scopes="$2"
                shift 2
                ;;
            --description)
                description="$2"
                shift 2
                ;;
            *)
                echo "Error: Invalid argument"
                return 1
                ;;
        esac
    done

    # Construct the command
    local command=("gcloud" "compute" "instances" "create" "$instance_name" \
        "--machine-type=$machine_type" \
        "--zone=$zone" \
        "--boot-disk-size=$boot_disk_size" \
        "--image=$image" \
        "--image-project=$image_project" \
        "--project=$project_id" \
        "--service-account=$service_account" \
        "--scopes=$scopes" \
        "--description=\"$description\"")

    # Execute the command
    "${command[@]}"
}

# Function to SSH into a Google Cloud VM instance
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

