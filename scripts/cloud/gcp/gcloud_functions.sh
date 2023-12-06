#!/bin/bash
# This script details various cli functions for Google Cloud SDK.
# curl -o gcloud_functions.sh https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/cloud/gcp/gcloud_functions.sh

# Google Cloud SDK 431.0.0
# bq 2.0.92
# core 2023.05.12
# gcloud-crc32c 1.0.0
# gsutil 5.23

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."

useful_commands() {
    gcloud services list --enabled
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

################################## Functions for Artifacts Registry ##################################

list_all_gar_repositories() {
    gcloud artifacts repositories list
    echo "gcloud artifacts docker images list <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>"
}

gar_docker_setup() {
    logger "INFO" "Step 1. Enable the Artifacts Registry API"

    service="artifactregistry.googleapis.com"
    if gcloud services list --enabled | grep -q "$service"; then
        logger "INFO" "$service is already enabled. Skipping to Step 2..."
    else
        logger "WARN" "$service is not enabled. Enabling now..."
        gcloud services enable "$service"
    fi

    empty_line

    logger "INFO" "Step 2. Create a Docker repository in Artifact Registry"
    logger "INFO" "Here's a sample command:"
    logger "BLOCK" \
        "$ gcloud artifacts repositories create <REPO_NAME> \\
            --repository-format=docker \\
            --location=<REGION> \\
            --description=\"Docker repository for storing images\""

    empty_line
    logger "INFO" "Run the following command to verify your repository was created:"
    logger "CODE" "$ gcloud artifacts repositories list"
    empty_line

    logger "INFO" "Step 3. Configure Docker to use Artifact Registry"
    logger "CODE" "$ gcloud auth configure-docker <LOCATION>-docker.pkg.dev"
    empty_line

    logger "INFO" "Step 4. Tag a Docker image"
    logger "BLOCK" \
        "$ docker tag <IMAGE_NAME> \\
            <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/<IMAGE_NAME>:<TAG>"
    empty_line

    logger "INFO" "Step 5. Push a Docker image"
    logger "CODE" "$ docker push <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/<IMAGE_NAME>:<TAG>"
    empty_line

    logger "TIP" "To check the images in your repository, run the following command:"
    logger "CODE" "$ gcloud artifacts docker images list <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>"
    empty_line

    logger "INFO" "Step 6. Pull a Docker image"
    logger "CODE" "$ docker pull <LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPO_NAME>/<IMAGE_NAME>:<TAG>"
    empty_line

    logger "INFO" "Step 7. Tear down - Delete a Docker repository"
    logger "CODE" "$ gcloud artifacts repositories delete <REPO_NAME> --location=<REGION>"
    empty_line

    logger "TIP" "For more detailed information, please refer to the official Google Cloud documentation:"
    logger "LINK" "https://cloud.google.com/artifact-registry/docs/docker/store-docker-container-images"
}

################################## Google Compute Engine (GCE)  ##################################

################################## Virtual Machine (VM) instances  ##################################
# https://cloud.google.com/compute/docs/instances/create-start-instance#publicimage

vm_other_commands() {
    gcloud compute instances delete --help
    gcloud compute instances start --help
    gcloud compute instances stop --help
    gcloud compute instances describe --help
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

    log_gcloud_global_flags_info "${command[@]}"

    logger "INFO" "Now, let's proceed with listing all VMs..."
    "${command[@]}"
}

vm_create_usage() {
    logger "INFO" "USAGE"
    empty_line
    echo "vm_create \\
    --instance-name INSTANCE_NAME \\
    --machine-type MACHINE_TYPE \\
    --zone ZONE \\
    --boot-disk-size BOOT_DISK_SIZE \\
    --image IMAGE \\
    --image-project IMAGE_PROJECT \\
    --project PROJECT_ID \\
    --service-account SERVICE_ACCOUNT \\
    --scopes SCOPES \\
    --description DESCRIPTION"

    empty_line

    logger "INFO" "If you want to add additional flags, you can do so by passing in the --additional-flags flag."
    logger "INFO" "For example, to add the --tags flag, you can run the following command:"
    logger "CODE" "$ vm_create --additional-flags \"--tags=tag1,tag2\""

    empty_line

    logger "INFO" "Options:"
    logger "INFO" "  --instance-name INSTANCE_NAME       Name of the instance (default: 'my-instance')"
    logger "INFO" "  --machine-type MACHINE_TYPE         Type of machine to use (default: 'e2-medium')"
    logger "INFO" "  --zone ZONE                         Zone to create the instance in (default: 'us-west2-a')"
    logger "INFO" "  --boot-disk-size BOOT_DISK_SIZE     Size of boot disk (default: '10GB')"
    logger "INFO" "  --image IMAGE                       Image to use for the instance (default: 'ubuntu-1804-bionic-v20230510')"
    logger "INFO" "  --image-project IMAGE_PROJECT       Image project to use (default: 'ubuntu-os-cloud')"
    logger "INFO" "  --project PROJECT_ID                Project ID to use"
    logger "INFO" "  --service-account SERVICE_ACCOUNT   Service account to assign to the instance"
    logger "INFO" "  --scopes SCOPES                     Scopes to assign to the instance (default: 'https://www.googleapis.com/auth/cloud-platform')"
    logger "INFO" "  --description DESCRIPTION           Description for the instance (default: 'A VM instance')"
    logger "INFO" "  --help                              Display this help and exit"
}

vm_check_required_args() {
    if [[ -z "${instance_name}" || -z "${machine_type}" || -z "${zone}" ]]; then
        logger "ERROR" "Missing required argument(s)"
        vm_create_usage
        echo "Sleeping for 60 seconds..."
        sleep 60
        exit 1
    fi
}

# Function to create a Google Cloud VM instance
vm_create() {
    # Define default parameters
    local instance_name="" # my-instance
    local machine_type="e2-medium"
    local zone="us-west2-a"
    local boot_disk_size="10GB"
    local image="ubuntu-1804-bionic-v20230510"
    local image_project="ubuntu-os-cloud"
    local project_id=""
    local service_account=""
    local scopes="https://www.googleapis.com/auth/cloud-platform"
    local description="A VM instance"
    local additional_flags=""

    # Check if the first argument is --help
    if check_for_help "$@"; then
        logger "INFO" "Help on the way..."
        gcloud compute instances create --help
        return
    fi

    # Process user-provided parameters
    while (( $# )); do
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
            --additional-flags)
                additional_flags="$2"
                shift 2
                ;;
            *)
                logger "ERROR" "Unknown argument: $1"
                vm_create_usage
                return 1
                ;;
        esac
    done

    # Check for required arguments
    vm_check_required_args

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

    # Append the additional flags if they are not empty
    if [[ -n "$additional_flags" ]]; then
        # Here we are using the additional_flags as string
        command+=($additional_flags)
    fi

    # Execute the command
    "${command[@]}"
}

# Function to SSH into a Google Cloud VM instance
vm_ssh_to_gcp_instance() {
    logger "INFO" "This function SSHes into a Google Cloud VM instance."
    logger "INFO" "Here's a sample command:"
    empty_line

    logger "BLOCK" \
        "$ gcloud compute ssh \\
            --project=<PROJECT_ID> \\
            --zone=<ZONE> \\
            <VM_NAME>"

    empty_line
    logger "INFO" "For more detailed information, please refer to the official Google Cloud documentation:"
    logger "LINK" "https://cloud.google.com/compute/docs/connect/standard-ssh#connect_to_vms"
}

