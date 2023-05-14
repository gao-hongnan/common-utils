#!/bin/bash
# curl -o dvc_setup_gcs.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/mlops/dvc/dvc_setup_gcs.sh

# TODO: to expand this script to support other cloud storage providers (e.g., AWS S3, Azure Blob Storage)

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m' # No Color

usage() {
    echo "Usage: $0 -d remote_name -r remote_url -b bucket_name -p bucket_path -c credentials"
    echo
    echo "Options:"
    echo "  -d REMOTE_NAME   The name of the remote in DVC (required)"
    echo "  -r REMOTE_URL    The remote storage URL (e.g., s3://, gs://, azure://) (required)"
    echo "  -b BUCKET_NAME   The bucket name for the remote storage (required)"
    echo "  -p BUCKET_PATH   The path within the bucket for the remote storage (required)"
    echo "  -c CREDENTIALS   The path to the credentials file for the remote storage (required)"
    echo "  -h               Display this help and exit"
}

check_required_args() {
    if [[ -z "${REMOTE_NAME}" || -z "${REMOTE_URL}" || -z "${BUCKET_NAME}" || -z "${BUCKET_PATH}" || -z "${CREDENTIALS}" ]]; then
        echo "Error: Missing required argument(s)"
        usage
        exit 1
    fi
}

parse_args() {
    while getopts "d:r:b:p:c:h" opt; do
        case ${opt} in
            d) REMOTE_NAME="$OPTARG" ;;
            r) REMOTE_URL="$OPTARG" ;;
            b) BUCKET_NAME="$OPTARG" ;;
            p) BUCKET_PATH="$OPTARG" ;;  # Change PATH to BUCKET_PATH
            c) CREDENTIALS="$OPTARG" ;;
            h) usage ;;
            *) usage ;;
        esac
    done
}

install_dvc() {
    python -m pip install dvc dvc-gs
}

init_dvc() {
    if [ ! -d .dvc ]; then
        dvc init
        echo -e "${GREEN}DVC is initialized. You should commit the DVC files to Git.${RESET}"
        echo -e "${GREEN}You can do this by running the following commands:${RESET}"
        echo -e "${YELLOW}  git add .dvc .dvcignore${RESET}"
        echo -e "${YELLOW}  git commit -m \"Initialize DVC\"${RESET}"
    else
        echo -e "${RED}DVC is already initialized. Skipping...${RESET}"
    fi
}

configure_remote_storage() {
    if [[ -n "${REMOTE_URL}" && -n "${BUCKET_NAME}" && -n "${BUCKET_PATH}" ]]; then
        echo -e "${YELLOW}Configuring DVC remote storage...${RESET}"
        dvc remote add -d "${REMOTE_NAME}" "${REMOTE_URL}${BUCKET_NAME}/${BUCKET_PATH}"
    fi
    dvc remote modify --local "${REMOTE_NAME}" credentialpath "${CREDENTIALS}"
}

# Main script execution
parse_args "$@"
check_required_args
install_dvc
init_dvc
configure_remote_storage

echo -e "${GREEN}DVC remote storage is configured.${RESET}"
