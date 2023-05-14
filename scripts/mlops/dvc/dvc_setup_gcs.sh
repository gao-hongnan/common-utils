#!/bin/bash
# curl -o dvc_setup_gcs.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/mlops/dvc/dvc_setup_gcs.sh

# TODO: to expand this script to support other cloud storage providers (e.g., AWS S3, Azure Blob Storage)

set -e

usage() {
    if [[ -z "${REMOTE_NAME}" || -z "${REMOTE_URL}" || -z "${BUCKET}" || -z "${PATH}" ]]; then
        echo "Error: Missing required argument(s)"
        echo "Usage: $0 -d remote_name -r remote_url -b bucket -p path"
        echo
        echo "Options:"
        echo "  -d REMOTE_NAME   The name of the remote in DVC (required)"
        echo "  -r REMOTE_URL    The remote storage URL (e.g., s3://, gs://, azure://) (required)"
        echo "  -b BUCKET        The bucket name for the remote storage (required)"
        echo "  -p PATH          The path within the bucket for the remote storage (required)"
        echo "  -h               Display this help and exit"
        exit 1
    fi
}

parse_args() {
    while getopts "d:r:b:p:n:h" opt; do
        case ${opt} in
            d) REMOTE_NAME="$OPTARG" ;;
            r) REMOTE_URL="$OPTARG" ;;
            b) BUCKET="$OPTARG" ;;
            p) PATH="$OPTARG" ;;
            h) usage ;;
            *) usage ;;
        esac
    done
}

install_dvc() {
    pip install dvc dvc-gc
    dvc init
}

configure_remote_storage() {
    if [[ -n "${REMOTE_URL}" && -n "${BUCKET}" && -n "${PATH}" ]]; then
        dvc remote add -d "${REMOTE_NAME}" "${REMOTE_URL}${BUCKET}/${PATH}"
    fi
    dvc remote modify --local "${REMOTE_NAME}" credentialpath <path-to-credentials-file>
}

# Main script execution
parse_args "$@"
usage
install_dvc
configure_remote_storage

echo "DVC setup complete"
