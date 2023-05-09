#!/bin/bash
# https://dvc.org/doc/start

set -e

function usage {
    echo "Usage: $0 [-d data_file] [-r remote_url] [-b bucket] [-p path]"
    echo
    echo "Options:"
    echo "  -d DATA_FILE   The path to the data file to track with DVC"
    echo "  -r REMOTE_URL  The remote storage URL (e.g., s3://, gs://, azure://)"
    echo "  -b BUCKET      The bucket name for the remote storage"
    echo "  -p PATH        The path within the bucket for the remote storage"
    exit 1
}

while getopts "d:r:b:p:h" opt; do
    case ${opt} in
        d) DATA_FILE="$OPTARG" ;;
        r) REMOTE_URL="$OPTARG" ;;
        b) BUCKET="$OPTARG" ;;
        p) PATH="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [[ -z "${DATA_FILE}" ]]; then
    echo "Error: Data file not provided"
    usage
fi

# Install DVC
pip install dvc

# Initialize DVC in the project
dvc init

# Configure remote storage if specified
if [[ -n "${REMOTE_URL}" && -n "${BUCKET}" && -n "${PATH}" ]]; then
    dvc remote add -d myremote "${REMOTE_URL}${BUCKET}/${PATH}"
fi

# Track the data file with DVC
dvc add "${DATA_FILE}"

# Commit the .dvc file to Git
DATA_DVC_FILE="${DATA_FILE}.dvc"
git add "${DATA_DVC_FILE}"
git commit -m "Track data file with DVC"

# Push the data to remote storage if configured
if [[ -n "${REMOTE_URL}" && -n "${BUCKET}" && -n "${PATH}" ]]; then
    dvc push
fi

echo "DVC setup complete"
