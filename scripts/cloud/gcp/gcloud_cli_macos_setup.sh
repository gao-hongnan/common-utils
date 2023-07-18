#!/bin/bash
# This script installs the Google Cloud SDK on a macOS M1 machine.
# Google Cloud SDK 431.0.0
# bq 2.0.92
# core 2023.05.12
# gcloud-crc32c 1.0.0
# gsutil 5.23

# Define colors as global variables
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RESET='\033[0m'

# Global variables for Google Cloud SDK
SDK_URL="https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-428.0.0-darwin-arm.tar.gz"
TAR_FILE="google-cloud-sdk.tar.gz"

logger "WARN" "This script is for installing Google Cloud SDK on a macOS M1 machine."
logger "WARN" "This script will install 428.0.0."

install_google_cloud_sdk() {
    logger "INFO" "Downloading Google Cloud SDK..."
    curl -o $TAR_FILE $SDK_URL

    logger "INFO" "Extracting Google Cloud SDK..."
    tar -xvf $TAR_FILE

    logger "INFO" "Installing Google Cloud SDK..."
    ./google-cloud-sdk/install.sh --install-python TRUE --path-update TRUE --quiet

    logger "INFO" "Cleaning up..."
    rm $TAR_FILE

    logger "INFO" "Google Cloud SDK installed successfully!"
}

install_google_cloud_sdk
