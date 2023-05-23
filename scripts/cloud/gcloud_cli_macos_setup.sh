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

install_google_cloud_sdk() {
    echo -e "${YELLOW}Installing Google Cloud SDK...${RESET}"
    curl -o $TAR_FILE $SDK_URL

    echo -e "${YELLOW}Extracting Google Cloud SDK...${RESET}"
    tar -xvf $TAR_FILE

    echo -e "${YELLOW}Installing Google Cloud SDK...${RESET}"
    ./google-cloud-sdk/install.sh --install-python TRUE --path-update TRUE --quiet

    echo -e "${YELLOW}Cleaning up...${RESET}"
    rm $TAR_FILE

    echo -e "${GREEN}Google Cloud SDK installed successfully!${RESET}"
}

install_google_cloud_sdk
