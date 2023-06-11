#!/bin/bash

# curl -o utils.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh

# Define the color variables
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
WHITE='\033[1;37m'
RESET='\033[0m' # No Color

# Logger function
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
        printf "${color}$TIMESTAMP [$PADDED_LOG_LEVEL]:\n\n    ${message}${RESET}\n"
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

empty_line() {
    printf "\n"
}

check_for_help() {
    for arg in "$@"; do
        if [ "$arg" = "--help" ] || [ "$arg" = "-h" ]; then
            return 0
        fi
    done
    return 1
}

check_for_pyproject_toml() {
    local tool=$1

    if [ ! -f "pyproject.toml" ]; then
        logger "WARN" "No pyproject.toml found in root directory."
        empty_line
    else
        logger "INFO" "Found pyproject.toml. $tool will use settings defined in it."
        empty_line
    fi
}

check_if_installed() {
    local tool=$1

    if ! command -v $tool &>/dev/null; then
        logger "ERROR" "$tool is not installed. Please install it and retry."
        exit 1
    fi
}
