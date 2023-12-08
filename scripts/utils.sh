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

LOG_LEVEL_WIDTH=10
MESSAGE_START=$(( LOG_LEVEL_WIDTH + 21 ))
DESIRED_WIDTH=79

logger() {
    local level=$1
    shift
    local message="$@"  # Ensures that the message is treated as a single string

    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Left-align the log level within the fixed width
    PADDED_LOG_LEVEL=$(printf "%-${LOG_LEVEL_WIDTH}s" "[$level]")

    # Determine the color based on the log level
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
    "BLOCK")
        color=$CYAN

        # Split the message into an array of lines
        readarray -t lines <<< "$message"

        # Print the first line of the message on the same line as the timestamp and log level
        printf "${color}$TIMESTAMP $PADDED_LOG_LEVEL ${lines[0]}"

        # Print the rest of the lines with the correct indentation
        for i in "${!lines[@]}"; do
            if [ $i -ne 0 ]; then  # Skip the first line
                printf "\n%-${MESSAGE_START}s%s" " " "${lines[i]}"
            fi
        done

        # Reset the color after printing all lines
        printf "${RESET}\n"
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

(
    IFS=$'\n'
    first=1
    printf -v msg "%s" "$message"
    for line in $(echo "$msg" | fold -w $DESIRED_WIDTH -s); do
        if [ $first -eq 1 ]; then
            printf "${color}$TIMESTAMP $PADDED_LOG_LEVEL $line${RESET}\n"
            first=0
        else
            printf "${color}%-${MESSAGE_START}s%s${RESET}\n" " " "$line"
        fi
    done
)
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
        return 1 # false
    else
        logger "INFO" "Found pyproject.toml. $tool will use settings defined in it."
        empty_line
        return 0 # true
    fi
}

check_if_installed() {
    local tool=$1

    if ! command -v $tool &>/dev/null; then
        logger "ERROR" "$tool is not installed. Please install it and retry."
        exit 1
    fi
}

check_bash_version() {
    local major_version=$(echo "$BASH_VERSION" | cut -d '.' -f1)
    if [ "$major_version" -lt 4 ]; then
        echo "This script requires Bash version 4.0 or higher. You are using Bash version $BASH_VERSION."
        exit 1
    fi
}
