#!/bin/bash
# curl -o dbt_debug_run_test.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/dbt/dbt_debug_run_test.sh

set -e # Exit immediately if a command exits with a non-zero status
set -o pipefail # Fail a pipe if any sub-command fails

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

# Change into the dbt project directory
cd_into_dbt_dir() {
    local project_dir="$1"
    logger "INFO" "Changing into dbt directory: $project_dir"
    cd "$project_dir"
}

# Debug dbt connection
debug_dbt() {
    logger "INFO" "Testing dbt connection..."
    dbt debug
}

# Run dbt model
run_dbt() {
    logger "INFO" "Running dbt model..."
    dbt run
}

# Test dbt model
test_dbt() {
    logger "INFO" "Testing dbt model..."
    dbt test
}

usage() {
    logger "INFO" "Sample Usage:"
    logger "CODE_MULTI" \
        "$ bash $0 \\
            <PATH_TO_DBT_PROJECT_DIRECTORY>"
    empty_line
    logger "INFO" "Example: bash $0 ~/my_dbt_project"
    exit 1
}

# Main function to call the other functions
main() {
    if [ -z "$1" ]; then
        logger "ERROR" "No dbt project directory provided."
        usage
    fi

    local project_dir="$1"
    cd_into_dbt_dir "$project_dir"
    debug_dbt
    run_dbt
    test_dbt
}

# Call the main function with your project directory as argument
main "$@"
