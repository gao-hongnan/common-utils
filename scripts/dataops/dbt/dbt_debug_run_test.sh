#!/bin/bash
# curl -o dbt_debug_run_test.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/dbt/dbt_debug_run_test.sh

set -e # Exit immediately if a command exits with a non-zero status
set -o pipefail # Fail a pipe if any sub-command fails

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."

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
    logger "BLOCK" \
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
