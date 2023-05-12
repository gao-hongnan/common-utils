#!/bin/bash
# curl -o scripts/dbt_debug_run_test.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/dataops/dbt/dbt_debug_run_test.sh


# Function to change into the dbt project directory
function cd_into_dbt_dir {
    local project_dir="$1"
    echo "Changing into dbt directory: $project_dir"
    cd "$project_dir"
}

# Function to debug dbt connection
function debug_dbt {
    echo "Testing dbt connection..."
    dbt debug
}

# Function to run dbt model
function run_dbt {
    echo "Running dbt model..."
    dbt run
}

# Function to test dbt model
function test_dbt {
    echo "Testing dbt model..."
    dbt test
}

# Main function to call the other functions
function main {
    local project_dir="$1"
    cd_into_dbt_dir "$project_dir"
    debug_dbt
    run_dbt
    test_dbt
}

# Call the main function with your project directory as argument
main "$@"