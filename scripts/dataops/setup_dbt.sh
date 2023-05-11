#!/bin/bash

# Function to print usage instructions
function usage {
    echo "Usage: $0"\
    " [-t type]"\
    " [-m method]"\
    " [-k keyfile]"\
    " [-j project]"\
    " [-d dataset]"\
    " [-f threads]"\
    " [-e job_execution_timeout_seconds]"\
    " [-l location]"\
    " [-u username]"\
    " [-p project_name]"\
    " [-q project_dir]"

    echo
    echo "Options:"
    echo "  -t TYPE          The type of the dbt project (e.g., bigquery)"
    echo "  -m METHOD        The authentication method for BigQuery (e.g., service-account, oauth)"
    echo "  -k KEYFILE       The path to the keyfile for BigQuery authentication"
    echo "  -j PROJECT       The ID of the GCP project"
    echo "  -d DATASET       The name of the dataset in BigQuery"
    echo "  -f THREADS       The number of threads for dbt execution"
    echo "  -e TIMEOUT       The job execution timeout seconds for dbt"
    echo "  -l LOCATION      The location for the BigQuery dataset (e.g., US, EU)"
    echo "  -u USERNAME      The username for the project"
    echo "  -p PROJECT_NAME  The name of the dbt project to initialize"
    echo "  -q PROJECT_DIR   The path to the dbt project directory"
    echo "  -h               Print out usage information"
    exit 1
}

# Parse command line arguments
while getopts ":t:m:k:j:d:f:e:l:u:p:q:h" opt; do
  case ${opt} in
    t )
      type=$OPTARG
      ;;
    m )
      method=$OPTARG
      ;;
    k )
      keyfile=$OPTARG
      ;;
    j )
      project=$OPTARG
      ;;
    d )
      dataset=$OPTARG
      ;;
    f )
      threads=$OPTARG
      ;;
    e )
      timeout=$OPTARG
      ;;
    l )
      location=$OPTARG
      ;;
    u )
      username=$OPTARG
      ;;
    p )
      project_name=$OPTARG
      ;;
    q )
      project_dir=$OPTARG
      ;;
    h )
      usage
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      usage
      ;;
    : )
      echo "Invalid option: $OPTARG requires an argument" 1>&2
      usage
      ;;
  esac
done
shift $((OPTIND -1))

# Check if all required arguments are provided
if [ -z "${type}" ] || [ -z "${method}" ] || [ -z "${keyfile}" ] || [ -z "${project}" ] || \
   [ -z "${dataset}" ] || [ -z "${threads}" ] || [ -z "${timeout}" ] || [ -z "${location}" ] || \
   [ -z "${username}" ] || [ -z "${project_name}" ] || [ -z "${project_dir}" ]; then
    echo "All arguments are required" 1>&2
    usage
fi

function check_venv() {
    # Set text color to yellow
    yellow='\033[1;33m'
    # Reset text color to default
    reset='\033[0m'

    # Display the "WARNING" text in yellow followed by the prompt
    printf "${yellow}WARNING${reset}: Are you in a virtual environment? [y/N] "
    read confirm
}

# Function to create .dbt directory
function create_dbt_directory() {
    local username="$1"
    mkdir -p "/Users/$username/.dbt"
    # mkdir -p ~/.dbt
}

# Function to install dbt CLI
function install_dbt() {
    architecture=$(uname -m)

    if [ "$architecture" = "arm64" ]; then
        # If system is arm64, use Homebrew for installation
        brew update
        brew install git # skip if you already have git installed
        brew tap dbt-labs/dbt
        brew install \
            dbt-core \
            dbt-bigquery \
            dbt-postgres \
            dbt-redshift \
            dbt-snowflake \
            dbt-trino
    else
        # Otherwise, use pip for installation
        pip install \
            dbt-core \
            dbt-bigquery \
            dbt-postgres \
            dbt-redshift \
            dbt-snowflake \
            dbt-trino
    fi
}

# Function to initialize a new dbt project
function initialize_dbt_project() {
    local type="$1"
    local method="$2"
    local keyfile="$3"
    local project="$4"
    local dataset="$5"
    local threads="$6"
    local timeout="$7"
    local location="$8"
    local project_name="${9}"
    local project_directory="${10}"

    # Initialize a new dbt project with predefined responses
    profile_path="/home/users/nus/gaohn/.dbt/profiles.yml"

    cat << EOF > "$profile_path"
$project_name:
  outputs:
    dev:
      type: $type
      method: $method
      keyfile: $keyfile
      project: $project
      dataset: $dataset
      threads: $threads
      job_execution_timeout_seconds: $timeout
      location: $location
      job_retries: 1
      priority: interactive
  target: dev
EOF

    echo "profiles.yml created successfully!"

    cd "$project_directory"
    echo -e 'N\n' | dbt init "$project_name" # predefine responses to dbt init.
}


# Main function to call the other functions
function main() {
    local type="$1"
    local method="$2"
    local keyfile="$3"
    local project="$4"
    local dataset="$5"
    local threads="$6"
    local timeout="$7"
    local location="$8"
    local username="$9"
    local project_name="${10}"
    local project_dir="${11}"

    check_venv

    create_dbt_directory \
        "$username"
    install_dbt
    # Call the initialize_dbt_project function with parsed command line arguments
    # Call the initialize_dbt_project function with parsed command line arguments
    initialize_dbt_project \
        "$type" \
        "$method" \
        "$keyfile" \
        "$project" \
        "$dataset" \
        "$threads" \
        "$timeout" \
        "$location" \
        "$project_name" \
        "$project_dir"
}


# Call the main function with command line arguments
main \
    "$type" \
    "$method" \
    "$keyfile" \
    "$project" \
    "$dataset" \
    "$threads" \
    "$timeout" \
    "$location" \
    "$username" \
    "$project_name" \
    "$project_dir"
