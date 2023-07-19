#!/bin/sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."
empty_line

usage() {
    logger "INFO" "Runs pytest with the specified options."
    logger "INFO" "Usage: ci_pytest_check <test_type> [--<option>=<value>] ..."
    empty_line
    logger "INFO" "All options available in pytest CLI can be used."
    logger "INFO" "For more details, see link below:"
    logger "LINK" "https://docs.pytest.org/en/latest/usage.html"
    logger "CODE" "$ pytest --help"
    empty_line
    logger "INFO" "Example:"
    logger "CODE" "$ ci_pytest_check unit --cov=package1 --cov=package2 | tee pytest.log"
}

ci_pytest_check() {
    check_if_installed "pytest"

    # Check if the first argument is --help
    if check_for_help "$@"; then
        usage
        exit 0
    fi

    # Define default flags
    local default_flags="--cov=./"

    # Process user-provided flags
    local test_type=$1
    shift
    local user_flags=""
    while (($#)); do
        case "$1" in
        --*)
            # Flags start with -- are treated as user-provided flags
            user_flags+="$1 "
            ;;
        *)
            # Any other argument is treated as a test type
            test_type="$1"
            ;;
        esac
        shift
    done

    VERSION=$(pytest --version)
    logger "INFO" "PYTEST VERSION: $VERSION"
    logger "LINK" "https://docs.pytest.org/en/latest/index.html"
    empty_line

    # Check if pyproject.toml exists
    check_for_pyproject_toml "pytest"
    logger "LINK" "https://docs.pytest.org/en/latest/reference/customize.html#adding-default-options"

    # If test_type is "all", run pytest on the entire tests directory
    if [ "$test_type" = "all" ]; then
        test_path="tests"
    else
        test_path="tests/$test_type"
    fi

    if ! pytest $test_path $default_flags $user_flags; then
        logger "ERROR" "❌ pytest check failed."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ pytest check passed."
}

ci_pytest_check "$@"
