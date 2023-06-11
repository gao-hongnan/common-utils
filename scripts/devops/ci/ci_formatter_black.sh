#!/bin/sh

# https://black.readthedocs.io/en/stable/index.html
# curl -o formatter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_formatter.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"
empty_line

usage() {
    logger "INFO" "Runs black with the specified options."
    logger "INFO" "Usage: ci_black_check [--<option>=<value>] ..."
    empty_line
    logger "INFO" "All options available in black CLI can be used."
    logger "INFO" "For more details, see link below:"
    logger "LINK" "https://black.readthedocs.io/en/stable/usage_and_configuration/the_basics.html#command-line-options"
    logger "CODE" "$ black --help"
    empty_line
    logger "INFO" "Example:"
    logger "CODE" "$ ci_black_check --diff --color --verbose --line-length=79"
}

ci_black_check() {
    check_if_installed "black"

    # Define default flags
    local default_flags="--check"

    # Process user-provided flags
    local user_flags=""
    while (($#)); do
        case "$1" in
        --help)
            usage
            return
            ;;
        *)
            # Any other flag is treated as a user-provided flag
            user_flags+="$1 "
            shift
            ;;
        esac
    done

    VERSION=$(black --version)
    logger "INFO" "BLACK VERSION: $VERSION"
    logger "LINK" "https://black.readthedocs.io/en/stable/index.html"
    empty_line

    # Check if pyproject.toml exists
    check_for_pyproject_toml "black"
    pyproject_exists=$?

    logger "TIP" "Note that all command-line options can also be configured" \
        "using a pyproject.toml file."

    if ! black $default_flags $user_flags .; then
        logger "ERROR" "BLACK ERROR: at least one file is poorly formatted."
        logger "INFO" "Consider running the following command to fix the formatting errors:"
        logger "CODE" "$ black ."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ Black check passed."
}

ci_black_check "$@"
