#!/bin/sh

# curl -o ci_formatter_isort.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_formatter_isort.sh

# https://pycqa.github.io/isort/

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"
empty_line

usage() {
    logger "INFO" "Runs isort with the specified options."
    logger "INFO" "Usage: ci_isort_check [--<option>=<value>] ..."
    empty_line
    logger "INFO" "All options available in isort CLI can be used."
    logger "INFO" "For more details, see link below:"
    logger "LINK" "https://pycqa.github.io/isort/docs/configuration/options"
    logger "CODE" "$ isort --help"
    empty_line
    logger "INFO" "Example:"
    logger "CODE" "$ ci_isort_check --diff --color --verbose --line-length=30"
}

ci_isort_check() {
    check_if_installed "isort"

    # Define default flags
    local default_flags="--check"

    # Process user-provided flags
    local user_flags=""
    local packages=""
    while (($#)); do
        case "$1" in
        --help)
            usage
            return
            ;;
        --*)
            # Flags start with -- are treated as user-provided flags
            user_flags+="$1 "
            ;;
        *)
            # Any other argument is treated as a package
            packages+="$1 "
            ;;
        esac
        shift
    done

    VERSION=$(isort --version)
    logger "INFO" "ISORT VERSION: $VERSION"
    logger "LINK" "https://pycqa.github.io/isort/"
    empty_line

    check_for_pyproject_toml "isort"
    logger "WARN" "Note that not all command-line options can also be configured" \
        "using a pyproject.toml file. See the link below for more details."
    logger "LINK" "https://pycqa.github.io/isort/docs/configuration/options"

    # Run isort with default and user flags
    if ! isort $default_flags $user_flags $packages; then
        logger "ERROR" "ISORT ERROR: at least one file has incorrect import order."
        logger "INFO" "Consider running the following command to fix the import order:"
        logger "CODE" "$ isort ."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ Isort check passed."
}

ci_isort_check "$@" # $@ is the list of arguments passed to this script
