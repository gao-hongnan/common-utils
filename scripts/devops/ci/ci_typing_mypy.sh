#!/bin/sh

# curl -o ci_typing_mypy.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_typing_mypy.sh

# https://mypy.readthedocs.io/en/stable/index.html

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."
empty_line

usage() {
    logger "INFO" "Runs mypy with the specified options."
    logger "INFO" "Usage: ci_mypy_check [--<option>=<value>] ..."
    empty_line
    logger "INFO" "All options available in mypy CLI can be used."
    logger "INFO" "For more details, see link below:"
    logger "LINK" "https://mypy.readthedocs.io/en/stable/command_line.html"
    logger "CODE" "$ mypy --help"
    empty_line
    logger "INFO" "Example:"
    logger "CODE" "$ ci_mypy_check --pretty --color-output package1 package2 | tee mypy.log"
}

ci_mypy_check() {
    check_if_installed "mypy"

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

    VERSION=$(mypy --version)
    logger "INFO" "MYPY VERSION: $VERSION"
    logger "LINK" "https://mypy.readthedocs.io/en/stable/index.html"
    empty_line

    # Check if pyproject.toml exists
    check_for_pyproject_toml "mypy"
    logger "WARN" "Note that not all command-line options can be configured" \
        "using a pyproject.toml file. See the link below for more details."
    logger "LINK" "https://mypy.readthedocs.io/en/stable/config_file.html"

    if ! mypy $default_flags $user_flags $packages; then
        logger "ERROR" "❌ mypy check failed."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ mypy check passed."
}

ci_mypy_check "$@"
