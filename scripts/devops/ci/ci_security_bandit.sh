#!/bin/sh

# https://bandit.readthedocs.io/en/latest/config.html
# curl -o formatter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_formatter_isort.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Successfully fetched and sourced the 'utils.sh' script from the 'common-utils' repository on GitHub."
empty_line

usage() {
    logger "INFO" "Runs Bandit with the specified options."
    logger "INFO" "Usage: ci_bandit_check [--<option>=<value>] ..."
    empty_line
    logger "INFO" "All options available in Bandit CLI can be used."
    logger "INFO" "For more details, see the link below:"
    logger "LINK" "https://bandit.readthedocs.io/en/latest/config.html"
    logger "CODE" "$ bandit --help"
    empty_line
    logger "INFO" "Example:"
    logger "CODE_MULTI" \
        "$ ci_bandit_check \\
            --severity-level=low \\
            --format=json \\
            --output=bandit_results.json \\
            --verbose"
}

ci_bandit_check() {
    check_if_installed "bandit"

    # Define default flags
    local default_flags="--recursive"

    # Process user-provided flags
    local user_flags=""
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

    VERSION=$(bandit --version)
    logger "INFO" "BANDIT VERSION: $VERSION"
    logger "LINK" "https://bandit.readthedocs.io/en/latest/index.html"
    empty_line

    check_for_pyproject_toml "bandit"
    pyproject_exists=$?
    logger "WARN" "Note that not all command-line options can also be configured" \
        "using a pyproject.toml file. See the link below for more details."
    logger "LINK" "https://bandit.readthedocs.io/en/latest/config.html"

    if [ $pyproject_exists -eq 0 ]; then
        # If pyproject.toml exists, then use it
        logger "INFO" "Appending --configfile pyproject.toml to the flags."
        default_flags+=" --configfile pyproject.toml"
    fi

    # Run bandit with default and user flags
    if ! bandit $default_flags $user_flags $packages; then
        logger "ERROR" "❌ Bandit check failed."
        logger "TIP" "Go through the logs and fix the issues."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ Bandit check passed."
}

ci_bandit_check "$@" # $@ is the list of arguments passed to this script
