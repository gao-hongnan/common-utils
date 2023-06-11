#!/bin/sh

# curl -o formatter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_formatter.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"
empty_line

ci_black_check() {
    check_if_installed "black"

    # Check if the first argument is --help
    if check_for_help "$@"; then
        logger "INFO" "Help on the way..."
        black --help
        return
    fi

    VERSION=$(black --version)
    logger "INFO" "BLACK VERSION: $VERSION"
    logger "LINK" "https://black.readthedocs.io/en/stable/index.html"
    empty_line

    check_for_pyproject_toml "black"
    logger "TIP" "Note that all command-line options can also be configured" \
        "using a pyproject.toml file."

    if ! black --check .; then
        logger "ERROR" "BLACK ERROR: at least one file is poorly formatted."
        logger "INFO" "Consider running the following command to fix the formatting errors:"
        logger "CODE" "$ black ."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ Black check passed."
}

ci_isort_check() {
    check_if_installed "isort"

    # Check if the first argument is --help
    if check_for_help "$@"; then
        logger "INFO" "Help on the way..."
        black --help
        return
    fi

    VERSION=$(isort --version)
    logger "INFO" "ISORT VERSION: $VERSION"
    logger "LINK" "https://pycqa.github.io/isort/"
    empty_line

    check_for_pyproject_toml "isort"
    logger "WARN" "Note that not all command-line options can also be configured" \
        "using a pyproject.toml file. See the link below for more details."
    logger "LINK" "https://pycqa.github.io/isort/docs/configuration/options"

    if ! isort --check .; then
        logger "ERROR" "ISORT ERROR: at least one file has incorrect import order."
        logger "INFO" "Consider running the following command to fix the import order:"
        logger "CODE" "$ isort ."
        exit 123
    fi

    empty_line
    logger "INFO" "✅ Isort check passed."
}

main() {
    ci_black_check "$@" # $@ is the list of arguments passed to this script
    #ci_isort_check "$@"
}

main "$@"
