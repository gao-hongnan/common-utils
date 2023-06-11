#!/bin/sh

# curl -o formatter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_formatter.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"
empty_line

ci_black_check() {
    if ! command -v black &>/dev/null; then
        logger "ERROR" "Black is not installed. Please install it and retry."
        exit 1
    fi

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
    VERSION=$(isort --version)
    logger "INFO" "ISORT VERSION: $VERSION"

    if [ ! -f "pyproject.toml" ]; then
        logger "WARN" "No pyproject.toml found. isort will use default settings."
    else
        logger "INFO" "Found pyproject.toml. isort will use settings defined in it."
    fi

    if ! isort --check --diff --verbose .; then
        logger "ERROR" "ISORT ERROR: at least one file has incorrect import order."
        logger "INFO" "Consider running the following command to fix the import order:"
        logger "CODE" "$ isort ."
        exit 123
    fi

    logger "INFO" "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
}

main() {
    ci_black_check "$@"
    #ci_isort_check
}

main "$@"
