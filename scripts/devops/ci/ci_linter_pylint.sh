#!/bin/sh

# curl -o ci_linter_pylint.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_linter_pylint.sh

# https://pylint.readthedocs.io/en/latest/user_guide/usage/run.html

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

usage() {
    logger "INFO" "Runs pylint with the specified options."
    logger "INFO" "Usage: ci_pylint_check [--<option>=<value>] <package1> <package2> ..."
    empty_line
    logger "INFO" "All options available in pylint CLI can be used."
    logger "INFO" "For more details, see the link below:"
    logger "LINK" "https://pylint.readthedocs.io/en/latest/user_guide/run.html"
    logger "CODE" "$ pylint --help"
    empty_line
    logger "INFO" "Example:"
    logger "CODE_MULTI" \
        "$ ci_pylint_check \\
            --rcfile=pyproject.toml \\
            --fail-under=10 \\
            --score=yes \\
            --output-format=colorized \\
            package1 package2"
}

ci_pylint_check() {
    check_if_installed "pylint"

    # Define default flags
    local default_flags="--fail-under=10"

    # Process user-provided flags and packages
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

    VERSION=$(pylint --version)
    logger "INFO" "PYLINT VERSION: $VERSION"
    logger "LINK" "https://pylint.readthedocs.io/en/latest/index.html"
    empty_line

    # Check if pyproject.toml exists
    check_for_pyproject_toml "pylint"

    if ! pylint $default_flags $user_flags $packages; then
        logger "ERROR" "PYLINT ERROR: Score less than 10."
        exit 123
    fi

    logger "INFO" "✅ Pylint check passed."
}

ci_pylint_check "$@"
