#!/bin/sh

# curl -o linter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci/ci_linter.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

MIN_LINT_SCORE=10
RCFILE="pyproject.toml"

usage() {
  logger "INFO" "Usage: $0 <package1> <package2> ..."
}

ci_pylint_check() {
  VERSION=$(pylint --version)
  logger "INFO" "PYLINT VERSION: $VERSION"

  if [ ! -f "pyproject.toml" ]; then
    logger "WARN" "No pyproject.toml found. Pylint will use default settings."
  else
    logger "INFO" "Found pyproject.toml. Pylint will use settings defined in it."
  fi

  for pkg in "$@"; do
    if ! pylint --rcfile=$RCFILE --fail-under=$MIN_LINT_SCORE --score=yes --output-format=colorized $pkg; then
      logger "ERROR" "PYLINT ERROR: $pkg has a score less than $MIN_LINT_SCORE."
      exit 123
    fi
  done

  logger "INFO" "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
}

# Check if the user asked for help
if check_for_help "$@"; then
    usage
fi

# Check if no arguments were provided
if [ $# -eq 0 ]; then
    logger "ERROR" "No packages provided."
    usage
    exit 1
fi

ci_pylint_check $@
