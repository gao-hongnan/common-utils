#!/bin/sh

# curl -o linter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci_linter.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

MIN_LINT_SCORE=10
RCFILE="pyproject.toml"

ci_pylint_check() {
  VERSION=$(pylint --version)
  logger "INFO" "PYLINT VERSION: $VERSION"

  if [ ! -f "pyproject.toml" ]; then
    logger "WARN" "No pyproject.toml found. Pylint will use default settings."
  else
    logger "INFO" "Found pyproject.toml. Pylint will use settings defined in it."
  fi

  if ! pylint --rcfile=$RCFILE --fail-under=$MIN_LINT_SCORE --score=yes --output-format=colorized .; then
    logger "ERROR" "PYLINT ERROR: at least one file has a score less than $MINLINTSCORE."
    exit 123
  fi

  logger "INFO" "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
}

ci_pylint_check