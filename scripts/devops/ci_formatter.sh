#!/bin/sh

# curl -o formatter.sh \
#    https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/ci_formatter.sh

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"

ci_black_check() {
  VERSION=$(black --version)
  logger "INFO" "BLACK VERSION: $VERSION"

  if [ ! -f "pyproject.toml" ]; then
    logger "WARN" "No pyproject.toml found. Black will use default settings."
  else
    logger "INFO" "Found pyproject.toml. Black will use settings defined in it."
  fi

  if ! black --check --diff --color .; then
    logger "ERROR" "BLACK ERROR: at least one file is poorly formatted."
    logger "INFO" "Consider running the following command to fix the formatting errors:"
    logger "CODE" "$ black ."
    exit 123
  fi

  logger "INFO" "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
}

ci_black_check
