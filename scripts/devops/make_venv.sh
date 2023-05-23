#!/bin/bash
# make venv without setup.cfg/setup.py/pyproject.toml.
# curl -o make_venv.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m' # No Color

usage() {
    echo "Usage: $0 <venv_name> [requirements_path] [dev_requirements_path]"
    echo
    echo "Creates a virtual environment and installs dependencies."
    echo
    echo "Arguments:"
    echo "  venv_name                The name of the virtual environment to create."
    echo "  requirements_path        The path to the requirements file. Defaults to 'requirements.txt'."
    echo "  dev_requirements_path    The path to the development requirements file."
    echo
    exit 1
}

check_input() {
    if [ -z "$1" ]; then
        echo "Error: Virtual environment name not provided."
        usage
        exit 1
    fi
}

create_venv() {
  local venv_name="$1"
  python3 -m venv "$venv_name"
}

activate_venv() {
  local venv_name="$1"
  source "$venv_name/bin/activate" || source "$venv_name/Scripts/activate"
}

upgrade_pip() {
  python3 -m pip3 install --upgrade pip3 setuptools wheel
}

install_dependencies() {
  local requirements_path="$1"
  local dev_requirements_path="$2"
  if [ -f "$requirements_path" ]; then
    python3 -m pip3 install -r "$requirements_path"
  fi
  if [ -f "$dev_requirements_path" ]; then
    python3 -m pip3 install -r "$dev_requirements_path"
  fi
}

main() {
  if [ $# -lt 1 ] || [ "$1" == "-h" ] || [ "$1" == "--help" ]; then
    usage
  fi

  local venv_name="${1}"
  local requirements_path="${2:-requirements.txt}"
  local dev_requirements_path="${3}"

  check_input "$venv_name"

  create_venv "$venv_name"
  activate_venv "$venv_name"
  upgrade_pip
  install_dependencies "$requirements_path" "$dev_requirements_path"
}

main "$@"
