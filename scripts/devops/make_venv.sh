#!/bin/bash
# make venv without setup.cfg/setup.py/pyproject.toml.
# curl -o scripts/make_venv.sh https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh

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

create_venv() {
  local venv_name="$1"
  python -m venv "$venv_name"
}

activate_venv() {
  local venv_name="$1"
  source "$venv_name/bin/activate" || source "$venv_name/Scripts/activate"
}

upgrade_pip() {
  python -m pip install --upgrade pip setuptools wheel
}

install_dependencies() {
  local requirements_path="$1"
  local dev_requirements_path="$2"
  if [ -f "$requirements_path" ]; then
    python -m pip install -r "$requirements_path"
  fi
  if [ -f "$dev_requirements_path" ]; then
    python -m pip install -r "$dev_requirements_path"
  fi
}

main() {
  if [ $# -lt 1 ]; then
    usage
  fi

  local venv_name="${1}"
  local requirements_path="${2:-requirements.txt}"
  local dev_requirements_path="${3}"

  if [ -z "$venv_name" ]; then
    echo "Error: Virtual environment name not provided."
    echo "Usage: ./setup_venv.sh <venv_name> [requirements_path] [dev_requirements_path]"
    exit 1
  fi

  create_venv "$venv_name"
  activate_venv "$venv_name"
  upgrade_pip
  install_dependencies "$requirements_path" "$dev_requirements_path"
}

main "$@"
