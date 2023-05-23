#!/bin/bash

function create_venv {
  local venv_name="$1"
  python3 -m venv "$venv_name"
}

function activate_venv {
  local venv_name="$1"
  source "$venv_name/bin/activate" || source "$venv_name/Scripts/activate"
}

function upgrade_pip {
  python3 -m pip install --upgrade pip setuptools wheel
}

function install_dependencies {
  local dev="$1"
  if [ "$dev" = "dev" ]; then
    python3 -m pip install -e .[dev]
  else
    python3 -m pip install -e .
  fi
}

function main {
  local venv_name="$1"
  local dev="$2"

  if [ -z "$venv_name" ]; then
    echo "Error: Virtual environment name not provided."
    echo "Usage: ./setup_venv.sh <venv_name> [dev]"
    exit 1
  fi

  create_venv "$venv_name"
  activate_venv "$venv_name"
  upgrade_pip
  install_dependencies "$dev"
}

main "$1" "$2"
