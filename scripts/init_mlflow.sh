#!/bin/bash

HOME_DIR=~
TMP_DIR=$HOME_DIR/tmp

function make_tmp_dir() {
    # Use the 'cd' command to change directory to the home directory
    mkdir -p $TMP_DIR && mkdir -p $TMP_DIR/log
}

function clone_mlflow_repo() {
    git clone https://github.com/mlflow/mlflow.git "$TMP_DIR/mlflow"
}

function create_requirements() {
  local requirements_path="$TMP_DIR/requirements.txt"

  touch "$requirements_path"
  echo "mlflow" >> "$requirements_path"
  echo "torchvision>=0.15.1" >> "$requirements_path"
  echo "torch>=2.0" >> "$requirements_path"
  echo "lightning==2.0.0" >> "$requirements_path"
  echo "jsonargparse[signatures]>=4.17.0" >> "$requirements_path"
  echo "protobuf<4.0.0" >> "$requirements_path"
}

function make_venv() {
    curl -o "$TMP_DIR/make_venv.sh" https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh
    bash "$TMP_DIR/make_venv.sh" $TMP_DIR/venv "$TMP_DIR/requirements.txt"
}

function main() {
    make_tmp_dir
    create_requirements
    make_venv
    clone_mlflow_repo
}

main
