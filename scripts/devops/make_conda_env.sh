#!/bin/bash

ENV_NAME=$1
PYTHON_VERSION=$2
conda create -n $ENV_NAME python=$PYTHON_VERSION