#!/bin/bash

rm -rf dist
rm -rf gaohn_common_utils.egg-info

# Install build package
python3 -m pip install --upgrade build

# Build package
python3 -m build

# Install twine package
python3 -m pip install --upgrade twine

# Upload package to Test PyPI
python3 -m twine upload --repository testpypi dist/*
