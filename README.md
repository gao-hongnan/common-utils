# Common Utils

[![Continuous Integration](https://github.com/gao-hongnan/common-utils/actions/workflows/continuous_integration.yaml/badge.svg)](https://github.com/gao-hongnan/common-utils/actions/workflows/continuous_integration.yaml)

## TODOs

1. Use own `Logger` once the class is finalized in all scripts.

## Continuous Integration

### Virtual Environment

First, make a virtual environment with `make_venv.sh`:

```bash
curl -s -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
source venv/bin/activate && \
rm make_venv.sh
```

### Continue on error vs If Always

See
[here](https://stackoverflow.com/questions/58858429/how-to-run-a-github-actions-step-even-if-the-previous-step-fails-while-still-f/58859404#58859404).

### Run Bandit Security Check

```bash
bash ./scripts/devops/ci/ci_security_bandit.sh \
  --severity-level=low \
  --format=json \
  --output=bandit_results.json \
  common_utils
```

### Run Linter Check

```bash
bash ./scripts/devops/ci/ci_linter_pylint.sh \
  --rcfile=pyproject.toml \
  --fail-under=10 \
  --score=yes \
  --output-format=json:pylint_results.json,colorized \
  common_utils
```

### Run Formatter Black Check

```bash
bash ./scripts/devops/ci/ci_formatter_black.sh \
  --check \
  --diff \
  --color \
  --verbose \
  common_utils
```

### Run Formatter Isort Check

```bash
bash ./scripts/devops/ci/ci_formatter_isort.sh \
  --check \
  --diff \
  --color \
  --verbose \
  common_utils
```

## Run MyPy Type Check

```bash
bash ./scripts/devops/ci/ci_typing_mypy.sh \
  --config-file=pyproject.toml \
  common_utils \
  | tee mypy_results.log
```

### Run Unit Test

## Run Integration Test

### Run System Test

### Run Acceptance Test

See [madewithml](https://madewithml.com/courses/mlops/testing/).

### Run Data Test (Great Expectations)

### Run Markdown Lint

```bash
npm install -g markdownlint-cli && \
touch .markdownlint.json && \
```

```bash
npm install --save-dev --save-exact prettier
```

```bash
# prettier
function pr() {
  if [ -z "$1" ]; then
    echo "Error: TARGET_DIR is mandatory."
    return 1
  fi

  TARGET_DIR="$1"
  prettier "$TARGET_DIR" --write \
    --prose-wrap always \
    --print-width 80 \
    --tab-width 4 \
    --use-tabs true
}
```

```bash
pr <TARGET_MARKDOWN_FILE>
markdownlint --fix <TARGET_MARKDOWN_FILE>
```
