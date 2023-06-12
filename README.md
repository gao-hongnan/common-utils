# Continuous Integration

[![Continuous Integration](https://github.com/gao-hongnan/common-utils/actions/workflows/ci.yaml/badge.svg?branch=continuous-integration)](https://github.com/gao-hongnan/common-utils/actions/workflows/ci.yaml)

## Virtual Environment

First, make a virtual environment with `make_venv.sh`:

```bash
curl -s -o make_venv.sh \
  https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/devops/make_venv.sh && \
bash make_venv.sh venv --pyproject --dev && \
source venv/bin/activate && \
rm make_venv.sh
```

## Run Bandit Security Check

```bash
bash ./scripts/devops/ci/ci_security_bandit.sh \
  --severity-level=low \
  --format=json \
  --output=bandit_results.json \
  common_utils
```

## Run Linter Check

```bash
bash ./scripts/devops/ci/ci_linter_pylint.sh \
  --rcfile=pyproject.toml \
  --fail-under=10 \
  --score=yes \
  --output-format=json:pylint_results.json,colorized \
  common_utils
```

## Run Formatter Black Check

```bash
bash ./scripts/devops/ci/ci_formatter_black.sh \
  --check \
  --diff \
  --color \
  --verbose \
  common_utils
```

## Run Formatter Isort Check

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
