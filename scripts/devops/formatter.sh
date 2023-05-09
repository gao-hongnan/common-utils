#!/bin/sh -eu

black_check() {
  fail="0"

  black --version

  if ! black --check .; then
    fail="1"
  fi

  if [ "$fail" = "1" ]; then
    echo "BLACK ERROR: at least one file is poorly formatted"
    exit 123
  else
    echo "BLACK SUCCESS!!"
  fi
}

main() {
  black_check
  echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
}

main