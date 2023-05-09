#!/bin/sh -eu

pylint_check() {
  if ! (pylint src); then
    echo "PYLINT ERROR: score below required lint score"
    exit 123
  else
    echo "PYLINT SUCCESS!!"
  fi
}

main() {
  pylint_check
  echo "CHECKCODE: CONGRATULATIONS, ALL TESTS SUCCESSFUL!!"
}

main