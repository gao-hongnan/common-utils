#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RESET='\033[0m' # No Color

SSH_ADDRESS=""
RETRY_INTERVAL=60 # Retry interval in seconds

usage() {
    echo "Usage: $0 -a ssh_address -i retry_interval"
    echo
    echo "Options:"
    echo "  -a SSH_ADDRESS   The SSH address to connect (required)"
    echo "  -i RETRY_INTERVAL   The retry interval in seconds (required)"
    echo "  -h               Display this help and exit"
}

check_required_args() {
    if [[ -z "${SSH_ADDRESS}" || -z "${RETRY_INTERVAL}" ]]; then
        echo -e "${RED}Error: Missing required argument(s)${RESET}"
        usage
        exit 1
    fi
}

say_connected() {
  if [[ "$(uname)" == "Darwin" ]]; then
    # MacOS
    say "SSH connection successful!"
  elif [[ "$(uname)" == "Linux" ]]; then
    # Linux
    spd-say "SSH connection successful!"
  else
    # Assuming Windows or other
    echo -e "${YELLOW}SSH connection successful! (Speech synthesis is not supported on this platform.)${RESET}"
  fi
}

parse_args() {
  while getopts ":a:i:" opt; do
    case ${opt} in
      a )
        SSH_ADDRESS=$OPTARG
        ;;
      i )
        RETRY_INTERVAL=$OPTARG
        ;;
      \? )
        echo -e "${RED}Invalid option: -$OPTARG${RESET}" 1>&2
        exit 1
        ;;
      : )
        echo -e "${RED}Option -$OPTARG requires an argument${RESET}" 1>&2
        exit 1
        ;;
    esac
  done
  shift $((OPTIND -1))
}

connect_ssh() {
  while true; do
    if ssh -o BatchMode=yes -o ConnectTimeout=30 $SSH_ADDRESS echo "Connected!" &> /dev/null
    then
      echo -e "${GREEN}SSH connection successful!${RESET}"
      say_connected
      break
    else
      echo -e "${YELLOW}Connection failed, retrying in $RETRY_INTERVAL seconds...${RESET}"
      sleep $RETRY_INTERVAL
    fi
  done
}

# Main script execution
parse_args "$@"
check_required_args
connect_ssh
