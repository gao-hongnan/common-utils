#!/bin/bash

SSH_ADDRESS=""
RETRY_INTERVAL=60 # Retry interval in seconds

say_connected() {
  if [[ "$(uname)" == "Darwin" ]]; then
    # MacOS
    say "SSH connection successful!"
  elif [[ "$(uname)" == "Linux" ]]; then
    # Linux
    spd-say "SSH connection successful!"
  else
    # Assuming Windows or other
    echo "SSH connection successful! (Speech synthesis is not supported on this platform.)"
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
        echo "Invalid option: -$OPTARG" 1>&2
        exit 1
        ;;
      : )
        echo "Option -$OPTARG requires an argument" 1>&2
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
      echo "SSH connection successful!"
      say_connected
      break
    else
      echo "Connection failed, retrying in $RETRY_INTERVAL seconds..."
      sleep $RETRY_INTERVAL
    fi
  done
}

# Call the functions
parse_args "$@"
connect_ssh
