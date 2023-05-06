#!/bin/bash

# Check if the user provided the required number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <number_of_minutes> <message>"
  exit 1
fi

# Get the number of minutes and the message from the command-line arguments
minutes="$1"
message="$2"

# Calculate the number of seconds from the provided minutes
seconds=$((minutes * 60))

# Function to display the progress bar
progress_bar() {
  local elapsed_seconds="$1"
  local total_seconds="$2"
  local elapsed_minutes=$((elapsed_seconds / 60))
  local total_minutes=$((total_seconds / 60))
  local width=50
  local progress=$((elapsed_seconds * width / total_seconds))
  local remaining=$((width - progress))

  printf "\r["
  printf "%0.s#" $(seq 1 $progress)
  printf "%0.s-" $(seq 1 $remaining)
  printf "] (%d/%d min)" $elapsed_minutes $total_minutes
}

# Infinite loop to display the message every N minutes
while true; do
  for ((i = 1; i <= seconds; i++)); do
    progress_bar $i $seconds
    sleep 1
  done
  printf "\n"
  osascript -e "display notification \"${message}\" with title \"Reminder\""
done
