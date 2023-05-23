#!/bin/bash
# curl -o setup_vm.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/cloud/setup_vm.sh

# TODO: NOT REFINED! NEED TO BE REFINED!

# Update the system
sudo apt-get update
sudo apt-get upgrade -y

# Configure firewall settings (example opens SSH (22), HTTP (80), and HTTPS (443))
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 5000
sudo ufw enable

# Install necessary software (example installs Python3 and pip)
sudo apt-get install -y python3
sudo apt-get install -y python3-pip
sudo apt-get install python3-venv

# Set up user (replace 'newuser' with actual username)
sudo adduser gaohn
sudo adduser gaohn sudo
