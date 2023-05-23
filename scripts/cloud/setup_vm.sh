#!/bin/bash
# curl -o setup_vm.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/cloud/setup_vm.sh

# Update the system
sudo apt-get update
sudo apt-get upgrade -y

# Configure firewall settings
sudo ufw allow 22
sudo ufw allow 80
sudo ufw allow 443
sudo ufw allow 5000
sudo ufw --force enable

# Install necessary software
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install -y python3.9
sudo apt-get install -y python3-pip
sudo apt-get install -y python3-venv

# Make python3 use the new Python 3.9 interpreter
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1

# Set up user
sudo adduser --quiet --disabled-password --shell /bin/bash --home /home/gaohn --gecos "User" gaohn
sudo adduser gaohn sudo
