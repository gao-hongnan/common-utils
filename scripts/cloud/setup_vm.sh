#!/bin/bash

# Variables
USER_NAME="gaohn"
HOME_DIR="/home/$USER_NAME"

# Function to update the system
update_system () {
    sudo apt-get update
    sudo apt-get upgrade -y
}

# Function to configure firewall settings
configure_firewall () {
    sudo ufw allow 22
    sudo ufw allow 80
    sudo ufw allow 443
    sudo ufw allow 5000
    sudo ufw --force enable
}

# Function to install necessary software
install_software () {
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.9
    sudo apt-get install -y python3-pip
    sudo apt-get install -y python3-venv
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
    sudo apt-get install sqlite3 libsqlite3-dev # for sqlite3 backend mlflow
}

# Function to set up user
setup_user () {
    sudo adduser --quiet --disabled-password --shell /bin/bash --home $HOME_DIR --gecos "User" $USER_NAME
    sudo adduser $USER_NAME sudo
}

# Execute the functions
update_system
configure_firewall
install_software
setup_user
