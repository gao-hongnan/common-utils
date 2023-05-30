#!/bin/bash
# curl -o docker_setup.sh \
#     https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/containerization/docker/docker_setup.sh
# Setup docker on a Linux machine (usually a VM instance on GCP).

# Fetch the utils.sh script from a URL and source it
UTILS_SCRIPT=$(curl -s https://raw.githubusercontent.com/gao-hongnan/common-utils/main/scripts/utils.sh)
source /dev/stdin <<<"$UTILS_SCRIPT"
logger "INFO" "Fetched the utils.sh script from a URL and sourced it"


install_docker() {
  logger "INFO" "Updating package lists for upgrades and new package installations..."
  sudo apt-get update -y

  logger "INFO" "Installing necessary packages for Docker installation..."
  sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

  logger "INFO" "Adding Docker's official GPG key..."
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

  logger "INFO" "Adding Docker repository..."
  sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"

  logger "INFO" "Updating package lists again..."
  sudo apt-get update -y

  logger "INFO" "Installing Docker..."
  sudo apt-get install -y docker-ce

  # Add the current user to the docker group
  logger "INFO" "Adding current user to docker group..."
  sudo usermod -aG docker ${USER}

  # Change the ownership of the .docker directory
  logger "INFO" "Changing the ownership of the .docker directory..."
  sudo chown "$USER":"$USER" "/home/$USER/.docker" -R
  sudo chmod g+rwx "$HOME/.docker" -R
}

install_docker_compose() {
  logger "INFO" "Getting Docker Compose latest version..."
  COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)

  logger "INFO" "Downloading Docker Compose..."
  sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

  logger "INFO" "Making Docker Compose executable..."
  sudo chmod +x /usr/local/bin/docker-compose
}


main() {
  install_docker
  install_docker_compose
}

main