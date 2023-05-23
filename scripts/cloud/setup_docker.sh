#!/bin/bash

# Setup docker on a Linux machine (usually a VM instance on GCP).

#!/bin/bash

install_docker() {
  sudo apt-get update -y
  sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
  sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
  sudo apt-get update -y
  sudo apt-get install -y docker-ce

  # Add the current user to the docker group
  sudo usermod -aG docker ${USER}
}


install_docker_compose() {
  COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
  sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
  sudo chmod +x /usr/local/bin/docker-compose
}

install_docker
install_docker_compose
