#!/bin/sh

# install Docker
sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu bionic stable"

sudo apt-get update

sudo apt-get install docker-ce #5:18.09.6~3-0~ubuntu-bionic

# install NVIDIA docker version 2
#curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
#distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
#curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
# sudo tee /etc/apt/sources.list.d/nvidia-docker.list
#sudo apt-get update
#sudo apt-get install -y nvidia-docker2
# reload the Docker daemon configuration
#sudo pkill -SIGHUP dockerd
# Test nvidia-smi with the latest official CUDA image
#sudo docker run -e NVIDIA_VISIBLE_DEVICES=void --runtime=nvidia --rm nvidia/cuda nvidia-smi
