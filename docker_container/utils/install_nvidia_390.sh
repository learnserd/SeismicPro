#!/bin/sh

sudo apt-get purge nvidia*
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y --no-install-recommends nvidia-driver-410
#sudo apt install nvidia-driver-410 libnvidia-gl-410 nvidia-utils-410 \
#xserver-xorg-video-nvidia-410 libnvidia-cfg1-410 libnvidia-ifr1-410 \
#libnvidia-decode-410 libnvidia-encode-410
nvidia-smi
