#!/bin/bash

set -e

segy_vol=${DS_SEGY_DIR:-$PWD/../SEGY}
config_vol=${DS_CONFIG_DIR:-`pwd`/config}
secret_vol=${DS_SECRET_DIR:-`pwd`/secret}
port=${DS_PORT:-8888}
image=${DS_IMAGE:-analysiscenter1/ds-py3:picking}


sudo docker run --rm -p ${port}:8888 \
  -v ${segy_vol}:/notebooks/SEGY \
  -v ${config_vol}:/jupyter/config \
  -v ${secret_vol}:/jupyter/secret \
  $@ \
  ${image}
