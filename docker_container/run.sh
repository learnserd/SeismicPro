#!/bin/bash

set -e

mkdir $PWD/data 

path="../../../data/"

data_vol=${DATA_DIR:-$PWD/data}
image=${IMAGE:-inference:latest}
segy=${SEGY:-filename.sgy}
model=${MODEL:-model.dill}
save_to=${DUMP_TO:-picking.csv}

sudo docker run --rm \
  -v ${data_vol}:/notebooks/data \
  $@ ${image} \
  -p $path$segy -m $path$model -d $path$save_to
