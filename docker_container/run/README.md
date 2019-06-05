# How to run a data science container

In order to start a container you just need `./run.sh`

Jupyter notebook config file is located in `./config` directory. You might want to change config, at least the password to access the notebook.


**Prerequisites**

You need [docker](https://docs.docker.com/engine/installation/linux/)
## Settings

The model for inference as well as jupyter notebook with inference pipeline located in 
Seismicpro/models/First_break_picking

The /SEGY folder in the containter mounted to /notebooks/SEGY folder on the host
Before running a container you might set some env variables:

### DS_SEGY_DIR
default: `../SEGY`

Directory in the host system where SEGY files are stored. 

### DS_CONFIG_DIR
default: `./config`

Directory in the host system where `jupyter_notebook_config.py` is stored. It is mapped to `/jupyter/config` directory in the container.

### DS_SECRET_DIR
default: `./secret`

Directory in the host system where TLS certs are stored. It is mapped to `/jupyter/secret` directory in the container.

### DS_PORT
default: `8888`

Host port where jupyter notebook is listening.

### DS_IMAGE
default: `analysiscenter1/ds-py3:cpu`

Docker image to run in a container.

## Examples
`DS_SEGY_DIR='some_path' ./run.sh` - to run a container with SEGY data folder mounted at 'some_path' on local host 

`DS_PORT=8889 ./run.sh` - to run a container which can be accessed at `http://localhost:8889`

You can pass additional docker options, for instance:

`DS_PORT=8889 ./run.sh -it --rm` - to run a container interactively and to remove it when it stops
