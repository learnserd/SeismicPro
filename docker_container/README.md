# Docker container with python 3 environment without GPU support for First break picking model inference


# Docker
To install Docker execute `utils/install_docker.sh`.


# Inference Image
To build the image for inference execute the following command from the parent docker_container directory

`docker build -t image_name -f docker_container/Dockerfile`


# How to run inference script
In order to run the container with inference script you need to specify some variables, see details bellow.


### DATA_DIR
default: `docker_container/data`.

Directory in the host system where SEGY files, model and inference results would be stored. 

### SEGY
Specify the name of the SEGY file located in `DATA_DIR` folder for which we predict picking.

### MODEL
Specify the model name located in  `DATA_DIR` folder.

### DUMP_TO
default: `picking.csv`

Specify the filename in the `DATA_DIR` folder where the results would be dumped.

The format of the resulted csv file is `FFID        TraceNumber        Predictions`

### IMAGE
Docker image to run in the container. Specify the `image_name` you assigned to the container when build it.

## Examples

`SEGY=segy_name.sgy MODEL=model_name.dill IMAGE=picking_inference ./run.sh` - to run the script on the **docker_container/data/segy_name.sgy** file using the **docker_container/data/model_name.dill** model. Result will be stored in **docker_container/data/picking.csv**