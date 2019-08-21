## Docker container with python 3 environment without GPU support for First break picking model inference


## Docker
To install Docker execute `../utils/install_docker.sh`


## Inference Image
To build the image for inference execute 2 following commands. First, we need to move 2 levels up to reach the repository level, which we want to put in the image, so it would be in the context of the image build. Then we build the image. This is the feature of the **Docker** and done in security purposes. You also can specify image name.   

default image name: `fb_inference`   

`cd ../..`   
`docker build -t fb_inference -f docker_containers/picking_docker/build/Dockerfile .`

Come back to the root picking_docker folder afterwards.

`cd docker_containers/picking_docker`

## How to run inference script
In order to run the container with inference script you need to specify some variables, see details bellow.

### DATA_DIR
default: `docker_containers/picking_docker/data`.

Directory in the host system where SEGY files, model and inference results would be stored. 

### SEGY
Specify the name of the SEGY file located in `DATA_DIR` folder for which picking is being predicted.

### MODEL
default: `fb_model.dill`

Specify the model name located in  `DATA_DIR` folder.

### DUMP_TO
default: `dump.csv`

Specify the filename in the `DATA_DIR` folder where the results would be dumped.

The format of the resulted csv file is `FFID        TraceNumber        Predictions`

### IMAGE
default: `fb_inference`

Docker image to run in the container. Specify the `image_name` you assigned to the container when building it.

### BATCH_SIZE
default: `1000`

The number of traces in the batch during inference stage.

### NUM_ZERO
default: `500`

Required number of zero values for the trace to contain to be dropped from the batch.

### TRACE_LEN
default: `751`

The number of first samples of the trace to load from SEGY.

### DEVICE
default: `cpu`

The device for inference stage. Can be 'cpu' or 'gpu' in case you have GPU device.

## Examples

`DATA_DIR=/home/user/data SEGY=segy_name.sgy MODEL=fb_model_2d.dill run/run.sh`    
 This command runs the inference script on the *home/user/data/segy_name.sgy* file using the *home/user/data/fb_model_2d.dill* model. Result will be stored in *home/user/data/dump.csv*
