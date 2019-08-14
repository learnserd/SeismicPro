path="data/"

data_vol=${DATA_DIR:-$PWD/data} 
image=${IMAGE:-fb_inference} 
segy=${SEGY:-filename.sgy} 
model=${MODEL:-fb_model.dill} 
save_to=${DUMP_TO:-dump.csv}
batch_size=${BATCH_SIZE:-1000} 
num_zero=${NUM_ZERO:-500} 
trace_len=${TRACE_LEN:-750} 
device=${DEVICE:-cpu} 

sudo docker run --rm \
  -v ${data_vol}:/notebooks/SeismicPro/docker_containers/picking_docker/data \
  $@ ${image} \
  -p $path$segy -m $path$model -d $path$save_to -n ${num_zero} \
  -bs ${batch_size} -ts ${trace_len} -dvc ${device}