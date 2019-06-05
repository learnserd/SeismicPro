# Docker container with python 3 environment without GPU support for First break picking model inference




# Docker
To install Docker  execute `utils/install_docker.sh`.


# Container
To prepare a docker environment run `utils/create_env.sh`.

Map additional disks to subdirectories within `/notebooks`.

Set a password in `run/config/jupyter_notebook_config.py`.


# Run jupyter
See [run/README.md](run/README.md) or just execute:
```
cd run
./run.sh
```
