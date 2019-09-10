FROM analysiscenter1/ds-py3:cpu
MAINTAINER Roman Kh <rhudor@gmail.com>

COPY docker_containers/ /notebooks/SeismicPro/docker_containers
COPY seismicpro/ /notebooks/SeismicPro/seismicpro

WORKDIR /notebooks/SeismicPro/docker_containers/picking_docker

ENTRYPOINT ["python3", "picking_inference.py"]

