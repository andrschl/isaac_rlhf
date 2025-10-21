ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base:latest

# we use the basic isaaclab image as the base
FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base

ARG DOCKER_ISAAC_RLHF_PATH_ARG
ENV DOCKER_ISAAC_RLHF_PATH=${DOCKER_ISAAC_RLHF_PATH_ARG}

USER root

# Copy the Isaac RLHF extension directory (files to exclude are defined in .dockerignore)
COPY ../ ${DOCKER_ISAAC_RLHF_PATH}

# Install additional dependencies from the Isaac RLHF extension
RUN bash -i -c "source ${HOME}/.bashrc && \
    cd ${DOCKER_ISAAC_RLHF_PATH}/source/isaac_rlhf && \
    pip install -e . && \
    pip install --no-cache-dir GPUtil==1.4.0"

# make working directory as the Isaac Lab directory
# this is the default directory when the container is run
WORKDIR /workspace
