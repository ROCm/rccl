## base docker image
ARG ROCM_IMAGE_NAME=rocm/dev-ubuntu-22.04
ARG ROCM_IMAGE_TAG=latest
FROM "${ROCM_IMAGE_NAME}:${ROCM_IMAGE_TAG}"

## rccl repo
ARG RCCL_REPO=https://github.com/ROCm/rccl
ARG RCCL_BRANCH=develop

## rccl-tests repo
ARG RCCL_TESTS_REPO=https://github.com/ROCmSoftwarePlatform/rccl-tests
ARG RCCL_TESTS_BRANCH=develop


## install dependencies and MPICH for running rccl-tests
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo \
    ca-certificates \
    git \
    make \
    cmake \
    rocm-cmake \
    ninja-build \
    gfortran \
    libomp5 \
    libomp-dev \
    llvm-dev \
    libbfd-dev \
    libboost-all-dev \
    libnuma1 \
    libnuma-dev \
    libpthread-stubs0-dev \
    libzstd-dev \
    lcov \
    zip \
    zlib1g-dev \
    wget \
    pkg-config \
    unzip \
    graphviz \
    clang \
    chrpath \
    doxygen \
    python3-pip \
    python3-setuptools \
    python3-venv \
    python3-dev \
    python3-tk \
    python3-yaml \
    lshw \
    build-essential \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libjpeg-dev \
    libsuitesparse-dev \
    vainfo \
    libva-dev \
    libdrm-dev \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    mpich \
    libmpich-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


## creating scratch space for installing rdma-core, rccl, and rccl-tests
RUN mkdir -p /workspace
WORKDIR /workspace


## building rccl (develop)
RUN git clone "${RCCL_REPO}" \
    && cd rccl \
    && git checkout "${RCCL_BRANCH}" \
    && mkdir build \
    && cd build \
    && CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -DCMAKE_INSTALL_PREFIX=/workspace/rccl/build/release .. \
    && make -j 16 \
    && make install

## building rccl-tests (develop)
RUN git clone "${RCCL_TESTS_REPO}" \
    && cd rccl-tests \
    && git checkout "${RCCL_TESTS_BRANCH}" \
    && mkdir build \
    && cd build \
    && CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH="/workspace/rccl/build/release;/opt/rocm/" .. \
    && make -j 16


## set environment variables
ENV PATH="/workspace/rccl/build/release/bin:/opt/rocm/bin:${PATH}"
ENV LD_LIBRARY_PATH="/workspace/rccl/build/release/lib:/opt/rocm/lib:/usr/lib:/usr/lib/x86_64-linux-gnu:/usr/local/lib:/lib:/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}"

