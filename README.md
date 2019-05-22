# RCCL

ROCm Communication Collectives Library

## Introduction

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, and reduce-scatter. It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node, and can be used in either single- or multi-process (e.g., MPI) applications. Multi node support is planned for a future release.

The collective operations are implemented using ring algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

## Requirements

1. ROCm supported GPUs
2. ROCm stack installed on the system (HIP runtime & HCC)
3. For building and running the unit tests, chrpath will need to be installed on your machine first. (sudo apt-get install chrpath)

## Quickstart RCCL Build

RCCL directly depends on HIP runtime & HCC C++ compiler which are part of the ROCm software stack.
In addition, HC Direct Function call support needs to be present on your machine.  There are binaries for hcc and HIP that need to be installed to get HC Direct Function call support.  These binaries are currently packaged with roc-master, and will be included in ROCm 2.4.

The root of this repository has a helper script 'install.sh' to build and install RCCL on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.

*  `./install.sh` -- builds library including unit tests
*  `./install.sh -i` -- builds and installs the library to /opt/rocm/rccl
*  `./install.sh -h` -- shows help
*  `./install.sh -t` -- builds library including unit tests

## Manual build
#### To build the library :

```shell
$ git clone https://github.com/ROCmSoftwarePlatform/rccl.git
$ cd rccl
$ mkdir build
$ cd build
$ CXX=/opt/rocm/bin/hcc cmake -DCMAKE_INSTALL_PREFIX=$PWD/rccl-install ..
$ make -j 8
```
You may substitute a path of your own choosing for CMAKE_INSTALL_PREFIX. Note: ensure rocm-cmake is installed, `apt install rocm-cmake`.

#### To build the RCCL package and install package :

Assuming you have already cloned this repository and built the library as shown in the previous section:

```shell
$ cd rccl/build
$ make package
$ sudo dpkg -i *.deb
```

RCCL package install requires sudo/root access because it creates a directory called "rccl" under /opt/rocm/. This is an optional step and RCCL can be used directly by including the path containing librccl.so.

## Tests

There are unit tests implemented with the Googletest framework in RCCL, which are currently a work-in-progress.  To invoke the unit tests, go to the rccl-install folder, then the test/ subfolder, and execute the appropriate unit test executable(s). Several notes for running the unit tests:

1. The LD_LIBRARY_PATH environment variable will need to be set to include /path/to/rccl-install/lib/ in order to run the unit tests.
2. The HSA_FORCE_FINE_GRAIN_PCIE environment variable will need to be set to 1 in order to run the unit tests.

An example call to the unit tests:
```shell
$ LD_LIBRARY_PATH=rccl-install/lib/ HSA_FORCE_FINE_GRAIN_PCIE=1 rccl-install/test/UnitTests
```

There are also other performance and error-checking tests for RCCL.  These are maintained separately at https://github.com/ROCmSoftwarePlatform/rccl-tests.
See the rccl-tests README for more information on how to build and run those tests.

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
