# RCCL

ROCm Communication Collectives Library

## Introduction

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is also initial support for direct GPU-to-GPU send and receive operations.  It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

The collective operations are implemented using ring and tree algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

## Requirements

1. ROCm supported GPUs
2. ROCm stack installed on the system (HIP runtime & HCC or HIP-Clang)

## Quickstart RCCL Build

RCCL directly depends on HIP runtime, plus the HCC C++ compiler or the HIP-Clang compiler which are part of the ROCm software stack.
For ROCm installation instructions, see https://github.com/RadeonOpenCompute/ROCm.

The root of this repository has a helper script 'install.sh' to build and install RCCL on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install.

```shell
./install.sh --help

 Options:
       --address-sanitizer     Build with address sanitizer enabled
       --build_allreduce_only  Build only AllReduce + sum + float kernel
    -d|--dependencies          Install RCCL depdencencies
       --debug                 Build debug library
       --disable_backtrace     Build without custom backtrace support
       --disable-colltrace     Build without collective trace
    -f|--fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)
    -h|--help                  Prints this help message
    -i|--install               Install RCCL library (see --prefix argument below)
    -j|--jobs                  Specify how many parallel compilation jobs to run (16 by default)
    -l|--local_gpu_only        Only compile for local GPU architecture
       --no_clean              Don't delete files if they already exist
       --npkit-enable          Compile with npkit enabled
    -p|--package_build         Build RCCL package
       --prefix                Specify custom directory to install RCCL to (default: /opt/rocm)
       --rm-legacy-include-dir Remove legacy include dir Packaging added for file/folder reorg backward compatibility
       --run_tests_all         Run all rccl unit tests (must be built already)
    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)
       --static                Build RCCL as a static library instead of shared library
    -t|--tests_build           Build rccl unit tests, but do not run
       --time-trace            Plot the build time of RCCL
       --verbose               Show compile commands
```

## Manual build

### To build the library :

```shell
$ git clone https://github.com/ROCmSoftwarePlatform/rccl.git
$ cd rccl
$ mkdir build
$ cd build
$ CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ ..
$ make -j 16      # Or some other suitable number of parallel jobs
```
You may substitute an installation path of your own choosing by passing CMAKE_INSTALL_PREFIX. For example:
```shell
$ CXX=/opt/rocm/bin/hipcc cmake -DCMAKE_PREFIX_PATH=/opt/rocm/ -DCMAKE_INSTALL_PREFIX=$PWD/rccl-install ..
```
Note: ensure rocm-cmake is installed, `apt install rocm-cmake`.

### To build the RCCL package and install package :

Assuming you have already cloned this repository and built the library as shown in the previous section:

```shell
$ cd rccl/build
$ make package
$ sudo dpkg -i *.deb
```

RCCL package install requires sudo/root access because it creates a directory called "rccl" under /opt/rocm/. This is an optional step and RCCL can be used directly by including the path containing librccl.so.

## Enabling peer-to-peer transport

In order to enable peer-to-peer access on machines with PCIe-connected GPUs, the HSA environment variable HSA_FORCE_FINE_GRAIN_PCIE=1 is required to be set, on top of requiring GPUs that support peer-to-peer access and proper large BAR addressing support.

## Tests

There are rccl unit tests implemented with the Googletest framework in RCCL.  The rccl unit tests require Googletest 1.10 or higher to build and execute properly (installed with the -d option to install.sh).
To invoke the rccl unit tests, go to the build folder, then the test subfolder, and execute the appropriate rccl unit test executable(s).

rccl unit test names are now of the format:

    CollectiveCall.[Type of test]

Filtering of rccl unit tests should be done with environment variable and by passing the --gtest_filter command line flag, for example:

```shell
UT_DATATYPES=ncclBfloat16 UT_REDOPS=prod ./rccl-UnitTests --gtest_filter="AllReduce.C*"
```
will run only AllReduce correctness tests with float16 datatype. A list of available filtering environment variables appears at the top of every run. See "Running a Subset of the Tests" at https://chromium.googlesource.com/external/github.com/google/googletest/+/HEAD/googletest/docs/advanced.md for more information on how to form more advanced filters.


There are also other performance and error-checking tests for RCCL.  These are maintained separately at https://github.com/ROCmSoftwarePlatform/rccl-tests.
See the rccl-tests README for more information on how to build and run those tests.

## NPKit

RCCL integrates [NPKit](https://github.com/microsoft/npkit), a profiler framework that enables collecting fine-grained trace events in RCCL components, especially in giant collective GPU kernels.

Please check [NPKit sample workflow for RCCL](https://github.com/microsoft/NPKit/tree/main/rccl_samples) as a fully automated usage example. It also provides good templates for the following manual instructions.

To manually build RCCL with NPKit enabled, pass `-DNPKIT_FLAGS="-DENABLE_NPKIT -DENABLE_NPKIT_...(other NPKit compile-time switches)"` with cmake command. All NPKit compile-time switches are declared in the RCCL code base as macros with prefix `ENABLE_NPKIT_`, and they control which information will be collected. Also note that currently NPKit only supports collecting non-overlapped events on GPU, and `-DNPKIT_FLAGS` should follow this rule.

To manually run RCCL with NPKit enabled, environment variable `NPKIT_DUMP_DIR` needs to be set as the NPKit event dump directory. Also note that currently NPKit only supports 1 GPU per process.

To manually analyze NPKit dump results, please leverage [npkit_trace_generator.py](https://github.com/microsoft/NPKit/blob/main/rccl_samples/npkit_trace_generator.py).

## Library and API Documentation

Please refer to the [RCCL Documentation Site](https://rocm.docs.amd.com/projects/rccl/en/latest/) for current documentation.

### How to build documentation

Run the steps below to build documentation locally.

```
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.
