.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API

.. _install:

***************
Installing RCCL
***************

Requirements
============

1. ROCm supported GPUs
2. ROCm stack installed on the system (HIP runtime & HIP-Clang)

Quickstart RCCL Build
=====================

RCCL directly depends on HIP runtime plus the HIP-Clang compiler, which are part of the ROCm software stack.
For ROCm installation instructions, see https://github.com/ROCm/ROCm.
The root of this repository has a helper script ``install.sh`` to build and install RCCL with a single command. It hard-codes configurations that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install RCCL.

To build the library using the install script:
----------------------------------------------

.. code-block:: shell

    ./install.sh

For more info on build options/flags when using the install script, use the following:

.. code-block:: shell

    ./install.sh --help

RCCL build & installation helper script options:

.. code-block:: shell

       --address-sanitizer     Build with address sanitizer enabled
    -d|--dependencies          Install RCCL depdencencies
       --debug                 Build debug library
       --enable_backtrace      Build with custom backtrace support
       --disable-colltrace     Build without collective trace
       --disable-msccl-kernel  Build without MSCCL kernels
       --disable-mscclpp       Build without MSCCL++ support
    -f|--fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)
    -h|--help                  Prints this help message
    -i|--install               Install RCCL library (see --prefix argument below)
    -j|--jobs                  Specify how many parallel compilation jobs to run ($nproc by default)
    -l|--local_gpu_only        Only compile for local GPU architecture
       --amdgpu_targets        Only compile for specified GPU architecture(s). For multiple targets, seperate by ';' (builds for all supported GPU architectures by default)
       --no_clean              Don't delete files if they already exist
       --npkit-enable          Compile with npkit enabled
       --openmp-test-enable    Enable OpenMP in rccl unit tests
       --roctx-enable          Compile with roctx enabled (example usage: rocprof --roctx-trace ./rccl-program)
    -p|--package_build         Build RCCL package
       --prefix                Specify custom directory to install RCCL to (default: `/opt/rocm`)
       --rm-legacy-include-dir Remove legacy include dir Packaging added for file/folder reorg backward compatibility
       --run_tests_all         Run all rccl unit tests (must be built already)
    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)
       --static                Build RCCL as a static library instead of shared library
    -t|--tests_build           Build rccl unit tests, but do not run
       --time-trace            Plot the build time of RCCL (requires `ninja-build` package installed on the system)
       --verbose               Show compile commands

.. tip::
    By default, RCCL builds for all GPU targets defined in ``DEFAULT_GPUS`` in `CMakeLists.txt <https://github.com/ROCm/rccl/blob/develop/CMakeLists.txt>`_. To target specific GPU(s), and potentially reduce build time, use ``--amdgpu_targets`` as a ``;`` separated string listing GPU(s) to target.

Manual build
============

To build the library using CMake:
---------------------------------

.. code-block:: shell

    $ git clone https://github.com/ROCm/rccl.git
    $ cd rccl
    $ mkdir build
    $ cd build
    $ cmake ..
    $ make -j 16      # Or some other suitable number of parallel jobs

You may substitute an installation path of your own choosing by passing ``CMAKE_INSTALL_PREFIX``. For example:

.. code-block:: shell

    $ cmake -DCMAKE_INSTALL_PREFIX=$PWD/rccl-install ..

.. note::
    Ensure rocm-cmake is installed, ``apt install rocm-cmake``.


To build the RCCL package and install package:
----------------------------------------------

Assuming you have already cloned this repository and built the library as shown in the previous section:

.. code-block:: shell

    $ cd rccl/build
    $ make package
    $ sudo dpkg -i *.deb

RCCL package install requires sudo/root access because it creates a directory called "rccl" under ``/opt/rocm/``. This is an optional step and RCCL can be used directly by including the path containing ``librccl.so``.

Enabling peer-to-peer transport
===============================

In order to enable peer-to-peer access on machines with PCIe-connected GPUs, the HSA environment variable ``HSA_FORCE_FINE_GRAIN_PCIE=1`` is required to be set, on top of requiring GPUs that support peer-to-peer access and proper large BAR addressing support.

Testing RCCL
============

There are rccl unit tests implemented with the Googletest framework in RCCL.  The rccl unit tests require Googletest 1.10 or higher to build and execute properly (installed with the ``-d`` option to ``install.sh``).
To invoke the rccl unit tests, go to the build folder, then the test subfolder, and execute the appropriate rccl unit test executable(s).

rccl unit test names are now of the format:

.. code-block:: shell

    CollectiveCall.[Type of test]

Filtering of rccl unit tests should be done with environment variable and by passing the ``--gtest_filter`` command line flag:

.. code-block:: shell

    UT_DATATYPES=ncclBfloat16 UT_REDOPS=prod ./rccl-UnitTests --gtest_filter="AllReduce.C*"

This will run only ``AllReduce`` correctness tests with float16 datatype. A list of available filtering environment variables appears at the top of every run. See https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests for more information on how to form more advanced filters.

There are also other performance and error-checking tests for RCCL.  These are maintained separately at https://github.com/ROCm/rccl-tests.

.. note::
    See the `rccl-tests/README <https://github.com/ROCm/rccl-tests/blob/develop/README.md>`_ for more information on how to build and run those tests.

NPKit
=====

RCCL integrates `NPKit <https://github.com/microsoft/npkit>`_, a profiler framework that enables collecting fine-grained trace events in RCCL components, especially in giant collective GPU kernels.
Please check `NPKit sample workflow for RCCL <https://github.com/microsoft/NPKit/tree/main/rccl_samples>`_ as a fully automated usage example. It also provides good templates for the following manual instructions.
To manually build RCCL with NPKit enabled, pass ``-DNPKIT_FLAGS="-DENABLE_NPKIT -DENABLE_NPKIT_...(other NPKit compile-time switches)"`` with ``cmake`` command. All NPKit compile-time switches are declared in the RCCL code base as macros with prefix ``ENABLE_NPKIT_``, and they control which information will be collected. Also note that currently NPKit only supports collecting non-overlapped events on GPU, and ``-DNPKIT_FLAGS`` should follow this rule.

To manually run RCCL with NPKit enabled, environment variable ``NPKIT_DUMP_DIR`` needs to be set as the NPKit event dump directory. Also note that currently NPKit only supports 1 GPU per process.
To manually analyze NPKit dump results, please leverage `npkit_trace_generator.py <https://github.com/microsoft/NPKit/blob/main/rccl_samples/npkit_trace_generator.py>`_.

MSCCL/MSCCL++
=============

RCCL integrates `MSCCL <https://github.com/microsoft/msccl>`_ and `MSCCL++ <https://github.com/microsoft/mscclpp>`_ to leverage the highly efficient GPU-GPU communication primitives for collective operations. Thanks to Microsoft Corporation for collaborating with us in this project.

MSCCL uses XMLs for different collective algorithms on different architectures. RCCL collectives can leverage those algorithms once the corresponding XML has been provided by the user. The XML files contain the sequence of send-recv and reduction operations to be executed by the kernel. On MI300X, MSCCL is enabled by default. On other platforms, the users may have to enable this by setting ``RCCL_MSCCL_FORCE_ENABLE=1``.
On the other hand, RCCL allreduce and allgather collectives can leverage the efficient MSCCL++ communication kernels for certain message sizes. MSCCL++ support is available whenever MSCCL support is available. Users need to set the RCCL environment variable ``RCCL_ENABLE_MSCCLPP=1`` to run RCCL workload with MSCCL++ support. It is also possible to set the message size threshold for using MSCCL++ by using the environment variable ``RCCL_MSCCLPP_THRESHOLD``. Once ``RCCL_MSCCLPP_THRESHOLD`` (the default value is 1MB) is set, RCCL will invoke MSCCL++ kernels for all message sizes less than or equal to the specified threshold.

Improving performance on MI300 when using less than 8 GPUs
==========================================================

On a system with 8\*MI300X GPUs, each pair of GPUs are connected with dedicated XGMI links in a fully-connected topology. So, for collective operations, one can achieve good performance when all 8 GPUs (and all XGMI links) are used. When using less than 8 GPUs, one can only achieve a fraction of the potential bandwidth on the system.
But, if your workload warrants using less than 8 MI300 GPUs on a system, you can set the run-time variable `NCCL_MIN_NCHANNELS` to increase the number of channels. 

For example: ``export NCCL_MIN_NCHANNELS=32``

Increasing the number of channels can be beneficial to performance, but it also increases GPU utilization for collective operations.
Additionally, we have pre-defined higher number of channels when using only 2 GPUs or 4 GPUs on a 8\*MI300 system. Here, RCCL will use **32 channels** for the 2 MI300 GPUs scenario and **24 channels** for the 4 MI300 GPUs scenario.
