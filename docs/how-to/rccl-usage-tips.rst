.. meta::
   :description: Usage tips for the RCCL library of collective communication primitives
   :keywords: RCCL, ROCm, library, API, peer-to-peer, transport

.. _rccl-usage-tips:


*****************************************
RCCL usage tips
*****************************************

This topic describes some of the more common RCCL extensions, such as NPKit and MSCCL, and provides tips on how to
configure and customize the application.

NPKit
=====

RCCL integrates `NPKit <https://github.com/microsoft/npkit>`_, a profiler framework that
enables the collection of fine-grained trace events in RCCL components, especially in giant collective GPU kernels.
See the `NPKit sample workflow for RCCL <https://github.com/microsoft/NPKit/tree/main/rccl_samples>`_ for
a fully-automated usage example. It also provides useful templates for the following manual instructions.

To manually build RCCL with NPKit enabled, pass ``-DNPKIT_FLAGS="-DENABLE_NPKIT -DENABLE_NPKIT_...(other NPKit compile-time switches)"`` to the ``cmake`` command. 
All NPKit compile-time switches are declared in the RCCL code base as macros with the prefix ``ENABLE_NPKIT_``.
These switches control the information that is collected.

.. note::
   
   NPKit only supports the collection of non-overlapped events on the GPU.
   The ``-DNPKIT_FLAGS`` settings must follow this rule.

To manually run RCCL with NPKit enabled, set the environment variable ``NPKIT_DUMP_DIR``
to the NPKit event dump directory. NPKit only supports one GPU per process.
To manually analyze the NPKit dump results, use `npkit_trace_generator.py <https://github.com/microsoft/NPKit/blob/main/rccl_samples/npkit_trace_generator.py>`_.

MSCCL/MSCCL++
=============

RCCL integrates `MSCCL <https://github.com/microsoft/msccl>`_ and `MSCCL++ <https://github.com/microsoft/mscclpp>`_ to
leverage these highly efficient GPU-GPU communication primitives for collective operations.
Microsoft Corporation collaborated with AMD for this project.

MSCCL uses XMLs for different collective algorithms on different architectures. 
RCCL collectives can leverage these algorithms after the user provides the corresponding XML.
The XML files contain sequences of send-recv and reduction operations for the kernel to run.

MSCCL is enabled by default on the AMD Instinct™ MI300X accelerator. On other platforms, users might have to enable it
using the setting ``RCCL_MSCCL_FORCE_ENABLE=1``. By default, MSCCL is only used if every rank belongs
to a unique process. To disable this restriction for multi-threaded or single-threaded configurations,
use the setting ``RCCL_MSCCL_ENABLE_SINGLE_PROCESS=1``.

RCCL allreduce and allgather collectives can leverage the efficient MSCCL++ communication kernels
for certain message sizes. MSCCL++ support is available whenever MSCCL support is available.
To run a RCCL workload with MSCCL++ support, set the following RCCL environment variable:

.. code-block:: shell

   RCCL_MSCCLPP_ENABLE=1

To set the message size threshold for using MSCCL++, use the environment variable ``RCCL_MSCCLPP_THRESHOLD``,
which has a default value of 1MB. After ``RCCL_MSCCLPP_THRESHOLD`` has been set,
RCCL invokes MSCCL++ kernels for all message sizes less than or equal to the specified threshold.

The following restrictions apply when using MSCCL++. If these restrictions are not met,
operations fall back to using MSCCL or RCCL.

*  The message size must be a non-zero multiple of 32 bytes
*  It does not support ``hipMallocManaged`` buffers
*  Allreduce only supports the ``float16``, ``int32``, ``uint32``, ``float32``, and ``bfloat16`` data types
*  Allreduce only supports the sum operation

Enabling peer-to-peer transport
===============================

To enable peer-to-peer access on machines with PCIe-connected GPUs,
set the HSA environment variable as follows:

.. code-block:: shell

   HSA_FORCE_FINE_GRAIN_PCIE=1

This feature requires GPUs that support peer-to-peer access along with
proper large BAR addressing support.

Improving performance on the MI300X accelerator when using fewer than 8 GPUs
============================================================================

On a system with 8\*MI300X accelerators, each pair of accelerators is connected with dedicated XGMI links
in a fully-connected topology. For collective operations, this can achieve good performance when
all 8 accelerators (and all XGMI links) are used. When fewer than 8 GPUs are used, however, this can only achieve a fraction
of the potential bandwidth on the system.
However, if your workload warrants using fewer than 8 MI300X accelerators on a system,
you can set the run-time variable ``NCCL_MIN_NCHANNELS`` to increase the number of channels. For example:

.. code-block:: shell

   export NCCL_MIN_NCHANNELS=32

Increasing the number of channels can benefit performance, but it also increases
GPU utilization for collective operations.
Additionally, RCCL pre-defines a higher number of channels when only 2 or
4 accelerators are in use on a 8\*MI300X system. In this situation, RCCL uses 32 channels with two MI300X accelerators
and 24 channels for four MI300X accelerators.