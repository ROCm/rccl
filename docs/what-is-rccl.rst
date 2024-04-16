.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API

.. _what-is-rccl:

=====================
What is RCCL?
=====================

RCCL (pronounced “Rickel”) is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs.
It implements routines such as `all-reduce`, `all-gather`, `reduce`, `broadcast`, `reduce-scatter`, `gather`, `scatter`, `all-to-allv`, and `all-to-all` as well as direct point-to-point (GPU-to-GPU) send and receive operations.
The provided collective communication routines are implemented using Ring and Tree algorithms. They are optimized to achieve high bandwidth and low latency by leveraging topology awareness, high-speed interconnects, and RDMA based collectives. 

RCCL utilizes PCIe and xGMI high-speed interconnects for intra-node communication as well as InfiniBand, RoCE, and TCP/IP for inter-node communication.
It supports an arbitrary number of GPUs installed in a single-node or multi-node platform and can be easily integrated into single- or multi-process (e.g., MPI) applications.
