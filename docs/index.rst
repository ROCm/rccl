.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API

.. _index:

******************
RCCL documentation
******************

The ROCm Communication Collectives Library (RCCL) is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs.
It implements routines such as ``all-reduce``, ``all-gather``, ``reduce``, ``broadcast``, ``reduce-scatter``, ``gather``, ``scatter``, ``all-to-allv``, and ``all-to-all`` as well as direct point-to-point (GPU-to-GPU) send and receive operations. It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

The collective operations are implemented using Ring and Tree algorithms, and have been optimized for throughput and latency by leveraging topology awareness, high-speed interconnects, and RDMA based collectives. For best performance, small operations can be either batched into larger operations or aggregated through the API.

RCCL utilizes PCIe and xGMI high-speed interconnects for intra-node communication as well as InfiniBand, RoCE, and TCP/IP for inter-node communication. It supports an arbitrary number of GPUs installed in a single-node or multi-node platform and can be easily integrated into single- or multi-process (e.g., MPI) applications.

You can access RCCL code on the `RCCL GitHub repository <https://github.com/ROCm/rccl>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Installation

    * :ref:`install`
       
  .. grid-item-card:: How to

    * :ref:`using-nccl`
       
  .. grid-item-card:: API reference

    * :ref:`Library specification<library-specification>`
    * :ref:`api-library`
       
To contribute to the documentation refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Licensing information can be found on the
`Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
