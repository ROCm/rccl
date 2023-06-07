****
RCCL
****

The ROCm Collective Communication Library (RCCL) is a stand-alone library which provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs.

RCCL (pronounced “Rickel”) implements routines such as all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, all-to-allv, and all-to-all as well as direct point-to-point (GPU-to-GPU) send and receive operations.

The provided collective communication routines are implemented using Ring and Tree algorithms. They are optimized to achieve high bandwidth and low latency by leveraging topology awareness, high-speed interconnects, RDMA based collectives. RCCL utilizes PCIe and xGMI high-speed interconnects for intra-node communication as well as InfiniBand, RoCE, and TCP/IP for inter-node communication.

RCCL supports an arbitrary number of GPUs installed in a single-node or multi-node platform. It can be easily integrated into either single- or multi-process (e.g., MPI) applications.
