#!/bin/bash
# Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.

FILES="
./src/nccl.h.in
./src/bootstrap.cu
./src/collectives/all_gather.cu
./src/collectives/all_reduce.cu
./src/collectives/broadcast.cu
./src/collectives/collectives.h
./src/collectives/device/all_gather.cu
./src/collectives/device/all_gather.h
./src/collectives/device/all_reduce.cu
./src/collectives/device/all_reduce.h
./src/collectives/device/broadcast.cu
./src/collectives/device/broadcast.h
./src/collectives/device/common.h
./src/collectives/device/common_kernel.h
./src/collectives/device/functions.cu
./src/collectives/device/ll_kernel.h
./src/collectives/device/primitives.h
./src/collectives/device/reduce.cu
./src/collectives/device/reduce.h
./src/collectives/device/reduce_kernel.h
./src/collectives/device/reduce_scatter.cu
./src/collectives/device/reduce_scatter.h
./src/collectives/reduce.cu
./src/collectives/reduce_scatter.cu
./src/include/bootstrap.h
./src/include/common_coll.h
./src/include/core.h
./src/include/debug.h
./src/include/enqueue.h
./src/include/group.h
./src/include/ibvwrap.h
./src/include/nccl_net.h
./src/include/net.h
./src/include/nvlink.h
./src/include/nvmlwrap.h
./src/include/param.h
./src/include/ring.h
./src/include/rings.h
./src/include/shm.h
./src/include/socket.h
./src/include/topo.h
./src/include/transport.h
./src/include/utils.h
./src/init.cu
./src/misc/enqueue.cu
./src/misc/group.cu
./src/misc/ibvwrap.cu
./src/misc/nvmlwrap.cu
./src/misc/rings.cu
./src/misc/utils.cu
./src/ring.cu
./src/transport.cu
./src/transport/net.cu
./src/transport/net_ib.cu
./src/transport/net_socket.cu
./src/transport/p2p.cu
./src/transport/shm.cu
"

for f in $FILES
do
    sed -i \
        -e 's@cuda_runtime.h@hip/hip_runtime_api.h@g' \
        -e 's@cuda_fp16.h@hip/hip_fp16.h@g' \
        -e 's/cudaDeviceCanAccessPeer/hipDeviceCanAccessPeer/g' \
        -e 's/cudaDeviceEnablePeerAccess/hipDeviceEnablePeerAccess/g' \
        -e 's/cudaDeviceGetPCIBusId/hipDeviceGetPCIBusId/g' \
        -e 's/cudaErrorPeerAccessAlreadyEnabled/hipErrorPeerAccessAlreadyEnabled/g' \
        -e 's/cudaError_t/hipError_t/g' \
        -e 's/cudaEventCreateWithFlags/hipEventCreateWithFlags/g' \
        -e 's/cudaEventDestroy/hipEventDestroy/g' \
        -e 's/cudaEventDisableTiming/hipEventDisableTiming/g' \
        -e 's/cudaEventRecord/hipEventRecord/g' \
        -e 's/cudaEvent_t/hipEvent_t/g' \
        -e 's/cudaFree/hipFree/g' \
        -e 's/cudaFreeHost/hipHostFree/g' \
        -e 's/cudaGetDevice/hipGetDevice/g' \
        -e 's/cudaGetErrorString/hipGetErrorString/g' \
        -e 's/cudaGetLastError/hipGetLastError/g' \
        -e 's/cudaHostAlloc/hipHostMalloc/g' \
        -e 's/cudaHostAllocMapped/hipHostMallocMapped/g' \
        -e 's/cudaHostGetDevicePointer/hipHostGetDevicePointer/g' \
        -e 's/cudaHostRegister/hipHostRegister/g' \
        -e 's/cudaHostRegisterMapped/hipHostRegisterMapped/g' \
        -e 's/cudaHostUnregister/hipHostUnregister/g' \
        -e 's/cudaIpcCloseMemHandle/hipIpcCloseMemHandle/g' \
        -e 's/cudaIpcGetMemHandle/hipIpcGetMemHandle/g' \
        -e 's/cudaIpcMemHandle_t/hipIpcMemHandle_t/g' \
        -e 's/cudaIpcMemLazyEnablePeerAccess/hipIpcMemLazyEnablePeerAccess/g' \
        -e 's/cudaIpcOpenMemHandle/hipIpcOpenMemHandle/g' \
        -e 's/cudaMalloc/hipMalloc/g' \
        -e 's/cudaMemcpy/hipMemcpy/g' \
        -e 's/cudaMemcpyAsync/hipMemcpyAsync/g' \
        -e 's/cudaMemcpyDefault/hipMemcpyDefault/g' \
        -e 's/cudaMemcpyDeviceToDevice/hipMemcpyDeviceToDevice/g' \
        -e 's/cudaMemoryTypeDevice/hipMemoryTypeDevice/g' \
        -e 's/cudaMemset/hipMemset/g' \
        -e 's/cudaPointerAttributes/hipPointerAttribute_t/g' \
        -e 's/cudaPointerGetAttributes/hipPointerGetAttributes/g' \
        -e 's/cudaSetDevice/hipSetDevice/g' \
        -e 's/cudaStreamCreateWithFlags/hipStreamCreateWithFlags/g' \
        -e 's/cudaStreamDestroy/hipStreamDestroy/g' \
        -e 's/cudaStreamNonBlocking/hipStreamNonBlocking/g' \
        -e 's/cudaStreamWaitEvent/hipStreamWaitEvent/g' \
        -e 's/cudaStream_t/hipStream_t/g' \
        -e 's/cudaSuccess/hipSuccess/g' \
        $f
done
