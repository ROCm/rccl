/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include"rcclKernelHelper.h"

__global__ void CheckPtrs(DeviceControl_t *currTrack) {
    if(hipThreadIdx_x == 0) {
        while(std::atomic_load_explicit(&(currTrack->dstBuffer), std::memory_order_seq_cst) == 0) {}
        while(std::atomic_load_explicit(&(currTrack->srcBuffer), std::memory_order_seq_cst) == 0) {}
    }
    __syncthreads();
}

__global__ void CopyRoot(DeviceControl_t *currTrack, uint32_t chunkId) {
    int tx = hipThreadIdx_x;
    int4* dst = reinterpret_cast<int4*>(std::atomic_load_explicit(&(currTrack->dstBuffer), std::memory_order_seq_cst));
    int4* src = reinterpret_cast<int4*>(std::atomic_load_explicit(&(currTrack->srcBuffer), std::memory_order_seq_cst));
    size_t offset = (chunkId-1)*CHUNK_DWORDx4;

    copyChunk((dst + offset), (src + offset), tx);
    __syncthreads();
    __threadfence_system();
}

template<typename DataType, typename VectorType>
__global__ void CopyRootCnt(DeviceControl_t *currTrack, uint32_t chunkId, size_t count) {
    int tx = hipThreadIdx_x;
    DataType* dst = reinterpret_cast<DataType*>(std::atomic_load_explicit(&(currTrack->dstBuffer), std::memory_order_seq_cst));
    DataType* src = reinterpret_cast<DataType*>(std::atomic_load_explicit(&(currTrack->srcBuffer), std::memory_order_seq_cst));
    size_t offset = (chunkId-1)*CHUNK_DWORD;

    static const int factor = sizeof(VectorType)/sizeof(DataType);
    copyChunkCnt<DataType, VectorType>((dst + offset), (src + offset), tx, count/factor, count%factor);
    __syncthreads();
    __threadfence_system();
}


__global__ void Copy(DeviceControl_t *currTrack, uint32_t chunkId) {
    int tx = hipThreadIdx_x;

    int4 *dst = reinterpret_cast<int4*>(std::atomic_load_explicit(&(currTrack->dstBuffer), std::memory_order_seq_cst));
    int4 *src = reinterpret_cast<int4*>(std::atomic_load_explicit(&(currTrack->srcBuffer), std::memory_order_seq_cst));

    size_t offset = (chunkId-1)*CHUNK_DWORDx4;

    for(int i=tx;i<CHUNK_DWORDx4;i=i+WI) {
        copyChunk((dst + offset), (src + offset), tx);
    }
    __syncthreads();
    __threadfence_system();
}

template<typename DataType, typename VectorType>
__global__ void CopyCnt(DeviceControl_t *currTrack, uint32_t chunkId, size_t count) {
    int tx = hipThreadIdx_x;

    DataType *dst = reinterpret_cast<DataType*>(std::atomic_load_explicit(&(currTrack->dstBuffer), std::memory_order_seq_cst));
    DataType *src = reinterpret_cast<DataType*>(std::atomic_load_explicit(&(currTrack->srcBuffer), std::memory_order_seq_cst));

    size_t offset = (chunkId-1)*CHUNK_DWORD;

    static const int factor = sizeof(VectorType)/sizeof(DataType);
    copyChunkCnt<DataType, VectorType>((dst + offset), (src + offset), tx, count/factor, count%factor);

    __syncthreads();
    __threadfence_system();

}

