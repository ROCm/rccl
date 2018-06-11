/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rcclTracker.h"
#include "rcclDataTypes.h"
#include "rccl.h"

#include <hip/hip_runtime.h>

#define WI 1024

__global__ void rcclWaitForPeerDst(DeviceControl_t *currTrack) {
    int tx = hipThreadIdx_x;
    if(tx == 0) {
        while(std::atomic_load_explicit(&(currTrack->nextPeer->dstBuffer), std::memory_order_seq_cst) == nullptr) {} 
    }
}

__global__ void rcclWaitForPeerSrc(DeviceControl_t *currTrack) {
    int tx = hipThreadIdx_x;
    if(tx == 0) {
        while(std::atomic_load_explicit(&(currTrack->nextPeer->srcBuffer), std::memory_order_seq_cst) == nullptr) {}
    }
}

__global__ void rcclWaitForChunk(DeviceControl_t *currTrack, uint32_t chunkId) {
    int tx = hipThreadIdx_x;
    if(tx == 0) {
        while(std::atomic_load_explicit(&(currTrack->chunkId), std::memory_order_seq_cst) < chunkId) {}
    }
    __syncthreads();
}

__global__ void rcclDoPeerChunk(DeviceControl_t *currTrack, uint32_t nextChunkId) {
    int tx = hipThreadIdx_x;
    if(tx == 0) {
        std::atomic_store_explicit(&(currTrack->nextPeer->chunkId), nextChunkId, std::memory_order_seq_cst);
    }
    __syncthreads();
}

template<typename T>
static inline __device__ void copyChunk(T *dst, T *src, int tx) {
    for(int i=tx;i<CHUNK_DWORDx4;i++) {
        dst[i] = src[i];
    }
}

template<typename DataType, typename VectorType>
static inline __device__ void copyChunkCnt(DataType *dst, DataType *src, int tx, size_t count, size_t offset) {
    VectorType* vDst = reinterpret_cast<VectorType*>(dst);
    VectorType* vSrc = reinterpret_cast<VectorType*>(src);
    if(tx < count) {
        vDst[tx] = vSrc[tx];
    }
    if(tx == 0) {
        static const int factor = sizeof(VectorType)/sizeof(DataType);
        int i = 0;
        while(i < offset) {
            (dst + count*factor)[i] = (src + count*factor)[i];
            i++;
        }
    }
}

template<typename DataType, typename VectorType>
static inline __device__ void copyCnt(DataType *dst, DataType *src, int tx, size_t count) {
    VectorType* vDst = reinterpret_cast<VectorType*>(dst);
    VectorType* vSrc = reinterpret_cast<VectorType*>(src);

    constexpr int factor = sizeof(VectorType) / sizeof(DataType);
    constexpr int increment = WI * factor;
    int loopLimit = count / factor;

    for(int i=tx; i < loopLimit; i=i+WI) {
        vDst[i] = vSrc[i];
    }

    int offset = count % factor;
    if(tx == 0) {
        int i = 0;
        while(i < offset) {
            (dst + loopLimit*factor)[i] = (src + loopLimit*factor)[i];
            i++;
        }
    }
}


template<typename DataType, typename VectorType>
static inline __device__ void reduceSumCnt(DataType *dst, DataType *src1, DataType *src2, int tx, size_t count) {
    VectorType* vDst = reinterpret_cast<VectorType*>(dst);
    VectorType* vSrc1 = reinterpret_cast<VectorType*>(src1);
    VectorType* vSrc2 = reinterpret_cast<VectorType*>(src2);

    constexpr int factor = sizeof(VectorType) / sizeof(DataType);
    constexpr int increment = WI * factor;
    int loopLimit = count / factor;

    for(int i=tx; i < loopLimit; i=i+WI) {
        vDst[i] = vSrc1[i] + vSrc2[i];
    }

    int offset = count % factor;
    if(tx == 0) {
        int i = 0;
        while(i < offset) {
            (dst + loopLimit*factor)[i] = (src1 + loopLimit*factor)[i] + (src2 + loopLimit*factor)[i];
            i++;
        }
    }
}



/* Sum kernel */

template<typename T>
static inline __device__ void reduceChunkSum(T *dst, T *src1, T *src2, int tx) {
    for(int i=tx;i<CHUNK_DWORDx4;i=i+WI) {
        dst[i] = src1[i] + src2[i];
    }
}

template<typename DataType, typename VectorType>
static inline __device__ void reduceChunkSumCnt(DataType *dst, DataType *src1, DataType *src2, int tx, size_t count, size_t offset) {
    VectorType *vDst = reinterpret_cast<VectorType*>(dst);
    VectorType *vSrc1 = reinterpret_cast<VectorType*>(src1);
    VectorType *vSrc2 = reinterpret_cast<VectorType*>(src2);

    for(int j=tx;j<count;j = j+WI) {
        vDst[j] = vSrc1[j] + vSrc2[j];
    }
    if(tx == 0) {
        static const int factor = sizeof(VectorType)/sizeof(DataType);
        int i = 0;
        while(i < offset) {
            (dst + count*factor)[i] = (src1 + count*factor)[i] + (src2 + count*factor)[i];
            i++;
        }
    }      
}

/* Prod kernel */

template<typename T>
static inline __device__ void reduceChunkProd(T *dst, T *src1, T *src2, int tx) {
    for(int i=tx; i<CHUNK_DWORDx4; i=i+WI) {
        dst[i] = src1[i] * src2[i];
    }
}


template<typename DataType, typename VectorType>
static inline __device__ void reduceChunkProdCnt(DataType *dst, DataType *src1, DataType *src2, int tx, size_t count, size_t offset) {
    VectorType *vDst = reinterpret_cast<VectorType*>(dst);
    VectorType *vSrc1 = reinterpret_cast<VectorType*>(src1);
    VectorType *vSrc2 = reinterpret_cast<VectorType*>(src2);

    for(int j=tx;j < count;j=j+WI) {
        vDst[j] = vSrc1[j] * vSrc2[j];
    }
    if(tx == 0) {
        static const int factor = sizeof(VectorType)/sizeof(DataType);
        int i = 0;
        while(i < offset) {
            (dst + count*factor)[i] = (src1 + count*factor)[i] * (src2 + count*factor)[i];
            i++;
        }
    }      
}


/* Max kernel */

template<typename DataType, typename VectorType>
inline __device__ void reduceChunkMax(VectorType *dst, VectorType *src1, VectorType *src2, int tx) {
    for(int i=tx; i<CHUNK_DWORDx4; i=i+WI) {
        VectorType d;
        VectorType s1 = src1[i];
        VectorType s2 = src2[i];
        for(int j=0;j<sizeof(VectorType)/sizeof(DataType);j++) {
            d[j] = s1[j] > s2[j] ? s1[j] : s2[j];
        }
        dst[i] = d;
    }
}

template<typename DataType, typename VectorType>
static inline __device__ void reduceChunkMaxCnt(DataType *dst, DataType *src1, DataType *src2, int tx, size_t count, size_t offset) {
    VectorType *vDst = reinterpret_cast<VectorType*>(dst);
    VectorType *vSrc1 = reinterpret_cast<VectorType*>(src1);
    VectorType *vSrc2 = reinterpret_cast<VectorType*>(src2);

    for(int j=tx; j < count; j=j+WI) {
        VectorType d;
        VectorType s1 = vSrc1[j];
        VectorType s2 = vSrc2[j];
        for(int i=0;i<sizeof(VectorType)/sizeof(DataType);i++) {
            d[i] = s1[i] > s2[i] ? s1[i] : s2[i];
        }
        vDst[j] = d;
    }
    if(tx == 0) {
        static const int factor = sizeof(VectorType)/sizeof(DataType);
        int i = 0;
        DataType rSrc1, rSrc2;
        while(i < offset) {
            rSrc1 = (src1 + count*factor)[i];
            rSrc2 = (src2 + count*factor)[i];
            (dst + count*factor)[i] = rSrc1 > rSrc2 ? rSrc1 : rSrc2;
            i++;
        }
    }
}



/* Min kernel */

template<typename DataType, typename VectorType>
inline __device__ void reduceChunkMin(VectorType *dst, VectorType *src1, VectorType *src2, int tx) {
    for(int i=tx; i<CHUNK_DWORDx4; i=i+WI) {
        VectorType d;
        VectorType s1 = src1[i];
        VectorType s2 = src2[i];
        for(int j=0;j<sizeof(VectorType)/sizeof(DataType);j++) {
            d[j] = s1[j] < s2[j] ? s1[j] : s2[j];
        }
        dst[i] = d;
    }
}



template<typename DataType, typename VectorType>
__device__ void reduceChunkMinCnt(DataType *dst, DataType *src1, DataType *src2, int tx, size_t count, size_t offset) {
    VectorType *vDst = reinterpret_cast<VectorType*>(dst);
    VectorType *vSrc1 = reinterpret_cast<VectorType*>(src1);
    VectorType *vSrc2 = reinterpret_cast<VectorType*>(src2);

    for(int j = tx;j < count; j=j+WI) {
        VectorType d;
        VectorType s1 = vSrc1[j];
        VectorType s2 = vSrc2[j];
        for(int i=0;i<sizeof(VectorType)/sizeof(DataType);i++) {
            d[i] = s1[i] < s2[i] ? s1[i] : s2[i];
        }
        vDst[j] = d;
    }
    if(tx == 0) {
        static const int factor = sizeof(VectorType)/sizeof(DataType);
        int i = 0;
        DataType rSrc1, rSrc2;
        while(i < offset) {
            rSrc1 = (src1 + count*factor)[i];
            rSrc2 = (src2 + count*factor)[i];
            (dst + count*factor)[i] = rSrc1 < rSrc2 ? rSrc1 : rSrc2;
            i++;
        }
    }
}


