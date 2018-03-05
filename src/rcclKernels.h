/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/


#pragma once

#include "rcclDataTypes.h"

#define WI 1024

template<typename VectorType, typename DataType>
__global__ void rcclKernelCopy(VectorType *vDst, VectorType *vSrc, size_t lenVec, size_t off) {
    /**
    * Using size_t as alu throughput is not the bottleneck in the kernel
    */
    int tx = hipThreadIdx_x;
    for(int i=tx;i<lenVec;i+=WI) {
        vDst[i] = vSrc[i];
    }

    DataType *sDst = reinterpret_cast<DataType*>(vDst + lenVec);
    DataType *sSrc = reinterpret_cast<DataType*>(vSrc + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc[tx];
    }

    __syncthreads();
}


template<typename VectorType>
__global__ void rcclKernelReduceSum(VectorType *vDst, VectorType *vSrc1, VectorType *vSrc2, size_t lenVec, size_t off);

template<typename VectorType>
__global__ void rcclKernelReduceProd(VectorType *vDst, VectorType *vSrc1, VectorType *vSrc2, size_t lenVec, size_t off);

template<typename VectorType>
__global__ void rcclKernelReduceMax(VectorType *vDst, VectorType *vSrc1, VectorType *vSrc2, size_t lenVec, size_t off);

template<typename VectorType>
__global__ void rcclKernelReduceMin(VectorType *vDst, VectorType *vSrc1, VectorType *vSrc2, size_t lenVec, size_t off);



__global__ void rcclKernelReduceSum(rccl_char16_t *vDst, rccl_char16_t *vSrc1, rccl_char16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_char16_t src1 = vSrc1[i];
        rccl_char16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_char16_t)/sizeof(signed char));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    signed char *sDst = reinterpret_cast<signed char*>(vDst + lenVec);
    signed char *sSrc1 = reinterpret_cast<signed char*>(vSrc1 + lenVec);
    signed char *sSrc2 = reinterpret_cast<signed char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_char16_t *vDst, rccl_char16_t *vSrc1, rccl_char16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_char16_t src1 = vSrc1[i];
        rccl_char16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_char16_t)/sizeof(signed char));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    signed char *sDst = reinterpret_cast<signed char*>(vDst + lenVec);
    signed char *sSrc1 = reinterpret_cast<signed char*>(vSrc1 + lenVec);
    signed char *sSrc2 = reinterpret_cast<signed char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_char16_t *vDst, rccl_char16_t *vSrc1, rccl_char16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_char16_t src1 = vSrc1[i];
        rccl_char16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_char16_t)/sizeof(signed char));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    signed char *sDst = reinterpret_cast<signed char*>(vDst + lenVec);
    signed char *sSrc1 = reinterpret_cast<signed char*>(vSrc1 + lenVec);
    signed char *sSrc2 = reinterpret_cast<signed char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_char16_t *vDst, rccl_char16_t *vSrc1, rccl_char16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_char16_t src1 = vSrc1[i];
        rccl_char16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_char16_t)/sizeof(signed char));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    signed char *sDst = reinterpret_cast<signed char*>(vDst + lenVec);
    signed char *sSrc1 = reinterpret_cast<signed char*>(vSrc1 + lenVec);
    signed char *sSrc2 = reinterpret_cast<signed char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}




__global__ void rcclKernelReduceSum(rccl_uchar16_t *vDst, rccl_uchar16_t *vSrc1, rccl_uchar16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uchar16_t src1 = vSrc1[i];
        rccl_uchar16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uchar16_t)/sizeof(unsigned char));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    unsigned char *sDst = reinterpret_cast<unsigned char*>(vDst + lenVec);
    unsigned char *sSrc1 = reinterpret_cast<unsigned char*>(vSrc1 + lenVec);
    unsigned char *sSrc2 = reinterpret_cast<unsigned char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_uchar16_t *vDst, rccl_uchar16_t *vSrc1, rccl_uchar16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uchar16_t src1 = vSrc1[i];
        rccl_uchar16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uchar16_t)/sizeof(unsigned char));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    unsigned char *sDst = reinterpret_cast<unsigned char*>(vDst + lenVec);
    unsigned char *sSrc1 = reinterpret_cast<unsigned char*>(vSrc1 + lenVec);
    unsigned char *sSrc2 = reinterpret_cast<unsigned char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_uchar16_t *vDst, rccl_uchar16_t *vSrc1, rccl_uchar16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uchar16_t src1 = vSrc1[i];
        rccl_uchar16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uchar16_t)/sizeof(unsigned char));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned char *sDst = reinterpret_cast<unsigned char*>(vDst + lenVec);
    unsigned char *sSrc1 = reinterpret_cast<unsigned char*>(vSrc1 + lenVec);
    unsigned char *sSrc2 = reinterpret_cast<unsigned char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_uchar16_t *vDst, rccl_uchar16_t *vSrc1, rccl_uchar16_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uchar16_t src1 = vSrc1[i];
        rccl_uchar16_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uchar16_t)/sizeof(unsigned char));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned char *sDst = reinterpret_cast<unsigned char*>(vDst + lenVec);
    unsigned char *sSrc1 = reinterpret_cast<unsigned char*>(vSrc1 + lenVec);
    unsigned char *sSrc2 = reinterpret_cast<unsigned char*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}



__global__ void rcclKernelReduceSum(rccl_short8_t *vDst, rccl_short8_t *vSrc1, rccl_short8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_short8_t src1 = vSrc1[i];
        rccl_short8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_short8_t)/sizeof(signed short));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    signed short *sDst = reinterpret_cast<signed short*>(vDst + lenVec);
    signed short *sSrc1 = reinterpret_cast<signed short*>(vSrc1 + lenVec);
    signed short *sSrc2 = reinterpret_cast<signed short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_short8_t *vDst, rccl_short8_t *vSrc1, rccl_short8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_short8_t src1 = vSrc1[i];
        rccl_short8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_short8_t)/sizeof(signed short));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    signed short *sDst = reinterpret_cast<signed short*>(vDst + lenVec);
    signed short *sSrc1 = reinterpret_cast<signed short*>(vSrc1 + lenVec);
    signed short *sSrc2 = reinterpret_cast<signed short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_short8_t *vDst, rccl_short8_t *vSrc1, rccl_short8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_short8_t src1 = vSrc1[i];
        rccl_short8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_short8_t)/sizeof(signed short));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    signed short *sDst = reinterpret_cast<signed short*>(vDst + lenVec);
    signed short *sSrc1 = reinterpret_cast<signed short*>(vSrc1 + lenVec);
    signed short *sSrc2 = reinterpret_cast<signed short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_short8_t *vDst, rccl_short8_t *vSrc1, rccl_short8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_short8_t src1 = vSrc1[i];
        rccl_short8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_short8_t)/sizeof(signed short));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    signed short *sDst = reinterpret_cast<signed short*>(vDst + lenVec);
    signed short *sSrc1 = reinterpret_cast<signed short*>(vSrc1 + lenVec);
    signed short *sSrc2 = reinterpret_cast<signed short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}



__global__ void rcclKernelReduceSum(rccl_ushort8_t *vDst, rccl_ushort8_t *vSrc1, rccl_ushort8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ushort8_t src1 = vSrc1[i];
        rccl_ushort8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ushort8_t)/sizeof(unsigned short));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    unsigned short *sDst = reinterpret_cast<unsigned short*>(vDst + lenVec);
    unsigned short *sSrc1 = reinterpret_cast<unsigned short*>(vSrc1 + lenVec);
    unsigned short *sSrc2 = reinterpret_cast<unsigned short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_ushort8_t *vDst, rccl_ushort8_t *vSrc1, rccl_ushort8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ushort8_t src1 = vSrc1[i];
        rccl_ushort8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ushort8_t)/sizeof(unsigned short));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    unsigned short *sDst = reinterpret_cast<unsigned short*>(vDst + lenVec);
    unsigned short *sSrc1 = reinterpret_cast<unsigned short*>(vSrc1 + lenVec);
    unsigned short *sSrc2 = reinterpret_cast<unsigned short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_ushort8_t *vDst, rccl_ushort8_t *vSrc1, rccl_ushort8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ushort8_t src1 = vSrc1[i];
        rccl_ushort8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ushort8_t)/sizeof(unsigned short));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned short *sDst = reinterpret_cast<unsigned short*>(vDst + lenVec);
    unsigned short *sSrc1 = reinterpret_cast<unsigned short*>(vSrc1 + lenVec);
    unsigned short *sSrc2 = reinterpret_cast<unsigned short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_ushort8_t *vDst, rccl_ushort8_t *vSrc1, rccl_ushort8_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ushort8_t src1 = vSrc1[i];
        rccl_ushort8_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ushort8_t)/sizeof(unsigned short));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned short *sDst = reinterpret_cast<unsigned short*>(vDst + lenVec);
    unsigned short *sSrc1 = reinterpret_cast<unsigned short*>(vSrc1 + lenVec);
    unsigned short *sSrc2 = reinterpret_cast<unsigned short*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}


__global__ void rcclKernelReduceSum(rccl_int4_t *vDst, rccl_int4_t *vSrc1, rccl_int4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_int4_t src1 = vSrc1[i];
        rccl_int4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_int4_t)/sizeof(signed int));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    signed int *sDst = reinterpret_cast<signed int*>(vDst + lenVec);
    signed int *sSrc1 = reinterpret_cast<signed int*>(vSrc1 + lenVec);
    signed int *sSrc2 = reinterpret_cast<signed int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_int4_t *vDst, rccl_int4_t *vSrc1, rccl_int4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_int4_t src1 = vSrc1[i];
        rccl_int4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_int4_t)/sizeof(signed int));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    signed int *sDst = reinterpret_cast<signed int*>(vDst + lenVec);
    signed int *sSrc1 = reinterpret_cast<signed int*>(vSrc1 + lenVec);
    signed int *sSrc2 = reinterpret_cast<signed int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_int4_t *vDst, rccl_int4_t *vSrc1, rccl_int4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_int4_t src1 = vSrc1[i];
        rccl_int4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_int4_t)/sizeof(signed int));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    signed int *sDst = reinterpret_cast<signed int*>(vDst + lenVec);
    signed int *sSrc1 = reinterpret_cast<signed int*>(vSrc1 + lenVec);
    signed int *sSrc2 = reinterpret_cast<signed int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_int4_t *vDst, rccl_int4_t *vSrc1, rccl_int4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_int4_t src1 = vSrc1[i];
        rccl_int4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_int4_t)/sizeof(signed int));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    signed int *sDst = reinterpret_cast<signed int*>(vDst + lenVec);
    signed int *sSrc1 = reinterpret_cast<signed int*>(vSrc1 + lenVec);
    signed int *sSrc2 = reinterpret_cast<signed int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}



__global__ void rcclKernelReduceSum(rccl_uint4_t *vDst, rccl_uint4_t *vSrc1, rccl_uint4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uint4_t src1 = vSrc1[i];
        rccl_uint4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uint4_t)/sizeof(unsigned int));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    unsigned int *sDst = reinterpret_cast<unsigned int*>(vDst + lenVec);
    unsigned int *sSrc1 = reinterpret_cast<unsigned int*>(vSrc1 + lenVec);
    unsigned int *sSrc2 = reinterpret_cast<unsigned int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_uint4_t *vDst, rccl_uint4_t *vSrc1, rccl_uint4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uint4_t src1 = vSrc1[i];
        rccl_uint4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uint4_t)/sizeof(unsigned int));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    unsigned int *sDst = reinterpret_cast<unsigned int*>(vDst + lenVec);
    unsigned int *sSrc1 = reinterpret_cast<unsigned int*>(vSrc1 + lenVec);
    unsigned int *sSrc2 = reinterpret_cast<unsigned int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_uint4_t *vDst, rccl_uint4_t *vSrc1, rccl_uint4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uint4_t src1 = vSrc1[i];
        rccl_uint4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uint4_t)/sizeof(unsigned int));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned int *sDst = reinterpret_cast<unsigned int*>(vDst + lenVec);
    unsigned int *sSrc1 = reinterpret_cast<unsigned int*>(vSrc1 + lenVec);
    unsigned int *sSrc2 = reinterpret_cast<unsigned int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_uint4_t *vDst, rccl_uint4_t *vSrc1, rccl_uint4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_uint4_t src1 = vSrc1[i];
        rccl_uint4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_uint4_t)/sizeof(unsigned int));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned int *sDst = reinterpret_cast<unsigned int*>(vDst + lenVec);
    unsigned int *sSrc1 = reinterpret_cast<unsigned int*>(vSrc1 + lenVec);
    unsigned int *sSrc2 = reinterpret_cast<unsigned int*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}




__global__ void rcclKernelReduceSum(rccl_float4_t *vDst, rccl_float4_t *vSrc1, rccl_float4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_float4_t src1 = vSrc1[i];
        rccl_float4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_float4_t)/sizeof(float));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    float *sDst = reinterpret_cast<float*>(vDst + lenVec);
    float *sSrc1 = reinterpret_cast<float*>(vSrc1 + lenVec);
    float *sSrc2 = reinterpret_cast<float*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_float4_t *vDst, rccl_float4_t *vSrc1, rccl_float4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_float4_t src1 = vSrc1[i];
        rccl_float4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_float4_t)/sizeof(float));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    float *sDst = reinterpret_cast<float*>(vDst + lenVec);
    float *sSrc1 = reinterpret_cast<float*>(vSrc1 + lenVec);
    float *sSrc2 = reinterpret_cast<float*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_float4_t *vDst, rccl_float4_t *vSrc1, rccl_float4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_float4_t src1 = vSrc1[i];
        rccl_float4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_float4_t)/sizeof(float));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    float *sDst = reinterpret_cast<float*>(vDst + lenVec);
    float *sSrc1 = reinterpret_cast<float*>(vSrc1 + lenVec);
    float *sSrc2 = reinterpret_cast<float*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_float4_t *vDst, rccl_float4_t *vSrc1, rccl_float4_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_float4_t src1 = vSrc1[i];
        rccl_float4_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_float4_t)/sizeof(float));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    float *sDst = reinterpret_cast<float*>(vDst + lenVec);
    float *sSrc1 = reinterpret_cast<float*>(vSrc1 + lenVec);
    float *sSrc2 = reinterpret_cast<float*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}





__global__ void rcclKernelReduceSum(rccl_long2_t *vDst, rccl_long2_t *vSrc1, rccl_long2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_long2_t src1 = vSrc1[i];
        rccl_long2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_long2_t)/sizeof(signed long));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    signed long *sDst = reinterpret_cast<signed long*>(vDst + lenVec);
    signed long *sSrc1 = reinterpret_cast<signed long*>(vSrc1 + lenVec);
    signed long *sSrc2 = reinterpret_cast<signed long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_long2_t *vDst, rccl_long2_t *vSrc1, rccl_long2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_long2_t src1 = vSrc1[i];
        rccl_long2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_long2_t)/sizeof(signed long));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    signed long *sDst = reinterpret_cast<signed long*>(vDst + lenVec);
    signed long *sSrc1 = reinterpret_cast<signed long*>(vSrc1 + lenVec);
    signed long *sSrc2 = reinterpret_cast<signed long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_long2_t *vDst, rccl_long2_t *vSrc1, rccl_long2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_long2_t src1 = vSrc1[i];
        rccl_long2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_long2_t)/sizeof(signed long));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    signed long *sDst = reinterpret_cast<signed long*>(vDst + lenVec);
    signed long *sSrc1 = reinterpret_cast<signed long*>(vSrc1 + lenVec);
    signed long *sSrc2 = reinterpret_cast<signed long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_long2_t *vDst, rccl_long2_t *vSrc1, rccl_long2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_long2_t src1 = vSrc1[i];
        rccl_long2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_long2_t)/sizeof(signed long));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    signed long *sDst = reinterpret_cast<signed long*>(vDst + lenVec);
    signed long *sSrc1 = reinterpret_cast<signed long*>(vSrc1 + lenVec);
    signed long *sSrc2 = reinterpret_cast<signed long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}



__global__ void rcclKernelReduceSum(rccl_ulong2_t *vDst, rccl_ulong2_t *vSrc1, rccl_ulong2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ulong2_t src1 = vSrc1[i];
        rccl_ulong2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ulong2_t)/sizeof(unsigned long));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    unsigned long *sDst = reinterpret_cast<unsigned long*>(vDst + lenVec);
    unsigned long *sSrc1 = reinterpret_cast<unsigned long*>(vSrc1 + lenVec);
    unsigned long *sSrc2 = reinterpret_cast<unsigned long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_ulong2_t *vDst, rccl_ulong2_t *vSrc1, rccl_ulong2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ulong2_t src1 = vSrc1[i];
        rccl_ulong2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ulong2_t)/sizeof(unsigned long));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    unsigned long *sDst = reinterpret_cast<unsigned long*>(vDst + lenVec);
    unsigned long *sSrc1 = reinterpret_cast<unsigned long*>(vSrc1 + lenVec);
    unsigned long *sSrc2 = reinterpret_cast<unsigned long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_ulong2_t *vDst, rccl_ulong2_t *vSrc1, rccl_ulong2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ulong2_t src1 = vSrc1[i];
        rccl_ulong2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ulong2_t)/sizeof(unsigned long));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned long *sDst = reinterpret_cast<unsigned long*>(vDst + lenVec);
    unsigned long *sSrc1 = reinterpret_cast<unsigned long*>(vSrc1 + lenVec);
    unsigned long *sSrc2 = reinterpret_cast<unsigned long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_ulong2_t *vDst, rccl_ulong2_t *vSrc1, rccl_ulong2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_ulong2_t src1 = vSrc1[i];
        rccl_ulong2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_ulong2_t)/sizeof(unsigned long));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    unsigned long *sDst = reinterpret_cast<unsigned long*>(vDst + lenVec);
    unsigned long *sSrc1 = reinterpret_cast<unsigned long*>(vSrc1 + lenVec);
    unsigned long *sSrc2 = reinterpret_cast<unsigned long*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}



__global__ void rcclKernelReduceSum(rccl_double2_t *vDst, rccl_double2_t *vSrc1, rccl_double2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_double2_t src1 = vSrc1[i];
        rccl_double2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_double2_t)/sizeof(double));j++) {
            vDst[i][j] = src1[j] + src2[j];
        }
    }

    double *sDst = reinterpret_cast<double*>(vDst + lenVec);
    double *sSrc1 = reinterpret_cast<double*>(vSrc1 + lenVec);
    double *sSrc2 = reinterpret_cast<double*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] + sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceProd(rccl_double2_t *vDst, rccl_double2_t *vSrc1, rccl_double2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_double2_t src1 = vSrc1[i];
        rccl_double2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_double2_t)/sizeof(double));j++) {
            vDst[i][j] = src1[j] * src2[j];
        }
    }

    double *sDst = reinterpret_cast<double*>(vDst + lenVec);
    double *sSrc1 = reinterpret_cast<double*>(vSrc1 + lenVec);
    double *sSrc2 = reinterpret_cast<double*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] * sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMax(rccl_double2_t *vDst, rccl_double2_t *vSrc1, rccl_double2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_double2_t src1 = vSrc1[i];
        rccl_double2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_double2_t)/sizeof(double));j++) {
            vDst[i][j] = src1[j] > src2[j] ? src1[j] : src2[j];
        }
    }

    double *sDst = reinterpret_cast<double*>(vDst + lenVec);
    double *sSrc1 = reinterpret_cast<double*>(vSrc1 + lenVec);
    double *sSrc2 = reinterpret_cast<double*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] > sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}

__global__ void rcclKernelReduceMin(rccl_double2_t *vDst, rccl_double2_t *vSrc1, rccl_double2_t *vSrc2, size_t lenVec, size_t off) {
    size_t tx = hipThreadIdx_x;
    for(size_t i=tx; i<lenVec;i+=WI) {
        rccl_double2_t src1 = vSrc1[i];
        rccl_double2_t src2 = vSrc2[i];
        for(int j=0;j<(sizeof(rccl_double2_t)/sizeof(double));j++) {
            vDst[i][j] = src1[j] < src2[j] ? src1[j] : src2[j];
        }
    }

    double *sDst = reinterpret_cast<double*>(vDst + lenVec);
    double *sSrc1 = reinterpret_cast<double*>(vSrc1 + lenVec);
    double *sSrc2 = reinterpret_cast<double*>(vSrc2 + lenVec);

    if(tx < off) {
        sDst[tx] = sSrc1[tx] < sSrc2[tx] ? sSrc1[tx] : sSrc2[tx];
    }
    __syncthreads();
}




