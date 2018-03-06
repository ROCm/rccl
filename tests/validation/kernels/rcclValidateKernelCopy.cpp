/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "rcclKernels.h"
#include "common.h"
#include "counts.h"
#include "validate.h"

template<typename VectorType, typename DataType>
inline void launchCopy(size_t length, int dstDevice, int srcDevice) {

    constexpr unsigned numElements = sizeof(VectorType) / sizeof(DataType);

    VectorType *dSrc, *dDst;
    std::vector<DataType> hSrc(length);
    std::vector<DataType> hDst(length);

    size_t size = sizeof(DataType) * length;

    HIPCHECK(hipSetDevice(dstDevice));
    HIPCHECK(hipMalloc(&dDst, size));
    HIPCHECK(hipMemcpy(dDst, hDst.data(), size, hipMemcpyHostToDevice));

    HIPCHECK(hipSetDevice(srcDevice));
    HIPCHECK(hipMalloc(&dSrc, size));
    HIPCHECK(hipMemcpy(dSrc, hSrc.data(), size, hipMemcpyHostToDevice));

    HIPCHECK(hipSetDevice(srcDevice));

    hipLaunchKernelGGL((rcclKernelCopy<VectorType, DataType>), dim3(1,1,1), dim3(WI,1,1), 0, 0, dDst, dSrc, length/numElements, length%numElements);

    HIPCHECK(hipDeviceSynchronize());

    HIPCHECK(hipSetDevice(dstDevice));
    HIPCHECK(hipMemcpy(hDst.data(), dDst, size, hipMemcpyDeviceToHost));

    validate(hDst.data(), hSrc.data(), length, 1, 0);

    HIPCHECK(hipFree(dSrc));
    HIPCHECK(hipFree(dDst));
}

int main(int argc, char* argv[]){
    if(argc != 3) {
        std::cerr<<"Usage: ./a.out <dst gpu> <src gpu>"<<std::endl;
        return 0;
    }

    int dstDevice = atoi(argv[1]);
    int srcDevice = atoi(argv[2]);

    HIPCHECK(hipSetDevice(dstDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(srcDevice, 0));

    HIPCHECK(hipSetDevice(srcDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(dstDevice, 0));

    for(auto &count: counts) {
        launchCopy<rccl_char16_t, signed char>(count, dstDevice, srcDevice);
        launchCopy<rccl_uchar16_t, unsigned char>(count, dstDevice, srcDevice);
        launchCopy<rccl_short8_t, signed short>(count, dstDevice, srcDevice);
        launchCopy<rccl_ushort8_t, unsigned short>(count, dstDevice, srcDevice);
        launchCopy<rccl_int4_t, signed int>(count, dstDevice, srcDevice);
        launchCopy<rccl_uint4_t, unsigned int>(count, dstDevice, srcDevice);
        launchCopy<rccl_long2_t, signed long>(count, dstDevice, srcDevice);
        launchCopy<rccl_ulong2_t, unsigned long>(count, dstDevice, srcDevice);
        launchCopy<rccl_half8_t, rccl_half_t>(count, dstDevice, srcDevice);
        launchCopy<rccl_float4_t, float>(count, dstDevice, srcDevice);
        launchCopy<rccl_double2_t, double>(count, dstDevice, srcDevice);
    }
}

