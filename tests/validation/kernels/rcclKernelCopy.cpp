/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "rcclKernels.h"
#include "common.h"
#include "validate.h"

constexpr size_t iter = 128;

template<typename VectorType, typename DataType>
inline void launchCopy(size_t length, int dstDevice, int srcDevice) {

    constexpr unsigned numElements = sizeof(VectorType) / sizeof(DataType);

    VectorType *dSrc, *dDst;
    std::vector<DataType> hSrc(length);
    std::vector<DataType> hDst(length);

    size_t size = sizeof(DataType) * length;

    HIPCHECK(hipSetDevice(dstDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(srcDevice, 0));
    HIPCHECK(hipMalloc(&dDst, size));
    HIPCHECK(hipMemcpy(dDst, hDst.data(), size, hipMemcpyHostToDevice));

    HIPCHECK(hipSetDevice(srcDevice));
    HIPCHECK(hipDeviceEnablePeerAccess(dstDevice, 0));
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
    if(argc != 5) {
        std::cerr<<"Usage: ./a.out <number of elements> <rcclDataType_t> <dst gpu> <src gpu>"<<std::endl;
        std::cerr<<"\n\
Example: ./a.out 123456 0 1 2       \n\
Does "<<__FILE__<<" of size 123456 bytes \n\
of rcclChar/rcclInt8 from GPU 2 to GPU 1"<<std::endl;
        return 0;
    }

    size_t count = atoi(argv[1]);
    int dataType = atoi(argv[2]);
    if(dataType > 10 || dataType < 0) {
        std::cerr<<"Bad Datatype requested. Use from 0 to 10"<<std::endl;
        return 0;
    }

    int dstDevice = atoi(argv[3]);
    int srcDevice = atoi(argv[4]);

    switch(dataType) {
        case 0:
            launchCopy<rccl_char16_t, signed char>(count, dstDevice, srcDevice);
            return 0;
        case 1:
            launchCopy<rccl_uchar16_t, unsigned char>(count, dstDevice, srcDevice);
            return 0;
        case 2:
            launchCopy<rccl_short8_t, signed short>(count, dstDevice, srcDevice);
            return 0;
        case 3:
            launchCopy<rccl_ushort8_t, unsigned short>(count, dstDevice, srcDevice);
            return 0;
        case 4:
            launchCopy<rccl_int4_t, signed int>(count, dstDevice, srcDevice);
            return 0;
        case 5:
            launchCopy<rccl_uint4_t, unsigned int>(count, dstDevice, srcDevice);
            return 0;
        case 6:
            launchCopy<rccl_long2_t, signed long>(count, dstDevice, srcDevice);
            return 0;
        case 7:
            launchCopy<rccl_ulong2_t, unsigned long>(count, dstDevice, srcDevice);
            return 0;
        case 8:
            launchCopy<rccl_half8_t, rccl_half_t>(count, dstDevice, srcDevice);
            return 0;
        case 9:
            launchCopy<rccl_float4_t, float>(count, dstDevice, srcDevice);
            return 0;
        case 10:
            launchCopy<rccl_double2_t, double>(count, dstDevice, srcDevice);
            return 0;

        default:
            return 0;
    }
}

