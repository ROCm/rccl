/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include "rcclKernels.h"
#include "performance.h"
#include "common.h"
#include "counts.h"

constexpr size_t iter = 128;

template<typename VectorType, typename DataType>
inline double launchCopy(size_t length, int dstDevice, int srcDevice) {

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

    hipLaunchKernelGGL((rcclKernelCopy<VectorType, DataType>), dim3(1,1,1), dim3(WI,1,1), 0, 0, dDst, dSrc, size_t(length/numElements), size_t(length%numElements));

    perf_marker mark;

    for(size_t i=0;i<iter;i++) {
        hipLaunchKernelGGL((rcclKernelCopy<VectorType, DataType>), dim3(1,1,1), dim3(WI,1,1), 0, 0, dDst, dSrc, size_t(length/numElements), size_t(length%numElements));
    }

    HIPCHECK(hipDeviceSynchronize());

    mark.done();

    HIPCHECK(hipFree(dSrc));
    HIPCHECK(hipFree(dDst));

    return mark.get_bw(size * iter);
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

    std::cout<<"Count: "<<" Char "<<" Uchar "<<" Short "<<" Ushort "<<" Int "<<" Uint "<<" Long "<<" Ulong "<<" Half "<<" Float "<<" Double "<<std::endl;
    for(auto &count: counts) {
            std::cout<<count<<" ";
            std::cout<<launchCopy<rccl_char16_t, signed char>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_uchar16_t, unsigned char>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_short8_t, signed short>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_ushort8_t, unsigned short>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_int4_t, signed int>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_uint4_t, unsigned int>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_long2_t, signed long>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_ulong2_t, unsigned long>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_half8_t, rccl_half_t>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_float4_t, float>(count, dstDevice, srcDevice)<<" ";
            std::cout<<launchCopy<rccl_double2_t, double>(count, dstDevice, srcDevice)<<" ";
            std::cout<<std::endl;
    }
}
