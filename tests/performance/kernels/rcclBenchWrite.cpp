/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
Benchmark write and reads over pcie
*/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

typedef float Float4 __attribute__((ext_vector_type(4)));

constexpr int WI = 1024;

__global__ void Reduce(Float4 *C, Float4 *A, Float4 *B, size_t len) {
    int tx = hipThreadIdx_x;
    for(int i=tx;i<len;i=i+WI) {
        C[i] = A[i] + B[i];
    }
}

constexpr unsigned iter = 128;
constexpr size_t length = 1 << 20;
constexpr size_t size = length * sizeof(Float4);

int main() {
    Float4 *Ad1, *Bd1, *Cd1;
    Float4 *Ad2, *Bd2, *Cd2;
    hipMalloc(&Ad1, size);
    hipMalloc(&Bd1, size);
    hipMalloc(&Cd2, size);
    hipMalloc(&Ad2, size);
    hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1);
    hipDeviceEnablePeerAccess(0, 0);
    hipMalloc(&Cd1, size);
    hipMalloc(&Bd2, size);
    hipSetDevice(0);

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0;i<iter;i++) {
    hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, 0, Cd1, Ad1, Bd1, length);
    }
    hipDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    double algbw = ((double)(size) * (double)(iter)) / 1.0E9 / elapsedSec;
    std::cout<<algbw<<std::endl;

    start = std::chrono::high_resolution_clock::now();

    for(int i=0;i<iter;i++) {
    hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, 0, Cd2, Ad2, Bd2, length);
    }
    hipDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    algbw = ((double)(size) * (double)(iter)) / 1.0E9 / elapsedSec;
    std::cout<<algbw<<std::endl;

}

