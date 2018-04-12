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

constexpr unsigned iter = 1024;
constexpr size_t length_2 = 2 * (1 << 15);
constexpr size_t length_1 = (1 << 15);
constexpr size_t size_2 = length_2 * sizeof(Float4);
constexpr size_t size_1 = length_1 * sizeof(Float4);

int main() {
    Float4 *Ad1, *Bd1, *Cd1;
    Float4 *Ad2, *Bd2, *Cd2;
    hipMalloc(&Ad2, size_2);
    hipMalloc(&Bd2, size_2);
    hipMalloc(&Cd1, size_1);
    hipMalloc(&Ad1, size_1);
    hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1);
    hipDeviceEnablePeerAccess(0, 0);
    hipMalloc(&Cd2, size_2);
    hipMalloc(&Bd1, size_1);
    hipSetDevice(0);

    hipStream_t stream1, stream2;

    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);

    auto start = std::chrono::high_resolution_clock::now();

    for(int i=0;i<iter;i++) {
        hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, 0, Cd2, Ad2, Bd2, length_2);
    }
    hipDeviceSynchronize();

    auto stop = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    double algbw = ((double)(size_2) * (double)(iter)) / 1.0E9 / elapsedSec;
    std::cout<<algbw<<" "<<elapsedSec<<std::endl;

    start = std::chrono::high_resolution_clock::now();

    for(int i=0;i<iter;i++) {
    hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, 0, Cd1, Ad1, Bd1, length_1);
    }
    hipDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    algbw = ((double)(size_1) * (double)(iter)) / 1.0E9 / elapsedSec;
    std::cout<<algbw<<" "<<elapsedSec<<std::endl;


    start = std::chrono::high_resolution_clock::now();

    for(int i=0;i<iter;i++) {
    hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, stream1, Cd2, Ad2, Bd2, length_2);
    hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, stream2, Cd1, Ad1, Bd1, length_1);
    }
    hipDeviceSynchronize();

    stop = std::chrono::high_resolution_clock::now();
    elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

    algbw = ((double)(size_1 + size_2) * (double)(iter)) / 1.0E9 / elapsedSec;
    std::cout<<algbw<<" "<<elapsedSec<<std::endl;

}

