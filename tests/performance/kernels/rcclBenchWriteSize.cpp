/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
Benchmark write and reads over pcie
*/

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <fstream>

typedef float Float4 __attribute__((ext_vector_type(4)));

constexpr int WI = 1024;

__global__ void Reduce(Float4 *C, Float4 *A, Float4 *B, size_t len) {
    int tx = hipThreadIdx_x;
    for(int i=tx;i<len;i=i+WI) {
        C[i] = A[i] + B[i];
    }
}

constexpr unsigned iter = 1024;

int main() {
    Float4 *Ad2, *Bd2, *Cd2;
    std::ofstream outfile;
    outfile.open("outfile.txt");

    for(size_t j=1;j<1024*1024*1024;j+=64) {
        hipMalloc(&Ad2, j * sizeof(Float4));
        hipMalloc(&Bd2, j * sizeof(Float4));
        hipDeviceEnablePeerAccess(1, 0);
        hipSetDevice(1);
        hipDeviceEnablePeerAccess(0, 0);
        hipMalloc(&Cd2, j * sizeof(Float4));
        hipSetDevice(0);

        auto start = std::chrono::high_resolution_clock::now();

        for(int i=0;i<iter;i++) {
            hipLaunchKernelGGL(Reduce, dim3(1,1,1), dim3(WI,1,1), 0, 0, Cd2, Ad2, Bd2, j);
        }
        hipDeviceSynchronize();

        auto stop = std::chrono::high_resolution_clock::now();
        double elapsedSec = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count();

        double algbw = ((double)(j * sizeof(Float4)) * (double)(iter)) / 1.0E9 / elapsedSec;

        outfile<<j*sizeof(Float4)<<" "<<algbw<<"\n";
        hipFree(Ad2);
        hipFree(Bd2);
        hipFree(Cd2);
    }
    outfile.close();
}

