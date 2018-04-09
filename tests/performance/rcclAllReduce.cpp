/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl.h"
#include <iostream>
#include "performance.h"
#include <vector>
#include "rcclCheck.h"

typedef float T;

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cout<<"Usage: ./a.out <num gpus> <length of buffer in ints>"<<std::endl;
        return 0;
    }

    int numGpus = atoi(argv[1]);
    std::vector<int> devs(numGpus);
    for(int i=0;i<numGpus;i++) {
        devs[i] = i;
    }

    size_t LEN = atoi(argv[2]);
    size_t SIZE = LEN * sizeof(T);

    T** hSrc = new T*[numGpus];
    for(int j=0;j<numGpus;j++) {
        hSrc[j] = new T[LEN];
        for(int i=0;i<LEN;i++) {
            hSrc[j][i] = 1;
        }
    }

    T** hDst = new T*[numGpus];
    for(int i=0;i<numGpus;i++) {
        hDst[i] = new T[LEN];
        memset(hDst[i], SIZE, 0);
    }

    T **dSrc, **dDst;
    dDst = new T*[numGpus];
    dSrc = new T*[numGpus];

    std::vector<hipStream_t> streams(numGpus);

    int root = 0;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        for(int j=0;j<numGpus;j++) {
            if(i!=j) {
                HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
            }
        }

        HIPCHECK(hipStreamCreate(&streams[i]));
        HIPCHECK(hipMalloc(&dSrc[i], SIZE));
        HIPCHECK(hipMemcpy(dSrc[i], hSrc[i], SIZE, hipMemcpyHostToDevice));
        HIPCHECK(hipMalloc(&dDst[i], SIZE));
        HIPCHECK(hipMemcpy(dDst[i], hDst[i], SIZE, hipMemcpyHostToDevice));
    }

    std::vector<rcclComm_t> comms(numGpus);
    RCCLCHECK(rcclCommInitAll(comms.data(), numGpus, devs.data()));

    perf_marker mark;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclAllReduce(dSrc[i], dDst[i], LEN, rcclFloat, rcclSum, comms[i], streams[i]));
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }
    mark.done();
    mark.bw(SIZE);


    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipFree(dDst[i]));
        HIPCHECK(hipFree(dSrc[i]));
        delete hDst[i];
        delete hSrc[i];
    }
}
