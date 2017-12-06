/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl.h"
#include "rcclCheck.h"
#include "performance.h"
#include <iostream>
#include <vector>

typedef int T;

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
        hDst[i] = new T[LEN*numGpus];
        memset(hDst[i], SIZE*numGpus, 0);
    }

    T **dSrc, **dDst;
    dDst = new T*[numGpus];
    dSrc = new T*[numGpus];

    std::vector<rcclComm_t> comms(numGpus);
    RCCLCHECK(rcclCommInitAll(comms.data(), numGpus, devs.data()));

    std::vector<hipStream_t> streams(numGpus);

    int root = 0;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamCreate(&streams[i]));
        HIPCHECK(hipMalloc(&dSrc[i], SIZE));
        HIPCHECK(hipMemcpy(dSrc[i], hSrc[i], SIZE, hipMemcpyHostToDevice));
        HIPCHECK(hipMalloc(&dDst[i], SIZE*numGpus));
        HIPCHECK(hipMemcpy(dDst[i], hDst[i], SIZE*numGpus, hipMemcpyHostToDevice));

        for(int j=0;j<numGpus;j++) {
            if(i!=j) {
                HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }


    perf_marker mark;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclAllGather(dSrc[i], LEN, rcclInt, dDst[i], comms[i], streams[i]));
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }
    mark.done();
    mark.bw(SIZE);

}
