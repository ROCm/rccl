/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl.h"
#include "rcclCheck.h"
#include <iostream>
#include <vector>
#include "performance.h"

typedef int T;

#define LEN  (1 << 22)
#define SIZE LEN * sizeof(T)

int main(int argc, char* argv[]) {
    if(argc < 2) {
        std::cout<<"Usage: ./a.out <num gpus>"<<std::endl;
        return 0;
    }

    int numGpus = atoi(argv[1]);
    std::vector<int> devs(numGpus);
    for(int i=0;i<numGpus;i++) {
        devs[i] = i;
    }

    T* hSrc = new T[LEN];
    for(int i=0;i<LEN;i++) {
        hSrc[i] = 1;
    }

    T** hDst = new T*[numGpus];
    for(int i=0;i<numGpus;i++) {
        hDst[i] = new T[LEN];
        memset(hDst[i], SIZE, 0);
    }

    T *dSrc, **dDst;
    dDst = new T*[numGpus];

    std::vector<rcclComm_t> comms(numGpus);
    RCCLCHECK(rcclCommInitAll(comms.data(), numGpus, devs.data()));

    std::vector<hipStream_t> streams(numGpus);

    int root = 0;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamCreate(&streams[i]));
        if(i==root) {
            HIPCHECK(hipMalloc(&dDst[i], SIZE));
            HIPCHECK(hipMemcpy(dDst[i], hSrc, SIZE, hipMemcpyHostToDevice));
        } else {
            HIPCHECK(hipMalloc(&dDst[i], SIZE));
            HIPCHECK(hipMemcpy(dDst[i], hDst[i], SIZE, hipMemcpyHostToDevice));
        }

        for(int j=0;j<numGpus;j++) {
            if(i!=j) {
                HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }


    perf_marker mark;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclBcast(dDst[i], LEN, rcclFloat, root, comms[i], streams[i]));
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }
    mark.done();
    mark.bw(SIZE);

}
