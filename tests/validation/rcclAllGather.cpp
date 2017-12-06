/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl.h"
#include <iostream>
#include <vector>
#include "rcclCheck.h"
#include "validate.h"

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

    T** hSrc = new T*[numGpus];
    for(int i=0;i<numGpus;i++) {
        hSrc[i] = new T[LEN];
        for(int j=0;j<LEN;j++){
            hSrc[i][j] = 1;
        }
    }

    T** hDst = new T*[numGpus];
    for(int i=0;i<numGpus;i++) {
        hDst[i] = new T[LEN*numGpus];
        memset(hDst[i], SIZE*numGpus, 0);
    }

    T **dSrc, **dDst;
    dSrc = new T*[numGpus];
    dDst = new T*[numGpus];

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

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclAllGather(dSrc[i], LEN, rcclInt, dDst[i], comms[i], streams[i]));
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipMemcpy(hDst[i], dDst[i], SIZE*numGpus, hipMemcpyDeviceToHost));
    }

    for(int i=0;i<numGpus;i++) {
        std::cout<<"Validating on GPU: "<<devs[i]<<std::endl;
        validate(hDst[i], hSrc[i], LEN, 5);
    }
}
