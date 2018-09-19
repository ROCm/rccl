/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl/rccl.h"
#include <iostream>
#include <vector>
//#include "rcclCheck.h"
#include "validate.h"
#include "common.h"

typedef int T;

#define LEN  (1<<21)
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

    rcclUniqueId id2;
/*
    RCCLCHECK(rcclGetUniqueId(&id1));
    std::vector<rcclComm_t> comms1(numGpus);
    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclCommInitRank(&comms1[i], numGpus, id1, i));
    }
*/
    RCCLCHECK(rcclGetUniqueId(&id2));
    std::vector<rcclComm_t> comms2(numGpus);
    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclCommInitRank(&comms2[i], numGpus, id2, i));
    }

    HIPCHECK(hipSetDevice(devs[0]));

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

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        RCCLCHECK(rcclBcast(dDst[i], LEN, rcclFloat, root, comms2[i], streams[i]));
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipMemcpy(hDst[i], dDst[i], SIZE, hipMemcpyDeviceToHost));
    }

    for(int i=0;i<numGpus;i++) {
        std::cout<<"Validating on GPU: "<<devs[i]<<std::endl;
        validate(hDst[i], 1, LEN, 5);
    }
}
