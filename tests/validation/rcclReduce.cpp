/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#include "rccl.h"
#include "rcclCheck.h"
#include <iostream>
#include <vector>
#include "validate.h"
#include <typeinfo>

template<typename T>
void doReduce(T *src, T *dst, size_t size, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream);


template<>
void doReduce(char *src, char *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclChar, op, comm, stream));
}

template<>
void doReduce(unsigned char *src, unsigned char *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUchar, rcclSum, comm, stream));
}

template<>
void doReduce(short *src, short *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclShort, op, comm, stream));
}

template<>
void doReduce(unsigned short *src, unsigned short *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUshort, op, comm, stream));
}

template<>
void doReduce(int *src, int *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclInt, op, comm, stream));
}

template<>
void doReduce(unsigned int *src, unsigned int *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUint, op, comm, stream));
}

template<>
void doReduce(long *src, long *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(src, dst, LEN, rcclLong, op, root, comm, stream));
}

template<>
void doReduce(unsigned long *src, unsigned long *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(src, dst, LEN, rcclUlong, op, root, comm, stream));
}

template<>
void doReduce(float *src, float *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(src, dst, LEN, rcclFloat, op, root, comm, stream));
}

template<>
void doReduce(double *src, double *dst, size_t LEN, rcclRedOp_t op, int root, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclReduce(src, dst, LEN, rcclDouble, op, root, comm, stream));
}


template<typename T>
void RunTest(size_t LEN, int numGpus, std::vector<int>& devs, int root) {
    size_t SIZE = LEN * sizeof(T);
    std::cout<<"Total Size on each GPU: "<<((double)2*SIZE)/(1024*1024)<<" MB"<<std::endl;

    T** hSrc = new T*[numGpus];
    for(int i=0;i<numGpus;i++) {
        hSrc[i] = new T[LEN];
        for(int j=0;j<LEN;j++) {
            hSrc[i][j] = 1;
        }
    }

    T* hDst = new T[LEN];
    memset(hDst, SIZE, 0);

    T **dSrc, *dDst;
    dSrc = new T*[numGpus];

    std::vector<rcclComm_t> comms(numGpus);
    RCCLCHECK(rcclCommInitAll(comms.data(), numGpus, devs.data()));

    std::vector<hipStream_t> streams(numGpus);


    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamCreate(&streams[i]));

        HIPCHECK(hipMalloc(&dSrc[i], SIZE));
        HIPCHECK(hipMemcpy(dSrc[i], hSrc[i], SIZE, hipMemcpyHostToDevice));

        if(i == root) {
            HIPCHECK(hipMalloc(&dDst, SIZE));
            HIPCHECK(hipMemcpy(dDst, hDst, SIZE, hipMemcpyHostToDevice));
        }

    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        doReduce<T>(dSrc[i], dDst, LEN, rcclSum, root, comms[i], streams[i]);
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }

    HIPCHECK(hipSetDevice(root));
    HIPCHECK(hipMemcpy(hDst, dDst, SIZE, hipMemcpyDeviceToHost));
    HIPCHECK(hipFree(dDst));
    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipFree(dSrc[i]));
    }

    std::cout<<"Validating..."<<std::endl;

    std::cout<<"Validating on GPU: "<<root<<std::endl;
    validate(hDst, T(numGpus), LEN, 1);

}

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

    int root = 0;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(devs[i]));
        for(int j=0;j<numGpus;j++) {
            if(i!=j) {
                HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }

    size_t LEN = atoi(argv[2]);
    RunTest<char>(LEN, numGpus, devs, root);
    RunTest<unsigned char>(LEN, numGpus, devs, root);
    RunTest<short>(LEN, numGpus, devs, root);
    RunTest<unsigned short>(LEN, numGpus, devs, root);
    RunTest<int>(LEN, numGpus, devs, root);
    RunTest<unsigned int>(LEN, numGpus, devs, root);
    RunTest<long>(LEN, numGpus, devs, root);
    RunTest<unsigned long>(LEN, numGpus, devs, root);

    RunTest<float>(LEN, numGpus, devs, root);
    RunTest<double>(LEN, numGpus, devs, root);
}
