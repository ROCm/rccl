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
void doAllReduce(T *src, T *dst, size_t size, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream);


template<>
void doAllReduce(char *src, char *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclChar, op, comm, stream));
}

template<>
void doAllReduce(unsigned char *src, unsigned char *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUchar, op, comm, stream));
}

template<>
void doAllReduce(short *src, short *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclShort, op, comm, stream));
}

template<>
void doAllReduce(unsigned short *src, unsigned short *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUshort, op, comm, stream));
}

template<>
void doAllReduce(int *src, int *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclInt, op, comm, stream));
}

template<>
void doAllReduce(unsigned int *src, unsigned int *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUint, op, comm, stream));
}

template<>
void doAllReduce(signed long *src, signed long *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclLong, op, comm, stream));
}

template<>
void doAllReduce(unsigned long *src, unsigned long *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclUlong, op, comm, stream));
}

template<>
void doAllReduce(float *src, float *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclFloat, op, comm, stream));
}

template<>
void doAllReduce(double *src, double *dst, size_t LEN, rcclRedOp_t op, rcclComm_t comm, hipStream_t stream) {
    RCCLCHECK(rcclAllReduce(src, dst, LEN, rcclDouble, op, comm, stream));
}


template<typename T>
void RunTest(size_t LEN, int numGpus, std::vector<int>& devs, rcclRedOp_t op) {
    size_t SIZE = LEN * sizeof(T);
    std::cout<<"Total Size on each GPU: "<<((double)2*SIZE)/(1024*1024)<<" MB"<<std::endl;

    T** hSrc = new T*[numGpus];
    for(int i=0;i<numGpus;i++) {
        hSrc[i] = new T[LEN];
        for(int j=0;j<LEN;j++) {
            hSrc[i][j] = T(2);
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

    std::vector<rcclComm_t> comms(numGpus);
    RCCLCHECK(rcclCommInitAll(comms.data(), numGpus, devs.data()));
    std::vector<hipStream_t> streams(numGpus);

    int root = 0;

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipStreamCreate(&streams[i]));

        HIPCHECK(hipMalloc(&dSrc[i], SIZE));
        HIPCHECK(hipMalloc(&dDst[i], SIZE));
        HIPCHECK(hipMemcpy(dSrc[i], hSrc[i], SIZE, hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(dDst[i], hDst[i], SIZE, hipMemcpyHostToDevice));
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        doAllReduce<T>(dSrc[i], dDst[i], LEN, op, comms[i], streams[i]);
    }

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipDeviceSynchronize());
    }


    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(i));
        HIPCHECK(hipMemcpy(hDst[i], dDst[i], SIZE, hipMemcpyDeviceToHost));
    }

    std::cout<<"Validating..."<<std::endl;

    if(op == rcclSum) {
    for(int i=0;i<numGpus;i++) {
        std::cout<<"Validating on GPU: "<<devs[i]<<std::endl;
        validate(hDst[i], T(numGpus*2), LEN, 1, 5);
    }
    }
    if(op == rcclProd) {
    T val = T(1);
    for(int i=0;i<numGpus;i++) {
        val = val * T(2);
    }
    for(int i=0;i<numGpus;i++) {
        std::cout<<"Validating on GPU: "<<devs[i]<<std::endl;
        validate(hDst[i], val, LEN, 1, 5);
    }
    }

    if(op == rcclMax) {
    for(int i=0;i<numGpus;i++) {
        std::cout<<"Validating on GPU: "<<devs[i]<<std::endl;
        validate(hDst[i], T(2), LEN, 1, 5);
    }
    }

    if(op == rcclMin) {
    for(int i=0;i<numGpus;i++) {
        std::cout<<"Validating on GPU: "<<devs[i]<<std::endl;
        validate(hDst[i], T(2), LEN, 1, 5);
    }
    }


    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipFree(dDst[i]));
        HIPCHECK(hipFree(dSrc[i]));
    }

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

    for(int i=0;i<numGpus;i++) {
        HIPCHECK(hipSetDevice(devs[i]));
        for(int j=0;j<numGpus;j++) {
            if(i!=j) {
                HIPCHECK(hipDeviceEnablePeerAccess(j, 0));
            }
        }
    }

    size_t LEN = atoi(argv[2]);

    std::cout<<"Testing: "<<"char"<<std::endl;
    RunTest<char>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"unsigned char"<<std::endl;
    RunTest<unsigned char>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"short"<<std::endl;
    RunTest<short>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"unsigned short"<<std::endl;
    RunTest<unsigned short>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"int"<<std::endl;
    RunTest<int>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"unsigned int"<<std::endl;
    RunTest<unsigned int>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"long"<<std::endl;
    RunTest<signed long>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"unsigned long"<<std::endl;
    RunTest<unsigned long>(LEN, numGpus, devs, rcclSum);

    std::cout<<"Testing: "<<"float"<<std::endl;
    RunTest<float>(LEN, numGpus, devs, rcclSum);
    std::cout<<"Testing: "<<"double"<<std::endl;
    RunTest<double>(LEN, numGpus, devs, rcclSum);


    std::cout<<"Testing: "<<"char"<<std::endl;
    RunTest<char>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"unsigned char"<<std::endl;
    RunTest<unsigned char>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"short"<<std::endl;
    RunTest<short>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"unsigned short"<<std::endl;
    RunTest<unsigned short>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"int"<<std::endl;
    RunTest<int>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"unsigned int"<<std::endl;
    RunTest<unsigned int>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"long"<<std::endl;
    RunTest<signed long>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"unsigned long"<<std::endl;
    RunTest<unsigned long>(LEN, numGpus, devs, rcclProd);

    std::cout<<"Testing: "<<"float"<<std::endl;
    RunTest<float>(LEN, numGpus, devs, rcclProd);
    std::cout<<"Testing: "<<"double"<<std::endl;
    RunTest<double>(LEN, numGpus, devs, rcclProd);


    std::cout<<"Testing: "<<"char"<<std::endl;
    RunTest<char>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"unsigned char"<<std::endl;
    RunTest<unsigned char>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"short"<<std::endl;
    RunTest<short>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"unsigned short"<<std::endl;
    RunTest<unsigned short>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"int"<<std::endl;
    RunTest<int>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"unsigned int"<<std::endl;
    RunTest<unsigned int>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"long"<<std::endl;
    RunTest<signed long>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"unsigned long"<<std::endl;
    RunTest<unsigned long>(LEN, numGpus, devs, rcclMax);

    std::cout<<"Testing: "<<"float"<<std::endl;
    RunTest<float>(LEN, numGpus, devs, rcclMax);
    std::cout<<"Testing: "<<"double"<<std::endl;
    RunTest<double>(LEN, numGpus, devs, rcclMax);



    std::cout<<"Testing: "<<"char"<<std::endl;
    RunTest<char>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"unsigned char"<<std::endl;
    RunTest<unsigned char>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"short"<<std::endl;
    RunTest<short>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"unsigned short"<<std::endl;
    RunTest<unsigned short>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"int"<<std::endl;
    RunTest<int>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"unsigned int"<<std::endl;
    RunTest<unsigned int>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"long"<<std::endl;
    RunTest<signed long>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"unsigned long"<<std::endl;
    RunTest<unsigned long>(LEN, numGpus, devs, rcclMin);

    std::cout<<"Testing: "<<"float"<<std::endl;
    RunTest<float>(LEN, numGpus, devs, rcclMin);
    std::cout<<"Testing: "<<"double"<<std::endl;
    RunTest<double>(LEN, numGpus, devs, rcclMin);


}
