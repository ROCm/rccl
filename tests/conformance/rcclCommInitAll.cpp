/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#include "rccl.h"
#include "rcclCheck.h"

int main() {
    int devCnt;
    HIPCHECK(hipGetDeviceCount(&devCnt));
    int *devList = new int[devCnt];
    rcclComm_t *commList = new rcclComm_t[devCnt];
    for(int i=0;i<devCnt;i++) {
        devList[i] = i;
    }
    if(rcclCommInitAll(commList, devCnt, devList) == rcclDeviceNotFound) {
        std::cout<<"Success"<<std::endl;
    }else{
        std::cout<<"Failed"<<std::endl;
    }
}
