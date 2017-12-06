/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

#pragma once

#include "rccl.h"

#define HIPCHECK(status) \
    if(status != hipSuccess) { \
        std::cout<<"Got: "<<hipGetErrorString(status)<<" at: "<<__LINE__<<" in file: "<<__FILE__<<std::endl; \
    }



#define RCCLCHECK(status) \
    if(status != rcclSuccess) { \
        std::cout<<"Got: "<<rcclGetErrorString(status)<<" at: "<<__LINE__<<" in file: "<<__FILE__<<std::endl; \
    }

