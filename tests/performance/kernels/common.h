/*
Copyright (c) 2017-Present Advanced Micro Devices, Inc. 
All rights reserved.
*/

#pragma once

#define HIPCHECK(status) \
    if(status != hipSuccess) { \
        std::cerr<<"Got: "<<hipGetErrorString(status)<<" at: "<<__LINE__<<" in file: "<<__FILE__<<std::endl; \
    }
