/*
Copyright (c) 2017 - Present Advanced Micro Devices, Inc.
All rights reserved.
*/

/**
 * @file rcclCheck.h
 * @brief This file contains macros for checking rcclResult_t and hipError_t
 *
 * Used in tests and rccl source to print proper error description
 *
 * @author Aditya Atluri
 */

#pragma once

#include <iostream>
#include "rccl.h"

#define HIPCHECK(status)                                             \
    if (status != hipSuccess) {                                      \
        std::cout << "Got: " << hipGetErrorString(status)            \
                  << " at: " << __LINE__ << " in file: " << __FILE__ \
                  << std::endl;                                      \
    }

#define RCCLCHECK(status)                                            \
    if (status != rcclSuccess) {                                     \
        std::cout << "Got: " << rcclGetErrorString(status)           \
                  << " at: " << __LINE__ << " in file: " << __FILE__ \
                  << std::endl;                                      \
    }
