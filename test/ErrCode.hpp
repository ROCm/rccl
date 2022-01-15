/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once

namespace RcclUnitTesting
{
  typedef enum
  {
    TEST_SUCCESS = 0,
    TEST_FAIL    = 1
  } ErrCode;

#define CHECK_CALL(func)                              \
  {                                                   \
    ErrCode status = func;                            \
    if (status != TEST_SUCCESS)                       \
    {                                                 \
      fprintf(stderr, "[ERROR] in call %s\n", #func); \
      return status;                                  \
    }                                                 \
  }
}

#define CHECK_HIP(func)                                                 \
  {                                                                     \
    hipError_t error = (func);                                          \
    if (error != hipSuccess)                                            \
    {                                                                   \
      fprintf(stderr, "[ERROR] HIP error: %s\n", hipGetErrorString(error)); \
      return TEST_FAIL;                                                 \
    }                                                                   \
  }
