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

#define ERROR(...) printf("\033[0;31m" "[ ERROR    ] " "\033[0m" __VA_ARGS__)
#define INFO(...)  printf("[ INFO     ] " __VA_ARGS__)

#define CHECK_CALL(func)                              \
  {                                                   \
    ErrCode status = func;                            \
    if (status != TEST_SUCCESS)                       \
    {                                                 \
      ERROR("Error in call %s\n", #func);             \
      return status;                                  \
    }                                                 \
  }

#define CHECK_HIP(func)                                                 \
  {                                                                     \
    hipError_t error = (func);                                          \
    if (error != hipSuccess)                                            \
    {                                                                   \
      fprintf(stderr, "\033[0;33" "[ ERROR    ] HIP error: %s\n" "\033[m", hipGetErrorString(error)); \
      return TEST_FAIL;                                                 \
    }                                                                   \
  }
}
