/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once
#include <cstring>

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
      fprintf(stderr, "\033[0;31m" "[ ERROR    ] HIP error: %s File:%s Line:%d\n" "\033[m", \
              hipGetErrorString(error), strrchr("/" __FILE__, '/') + 1, __LINE__); \
      return TEST_FAIL;                                                 \
    }                                                                   \
  }
}
