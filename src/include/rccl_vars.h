/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RCCL_VARS_H_
#define RCCL_VARS_H_

#include "param.h"

RCCL_PARAM_DECLARE(EnableHipGraph);      // Opt-in environment variable for enabling hipGraph
RCCL_PARAM_DECLARE(ExperimentalXccMode); // Opt-in environment variable for enabling experimental XCC filtering

int32_t __inline__ IsValidRcclXccModeStr()
{
  char* remapTable = getenv("RCCL_EXPERIMENTAL_XCC_MODE");
  if (remapTable == NULL || strlen(remapTable) != 8 || strspn(remapTable, "01234567") != 8)
    return 0;

  for (int i = 0; i < 7; i++)
    if (!strchr(remapTable, i + '0')) return 0;

  return 1;
}

int32_t __inline__ rcclGetPreferredXcc(int srcBusIdIdx, int dstBusIdIdx)
{
  constexpr int32_t table[8][8] = {{0,7,6,1,2,4,5,3},
                                   {7,0,1,5,4,2,3,6},
                                   {5,1,0,6,7,3,2,4},
                                   {1,6,5,0,3,7,4,2},
                                   {2,4,7,3,0,5,6,1},
                                   {4,2,3,7,6,0,1,5},
                                   {5,3,2,4,6,1,0,7},
                                   {3,6,4,2,1,5,7,0}};

  if (srcBusIdIdx < 0 || srcBusIdIdx >= 8 || dstBusIdIdx < 0 || dstBusIdIdx >= 8) return -1;

  if (IsValidRcclXccModeStr())
  {
    char* remapTable = getenv("RCCL_EXPERIMENTAL_XCC_MODE");
    return table[remapTable[srcBusIdIdx]-'0'][remapTable[dstBusIdIdx]-'0'];
  }
  return -1;
}

#endif
