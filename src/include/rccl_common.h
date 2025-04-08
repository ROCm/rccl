/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RCCL_COMMON_H_
#define RCCL_COMMON_H_
#include "nccl_common.h"

#define RCCL_TUNABLE_COLLS 4 // LL/LL64/LL128 tunable Collectives
#define RCCL_RS_TUNABLE 0       // reduce_scatter index
#define RCCL_AG_TUNABLE 1       // all_gather index
#define RCCL_AR_TUNABLE 2       // all_reduce index
#define RCCL_RE_TUNABLE 3       // reduce index
#define RCCL_LL_LIMITS_UNDEFINED 0
#define RCCL_PROTOCOL_ENTRY_SIZE 3
#define RCCL_PROTOCOL_MIN_IDX 0
#define RCCL_PROTOCOL_MAX_IDX 1
#define RCCL_PROTOCOL_FACTOR_IDX 2

inline int getRcclTunableIndex(ncclFunc_t& func) {
  switch (func) {
    case ncclFuncReduceScatter:
      return RCCL_RS_TUNABLE;
    case ncclFuncAllGather:
      return RCCL_AG_TUNABLE;
    case ncclFuncAllReduce:
      return RCCL_AR_TUNABLE;
    case ncclFuncReduce:
      return RCCL_RE_TUNABLE;
    default:
      return -1; // Invalid or unsupported function
  }
}

#endif