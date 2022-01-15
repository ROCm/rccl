/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#pragma once
#include "ErrCode.hpp"
#include "rccl.h"
#include "rccl_bfloat16.h"

namespace RcclUnitTesting
{
  // Performs the various basic reduction operations
  template <typename T>
  T ReduceOp(ncclRedOp_t const op, T const A, T const B)
  {
    switch (op)
    {
    case ncclSum:  return A + B;
    case ncclProd: return A * B;
    case ncclMax:  return std::max(A, B);
    case ncclMin:  return std::min(A, B);
    default:
      fprintf(stderr, "[ERROR] Unsupported reduction operator (%d)\n", op);
      exit(0);
    }
  }

  size_t DataTypeToBytes(ncclDataType_t const dataType);

  // PtrUnion encapsulates a pointer of all the different supported datatypes
  // NOTE: Currently half-precision float tests are unsupported due to half
  //       being supported on GPU only and not host
  union PtrUnion
  {
    void*          ptr;
    int8_t*        I1; // ncclInt8
    uint8_t*       U1; // ncclUint8
    int32_t*       I4; // ncclInt32
    uint32_t*      U4; // ncclUint32
    int64_t*       I8; // ncclInt64
    uint64_t*      U8; // ncclUint64
    float*         F4; // ncclFloat32
    double*        F8; // ncclFloat64
    rccl_bfloat16* B2; // ncclBfloat16

    void Attach(void *ptr);

    ErrCode AllocateGpuMem(size_t const numBytes, bool const useManagedMem);
    ErrCode AllocateCpuMem(size_t const numBytes);

    void FreeGpuMem();
    void FreeCpuMem();

    ErrCode ClearGpuMem(size_t const numBytes);
    ErrCode ClearCpuMem(size_t const numBytes);

    ErrCode FillPattern(ncclDataType_t const dataType,
                        size_t         const numElementsAllocated,
                        int            const globalRank,
                        bool           const isGpuMem);

    ErrCode Set(ncclDataType_t const dataType, int const idx, int valueI, double valueF);
    ErrCode Get(ncclDataType_t const dataType, int const idx, int& valueI, double& valueF);

    bool IsEqual(PtrUnion const& expected, ncclDataType_t const dataType, int const idx, bool const verbose);
    void Reduce(PtrUnion const& input, ncclDataType_t const dataType, size_t const idx, ncclRedOp_t const op);
    void Divide(ncclDataType_t const dataType, size_t const idx, int const divisor);
  };
}
